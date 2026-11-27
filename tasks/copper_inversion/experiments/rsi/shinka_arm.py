"""ShinkaEvolve external-arm integration for the RSI copper-inversion benchmark.

Integrates ``SakanaAI/ShinkaEvolve`` (Apache-2.0, pinned commit ``91b6e8c``)
in its dedicated virtualenv (``$R/env/shinka``) under the identical fairness
contract as LDM4Geo from scratch (protocol ``r2-pilot-h12``, Tier ``S``,
standard aggregate-only feedback):
  - Same release ``h12-formal`` Tier ``S`` (12 train bodies, 30 s per solve),
    same ``RiskOnlyService`` (``objective_v3``, case isolation, ``case_feedback='none'``).
  - Same starting program ``A0`` (``INITIAL_SOURCE`` wrapped in
    ``# EVOLVE-BLOCK-START`` / ``# EVOLVE-BLOCK-END`` comments, which
    ``budget_gate.prepare_submission`` strips prior to submission).
  - Same LLM ``deepseek-flash`` (``https://api.deepseek.com``) with
    ``reasoning_effort='high'`` and ``max_tokens=65536``.
  - Exact ``321`` service evaluations per run via ``BudgetGate`` plus a
    resume/top-up loop over ``programs.sqlite`` when proposals fail pre-eval
    or hit the duplicate cache, bounded by ``max_llm_calls = 2 * slots`` (642).
  - Top-5 training finalists -> ``freeze.py`` validation selection -> one-shot
    sealed ``unseal.py`` & ``export_outcomes.py`` under experiment ``SHINKA``.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import pwd
import random
import shutil
import signal
import sqlite3
import subprocess
import sys
import time
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from ldm_tts.engine.run_store import atomic_json_write
from tasks.copper_inversion.core.execution import digest
from tasks.copper_inversion.experiments.rsi.budget_gate import (
    DEFAULT_SLOTS,
    BudgetGate,
    prepare_submission,
    read_submissions,
)
from tasks.copper_inversion.experiments.rsi.freeze import FULL, layout
from tasks.copper_inversion.experiments.rsi.unseal import REVIEW_FILE
from tasks.copper_inversion.rsi import ledger as ledgers, protocol as protocols
from tasks.copper_inversion.rsi.client import RiskClient
from tasks.copper_inversion.rsi.controller import INITIAL_SOURCE
from tasks.copper_inversion.rsi.integrity import shared_identity
from tasks.copper_inversion.v2.public_catalog import catalog_digest, load_catalog

TASK_BRIEF_PATH = Path(__file__).resolve().parents[2] / "resources/rsi/TASK.md"
PATCH_PATH = Path(__file__).resolve().parents[2] / "resources/rsi/shinka-91b6e8c.patch"
DEFAULT_API_KEY_FILE = Path("/Aidata/shared/ldm/async-ldm-core/secrets/deepseek.key")
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-flash"
DEFAULT_THINKING = "high"
DEFAULT_MAX_TOKENS = 65536
DEFAULT_SEEDS = (401, 402, 403, 404)

SHINKA_REPO = "https://github.com/SakanaAI/ShinkaEvolve"
SHINKA_PINNED_COMMIT = "91b6e8c7ee4a773c84710a197abf873056396e45"
SHINKA_PINNED_VERSION = "0.0.8+91b6e8c"
SHINKA_LICENSE = "Apache-2.0"

RUNTIME_MODULE = "tasks.copper_inversion.experiments.v2.parallel_runtime"
LLM_CALLS_FILENAME = "shinka-llm-calls.jsonl"
PROXY_ENV_VARS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)
NUMERIC_THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}

_TELEMETRY_LOCK = None
_ORIGINAL_DEEPSEEK_SYNC = None
_ORIGINAL_DEEPSEEK_ASYNC = None
_ACTIVE_TELEMETRY = None


def wrap_initial_source(source_bytes=INITIAL_SOURCE) -> str:
    """Wrap A0 in ShinkaEvolve EVOLVE-BLOCK comment markers (stripped before evaluation)."""
    text = (
        source_bytes.decode("utf-8")
        if isinstance(source_bytes, (bytes, bytearray))
        else str(source_bytes)
    )
    if not text.endswith("\n"):
        text += "\n"
    return f"# EVOLVE-BLOCK-START\n{text}# EVOLVE-BLOCK-END\n"


def build_system_prompt(task_brief_path=TASK_BRIEF_PATH) -> str:
    """Build ShinkaEvolve's task system prompt from TASK.md (aggregate-only, EVOLVE-BLOCK contract)."""
    raw = Path(task_brief_path).read_text(encoding="utf-8")
    marker = "Write the requested complete programs to"
    core = raw.split(marker, 1)[0].strip() if marker in raw else raw.strip()
    footer = (
        "Edit the Python solver code inside the `# EVOLVE-BLOCK-START` ... `# EVOLVE-BLOCK-END` region "
        "so the module defines `solve(problem, budget, seed)`.\n"
        "Keep both `# EVOLVE-BLOCK-START` and `# EVOLVE-BLOCK-END` markers intact.\n"
        "Each solver must stay within 32,000 bytes once comments are removed and the code is re-formatted.\n"
        "The evaluator returns only aggregate `combined_score = 1.0 - risk` in [0, 1] "
        "(higher is better; failed or illegal solves score 0.0 and `correct = false`).\n"
        "No per-case metrics, tracebacks, or withheld observations are provided. "
        "Do not use external files or network access."
    )
    return core + "\n\n" + footer + "\n"


def clear_dead_proxy_env() -> None:
    """Remove ambient HTTP(S) proxy variables so DeepSeek API calls connect directly."""
    for name in PROXY_ENV_VARS:
        os.environ.pop(name, None)


def is_deepseek_offpeak(now_utc: datetime | None = None) -> tuple[bool, str]:
    """Check DeepSeek off-peak window in China Standard Time (UTC+8).

    Peak hours to avoid (frameworks-common.md): Mon-Fri 09:00-12:00 and 14:00-18:00 CST.
    """
    utc = now_utc or datetime.now(timezone.utc)
    cst = utc.astimezone(timezone(timedelta(hours=8)))
    weekday = cst.weekday()  # 0=Mon .. 6=Sun
    minutes = cst.hour * 60 + cst.minute
    stamp = cst.strftime("%Y-%m-%d %H:%M CST (weekday=%a)")
    if weekday < 5 and ((9 * 60 <= minutes < 12 * 60) or (14 * 60 <= minutes < 18 * 60)):
        return False, f"inside DeepSeek weekday peak window at {stamp}"
    return True, f"off-peak at {stamp}"


def evaluate_candidate(program_path, results_dir, gate_config_path) -> dict:
    """Subprocess evaluator entrypoint invoked by ShinkaEvolve per candidate."""
    cfg = json.loads(Path(gate_config_path).read_text(encoding="utf-8"))
    client = RiskClient(
        cfg["run_dir"],
        cfg["evaluation_socket"],
        cfg["evaluation_contract"],
        service_uid=int(cfg["evaluation_service_uid"]),
    )
    gate = BudgetGate(
        cfg["run_dir"],
        client,
        slots=int(cfg["slots"]),
        case_feedback=cfg.get("case_feedback", "none"),
        submissions_filename=cfg.get("submissions_filename", "shinka-submissions.jsonl"),
    )
    detail = gate.submit_detailed(Path(program_path))
    out = Path(results_dir)
    out.mkdir(parents=True, exist_ok=True)
    metrics_payload = {
        "combined_score": float(detail["combined_score"]),
        "public": {},
        "private": {},
    }
    correct_payload = {
        "correct": bool(detail["accepted"]),
    }
    atomic_json_write(out / "metrics.json", metrics_payload)
    atomic_json_write(out / "correct.json", correct_payload)
    return {"metrics": metrics_payload, "correct": correct_payload, "detail": detail}


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _append_jsonl_locked(path: Path, payload: dict) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.seek(0)
        count = sum(1 for line in stream if line.strip())
        row = {"call_index": count + 1, **payload}
        stream.seek(0, os.SEEK_END)
        stream.write(
            json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        )
        stream.flush()
        os.fsync(stream.fileno())
        return count + 1


def _count_jsonl_locked(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("r", encoding="utf-8") as stream:
        fcntl.flock(stream, fcntl.LOCK_SH)
        return sum(1 for line in stream if line.strip())


def install_deepseek_telemetry(calls_log_path: Path, max_llm_calls: int) -> None:
    """Wrap ShinkaEvolve's DeepSeek provider functions to record exact tokens and cap total LLM calls."""
    global _ORIGINAL_DEEPSEEK_SYNC, _ORIGINAL_DEEPSEEK_ASYNC, _ACTIVE_TELEMETRY
    from shinka.llm import query as shinka_query
    from shinka.llm.providers import deepseek as shinka_deepseek
    from shinka.llm.providers.errors import NonRetryableLLMError

    _ACTIVE_TELEMETRY = {
        "path": Path(calls_log_path),
        "max_llm_calls": int(max_llm_calls),
    }
    if _ORIGINAL_DEEPSEEK_SYNC is not None:
        return

    _ORIGINAL_DEEPSEEK_SYNC = shinka_deepseek.query_deepseek
    _ORIGINAL_DEEPSEEK_ASYNC = shinka_deepseek.query_deepseek_async

    def _check_cap():
        state = _ACTIVE_TELEMETRY
        if state is None:
            return
        used = _count_jsonl_locked(state["path"])
        if used >= state["max_llm_calls"]:
            raise NonRetryableLLMError(
                f"ShinkaEvolve LLM call cap ({state['max_llm_calls']}) reached at {used} calls"
            )

    def _record_result(res, *, error=None, kwargs=None):
        state = _ACTIVE_TELEMETRY
        if state is None:
            return
        if res is not None:
            in_tok = int(getattr(res, "input_tokens", 0) or 0)
            out_tok = int(getattr(res, "output_tokens", 0) or 0)
            think_tok = int(getattr(res, "thinking_tokens", 0) or 0)
            kw = getattr(res, "kwargs", None) or kwargs or {}
            _append_jsonl_locked(
                state["path"],
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "status": "ok",
                    "model_name": getattr(res, "model_name", None),
                    "reasoning_effort": kw.get("reasoning_effort"),
                    "max_tokens": kw.get("max_tokens"),
                    "input_tokens": in_tok,
                    "output_tokens": out_tok,
                    "thinking_tokens": think_tok,
                    "completion_tokens": out_tok + think_tok,
                    "total_tokens": in_tok + out_tok + think_tok,
                    "num_total_queries": int(getattr(res, "num_total_queries", 1) or 1),
                    "cost": float(getattr(res, "cost", 0.0) or 0.0),
                    "incomplete_reason": getattr(res, "incomplete_reason", None),
                    "content_chars": len(getattr(res, "content", "") or ""),
                },
            )
        else:
            cost_info = getattr(error, "cost_results", None) or {}
            in_tok = int(cost_info.get("input_tokens", 0) or 0)
            out_tok = int(cost_info.get("output_tokens", 0) or 0)
            think_tok = int(cost_info.get("thinking_tokens", 0) or 0)
            kw = kwargs or {}
            _append_jsonl_locked(
                state["path"],
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "status": "error",
                    "error": f"{type(error).__name__}: {error}"[:240],
                    "reasoning_effort": kw.get("reasoning_effort"),
                    "max_tokens": kw.get("max_tokens"),
                    "input_tokens": in_tok,
                    "output_tokens": out_tok,
                    "thinking_tokens": think_tok,
                    "completion_tokens": out_tok + think_tok,
                    "total_tokens": in_tok + out_tok + think_tok,
                    "num_total_queries": 1,
                    "cost": float(cost_info.get("cost", getattr(error, "cost", 0.0)) or 0.0),
                    "incomplete_reason": getattr(error, "incomplete_reason", None),
                    "content_chars": 0,
                },
            )

    def wrapped_sync(*args, **kwargs):
        _check_cap()
        try:
            res = _ORIGINAL_DEEPSEEK_SYNC(*args, **kwargs)
        except Exception as exc:
            if not isinstance(exc, NonRetryableLLMError) or "LLM call cap" not in str(exc):
                _record_result(None, error=exc, kwargs=kwargs)
            raise
        _record_result(res, kwargs=kwargs)
        return res

    async def wrapped_async(*args, **kwargs):
        _check_cap()
        try:
            res = await _ORIGINAL_DEEPSEEK_ASYNC(*args, **kwargs)
        except Exception as exc:
            if not isinstance(exc, NonRetryableLLMError) or "LLM call cap" not in str(exc):
                _record_result(None, error=exc, kwargs=kwargs)
            raise
        _record_result(res, kwargs=kwargs)
        return res

    shinka_deepseek.query_deepseek = wrapped_sync
    shinka_deepseek.query_deepseek_async = wrapped_async
    shinka_query.query_deepseek = wrapped_sync
    shinka_query.query_deepseek_async = wrapped_async


def summarize_token_usage(shinka_dir: Path) -> dict:
    """Aggregate LLM call and token counts from shinka-llm-calls.jsonl and programs.sqlite."""
    shinka_dir = Path(shinka_dir)
    rows = _read_jsonl(shinka_dir / LLM_CALLS_FILENAME)
    prompt_tokens = sum(int(r.get("input_tokens", 0)) for r in rows)
    output_tokens = sum(int(r.get("output_tokens", 0)) for r in rows)
    thinking_tokens = sum(int(r.get("thinking_tokens", 0)) for r in rows)
    completion_tokens = sum(int(r.get("completion_tokens", 0)) for r in rows)
    total_tokens = sum(int(r.get("total_tokens", 0)) for r in rows)
    ok_calls = sum(1 for r in rows if r.get("status") == "ok")
    err_calls = sum(1 for r in rows if r.get("status") != "ok")
    length_truncations = sum(
        1 for r in rows if r.get("incomplete_reason") == "max_output_tokens"
    )
    est_cost = sum(float(r.get("cost", 0.0) or 0.0) for r in rows)

    db_stats = inspect_shinka_db(shinka_dir / "programs.sqlite")
    return {
        "calls": len(rows),
        "succeeded_calls": ok_calls,
        "failed_calls": err_calls,
        "length_truncated_calls": length_truncations,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "thinking_tokens": thinking_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "estimated_cost_usd": round(est_cost, 6),
        "persisted_generations": len(db_stats["persisted_generations"]),
        "failed_proposal_generations": len(db_stats["failed_generations"]),
    }


def inspect_shinka_db(db_path: Path) -> dict:
    """Inspect programs.sqlite to count persisted generations and terminal pre-eval failures."""
    db_path = Path(db_path)
    if not db_path.is_file():
        return {"persisted_generations": set(), "failed_generations": set(), "settled_count": 0}
    persisted: set[int] = set()
    failed: set[int] = set()
    with closing(sqlite3.connect(str(db_path), timeout=30.0)) as conn:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        if "programs" in tables:
            for gen, raw_meta in conn.execute(
                "SELECT generation, metadata FROM programs"
            ).fetchall():
                meta = {}
                if raw_meta:
                    try:
                        meta = json.loads(raw_meta)
                    except json.JSONDecodeError:
                        meta = {}
                if isinstance(meta, dict) and (
                    meta.get("_is_island_copy") or meta.get("_spawned_island")
                ):
                    continue
                persisted.add(int(gen))
        if "attempt_log" in tables:
            for gen, raw_details in conn.execute(
                "SELECT generation, details FROM attempt_log WHERE status = 'failed'"
            ).fetchall():
                try:
                    details = json.loads(raw_details or "{}")
                except json.JSONDecodeError:
                    continue
                if isinstance(details, dict) and details.get("node_kind") == "failed_proposal":
                    if int(gen) not in persisted:
                        failed.add(int(gen))
    settled = persisted | failed
    return {
        "persisted_generations": persisted,
        "failed_generations": failed,
        "settled_count": (max(settled) + 1) if settled else 0,
    }


def verify_shinka_patches(model_name: str = DEFAULT_MODEL, thinking: str = DEFAULT_THINKING) -> dict:
    """Verify the two required local ShinkaEvolve patches are active in the running venv."""
    from shinka.llm.kwargs import sample_model_kwargs
    from shinka.llm.providers.model_resolver import resolve_model_backend

    resolved = resolve_model_backend(model_name)
    if resolved.provider != "deepseek" or resolved.api_model_name != model_name:
        raise RuntimeError(
            f"ShinkaEvolve catalog patch missing for {model_name!r} (got {resolved}); "
            f"apply {PATCH_PATH}"
        )
    sampled_high = sample_model_kwargs(
        model_names=model_name, max_tokens=DEFAULT_MAX_TOKENS, reasoning_efforts="high"
    )
    sampled_low = sample_model_kwargs(
        model_names=model_name, max_tokens=DEFAULT_MAX_TOKENS, reasoning_efforts="low"
    )
    if sampled_low.get("reasoning_effort") != "low" or sampled_high.get("reasoning_effort") != "high":
        raise RuntimeError(
            f"ShinkaEvolve reasoning_effort pass-through patch missing "
            f"(high={sampled_high}, low={sampled_low}); apply {PATCH_PATH}"
        )
    active = sample_model_kwargs(
        model_names=model_name, max_tokens=DEFAULT_MAX_TOKENS, reasoning_efforts=thinking
    )
    return {"resolved_provider": resolved.provider, "sampled_kwargs": active}


def build_shinka_configs(
    *,
    shinka_dir: Path,
    eval_script_path: Path,
    gate_config_path: Path,
    num_generations: int,
    model_name: str = DEFAULT_MODEL,
    thinking: str = DEFAULT_THINKING,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    meta_rec_interval: int | None = 10,
    eval_timeout_seconds: int = 600,
    task_brief_path: Path = TASK_BRIEF_PATH,
):
    """Build ShinkaEvolve EvolutionConfig, DatabaseConfig, and LocalJobConfig (no credentials stored)."""
    from shinka.core import EvolutionConfig
    from shinka.database import DatabaseConfig
    from shinka.launch import LocalJobConfig

    effort = "disabled" if thinking in (None, "", "off") else str(thinking)
    llm_kwargs = {
        "max_tokens": int(max_tokens),
        "reasoning_efforts": effort,
    }
    meta_interval = None if (meta_rec_interval is None or int(meta_rec_interval) <= 0) else int(meta_rec_interval)
    evo_config = EvolutionConfig(
        task_sys_msg=build_system_prompt(task_brief_path),
        patch_types=["diff", "full", "cross"],
        patch_type_probs=[0.6, 0.3, 0.1],
        num_generations=int(num_generations),
        max_patch_resamples=3,
        max_patch_attempts=1,
        job_type="local",
        language="python",
        llm_models=[model_name],
        llm_dynamic_selection=None,
        llm_kwargs=dict(llm_kwargs),
        meta_rec_interval=meta_interval,
        meta_llm_models=[model_name] if meta_interval else None,
        meta_llm_kwargs=dict(llm_kwargs) if meta_interval else {},
        meta_max_recommendations=5,
        sample_single_meta_rec=True,
        embedding_model=None,
        init_program_path=str((Path(shinka_dir) / "initial.py").resolve()),
        results_dir=str(Path(shinka_dir).resolve()),
        enable_wandb_logging=False,
        novelty_llm_models=None,
        use_text_feedback=False,
        evolve_prompts=False,
    )
    # ShinkaEvolve documented defaults (DatabaseConfig / island_medium.yaml).
    db_config = DatabaseConfig(
        num_islands=2,
        archive_size=40,
        elite_selection_ratio=0.3,
        num_archive_inspirations=1,
        num_top_k_inspirations=1,
        migration_interval=10,
        migration_rate=0.0,
        island_elitism=True,
        enforce_island_separation=True,
        parent_selection_strategy="weighted",
        parent_selection_lambda=10.0,
    )
    hours, rem = divmod(max(60, int(eval_timeout_seconds)), 3600)
    minutes, seconds = divmod(rem, 60)
    job_config = LocalJobConfig(
        eval_program_path=str(Path(eval_script_path).resolve()),
        extra_cmd_args={"gate_config": str(Path(gate_config_path).resolve())},
        eval_verbose=False,
        numeric_threads_per_job=1,
        time=f"{hours:02d}:{minutes:02d}:{seconds:02d}",
        python_executable=sys.executable,
    )
    return evo_config, db_config, job_config


def sanitized_hyperparameters(evo_config, db_config, job_config, *, max_eval_jobs, max_prop_jobs, max_db_workers, max_llm_calls, seed) -> dict:
    return {
        "seed": int(seed),
        "llm_models": list(evo_config.llm_models),
        "llm_dynamic_selection": evo_config.llm_dynamic_selection,
        "llm_kwargs": dict(evo_config.llm_kwargs),
        "patch_types": list(evo_config.patch_types),
        "patch_type_probs": list(evo_config.patch_type_probs),
        "max_patch_resamples": evo_config.max_patch_resamples,
        "max_patch_attempts": evo_config.max_patch_attempts,
        "meta_rec_interval": evo_config.meta_rec_interval,
        "meta_llm_models": list(evo_config.meta_llm_models) if evo_config.meta_llm_models else None,
        "meta_max_recommendations": evo_config.meta_max_recommendations,
        "embedding_model": evo_config.embedding_model,
        "novelty_llm_models": evo_config.novelty_llm_models,
        "use_text_feedback": evo_config.use_text_feedback,
        "evolve_prompts": evo_config.evolve_prompts,
        "num_islands": db_config.num_islands,
        "archive_size": db_config.archive_size,
        "elite_selection_ratio": db_config.elite_selection_ratio,
        "num_archive_inspirations": db_config.num_archive_inspirations,
        "num_top_k_inspirations": db_config.num_top_k_inspirations,
        "migration_interval": db_config.migration_interval,
        "migration_rate": db_config.migration_rate,
        "island_elitism": db_config.island_elitism,
        "enforce_island_separation": db_config.enforce_island_separation,
        "parent_selection_strategy": db_config.parent_selection_strategy,
        "parent_selection_lambda": db_config.parent_selection_lambda,
        "numeric_threads_per_job": job_config.numeric_threads_per_job,
        "eval_timeout": job_config.time,
        "max_evaluation_jobs": int(max_eval_jobs),
        "max_proposal_jobs": int(max_prop_jobs),
        "max_db_workers": int(max_db_workers),
        "max_llm_calls": int(max_llm_calls),
    }


def _write_evaluator_stub(stub_path: Path, repo_root: Path) -> None:
    content = (
        "import argparse\n"
        "import sys\n"
        f"sys.path.insert(0, {str(Path(repo_root).resolve())!r})\n"
        "from tasks.copper_inversion.experiments.rsi.shinka_arm import evaluate_candidate\n\n"
        "def main():\n"
        "    p = argparse.ArgumentParser()\n"
        "    p.add_argument('--program_path', required=True)\n"
        "    p.add_argument('--results_dir', required=True)\n"
        "    p.add_argument('--gate_config', required=True)\n"
        "    args = p.parse_args()\n"
        "    evaluate_candidate(args.program_path, args.results_dir, args.gate_config)\n\n"
        "if __name__ == '__main__':\n"
        "    main()\n"
    )
    stub_path.write_text(content, encoding="utf-8")


def run_shinka_arm(args) -> dict:
    """Execute one ShinkaEvolve external-arm run to exact slot budget and seal training."""
    if not args.allow_peak and not args.fixture_only:
        ok, msg = is_deepseek_offpeak()
        if not ok:
            raise RuntimeError(f"refusing to run during DeepSeek peak hours ({msg}); pass --allow-peak to override")

    clear_dead_proxy_env()
    os.environ.setdefault("SHINKA_PRICING_MODE", "offline")
    for k, v in NUMERIC_THREAD_ENV.items():
        os.environ[k] = v

    if not 1 <= int(args.max_evaluation_jobs) <= 16:
        raise ValueError("max_evaluation_jobs must be in 1..16 (r2-common.md CPU rule)")
    if not 1 <= int(args.max_proposal_jobs) <= 16:
        raise ValueError("max_proposal_jobs must be in 1..16 (r2-common.md CPU rule)")

    protocol = json.loads(Path(args.protocol).read_text(encoding="utf-8"))
    pid = args.protocol_id or protocols.protocol_id(protocol)
    tier_spec = protocol["tiers"][args.tier]
    judge_spec = protocol["judges"][args.judge]
    isolation = protocols.case_isolation(protocol)
    if isolation["case_feedback"] != "none":
        raise ValueError("ShinkaEvolve external arm requires case_isolation.case_feedback == 'none'")

    slots = int(args.slots) if args.slots is not None else int(tier_spec["slots"])
    default_call_mult = 3 if (args.meta_rec_interval is not None and int(args.meta_rec_interval) > 0) else 2
    max_llm_calls = (
        int(args.max_llm_calls) if args.max_llm_calls is not None else default_call_mult * slots
    )
    solver_seconds = int(tier_spec["solver_seconds"])
    finalists_count = int((protocol.get("loop") or {}).get("finalists", 5))
    catalog = load_catalog(args.public_catalog)
    public_digest = catalog_digest(args.public_catalog)

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    shinka_dir = out_dir / "shinka"
    shinka_dir.mkdir(parents=True, exist_ok=True)

    client = RiskClient(
        out_dir,
        args.evaluation_socket,
        args.evaluation_contract,
        service_uid=args.evaluation_service_uid,
    )
    raw_info = client.request({"operation": "info"})
    expected_service_slots = int(tier_spec["slots"])
    if args.slots is not None and isinstance(raw_info, dict) and raw_info.get("slots") == int(args.slots):
        expected_service_slots = int(args.slots)
    info = client.verify_contract(
        public_digest,
        solver_seconds,
        expected_service_slots,
        objective=judge_spec.get("objective", "confirmation"),
        tier=args.tier,
        import_ledger_prefix_sha256=None,
        case_isolation=isolation,
    )
    existing_ledger = ledgers.read(out_dir / "ledger.jsonl")
    if not args.resume and (info["used_slots"] != 0 or existing_ledger):
        raise ValueError("a new ShinkaEvolve run requires an unused private evaluator (or pass --resume)")
    if info["used_slots"] != len(existing_ledger):
        raise ValueError(
            f"service used_slots ({info['used_slots']}) != public ledger ({len(existing_ledger)})"
        )

    # Read API key in process memory only; never print, log, or write to disk.
    api_key = Path(args.api_key_file).read_text(encoding="utf-8").strip()
    if not api_key:
        raise ValueError("protected model API key file is empty")
    os.environ["DEEPSEEK_API_KEY"] = api_key

    model_cfg = protocol.get("model") or {}
    base_url = args.base_url or model_cfg.get("base_url") or DEFAULT_BASE_URL
    model_name = args.model or model_cfg.get("name") or DEFAULT_MODEL
    thinking = args.thinking if args.thinking is not None else model_cfg.get("thinking", DEFAULT_THINKING)
    if base_url.rstrip("/") != DEFAULT_BASE_URL:
        raise ValueError(f"ShinkaEvolve DeepSeek provider expects {DEFAULT_BASE_URL}, got {base_url!r}")

    patch_check = verify_shinka_patches(model_name=model_name, thinking=thinking)

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))

    initial_path = shinka_dir / "initial.py"
    wrapped_a0 = wrap_initial_source(INITIAL_SOURCE)
    if not initial_path.is_file():
        initial_path.write_text(wrapped_a0, encoding="utf-8")
    _, a0_sha = prepare_submission(INITIAL_SOURCE)
    _, wrapped_sha = prepare_submission(wrapped_a0.encode("utf-8"))
    if a0_sha != wrapped_sha:
        raise RuntimeError("EVOLVE-BLOCK wrapper altered A0 canonical SHA-256")

    gate_cfg_path = shinka_dir / "gate-config.json"
    atomic_json_write(
        gate_cfg_path,
        {
            "run_dir": str(out_dir),
            "evaluation_socket": str(Path(args.evaluation_socket).resolve()),
            "evaluation_contract": args.evaluation_contract,
            "evaluation_service_uid": int(args.evaluation_service_uid),
            "slots": slots,
            "case_feedback": isolation["case_feedback"],
            "submissions_filename": "shinka-submissions.jsonl",
        },
    )
    repo_root = Path(__file__).resolve().parents[4]
    eval_stub_path = shinka_dir / "evaluate_candidate.py"
    _write_evaluator_stub(eval_stub_path, repo_root)

    evo_preview, db_preview, job_preview = build_shinka_configs(
        shinka_dir=shinka_dir,
        eval_script_path=eval_stub_path,
        gate_config_path=gate_cfg_path,
        num_generations=slots,
        model_name=model_name,
        thinking=thinking,
        max_tokens=args.llm_max_tokens,
        meta_rec_interval=args.meta_rec_interval,
        eval_timeout_seconds=args.eval_timeout,
    )
    atomic_json_write(
        shinka_dir / "config-sanitized.json",
        {
            **sanitized_hyperparameters(
                evo_preview,
                db_preview,
                job_preview,
                max_eval_jobs=args.max_evaluation_jobs,
                max_prop_jobs=args.max_proposal_jobs,
                max_db_workers=args.max_db_workers,
                max_llm_calls=max_llm_calls,
                seed=args.seed,
            ),
            "patch_verification": patch_check,
        },
    )

    import importlib.metadata

    try:
        shinka_version = importlib.metadata.version("shinka-evolve")
    except Exception:
        shinka_version = SHINKA_PINNED_VERSION

    manifest_path = out_dir / "manifest.json"
    if not manifest_path.is_file():
        manifest = {
            "schema": "rsi-run-manifest-v1",
            "external_system": "shinka_evolve",
            "shinka_repo": SHINKA_REPO,
            "shinka_version": shinka_version,
            "shinka_commit": SHINKA_PINNED_COMMIT,
            "shinka_license": SHINKA_LICENSE,
            "shinka_patch_sha256": hashlib.sha256(PATCH_PATH.read_bytes()).hexdigest(),
            "run_id": out_dir.name,
            "experiment": args.experiment,
            "protocol_id": pid,
            "protocol_sha256": protocols.protocol_sha256(protocol),
            "arm": "shinka",
            "tier": args.tier,
            "judge": args.judge,
            "seed": int(args.seed),
            "slots": slots,
            "max_llm_calls": max_llm_calls,
            "solver_seconds": solver_seconds,
            "finalists": finalists_count,
            "model": {"base_url": base_url, "name": model_name, "thinking": thinking},
            "case_isolation": isolation,
            "public_digest": public_digest,
            "shared_implementation_sha256": shared_identity(),
            "code_commit": protocols.code_commit(),
            "stage": catalog["stage"],
            "synthetic_test": bool(args.fixture_only),
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        atomic_json_write(manifest_path, manifest)

    calls_log_path = shinka_dir / LLM_CALLS_FILENAME
    install_deepseek_telemetry(calls_log_path, max_llm_calls=max_llm_calls)

    from shinka.core import ShinkaEvolveRunner
    from shinka.llm.providers.errors import NonRetryableLLMError

    started = time.monotonic()
    passes = 0
    stop_reason = "completed"
    max_topup_passes = max(8, slots + 4)

    while True:
        used_slots = len(ledgers.read(out_dir / "ledger.jsonl"))
        if used_slots >= slots:
            stop_reason = "completed"
            break
        used_calls = _count_jsonl_locked(calls_log_path)
        if used_calls >= max_llm_calls:
            stop_reason = "llm_call_cap_reached"
            break
        if passes >= max_topup_passes:
            stop_reason = "topup_pass_cap_reached"
            break

        passes += 1
        remaining_slots = slots - used_slots
        db_state = inspect_shinka_db(shinka_dir / "programs.sqlite")
        settled_count = db_state["settled_count"]
        target_generations = max(settled_count + remaining_slots, slots)

        evo_cfg, db_cfg, job_cfg = build_shinka_configs(
            shinka_dir=shinka_dir,
            eval_script_path=eval_stub_path,
            gate_config_path=gate_cfg_path,
            num_generations=target_generations,
            model_name=model_name,
            thinking=thinking,
            max_tokens=args.llm_max_tokens,
            meta_rec_interval=args.meta_rec_interval,
            eval_timeout_seconds=args.eval_timeout,
        )
        runner = ShinkaEvolveRunner(
            evo_config=evo_cfg,
            job_config=job_cfg,
            db_config=db_cfg,
            verbose=bool(args.verbose),
            max_evaluation_jobs=int(args.max_evaluation_jobs),
            max_proposal_jobs=int(args.max_proposal_jobs),
            max_db_workers=int(args.max_db_workers),
        )
        try:
            runner.run()
        except NonRetryableLLMError as exc:
            if "LLM call cap" in str(exc):
                stop_reason = "llm_call_cap_reached"
                break
            raise

        new_used_slots = len(ledgers.read(out_dir / "ledger.jsonl"))
        new_db_state = inspect_shinka_db(shinka_dir / "programs.sqlite")
        if new_used_slots == used_slots and new_db_state["settled_count"] == settled_count:
            stop_reason = "no_progress"
            break

    wall_seconds = time.monotonic() - started
    records = ledgers.read(out_dir / "ledger.jsonl")
    submissions = read_submissions(out_dir / "shinka-submissions.jsonl")
    usage = summarize_token_usage(shinka_dir)
    usage["topup_passes"] = passes
    atomic_json_write(out_dir / "usage.json", usage)

    valid_records = [r for r in records if r["status"] == "valid"]
    finalists = sorted(
        valid_records,
        key=lambda r: (r["risk"], -r["legal_rate"], r["seconds_per_solve"], r["program_id"]),
    )[:finalists_count]
    training_finalists = [r["program_id"] for r in finalists]

    sealed = client.seal_training()
    if sealed["used_slots"] != len(records) or sealed["source_sha256s"] != training_finalists:
        raise ValueError(
            f"public and private training records or finalists differ: "
            f"sealed_slots={sealed['used_slots']} ledger={len(records)} "
            f"sealed_finalists={sealed['source_sha256s']} ledger_finalists={training_finalists}"
        )

    status = "completed" if len(records) == slots else "stopped"
    result = {
        "status": status,
        "stop_reason": stop_reason,
        "external_system": "shinka_evolve",
        "shinka_version": shinka_version,
        "shinka_commit": SHINKA_PINNED_COMMIT,
        "charged_records": len(records),
        "target_slots": slots,
        "counted_submissions": sum(1 for s in submissions if s.get("counted", False)),
        "total_submissions": len(submissions),
        "pre_eval_rejected_submissions": sum(
            1 for s in submissions if s.get("status") in ("invalid_source", "oversized")
        ),
        "duplicate_submissions": sum(1 for s in submissions if s.get("status") == "duplicate"),
        "valid_records": len(valid_records),
        "best_risk": finalists[0]["risk"] if finalists else 1.0,
        "training_finalists": training_finalists,
        "topup_passes": passes,
        "token_usage": usage,
        "wall_seconds": round(wall_seconds, 2),
        "scope": (
            "dry run / smoke test only; not a scientific result"
            if args.fixture_only
            else "training feedback only; validation selection and sealed evaluation not run"
        ),
    }
    atomic_json_write(out_dir / "result.json", result)
    atomic_json_write(
        out_dir / "run-status.json",
        {"status": status, "stop_reason": stop_reason},
    )
    return result


# Multi-seed batch orchestrator (root operator, mirrors launch.py) ------------------


def _readable(root: Path) -> None:
    for path in [Path(root), *Path(root).rglob("*")]:
        if not path.is_symlink():
            path.chmod(path.stat().st_mode | (0o555 if path.is_dir() else 0o444))


def _account(user: str) -> tuple[int, int, list[int]]:
    entry = pwd.getpwnam(user)
    return entry.pw_uid, entry.pw_gid, os.getgrouplist(user, entry.pw_gid)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _stop_pg(pid: int) -> None:
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass


class ShinkaBatchLauncher:
    """Trusted-operator 4-seed launcher for experiment SHINKA (mirrors launch.py)."""

    def __init__(
        self,
        site_path: Path,
        *,
        protocol_path: Path | None = None,
        shinka_python: Path | None = None,
        source_dir: Path | None = None,
        experiment: str = "SHINKA",
        tier: str = "S",
        judge: str = "standard",
        seeds: list[int] = list(DEFAULT_SEEDS),
        slots: int | None = None,
        max_eval_jobs: int = 3,
        max_prop_jobs: int = 3,
        batch_name: str = "r2-h12-shinka",
        allow_peak: bool = False,
    ):
        self.site = json.loads(Path(site_path).read_text(encoding="utf-8"))
        self.root = Path(self.site["root"]).resolve()
        self.source = Path(source_dir or Path(__file__).resolve().parents[4]).resolve()
        self.shinka_python = Path(os.path.abspath(shinka_python or sys.executable))
        default_proto = self.root / "protocols/r2-pilot-h12/protocol.json"
        self.protocol_path = Path(protocol_path or default_proto).resolve()
        self.protocol = protocols.load(self.protocol_path)
        self.pid = protocols.protocol_id(self.protocol)
        self.policy_path = Path(self.site["policy"]).resolve()
        self.policy = json.loads(self.policy_path.read_text(encoding="utf-8"))
        self.experiment = experiment
        self.tier = tier
        self.judge = judge
        self.seeds = [int(s) for s in seeds]
        self.slots = int(slots) if slots is not None else int(self.protocol["tiers"][tier]["slots"])
        self.max_eval_jobs = int(max_eval_jobs)
        self.max_prop_jobs = int(max_prop_jobs)
        self.allow_peak = bool(allow_peak)
        self.batch_name = batch_name
        self.home = self.root / "launcher" / batch_name
        self.state_path = self.home / "state.json"

    def run_ids(self) -> list[tuple[str, int]]:
        return [(f"shinka-s{seed}", seed) for seed in self.seeds]

    def paths(self, run_id: str) -> dict[str, Path]:
        p = layout(self.root, self.pid, self.experiment, run_id)
        p["socket_dir"] = self.root / "rpc" / digest([self.pid, self.experiment, run_id])[:12]
        return p

    def _base_env(self) -> dict[str, str]:
        env = {
            k: v
            for k, v in os.environ.items()
            if k not in ("LLM_API_KEY", "DEEPSEEK_API_KEY", *PROXY_ENV_VARS)
        }
        env.update(
            PYTHONPATH=str(self.source),
            PYTHONDONTWRITEBYTECODE="1",
            SHINKA_PRICING_MODE="offline",
            **NUMERIC_THREAD_ENV,
        )
        return env

    def _spawn_as(self, user: str, argv: list[str], *, env: dict[str, str], log: Path, nice_level: int = 10):
        uid, gid, groups = _account(user)
        cmd = ["nice", "-n", str(nice_level), *argv]
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("ab") as stream:
            os.chown(log, uid, gid)
            return subprocess.Popen(
                cmd,
                cwd=str(self.source),
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=stream,
                stderr=subprocess.STDOUT,
                user=uid,
                group=gid,
                extra_groups=groups,
                start_new_session=True,
            )

    def start_service(self, run_id: str) -> dict:
        paths = self.paths(run_id)
        private, sockets = paths["service"], paths["socket_dir"]
        uid, gid, _ = _account(self.site["service_user"])
        client_uid, client_gid, _ = _account(self.site["client_user"])
        config_path = private / "service-config.json"
        socket_path = sockets / "e.sock"
        if not config_path.exists():
            self.root.mkdir(parents=True, exist_ok=True)
            for folder in (self.root / "private", self.root / "private" / self.pid, private.parent):
                if not folder.exists():
                    folder.mkdir(parents=True, mode=0o700)
                    os.chown(folder, uid, gid)
            private.mkdir(mode=0o700)
            os.chown(private, uid, gid)
            sockets.mkdir(parents=True, mode=0o750)
            os.chown(sockets, uid, client_gid)
            tier_spec = self.protocol["tiers"][self.tier]
            judge_spec = self.protocol["judges"][self.judge]
            manifest_rel = judge_spec.get("train_manifest", tier_spec["train_manifest"])
            config = {
                "manifest": str((Path(self.site["release"]) / manifest_rel).resolve()),
                "root": str(private),
                "image": self.site["image"],
                "seconds": int(tier_spec["solver_seconds"]),
                "slots": int(tier_spec["slots"]),
                "caller_uid": client_uid,
                "socket": str(socket_path),
                "objective": judge_spec["objective"],
                "tier": self.tier,
            }
            if "resample" in judge_spec:
                config["resample"] = dict(judge_spec["resample"])
            if "confirmation_packs" in judge_spec:
                packs = judge_spec["confirmation_packs"]
                config["confirmation_packs"] = list(packs) if isinstance(packs, list) else packs
            if self.protocol["loop"]["finalists"] != 3:
                config["finalists"] = int(self.protocol["loop"]["finalists"])
            if "case_isolation" in self.protocol:
                config["case_isolation"] = protocols.case_isolation(self.protocol)
            config_path.write_text(json.dumps(config, sort_keys=True), encoding="utf-8")
            os.chown(config_path, uid, gid)
            config_path.chmod(0o600)
            for folder in (private / "tmp", private / "tmp" / "docker"):
                folder.mkdir(mode=0o700)
                os.chown(folder, uid, gid)
        elif socket_path.exists():
            socket_path.unlink()

        env = {
            **self._base_env(),
            "TMPDIR": str(private / "tmp"),
            "DOCKER_CONFIG": str(private / "tmp" / "docker"),
        }
        log = private / "service.log"
        argv = [
            str(self.site["service_python"]),
            "-m",
            RUNTIME_MODULE,
            "--policy",
            str(self.policy_path),
            "--tier",
            self.tier,
            "rsi-service",
            "--config",
            str(config_path),
        ]
        proc = self._spawn_as(self.site["service_user"], argv, env=env, log=log)
        deadline = time.monotonic() + 120.0
        ready = None
        while ready is None:
            if socket_path.exists() and log.exists():
                for line in log.read_text(encoding="utf-8").splitlines():
                    if line.startswith("{") and '"listening"' in line:
                        row = json.loads(line)
                        if row.get("pid") == proc.pid:
                            ready = row
            if ready is None:
                if proc.poll() is not None or time.monotonic() >= deadline:
                    if proc.poll() is None:
                        _stop_pg(proc.pid)
                    raise RuntimeError(f"evaluator service did not start: {run_id}")
                time.sleep(0.2)
        os.chown(socket_path, uid, client_gid)
        socket_path.chmod(0o660)
        deployment = {
            "socket": str(socket_path),
            "contract_digest": ready["contract_digest"],
            "service_uid": uid,
            "pid": proc.pid,
        }
        (sockets / "deployment.json").write_text(json.dumps(deployment, sort_keys=True) + "\n", encoding="utf-8")
        return deployment

    def start_controller(self, run_id: str, seed: int, deployment: dict, *, resume: bool):
        paths = self.paths(run_id)
        client_uid, client_gid, _ = _account(self.site["client_user"])
        if not paths["runs"].exists():
            paths["runs"].mkdir(parents=True)
            for folder in [paths["runs"], *paths["runs"].parents]:
                if folder == self.root:
                    break
                os.chown(folder, client_uid, client_gid)
        argv = [
            str(self.shinka_python),
            "-m",
            "tasks.copper_inversion.experiments.rsi.shinka_arm",
            "run",
            "--protocol",
            str(self.protocol_path),
            "--experiment",
            self.experiment,
            "--tier",
            self.tier,
            "--judge",
            self.judge,
            "--seed",
            str(seed),
            "--slots",
            str(self.slots),
            "--out-dir",
            str(paths["run"]),
            "--public-catalog",
            str(self.site["public_catalogs"][self.tier]),
            "--evaluation-socket",
            deployment["socket"],
            "--evaluation-contract",
            deployment["contract_digest"],
            "--evaluation-service-uid",
            str(deployment["service_uid"]),
            "--api-key-file",
            str(self.site["api_key_file"]),
            "--max-evaluation-jobs",
            str(self.max_eval_jobs),
            "--max-proposal-jobs",
            str(self.max_prop_jobs),
        ]
        if resume:
            argv.append("--resume")
        if self.allow_peak:
            argv.append("--allow-peak")
        log = paths["runs"] / f"{run_id}.controller.log"
        return self._spawn_as(self.site["client_user"], argv, env=self._base_env(), log=log)

    def _save_state(self, state: dict) -> None:
        self.home.mkdir(parents=True, exist_ok=True)
        state["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        atomic_json_write(self.state_path, state)
        self.state_path.chmod(0o644)

    def start(self) -> dict:
        if os.geteuid() != 0:
            raise PermissionError("batch start runs as root (spawns ldm-copper-eval services + zsgpu controllers)")
        if not self.allow_peak:
            ok, msg = is_deepseek_offpeak()
            if not ok:
                raise RuntimeError(f"refusing to start during DeepSeek peak hours ({msg})")
        if self.state_path.exists():
            raise ValueError(f"batch state already exists at {self.state_path}; use resume")
        disk_probe = self.root
        while not disk_probe.exists() and disk_probe != disk_probe.parent:
            disk_probe = disk_probe.parent
        free_gib = shutil.disk_usage(disk_probe).free / (1024**3)
        if free_gib < 150.0:
            raise RuntimeError(f"insufficient free disk space ({free_gib:.1f} GiB < 150 GiB)")
        if self.max_eval_jobs * len(self.seeds) > 16 or self.max_prop_jobs * len(self.seeds) > 16:
            raise ValueError("aggregate jobs across seeds must stay <= 16 (r2-common.md)")

        state = {
            "schema": "rsi-shinka-batch-v1",
            "batch": self.batch_name,
            "experiment": self.experiment,
            "protocol_id": self.pid,
            "seeds": self.seeds,
            "slots": self.slots,
            "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "runs": {},
        }
        processes = {}
        for run_id, seed in self.run_ids():
            dep = self.start_service(run_id)
            state["runs"][run_id] = {"seed": seed, "service": dep, "status": "starting"}
            self._save_state(state)
        for run_id, seed in self.run_ids():
            proc = self.start_controller(run_id, seed, state["runs"][run_id]["service"], resume=False)
            processes[run_id] = proc
            state["runs"][run_id].update(controller_pid=proc.pid, status="running")
            self._save_state(state)
        return self._monitor(state, processes)

    def resume(self) -> dict:
        if os.geteuid() != 0:
            raise PermissionError("batch resume runs as root")
        if not self.allow_peak:
            ok, msg = is_deepseek_offpeak()
            if not ok:
                raise RuntimeError(f"refusing to resume during DeepSeek peak hours ({msg})")
        state = json.loads(self.state_path.read_text(encoding="utf-8"))
        processes = {}
        for run_id, seed in self.run_ids():
            entry = state["runs"][run_id]
            if entry["status"] == "finished":
                continue
            if not _alive(entry["service"]["pid"]):
                restarted = self.start_service(run_id)
                if restarted["contract_digest"] != entry["service"]["contract_digest"]:
                    raise RuntimeError(f"restarted service contract mismatch: {run_id}")
                entry["service"] = restarted
            proc = self.start_controller(
                run_id, seed, entry["service"], resume=self.paths(run_id)["run"].exists()
            )
            processes[run_id] = proc
            entry.update(controller_pid=proc.pid, status="running", resumes=entry.get("resumes", 0) + 1)
            self._save_state(state)
        return self._monitor(state, processes)

    def _monitor(self, state: dict, processes: dict) -> dict:
        while True:
            running = [rid for rid, p in processes.items() if p.poll() is None]
            for rid, proc in processes.items():
                if proc.poll() is not None and state["runs"][rid]["status"] == "running":
                    st_path = self.paths(rid)["run"] / "run-status.json"
                    st = json.loads(st_path.read_text(encoding="utf-8")) if st_path.exists() else {}
                    ok = proc.returncode == 0 and st.get("status") == "completed"
                    state["runs"][rid].update(
                        status="finished" if ok else "needs_resume",
                        exit_code=proc.returncode,
                        run_status=st,
                    )
            self._save_state(state)
            if not running:
                break
            time.sleep(int(self.site.get("poll_seconds", 15)))
        failed = sorted(r for r, e in state["runs"].items() if e["status"] != "finished")
        state["stage"] = "runs_finished" if not failed else "needs_resume"
        self._save_state(state)
        if failed:
            raise RuntimeError("unfinished ShinkaEvolve runs (re-run batch resume): " + ", ".join(failed))
        return state

    def finish(self) -> dict:
        if os.geteuid() != 0:
            raise PermissionError("batch finish runs as root")
        from tasks.copper_inversion.experiments.rsi import q1
        from tasks.copper_inversion.experiments.rsi.launch import reap
        from tasks.copper_inversion.rsi import audit

        state = json.loads(self.state_path.read_text(encoding="utf-8"))
        unfinished = sorted(r for r, e in state["runs"].items() if e["status"] != "finished")
        if unfinished:
            raise RuntimeError("unfinished runs: " + ", ".join(unfinished))
        uid, gid, _ = _account(self.site["service_user"])
        reconciliation = {}
        for run_id, _ in self.run_ids():
            paths = self.paths(run_id)
            budget_res = audit.reconcile_budget(paths["run"], paths["service"])
            if budget_res["charged_sources"] != self.slots:
                raise ValueError(
                    f"{run_id}: expected exactly {self.slots} charged evaluations, got {budget_res['charged_sources']}"
                )
            blind = audit.scan_blind(
                paths["run"], paths["service"], require_requests=False, key_file=self.site["api_key_file"]
            )
            if blind["review_required"]:
                rev_path = paths["service"] / REVIEW_FILE
                atomic_json_write(
                    rev_path,
                    {
                        "run_id": run_id,
                        "review_sha256": blind["review_sha256"],
                        "hits": blind["review_required"],
                    },
                )
                os.chown(rev_path, uid, gid)
            reconciliation[run_id] = {
                "budget": budget_res,
                "blind": {
                    k: blind[k]
                    for k in (
                        "status",
                        "files_checked",
                        "marker_hits",
                        "iou_hits_by_kind",
                        "leaks",
                        "review_required",
                        "review_sha256",
                    )
                },
            }
            _readable(paths["run"])
        state["reconciliation"] = reconciliation
        self._save_state(state)

        procs = {}
        for run_id, _ in self.run_ids():
            argv = [
                str(self.site["service_python"]),
                "-m",
                RUNTIME_MODULE,
                "--policy",
                str(self.policy_path),
                "--tier",
                self.tier,
                "rsi-freeze",
                "--root",
                str(self.root),
                "--protocol-id",
                self.pid,
                "--experiment",
                self.experiment,
                "--run",
                run_id,
                "--release",
                str(self.site["release"]),
                "--protocol",
                str(self.protocol_path),
                "--prefix",
                FULL,
            ]
            log = self.paths(run_id)["service"] / "freeze.log"
            procs[run_id] = self._spawn_as(self.site["service_user"], argv, env=self._base_env(), log=log)
        codes = {rid: p.wait() for rid, p in procs.items()}
        if any(codes.values()):
            state["stage"] = "freeze_failed"
            state["freeze_exit_codes"] = codes
            self._save_state(state)
            raise RuntimeError(f"freeze/validation failed: {codes}")

        exp_paths = layout(self.root, self.pid, self.experiment)
        solvers = [
            {"run_id": run_id, "prefix": FULL}
            for run_id, _ in self.run_ids()
            if (exp_paths["selection"] / run_id / FULL / "prefix.json").is_file()
        ]
        target = self.root / "private" / self.pid / "frozen-lists" / f"{self.experiment}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        atomic_json_write(
            target,
            {"experiment": self.experiment, "protocol_id": self.pid, "solvers": solvers},
        )
        for path in (target.parent, target):
            os.chown(path, uid, gid)

        ref_dir = self.root / "private" / self.pid / "refshinka-inputs"
        ref_dir.mkdir(parents=True, exist_ok=True)
        refs, selections = {}, {}
        for run_id, seed in self.run_ids():
            paths = self.paths(run_id)
            sel_target = paths["run_selection"] / FULL
            report = json.loads((sel_target / "selection.json").read_text(encoding="utf-8"))
            selected_sha = report["selected_sha256"]
            src_file = paths["run"] / "candidates" / selected_sha / "solver.py"
            refs[run_id] = {"source": str(src_file.resolve()), "sha256": selected_sha, "seed": int(seed)}
            selections[run_id] = str(sel_target.resolve())
        refs_path = ref_dir / "references.json"
        sels_path = ref_dir / "selections.json"
        atomic_json_write(refs_path, refs)
        atomic_json_write(sels_path, selections)
        q1.references(refs_path)
        for path in (ref_dir, refs_path, sels_path):
            os.chown(path, uid, gid)

        for entry in state["runs"].values():
            _stop_pg(entry["service"]["pid"])
        state["frozen_list"] = str(target)
        state["reference_unseal_inputs"] = {
            "reference_experiment": "REFSHINKA",
            "references": str(refs_path),
            "selections": str(sels_path),
        }
        state["containers"] = reap(self.policy["queue_root"])
        state["stage"] = "frozen_and_validated"
        state["finished_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self._save_state(state)
        return state


def _add_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--protocol-id", default=None)
    parser.add_argument("--experiment", default="SHINKA", choices=protocols.EXPERIMENTS)
    parser.add_argument("--tier", default="S", choices=("S", "M", "F"))
    parser.add_argument("--judge", default="standard")
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--public-catalog", required=True, type=Path)
    parser.add_argument("--evaluation-socket", required=True, type=Path)
    parser.add_argument("--evaluation-contract", required=True)
    parser.add_argument("--evaluation-service-uid", required=True, type=int)
    parser.add_argument("--api-key-file", type=Path, default=DEFAULT_API_KEY_FILE)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--thinking", default=None)
    parser.add_argument("--slots", type=int, default=None)
    parser.add_argument("--max-llm-calls", type=int, default=None)
    parser.add_argument("--llm-max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--meta-rec-interval", type=int, default=10)
    parser.add_argument("--max-evaluation-jobs", type=int, default=4)
    parser.add_argument("--max-proposal-jobs", type=int, default=4)
    parser.add_argument("--max-db-workers", type=int, default=2)
    parser.add_argument("--eval-timeout", type=int, default=600)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-peak", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--fixture-only", action="store_true")


def main(argv=None) -> int:
    args_list = list(sys.argv[1:] if argv is None else argv)
    # Direct subprocess evaluator mode (`--program_path ... --results_dir ... --gate_config ...`)
    if "--program_path" in args_list and (not args_list or args_list[0] not in ("run", "evaluate", "batch")):
        args_list = ["evaluate", *args_list]
    elif args_list and args_list[0].startswith("--") and args_list[0] not in ("-h", "--help"):
        args_list = ["run", *args_list]

    parser = argparse.ArgumentParser(description="ShinkaEvolve external-arm runner & batch launcher.")
    sub = parser.add_subparsers(dest="subcommand", required=True)

    p_run = sub.add_parser("run", help="Run one ShinkaEvolve seed against a live RiskOnlyService.")
    _add_run_args(p_run)

    p_eval = sub.add_parser("evaluate", help="Subprocess evaluator called by ShinkaEvolve.")
    p_eval.add_argument("--program_path", required=True, type=Path)
    p_eval.add_argument("--results_dir", required=True, type=Path)
    p_eval.add_argument("--gate_config", required=True, type=Path)

    p_batch = sub.add_parser("batch", help="Start, resume, finish, or inspect multi-seed SHINKA batch.")
    p_batch.add_argument("action", choices=("start", "resume", "finish", "status"))
    p_batch.add_argument("--site", required=True, type=Path)
    p_batch.add_argument("--protocol", type=Path, default=None)
    p_batch.add_argument("--shinka-python", type=Path, default=None)
    p_batch.add_argument("--source", type=Path, default=None)
    p_batch.add_argument("--experiment", default="SHINKA", choices=protocols.EXPERIMENTS)
    p_batch.add_argument("--tier", default="S", choices=("S", "M", "F"))
    p_batch.add_argument("--judge", default="standard")
    p_batch.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    p_batch.add_argument("--slots", type=int, default=None)
    p_batch.add_argument("--max-evaluation-jobs", type=int, default=3)
    p_batch.add_argument("--max-proposal-jobs", type=int, default=3)
    p_batch.add_argument("--batch-name", default="r2-h12-shinka")
    p_batch.add_argument("--allow-peak", action="store_true")

    args = parser.parse_args(args_list)
    if args.subcommand == "evaluate":
        evaluate_candidate(args.program_path, args.results_dir, args.gate_config)
        return 0
    if args.subcommand == "run":
        result = run_shinka_arm(args)
        print(json.dumps({"run_dir": str(Path(args.out_dir).resolve()), **result}, sort_keys=True))
        return 0
    launcher = ShinkaBatchLauncher(
        args.site,
        protocol_path=args.protocol,
        shinka_python=args.shinka_python,
        source_dir=args.source,
        experiment=args.experiment,
        tier=args.tier,
        judge=args.judge,
        seeds=args.seeds,
        slots=args.slots,
        max_eval_jobs=args.max_evaluation_jobs,
        max_prop_jobs=args.max_proposal_jobs,
        batch_name=args.batch_name,
        allow_peak=args.allow_peak,
    )
    if args.action == "status":
        out = (
            json.loads(launcher.state_path.read_text(encoding="utf-8"))
            if launcher.state_path.exists()
            else {"status": "not_started"}
        )
    else:
        out = getattr(launcher, args.action)()
    print(json.dumps(out, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
