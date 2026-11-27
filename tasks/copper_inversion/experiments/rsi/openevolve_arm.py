"""OpenEvolve external-arm integration for the RSI copper-inversion benchmark.

Submits OpenEvolve-generated solver programs through the shared 321-evaluation
``BudgetGate`` and ``RiskOnlyService`` contract (comment stripping, 32 kB
normalized-code limit, interface checks, exact 321 service evaluations with a
checkpoint top-up loop and LLM-call safety cap, aggregate-only feedback with
artefacts disabled) and writes a ledger/run layout compatible with
``experiments/rsi/freeze.py`` (top-5 training finalists -> standard validation ->
one-shot ``REFOE`` sealed test evaluation).

OpenEvolve (``algorithmicsuperintelligence/openevolve``, Apache-2.0) runs in its
own venv (``$R/env/openevolve``) pinned at commit ``0d4a580`` (PR #599 truncation
logging + PR #620 Responses API support).
"""

import argparse
import asyncio
import hashlib
import json
import os
import re
import socket
import time
from datetime import datetime, timezone
from pathlib import Path

from ldm_tts.engine.run_store import atomic_json_write
from tasks.copper_inversion.core.encoding import MAX_NORMALIZED_BYTES
from tasks.copper_inversion.experiments.rsi.budget_gate import (
    DEFAULT_SLOTS,
    BudgetGate,
    append_submission as _append_submission,
    prepare_submission,
    read_submissions as _read_submissions,
)
from tasks.copper_inversion.rsi import ledger as ledgers, protocol as protocols
from tasks.copper_inversion.rsi.client import RiskClient
from tasks.copper_inversion.rsi.controller import INITIAL_SOURCE
from tasks.copper_inversion.rsi.integrity import shared_identity
from tasks.copper_inversion.v2.public_catalog import catalog_digest, load_catalog

TASK_BRIEF_PATH = Path(__file__).resolve().parents[2] / "resources/rsi/TASK.md"
DEFAULT_API_KEY_FILE = Path("/Aidata/shared/ldm/async-ldm-core/secrets/deepseek.key")
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-flash"
DEFAULT_THINKING = "high"
DEFAULT_API_TYPE = "responses"
DEFAULT_LLM_MAX_TOKENS = 65536
DEFAULT_MAX_ITERATIONS = DEFAULT_SLOTS - 1
MAX_PARALLEL_EVALUATIONS = 16
OPENEVOLVE_PINNED_COMMIT = "0d4a5807499be3446eba4069932b80c27bcfab4b"
OPENEVOLVE_PINNED_VERSION = "0.4.1+0d4a580"
OPENEVOLVE_LICENSE = "Apache-2.0"

THREAD_ENV_CAPS = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}

PUBLIC_CONTRACT_SUMMARY = (
    "Public problem & helper contract (`tasks/copper_inversion/v2/`):\n"
    "- `problem`: dict with keys `'schema_version'`, `'geometry_version'`, `'objective_version'`, `'case_id'`, "
    "`'shape'` (3 ints `[nx, ny, nz]`), `'origin'` (3 floats `[ox, oy, oz]`), `'spacing'` (3 positive floats `[dx, dy, dz]`), "
    "`'domain_mask'` (bool 3D ndarray of `shape`), `'forbidden_mask'` (bool 3D ndarray of `shape`), "
    "`'open_boundary_faces'` (list of faces in `{'x-','x+','y-','y+','z-','z+'}`), "
    "`'observations'` (non-empty list of modality dicts), and `'priors'` (dict).\n"
    "- Each entry in `problem['observations']` has `'name'` (str), `'points'` (`(N, 3)` float physical coordinates), "
    "`'labels'` (`(N,)` float in `{0.0, 0.5, 1.0}` where `1.0` = positive/ore, `0.0` = negative/host, `0.5` = contact), "
    "`'groups'` (`(N,)` non-negative int borehole/line group IDs), `'weight'` (positive float), "
    "`'transition_scale'` (positive float), `'contact_sigma'` (positive float), `'label_error'` (float in `[0, 0.5)`).\n"
    "- `problem['priors']` has `'area_weight'`, `'volume_weight'`, `'volume_bounds'` (`[lo, hi]` or `None`), "
    "`'fragment_weight'`, `'fragment_volume'`, `'hard_volume_bounds'` (`[lo, hi]` physical volume or `None`), "
    "and `'max_components'` (int >= 1 or `None`, 6-connected components on `field > 0`).\n"
    "- Voxel `(i, j, k)` center in physical coordinates is `origin + ([i, j, k] + 0.5) * spacing`. "
    "Feasible voxels are `feasible = problem['domain_mask'] & ~problem['forbidden_mask']`. Any `field > 0` outside "
    "`feasible`, or violating `hard_volume_bounds` or `max_components`, is rejected as illegal (risk 1.0).\n"
    "- `budget`: dict including `'wall_time_seconds'` (float; reserve ~10% margin for return/serialization).\n"
    "- Public v2 modules importable in the solver container: "
    "`tasks.copper_inversion.v2.contract.validate(problem)`, "
    "`tasks.copper_inversion.v2.geometry.{canonicalize(problem, field), components(mask, spacing), diagnostics(problem, mask)}`, "
    "`tasks.copper_inversion.v2.objective.score(problem, field)`, and "
    "`tasks.copper_inversion.v2.baselines.{initialize(problem), constrained_initial(problem, mask, rank)}`."
)


def _enforce_thread_caps():
    for key, val in THREAD_ENV_CAPS.items():
        os.environ[key] = val


def build_system_prompt(task_brief_path=TASK_BRIEF_PATH, *, diff_based=True):
    """Build OpenEvolve's system prompt from TASK.md and the public v2 problem contract."""
    raw = Path(task_brief_path).read_text(encoding="utf-8")
    marker = "Write the requested complete programs to"
    core = raw.split(marker, 1)[0].strip() if marker in raw else raw.strip()
    format_note = (
        "Keep reasoning focused and leave ample output room for the complete SEARCH/REPLACE diff blocks."
        if diff_based
        else "Return ONLY the complete rewritten Python file inside a single ```python ... ``` code block, without prose outside the code block."
    )
    footer = (
        f"{PUBLIC_CONTRACT_SUMMARY}\n\n"
        "Edit the Python solver program directly so it defines `solve(problem, budget, seed)`.\n"
        f"Each solver must stay within {MAX_NORMALIZED_BYTES:,} bytes once comments are removed and the code is re-formatted.\n"
        "The evaluator returns only aggregate metrics: `risk` in [0, 1] (lower is better; failed or illegal solves score 1.0) "
        "and `combined_score = 1.0 - risk` (higher is better; maximize `combined_score`).\n"
        "No per-case metrics, tracebacks, or withheld observations are provided. Do not use external files or network access.\n"
        f"{format_note}"
    )
    return core + "\n\n" + footer + "\n"


def _aggregate_metrics(risk):
    risk = float(risk)
    return {"combined_score": float(max(0.0, min(1.0, 1.0 - risk))), "risk": risk}


class OpenEvolveAdapter:
    """Process-safe, concurrent evaluator bridge between OpenEvolve and RiskOnlyService."""

    def __init__(self, run_dir, evaluator, *, slots=DEFAULT_SLOTS, case_feedback="none"):
        if case_feedback != "none":
            raise ValueError("OpenEvolve arm requires aggregate-only feedback (case_feedback='none')")
        self.gate = BudgetGate(
            run_dir,
            evaluator,
            slots=slots,
            case_feedback=case_feedback,
            submissions_filename="openevolve-submissions.jsonl",
        )
        self.run_dir = self.gate.run_dir
        self.evaluator = evaluator
        self.slots = self.gate.slots
        self.case_feedback = case_feedback
        self.ledger_path = self.gate.ledger_path
        self.submissions_path = self.gate.submissions_path

    def evaluate_program(self, program_path):
        """Evaluate one candidate file concurrently; return ONLY aggregate metrics."""
        _enforce_thread_caps()
        risk, _status = self.gate.submit(Path(program_path), raise_on_leak=True)
        return _aggregate_metrics(risk)


def evaluate(program_path, adapter_config=None):
    """OpenEvolve evaluator entrypoint; returns ONLY aggregate metrics (no artefacts)."""
    _enforce_thread_caps()
    cfg_path = adapter_config or os.environ.get("OPENEVOLVE_ADAPTER_CONFIG")
    if not cfg_path:
        raise ValueError("missing adapter_config for OpenEvolve evaluator")
    cfg = json.loads(Path(cfg_path).read_text(encoding="utf-8"))
    client = RiskClient(
        cfg["run_dir"],
        cfg["evaluation_socket"],
        cfg["evaluation_contract"],
        service_uid=int(cfg["evaluation_service_uid"]),
    )
    adapter = OpenEvolveAdapter(
        cfg["run_dir"],
        client,
        slots=int(cfg["slots"]),
        case_feedback=cfg.get("case_feedback", "none"),
    )
    return adapter.evaluate_program(program_path)


def build_openevolve_config(
    *,
    api_key,
    adapter_config_path,
    output_dir,
    base_url=DEFAULT_BASE_URL,
    model=DEFAULT_MODEL,
    thinking=DEFAULT_THINKING,
    api_type=DEFAULT_API_TYPE,
    diff_based_evolution=True,
    full_rewrite_probability=0.0,
    max_iterations=DEFAULT_MAX_ITERATIONS,
    seed=401,
    parallel_evaluations=1,
    eval_timeout=600,
    llm_timeout=600,
    llm_max_tokens=DEFAULT_LLM_MAX_TOKENS,
    llm_retries=3,
    task_brief_path=TASK_BRIEF_PATH,
):
    """Build an in-memory OpenEvolve Config (never persists api_key to disk)."""
    from openevolve.config import (
        Config,
        DatabaseConfig,
        EvaluatorConfig,
        EvolutionTraceConfig,
        LLMConfig,
        LLMModelConfig,
        PromptConfig,
    )

    if not api_key or not str(api_key).strip():
        raise ValueError("non-empty API key is required")
    if not 1 <= int(parallel_evaluations) <= MAX_PARALLEL_EVALUATIONS:
        raise ValueError(f"parallel_evaluations must lie in 1..{MAX_PARALLEL_EVALUATIONS}")
    if api_type not in ("chat_completions", "responses"):
        raise ValueError("api_type must be 'chat_completions' or 'responses'")

    diff_mode = bool(diff_based_evolution)
    rewrite_prob = float(full_rewrite_probability) if diff_mode else 0.0
    effort = thinking if thinking and thinking != "off" else None
    temp = None if effort is not None else 0.7
    system_message = build_system_prompt(task_brief_path, diff_based=diff_mode)
    out = Path(output_dir)

    model_kwargs = dict(
        name=model,
        api_base=base_url,
        api_key=str(api_key).strip(),
        weight=1.0,
        system_message=system_message,
        temperature=temp,
        max_tokens=int(llm_max_tokens),
        timeout=int(llm_timeout),
        retries=int(llm_retries),
        retry_delay=10,
        random_seed=int(seed),
        reasoning_effort=effort,
    )
    if hasattr(LLMModelConfig, "api_type"):
        model_kwargs["api_type"] = api_type
    model_cfg = LLMModelConfig(**model_kwargs)

    llm_kwargs = dict(
        api_base=base_url,
        api_key=str(api_key).strip(),
        system_message=system_message,
        temperature=temp,
        max_tokens=int(llm_max_tokens),
        timeout=int(llm_timeout),
        retries=int(llm_retries),
        retry_delay=10,
        random_seed=int(seed),
        reasoning_effort=effort,
        models=[model_cfg],
        evaluator_models=[model_cfg],
    )
    if hasattr(LLMConfig, "api_type"):
        llm_kwargs["api_type"] = api_type
    llm_cfg = LLMConfig(**llm_kwargs)

    prompt_cfg = PromptConfig(
        system_message=system_message,
        include_artifacts=False,
    )
    # Keep OpenEvolve documented defaults (num_islands=5, population_size=1000, archive_size=100, migration_interval=50).
    db_cfg = DatabaseConfig(
        random_seed=int(seed),
        log_prompts=True,
    )
    eval_cfg = EvaluatorConfig(
        timeout=int(eval_timeout),
        max_retries=0,
        cascade_evaluation=False,
        parallel_evaluations=int(parallel_evaluations),
        use_llm_feedback=False,
        enable_artifacts=False,
        evaluator_args={"adapter_config": str(Path(adapter_config_path).resolve())},
    )
    trace_cfg = EvolutionTraceConfig(
        enabled=True,
        format="jsonl",
        include_code=False,
        include_prompts=False,
        include_failures=True,
        output_path=str((out / "evolution_trace.jsonl").resolve()),
        buffer_size=1,
    )
    return Config(
        max_iterations=max(1, int(max_iterations)),
        checkpoint_interval=max(1, min(10, int(max_iterations))),
        log_dir=str((out / "logs").resolve()),
        random_seed=int(seed),
        language="python",
        file_suffix=".py",
        llm=llm_cfg,
        prompt=prompt_cfg,
        database=db_cfg,
        evaluator=eval_cfg,
        evolution_trace=trace_cfg,
        diff_based_evolution=diff_mode,
        full_rewrite_probability=rewrite_prob,
        max_code_length=MAX_NORMALIZED_BYTES,
        enforce_evolve_blocks=False,
    )


def summarize_token_usage(trace_dir):
    """Aggregate prompt/completion/total token usage and truncation counts from OpenEvolve trace JSONL files."""
    trace_dir = Path(trace_dir)
    totals = {
        "calls": 0,
        "succeeded_iterations": 0,
        "failed_iterations": 0,
        "length_truncated_calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "missing_usage_calls": 0,
    }
    for fname, bucket in (("evolution_trace.jsonl", "succeeded_iterations"),
                          ("evolution_trace_failures.jsonl", "failed_iterations")):
        path = trace_dir / fname
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            totals[bucket] += 1
            usage = row.get("token_usage") or (row.get("metadata") or {}).get("token_usage")
            if isinstance(usage, dict):
                totals["calls"] += 1
                totals["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
                totals["completion_tokens"] += int(usage.get("completion_tokens") or 0)
                totals["total_tokens"] += int(usage.get("total_tokens") or 0)
                if usage.get("finish_reason") == "length":
                    totals["length_truncated_calls"] += 1
            else:
                totals["missing_usage_calls"] += 1
    return totals


def latest_checkpoint(checkpoints_dir):
    """Return ``(iteration, path_str)`` for the highest valid checkpoint_<N>, or ``(0, None)``."""
    checkpoints_dir = Path(checkpoints_dir)
    if not checkpoints_dir.is_dir():
        return 0, None
    found = []
    for entry in checkpoints_dir.iterdir():
        if not entry.is_dir():
            continue
        m = re.fullmatch(r"checkpoint_(\d+)", entry.name)
        if m and (entry / "metadata.json").is_file():
            found.append((int(m.group(1)), str(entry.resolve())))
    if not found:
        return 0, None
    return max(found, key=lambda item: item[0])


def _sanitized_config_summary(config):
    return {
        "max_iterations": config.max_iterations,
        "checkpoint_interval": config.checkpoint_interval,
        "random_seed": config.random_seed,
        "max_code_length": config.max_code_length,
        "diff_based_evolution": config.diff_based_evolution,
        "full_rewrite_probability": getattr(config, "full_rewrite_probability", 0.0),
        "model": config.llm.models[0].name if config.llm.models else None,
        "api_base": config.llm.api_base,
        "api_type": getattr(config.llm.models[0], "api_type", "chat_completions") if config.llm.models else None,
        "reasoning_effort": config.llm.reasoning_effort,
        "max_tokens": config.llm.max_tokens,
        "temperature": config.llm.temperature,
        "num_islands": config.database.num_islands,
        "migration_interval": config.database.migration_interval,
        "migration_rate": config.database.migration_rate,
        "population_size": config.database.population_size,
        "archive_size": config.database.archive_size,
        "feature_dimensions": list(config.database.feature_dimensions),
        "evaluator_max_retries": config.evaluator.max_retries,
        "enable_artifacts": config.evaluator.enable_artifacts,
        "include_artifacts": config.prompt.include_artifacts,
        "cascade_evaluation": config.evaluator.cascade_evaluation,
        "use_llm_feedback": config.evaluator.use_llm_feedback,
        "parallel_evaluations": config.evaluator.parallel_evaluations,
    }


def run_openevolve_arm(args):
    """Execute one OpenEvolve external-arm run to exact ``slots`` evaluations and seal training."""
    _enforce_thread_caps()
    if not 1 <= int(args.parallel_evaluations) <= MAX_PARALLEL_EVALUATIONS:
        raise ValueError(f"parallel_evaluations must lie in 1..{MAX_PARALLEL_EVALUATIONS}")

    protocol = json.loads(Path(args.protocol).read_text(encoding="utf-8"))
    pid = args.protocol_id or protocols.protocol_id(protocol)
    tier_spec = protocol["tiers"][args.tier]
    judge_spec = protocol["judges"][args.judge]
    isolation = protocols.case_isolation(protocol)
    if isolation["case_feedback"] != "none":
        raise ValueError("OpenEvolve external arm requires case_isolation.case_feedback == 'none'")
    service_slots = int(args.slots) if args.slots is not None else int(tier_spec["slots"])
    slots = int(args.target_evaluations) if args.target_evaluations is not None else service_slots
    if not 1 <= slots <= service_slots:
        raise ValueError(f"target_evaluations ({slots}) must lie in 1..{service_slots}")
    max_llm_calls = int(args.max_llm_calls) if args.max_llm_calls is not None else 2 * slots
    solver_seconds = int(tier_spec["solver_seconds"])
    finalists_count = int((protocol.get("loop") or {}).get("finalists", 5))
    catalog = load_catalog(args.public_catalog)
    public_digest = catalog_digest(args.public_catalog)

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    oe_dir = out_dir / "openevolve"
    oe_dir.mkdir(parents=True, exist_ok=True)

    client = RiskClient(
        out_dir,
        args.evaluation_socket,
        args.evaluation_contract,
        service_uid=args.evaluation_service_uid,
    )
    info = client.verify_contract(
        public_digest,
        solver_seconds,
        service_slots,
        objective=judge_spec.get("objective", "confirmation"),
        tier=args.tier,
        import_ledger_prefix_sha256=None,
        case_isolation=isolation,
    )
    existing_records = ledgers.read(out_dir / "ledger.jsonl")
    if args.resume:
        if info["used_slots"] != len(existing_records):
            raise ValueError(
                f"cannot resume: private used_slots={info['used_slots']} != public ledger={len(existing_records)}"
            )
    else:
        if info["used_slots"] != 0 or existing_records:
            raise ValueError("a new OpenEvolve run requires an unused private evaluator and empty ledger")

    # Read API key in process memory only; never print, log, or copy to any file.
    api_key = Path(args.api_key_file).read_text(encoding="utf-8").strip()
    if not api_key:
        raise ValueError("protected model API key file is empty")

    model_cfg = protocol.get("model") or {}
    base_url = args.base_url or model_cfg.get("base_url") or DEFAULT_BASE_URL
    model_name = args.model or model_cfg.get("name") or DEFAULT_MODEL
    thinking = args.thinking if args.thinking is not None else model_cfg.get("thinking", DEFAULT_THINKING)
    if thinking == "high" and int(args.llm_max_tokens) < 65536 and not args.allow_small_max_tokens:
        raise ValueError("thinking='high' requires --llm-max-tokens >= 65536 per frameworks-common.md")

    adapter_cfg_path = oe_dir / "adapter-config.json"
    atomic_json_write(adapter_cfg_path, {
        "run_dir": str(out_dir),
        "evaluation_socket": str(Path(args.evaluation_socket).resolve()),
        "evaluation_contract": args.evaluation_contract,
        "evaluation_service_uid": int(args.evaluation_service_uid),
        "slots": slots,
        "case_feedback": isolation["case_feedback"],
    })
    initial_program = oe_dir / "initial_solver.py"
    initial_program.write_bytes(INITIAL_SOURCE)

    initial_max_iters = int(args.max_iterations) if args.max_iterations is not None else max(1, slots - 1)
    config = build_openevolve_config(
        api_key=api_key,
        adapter_config_path=adapter_cfg_path,
        output_dir=oe_dir,
        base_url=base_url,
        model=model_name,
        thinking=thinking,
        api_type=args.api_type,
        diff_based_evolution=not args.full_rewrite_only,
        full_rewrite_probability=args.full_rewrite_probability,
        max_iterations=initial_max_iters,
        seed=args.seed,
        parallel_evaluations=args.parallel_evaluations,
        eval_timeout=args.eval_timeout,
        llm_timeout=args.llm_timeout,
        llm_max_tokens=args.llm_max_tokens,
        llm_retries=args.llm_retries,
    )
    atomic_json_write(oe_dir / "config-sanitized.json", _sanitized_config_summary(config))

    task_brief_text = config.prompt.system_message
    (out_dir / "task_brief.txt").write_text(task_brief_text, encoding="utf-8")

    import importlib.metadata
    try:
        oe_version = importlib.metadata.version("openevolve")
    except Exception:
        oe_version = OPENEVOLVE_PINNED_VERSION

    prev_manifest = json.loads((out_dir / "manifest.json").read_text()) if (out_dir / "manifest.json").is_file() else {}
    manifest = {
        "schema": "rsi-run-manifest-v1",
        "external_system": "openevolve",
        "openevolve_version": oe_version,
        "openevolve_commit": OPENEVOLVE_PINNED_COMMIT,
        "openevolve_license": OPENEVOLVE_LICENSE,
        "run_id": out_dir.name,
        "experiment": args.experiment,
        "protocol_id": pid,
        "protocol_sha256": protocols.protocol_sha256(protocol),
        "arm": "openevolve",
        "tier": args.tier,
        "judge": args.judge,
        "objective": judge_spec.get("objective", "confirmation"),
        "blind": judge_spec.get("objective", "confirmation") != "oracle_iou",
        "seed": int(args.seed),
        "slots": slots,
        "max_llm_calls": max_llm_calls,
        "solver_seconds": solver_seconds,
        "finalists": finalists_count,
        "model": {
            "base_url": base_url,
            "name": model_name,
            "thinking": thinking,
            "api_type": args.api_type,
            "max_tokens": int(args.llm_max_tokens),
        },
        "case_isolation": isolation,
        "public_digest": public_digest,
        "task_brief_sha256": hashlib.sha256(task_brief_text.encode("utf-8")).hexdigest(),
        "shared_implementation_sha256": shared_identity(),
        "code_commit": protocols.code_commit(),
        "host": prev_manifest.get("host") or socket.gethostname(),
        "stage": catalog["stage"],
        "synthetic_test": bool(args.fixture_only),
        "started_at": prev_manifest.get("started_at") or datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    atomic_json_write(out_dir / "manifest.json", manifest)

    from openevolve import OpenEvolve
    from openevolve.controller import EvolutionFailedError

    started = time.monotonic()
    evaluator_file = str(Path(__file__).resolve())
    evolve = OpenEvolve(
        initial_program_path=str(initial_program),
        evaluation_file=evaluator_file,
        config=config,
        output_dir=str(oe_dir),
    )
    evolve.database.novelty_llm = None

    passes = 0
    ckpt_iter, ckpt_path = latest_checkpoint(oe_dir / "checkpoints")
    cumulative_iter = ckpt_iter

    if slots == 1 and len(ledgers.read(out_dir / "ledger.jsonl")) == 0:
        evaluate(initial_program, adapter_config=str(adapter_cfg_path))
    else:
        while True:
            used = len(ledgers.read(out_dir / "ledger.jsonl"))
            usage_now = summarize_token_usage(oe_dir)
            calls_so_far = usage_now["calls"]
            if used >= slots or calls_so_far >= max_llm_calls:
                break
            if args.max_iterations is not None and passes >= 1 and not args.top_up_to_slots:
                break
            remaining_slots = slots - used
            fresh_start = (ckpt_path is None and used == 0)
            desired_iters = max(1, remaining_slots - 1) if fresh_start else remaining_slots
            if args.max_iterations is not None and passes == 0 and not args.top_up_to_slots:
                desired_iters = int(args.max_iterations)
            pass_iters = min(desired_iters, max(1, max_llm_calls - calls_so_far))
            passes += 1
            try:
                asyncio.run(evolve.run(iterations=int(pass_iters), checkpoint_path=ckpt_path))
            except EvolutionFailedError:
                pass
            finally:
                if evolve.evolution_tracer is not None:
                    evolve.evolution_tracer.flush()
            cumulative_iter = max(cumulative_iter + int(pass_iters), int(getattr(evolve.database, "last_iteration", 0)))
            evolve.database.last_iteration = cumulative_iter
            evolve._save_checkpoint(cumulative_iter)
            ckpt_iter, ckpt_path = latest_checkpoint(oe_dir / "checkpoints")

    if evolve.evolution_tracer is not None:
        evolve.evolution_tracer.close()
    wall_seconds = time.monotonic() - started

    records = ledgers.read(out_dir / "ledger.jsonl")
    submissions = _read_submissions(out_dir / "openevolve-submissions.jsonl")
    usage = summarize_token_usage(oe_dir)
    atomic_json_write(out_dir / "usage.json", usage)

    valid_records = [r for r in records if r["status"] == "valid"]
    ranked_valid = sorted(
        valid_records,
        key=lambda r: (r["risk"], -r["legal_rate"], r["seconds_per_solve"], r["program_id"]),
    )
    finalists = ranked_valid[:finalists_count]
    training_finalists = [r["program_id"] for r in finalists]
    oe_best = ranked_valid[0] if ranked_valid else None

    sealed = client.seal_training()
    if sealed["used_slots"] != len(records) or sealed["source_sha256s"] != training_finalists:
        raise ValueError(
            f"public and private training records or finalists differ: "
            f"sealed_slots={sealed['used_slots']} ledger={len(records)} "
            f"sealed_finalists={sealed['source_sha256s']} ledger_finalists={training_finalists}"
        )

    reached_target = len(records) == slots or (args.max_iterations is not None and not args.top_up_to_slots)
    status = "completed" if reached_target else "stopped"
    stop_reason = "completed" if reached_target else "max_llm_calls_reached"

    result = {
        "status": status,
        "stop_reason": stop_reason,
        "external_system": "openevolve",
        "openevolve_version": oe_version,
        "openevolve_commit": OPENEVOLVE_PINNED_COMMIT,
        "charged_records": len(records),
        "target_slots": slots,
        "counted_submissions": sum(1 for s in submissions if s.get("counted", False)),
        "total_submissions": len(submissions),
        "duplicate_submissions": sum(1 for s in submissions if s.get("status") == "duplicate"),
        "invalid_submissions": sum(1 for s in submissions if s.get("status") in ("invalid_source", "oversized")),
        "valid_records": len(valid_records),
        "openevolve_best_program_id": oe_best["program_id"] if oe_best else None,
        "openevolve_best_risk": oe_best["risk"] if oe_best else 1.0,
        "best_risk": finalists[0]["risk"] if finalists else 1.0,
        "training_finalists": training_finalists,
        "selection_rule": "top-5 valid candidates by training risk -> freeze.py standard validation selection -> one-shot REFOE sealed test",
        "topup_passes": passes,
        "llm_calls": usage["calls"],
        "max_llm_calls": max_llm_calls,
        "token_usage": usage,
        "wall_seconds": round(wall_seconds, 2),
        "scope": "dry run / integration fixture only; not a scientific result"
                 if args.fixture_only else "training feedback only; validation selection and sealed evaluation run via openevolve_launch finish / unseal",
    }
    atomic_json_write(out_dir / "result.json", result)
    atomic_json_write(out_dir / "run-status.json", {"status": status, "stop_reason": stop_reason})
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run OpenEvolve external arm against RiskOnlyService.")
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--protocol-id", default=None, help="Override protocol_id (defaults to protocol hash id)")
    parser.add_argument("--experiment", required=True, choices=protocols.EXPERIMENTS)
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
    parser.add_argument("--thinking", default=None,
                        help="Reasoning effort override (defaults to protocol.model.thinking or 'high')")
    parser.add_argument("--api-type", default=DEFAULT_API_TYPE, choices=("chat_completions", "responses"))
    parser.add_argument("--full-rewrite-only", action="store_true",
                        help="Use full-file rewrite on every iteration (diff_based_evolution=False)")
    parser.add_argument("--full-rewrite-probability", type=float, default=0.0,
                        help="Probability of full-file rewrite per iteration when diff_based_evolution=True")
    parser.add_argument("--max-iterations", type=int, default=None,
                        help="Explicit evolutionary iterations (defaults to slots - 1 with top-up to exact slots)")
    parser.add_argument("--no-top-up", dest="top_up_to_slots", action="store_false", default=True,
                        help="Disable checkpoint top-up loop when --max-iterations is set explicitly")
    parser.add_argument("--slots", type=int, default=None, help="Override service slot limit (defaults to tier slots)")
    parser.add_argument("--target-evaluations", type=int, default=None,
                        help="Stop gate at exact target evaluations <= service slots (defaults to service slots)")
    parser.add_argument("--max-llm-calls", type=int, default=None,
                        help="Hard cap on total LLM calls (defaults to 2 * slots)")
    parser.add_argument("--parallel-evaluations", type=int, default=1)
    parser.add_argument("--eval-timeout", type=int, default=600)
    parser.add_argument("--llm-timeout", type=int, default=900)
    parser.add_argument("--llm-max-tokens", type=int, default=DEFAULT_LLM_MAX_TOKENS)
    parser.add_argument("--allow-small-max-tokens", action="store_true")
    parser.add_argument("--llm-retries", type=int, default=3)
    parser.add_argument("--fixture-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    result = run_openevolve_arm(args)
    print(json.dumps({"run_dir": str(Path(args.out_dir).resolve()), **result}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
