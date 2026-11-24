"""OpenEvolve external-arm integration for the RSI copper-inversion benchmark.

Submits OpenEvolve-generated solver programs through the same RiskOnlyService
contract as our controlled arms (comment stripping, 32 kB normalized-code limit,
interface checks, slot-counted budget including rejections, aggregate-only
feedback with artefacts disabled) and writes a ledger/run layout compatible with
``experiments/rsi/freeze.py`` (five finalists, standard validation, sealed path).

OpenEvolve (``algorithmicsuperintelligence/openevolve``, Apache-2.0) runs in its
own venv (``$R/env/openevolve``) so no existing environment is modified.
"""

import argparse
import asyncio
import fcntl
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from ldm_tts.engine.run_store import BudgetExceededError, atomic_json_write
from tasks.copper_inversion.core.candidate import seal, validate_source
from tasks.copper_inversion.core.encoding import MAX_NORMALIZED_BYTES, normalized_code
from tasks.copper_inversion.rsi import ledger as ledgers, protocol as protocols
from tasks.copper_inversion.rsi.client import RiskClient
from tasks.copper_inversion.rsi.controller import INITIAL_SOURCE
from tasks.copper_inversion.rsi.integrity import shared_identity
from tasks.copper_inversion.rsi.render import stripped_source
from tasks.copper_inversion.v2.public_catalog import catalog_digest, load_catalog

TASK_BRIEF_PATH = Path(__file__).resolve().parents[2] / "resources/rsi/TASK.md"
DEFAULT_API_KEY_FILE = Path("/Aidata/shared/ldm/async-ldm-core/secrets/deepseek.key")
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-flash"
DEFAULT_THINKING = "high"
DEFAULT_MAX_ITERATIONS = 321
DEFAULT_SLOTS = 321
OPENEVOLVE_PINNED_COMMIT = "73448d715968b3578d8458b16295e827edd2b6bf"
OPENEVOLVE_PINNED_VERSION = "0.4.1"
OPENEVOLVE_LICENSE = "Apache-2.0"


def build_system_prompt(task_brief_path=TASK_BRIEF_PATH):
    """Build OpenEvolve's system prompt from TASK.md (aggregate-only, direct solver.py evolution)."""
    raw = Path(task_brief_path).read_text(encoding="utf-8")
    marker = "Write the requested complete programs to"
    core = raw.split(marker, 1)[0].strip() if marker in raw else raw.strip()
    footer = (
        "Edit the Python solver program directly so it defines `solve(problem, budget, seed)`.\n"
        f"Each solver must stay within {MAX_NORMALIZED_BYTES:,} bytes once comments are removed and the code is re-formatted.\n"
        "The evaluator returns only aggregate metrics: `risk` in [0, 1] (lower is better; failed or illegal solves score 1.0) "
        "and `combined_score = 1.0 - risk` (higher is better; maximize `combined_score`).\n"
        "No per-case metrics, tracebacks, or withheld observations are provided. Do not use external files or network access."
    )
    return core + "\n\n" + footer + "\n"


def prepare_submission(source_bytes):
    """Validate Python syntax + solve() entrypoint, strip comments/docstrings, and enforce 32 kB limit."""
    if not isinstance(source_bytes, (bytes, bytearray)):
        raise ValueError("solver source must be bytes")
    raw = bytes(source_bytes)
    validate_source(raw)
    stripped_text = stripped_source(raw.decode("utf-8"))
    stripped_bytes = stripped_text.encode("utf-8")
    sha = validate_source(stripped_bytes)
    norm_size = len(normalized_code(stripped_bytes).encode("utf-8"))
    if norm_size > MAX_NORMALIZED_BYTES:
        raise ValueError(
            f"normalized solver is {norm_size} bytes (limit {MAX_NORMALIZED_BYTES} bytes)"
        )
    return stripped_bytes, sha


def _read_submissions(path):
    path = Path(path)
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _append_submission(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _aggregate_metrics(risk):
    risk = float(risk)
    return {"combined_score": float(max(0.0, min(1.0, 1.0 - risk))), "risk": risk}


class OpenEvolveAdapter:
    """Process-safe evaluator bridge between OpenEvolve and RiskOnlyService."""

    def __init__(self, run_dir, evaluator, *, slots=DEFAULT_SLOTS, case_feedback="none"):
        if case_feedback != "none":
            raise ValueError("OpenEvolve arm requires aggregate-only feedback (case_feedback='none')")
        if type(slots) is not int or slots < 1:
            raise ValueError("slots must be a positive integer")
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.evaluator = evaluator
        self.slots = slots
        self.case_feedback = case_feedback
        self.ledger_path = self.run_dir / "ledger.jsonl"
        self.submissions_path = self.run_dir / "openevolve-submissions.jsonl"
        self.lock_path = self.run_dir / ".oe-eval.lock"

    def evaluate_program(self, program_path):
        """Evaluate one candidate file; enforce contract, count against slots, return aggregate metrics only."""
        source_bytes = Path(program_path).read_bytes()
        with self.lock_path.open("a+", encoding="utf-8") as lock_stream:
            fcntl.flock(lock_stream, fcntl.LOCK_EX)
            return self._evaluate_locked(source_bytes)

    def _evaluate_locked(self, source_bytes):
        existing = ledgers.read(self.ledger_path)
        submissions = _read_submissions(self.submissions_path)
        by_sha = {row["program_id"]: row for row in existing}
        charged_attempts = sum(1 for s in submissions if s.get("counted", False))
        raw_sha = hashlib.sha256(source_bytes).hexdigest()

        try:
            stripped_bytes, sha = prepare_submission(source_bytes)
        except (ValueError, SyntaxError, UnicodeError) as exc:
            if charged_attempts >= self.slots:
                _append_submission(self.submissions_path, {
                    "attempt": len(submissions) + 1, "raw_sha256": raw_sha,
                    "status": "budget_exhausted", "counted": False, "risk": 1.0,
                })
                return _aggregate_metrics(1.0)
            reason = "oversized" if "bytes" in str(exc) else "invalid_source"
            _append_submission(self.submissions_path, {
                "attempt": len(submissions) + 1, "raw_sha256": raw_sha,
                "status": reason, "counted": True, "service_slot": False, "risk": 1.0,
            })
            return _aggregate_metrics(1.0)

        if sha in by_sha:
            prev = by_sha[sha]
            _append_submission(self.submissions_path, {
                "attempt": len(submissions) + 1, "raw_sha256": raw_sha, "program_id": sha,
                "status": "duplicate", "counted": False, "risk": prev["risk"],
            })
            return _aggregate_metrics(prev["risk"])

        if charged_attempts >= self.slots or len(existing) >= self.slots:
            _append_submission(self.submissions_path, {
                "attempt": len(submissions) + 1, "raw_sha256": raw_sha, "program_id": sha,
                "status": "budget_exhausted", "counted": False, "risk": 1.0,
            })
            return _aggregate_metrics(1.0)

        sealed = seal(self.run_dir, stripped_bytes)
        source_file = self.run_dir / sealed["source_path"]
        try:
            feedback = self.evaluator.evaluate_source(source_file)
        except BudgetExceededError:
            _append_submission(self.submissions_path, {
                "attempt": len(submissions) + 1, "raw_sha256": raw_sha, "program_id": sha,
                "status": "budget_exhausted", "counted": False, "risk": 1.0,
            })
            return _aggregate_metrics(1.0)

        if feedback.get("risk_by_case"):
            raise ValueError("service returned per-case risks under aggregate-only mode")
        seq = len(existing)
        record = {
            "seq": seq,
            "program_id": sha,
            "epoch": 0,
            "round": 0 if seq == 0 else seq,
            "branch": -1 if seq == 0 else 0,
            "status": feedback["status"],
            "risk": float(feedback["risk"]),
            "risk_by_case": {},
            "legal_rate": float(feedback["legal_rate"]),
            "failed_solves": int(feedback["failed_solves"]),
            "seconds_per_solve": float(feedback["seconds_per_solve"]),
            "message": feedback["message"],
            "selection": "initial" if seq == 0 else "acquisition",
        }
        ledgers.append(self.ledger_path, record)
        _append_submission(self.submissions_path, {
            "attempt": len(submissions) + 1, "seq": seq, "raw_sha256": raw_sha,
            "program_id": sha, "status": record["status"], "counted": True,
            "service_slot": True, "risk": record["risk"],
        })
        return _aggregate_metrics(record["risk"])


def evaluate(program_path, adapter_config=None):
    """OpenEvolve evaluator entrypoint; returns ONLY aggregate metrics (no artefacts)."""
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
    max_iterations=DEFAULT_MAX_ITERATIONS,
    seed=42,
    parallel_evaluations=1,
    eval_timeout=600,
    llm_timeout=600,
    llm_max_tokens=32768,
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
    system_message = build_system_prompt(task_brief_path)
    out = Path(output_dir)
    model_cfg = LLMModelConfig(
        name=model,
        api_base=base_url,
        api_key=str(api_key).strip(),
        weight=1.0,
        system_message=system_message,
        temperature=0.7,
        max_tokens=int(llm_max_tokens),
        timeout=int(llm_timeout),
        retries=int(llm_retries),
        retry_delay=10,
        random_seed=int(seed),
        reasoning_effort=thinking if thinking and thinking != "off" else None,
    )
    llm_cfg = LLMConfig(
        api_base=base_url,
        api_key=str(api_key).strip(),
        system_message=system_message,
        temperature=0.7,
        max_tokens=int(llm_max_tokens),
        timeout=int(llm_timeout),
        retries=int(llm_retries),
        retry_delay=10,
        random_seed=int(seed),
        reasoning_effort=thinking if thinking and thinking != "off" else None,
        models=[model_cfg],
        evaluator_models=[model_cfg],
    )
    prompt_cfg = PromptConfig(
        system_message=system_message,
        include_artifacts=False,
    )
    # Keep OpenEvolve default islands (num_islands=5, migration_interval=50, etc.).
    db_cfg = DatabaseConfig(
        random_seed=int(seed),
        log_prompts=True,
    )
    eval_cfg = EvaluatorConfig(
        timeout=int(eval_timeout),
        max_retries=1,
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
        max_iterations=int(max_iterations),
        checkpoint_interval=max(1, min(50, int(max_iterations))),
        log_dir=str((out / "logs").resolve()),
        random_seed=int(seed),
        language="python",
        file_suffix=".py",
        llm=llm_cfg,
        prompt=prompt_cfg,
        database=db_cfg,
        evaluator=eval_cfg,
        evolution_trace=trace_cfg,
        diff_based_evolution=True,
        max_code_length=MAX_NORMALIZED_BYTES,
        enforce_evolve_blocks=False,
    )


def summarize_token_usage(trace_dir):
    """Aggregate prompt/completion/total token usage from OpenEvolve trace JSONL files."""
    trace_dir = Path(trace_dir)
    totals = {
        "calls": 0,
        "succeeded_iterations": 0,
        "failed_iterations": 0,
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
            else:
                totals["missing_usage_calls"] += 1
    return totals


def _sanitized_config_summary(config):
    return {
        "max_iterations": config.max_iterations,
        "random_seed": config.random_seed,
        "max_code_length": config.max_code_length,
        "diff_based_evolution": config.diff_based_evolution,
        "model": config.llm.models[0].name if config.llm.models else None,
        "api_base": config.llm.api_base,
        "reasoning_effort": config.llm.reasoning_effort,
        "max_tokens": config.llm.max_tokens,
        "num_islands": config.database.num_islands,
        "migration_interval": config.database.migration_interval,
        "migration_rate": config.database.migration_rate,
        "population_size": config.database.population_size,
        "archive_size": config.database.archive_size,
        "feature_dimensions": list(config.database.feature_dimensions),
        "enable_artifacts": config.evaluator.enable_artifacts,
        "include_artifacts": config.prompt.include_artifacts,
        "cascade_evaluation": config.evaluator.cascade_evaluation,
        "use_llm_feedback": config.evaluator.use_llm_feedback,
        "parallel_evaluations": config.evaluator.parallel_evaluations,
    }


def run_openevolve_arm(args):
    """Execute one OpenEvolve external-arm run against a live RiskOnlyService and seal training."""
    protocol = json.loads(Path(args.protocol).read_text(encoding="utf-8"))
    pid = args.protocol_id or protocols.protocol_id(protocol)
    tier_spec = protocol["tiers"][args.tier]
    judge_spec = protocol["judges"][args.judge]
    isolation = protocols.case_isolation(protocol)
    if isolation["case_feedback"] != "none":
        raise ValueError("OpenEvolve external arm requires case_isolation.case_feedback == 'none'")
    slots = args.slots if args.slots is not None else int(tier_spec["slots"])
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
        slots,
        objective=judge_spec.get("objective", "confirmation"),
        tier=args.tier,
        import_ledger_prefix_sha256=None,
        case_isolation=isolation,
    )
    if info["used_slots"] != 0:
        raise ValueError("a new OpenEvolve run requires an unused private evaluator")

    # Read API key in process memory only; never print, log, or copy to any file.
    api_key = Path(args.api_key_file).read_text(encoding="utf-8").strip()
    if not api_key:
        raise ValueError("protected model API key file is empty")

    model_cfg = protocol.get("model") or {}
    base_url = args.base_url or model_cfg.get("base_url") or DEFAULT_BASE_URL
    model_name = args.model or model_cfg.get("name") or DEFAULT_MODEL
    thinking = args.thinking if args.thinking is not None else model_cfg.get("thinking", DEFAULT_THINKING)

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

    config = build_openevolve_config(
        api_key=api_key,
        adapter_config_path=adapter_cfg_path,
        output_dir=oe_dir,
        base_url=base_url,
        model=model_name,
        thinking=thinking,
        max_iterations=args.max_iterations,
        seed=args.seed,
        parallel_evaluations=args.parallel_evaluations,
        eval_timeout=args.eval_timeout,
        llm_timeout=args.llm_timeout,
        llm_max_tokens=args.llm_max_tokens,
        llm_retries=args.llm_retries,
    )
    atomic_json_write(oe_dir / "config-sanitized.json", _sanitized_config_summary(config))

    import importlib.metadata
    try:
        oe_version = importlib.metadata.version("openevolve")
    except Exception:
        oe_version = OPENEVOLVE_PINNED_VERSION

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
        "seed": int(args.seed),
        "slots": slots,
        "max_iterations": int(args.max_iterations),
        "solver_seconds": solver_seconds,
        "finalists": finalists_count,
        "model": {"base_url": base_url, "name": model_name, "thinking": thinking},
        "case_isolation": isolation,
        "public_digest": public_digest,
        "shared_implementation_sha256": shared_identity(),
        "stage": catalog["stage"],
        "synthetic_test": bool(args.fixture_only),
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json_write(out_dir / "manifest.json", manifest)

    from openevolve import OpenEvolve

    started = time.monotonic()
    evaluator_file = str(Path(__file__).resolve())
    evolve = OpenEvolve(
        initial_program_path=str(initial_program),
        evaluation_file=evaluator_file,
        config=config,
        output_dir=str(oe_dir),
    )
    evolve.database.novelty_llm = None
    asyncio.run(evolve.run(iterations=int(args.max_iterations)))
    wall_seconds = time.monotonic() - started

    records = ledgers.read(out_dir / "ledger.jsonl")
    submissions = _read_submissions(out_dir / "openevolve-submissions.jsonl")
    usage = summarize_token_usage(oe_dir)
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

    result = {
        "status": "completed",
        "stop_reason": "completed",
        "external_system": "openevolve",
        "openevolve_version": oe_version,
        "charged_records": len(records),
        "counted_submissions": sum(1 for s in submissions if s.get("counted", False)),
        "total_submissions": len(submissions),
        "valid_records": len(valid_records),
        "best_risk": finalists[0]["risk"] if finalists else 1.0,
        "training_finalists": training_finalists,
        "token_usage": usage,
        "wall_seconds": round(wall_seconds, 2),
        "scope": "dry run / integration fixture only; not a scientific result"
                 if args.fixture_only else "training feedback only; validation selection and sealed evaluation not run",
    }
    atomic_json_write(out_dir / "result.json", result)
    atomic_json_write(out_dir / "run-status.json", {"status": "completed", "stop_reason": "completed"})
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
    parser.add_argument("--max-iterations", type=int, default=DEFAULT_MAX_ITERATIONS)
    parser.add_argument("--slots", type=int, default=None, help="Override slot limit (defaults to tier slots)")
    parser.add_argument("--parallel-evaluations", type=int, default=1)
    parser.add_argument("--eval-timeout", type=int, default=600)
    parser.add_argument("--llm-timeout", type=int, default=600)
    parser.add_argument("--llm-max-tokens", type=int, default=32768)
    parser.add_argument("--llm-retries", type=int, default=3)
    parser.add_argument("--fixture-only", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    result = run_openevolve_arm(args)
    print(json.dumps({"run_dir": str(Path(args.out_dir).resolve()), **result}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
