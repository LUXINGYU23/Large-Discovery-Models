"""Config-driven NucleoBench workflow using the shared LDM engine."""

from __future__ import annotations

import argparse
import json
from argparse import Namespace
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from ldm_tts.contracts import LDMTaskSpec
from ldm_tts.data import DataCollectionSink
from ldm_tts.engine import LDMEngineConfig
from ldm_tts.engine.reporting import (
    build_campaign_result,
    build_trajectory_rows,
    load_successful_observations,
    write_trajectory_csv,
)
from ldm_tts.engine.run_store import CampaignRuntime, atomic_json_write, unique_run_dir
from ldm_tts.registration.experiment import (
    load_active_experiment_contract,
    load_experiment_contract,
    snapshot_experiment_contract,
)
from tasks.nucleobench.core.cases import get_case
from tasks.nucleobench.core.constants import TASK_ID
from tasks.nucleobench.core.factory import build_mock_engine
from tasks.nucleobench.core.mock import MOCK_CASE, build_mock_task_spec
from tasks.nucleobench.core.reporting import inventory_official_outputs
from tasks.nucleobench.core.task_spec import build_task_spec


TASK_ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", default="malinois_k562")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--reservoir-size", type=int, default=4)
    parser.add_argument("--evaluations-per-round", type=int, default=1)
    parser.add_argument("--out-dir", type=Path, default=Path("runs"))
    parser.add_argument("--run-name")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.iterations < 1:
        parser.error("--iterations must be positive")
    if args.mock and args.iterations < 2:
        parser.error(
            "--iterations must be at least 2 for initialization and active search"
        )
    if args.reservoir_size < 1 or args.evaluations_per_round < 1:
        parser.error("reservoir and evaluation counts must be positive")
    if args.mock and args.case_id != MOCK_CASE.case_id:
        parser.error(f"mock execution requires --case-id {MOCK_CASE.case_id}")
    return args


def describe_ldm_task(args: argparse.Namespace) -> LDMTaskSpec:
    return build_mock_task_spec() if args.mock else build_task_spec(get_case(args.case_id))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    case = MOCK_CASE if args.mock else get_case(args.case_id)
    task_spec = describe_ldm_task(args)
    contract, profile_name = load_active_experiment_contract()
    if contract is None:
        contract = load_experiment_contract(TASK_ROOT / "experiment.json")
    payload = {
        "task": TASK_ID,
        "mode": "mock" if args.mock else "official",
        "case_id": args.case_id,
        "case": case.to_dict(),
        "contract_profile": profile_name,
        "contract_sha256": contract.digest,
        "ldm_task_spec": task_spec.to_dict(),
    }
    if args.dry_run:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if not args.mock:
        raise SystemExit(
            "NucleoBench official execution is not qualified; "
            "run the mock profile or use --dry-run."
        )
    return _run_mock(args, contract, profile_name, payload)


def _run_mock(args, contract, profile_name: str, payload: dict[str, Any]) -> int:
    run_dir = unique_run_dir(args.out_dir / (args.run_name or "mock"))
    task_spec = build_mock_task_spec()
    evaluation_limit = args.iterations * args.evaluations_per_round
    runtime = CampaignRuntime.open(
        run_dir,
        task=TASK_ID,
        config=_jsonable_args(args),
        task_spec=task_spec,
        budget_limits={
            "outer_iterations": args.iterations,
            "valid_search_candidates": 1
            + (args.iterations - 1) * args.reservoir_size,
            "selected_candidates": evaluation_limit,
            "external_evaluations": evaluation_limit,
            "expensive_evaluation_attempts": evaluation_limit,
        },
        contract_snapshot=contract.to_dict(),
        contract_sha256=contract.digest,
        contract_profile=profile_name,
    )
    snapshot_experiment_contract(contract, run_dir, profile=profile_name)
    sink = DataCollectionSink.from_env(default_root=run_dir / "ldm_data")
    engine = build_mock_engine(runtime, sink, task_spec)
    result = engine.run(
        LDMEngineConfig(
            iterations=args.iterations,
            reservoir_size=args.reservoir_size,
            evaluations_per_round=args.evaluations_per_round,
        )
    )
    observations = load_successful_observations(run_dir / "checkpoint.json")
    trajectory = build_trajectory_rows(
        observations,
        objective_name="utility",
        direction="maximize",
    )
    write_trajectory_csv(
        run_dir / "trajectory.csv",
        trajectory,
        fieldnames=("evaluation", "round", "candidate_id", "utility", "best_utility"),
    )
    report = {
        **build_campaign_result(
            run_dir,
            objective_name="utility",
            direction="maximize",
        ),
        "mode": "mock",
        "case_id": MOCK_CASE.case_id,
        "engine_summary": result.summary,
        "artifacts": {"trajectory": "trajectory.csv", "summary": "summary.json"},
    }
    atomic_json_write(run_dir / "result.json", report)
    payload.update(run_dir=str(run_dir.resolve()), result=report)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if result.summary["successful_evaluation_count"] else 1


def _jsonable_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def build_official_runner_args(
    parsed_args_type: type,
    *,
    model_name: str,
    optimization_name: str,
    start_sequence: str,
    positions_to_mutate: Sequence[int] | None,
    output_path: Path,
    proposals_per_round: int,
    max_seconds: int | None = None,
    max_number_of_rounds: int | None = None,
    model_init_args: Mapping[str, Any] | None = None,
    optimizer_init_args: Mapping[str, Any] | None = None,
) -> Any:
    if (max_seconds is None) == (max_number_of_rounds is None):
        raise ValueError("official runner requires exactly one termination mode")
    if max_seconds is not None and max_seconds < 1:
        raise ValueError("max_seconds must be positive")
    if max_number_of_rounds is not None and max_number_of_rounds < 1:
        raise ValueError("max_number_of_rounds must be positive")
    if proposals_per_round < 1:
        raise ValueError("proposals_per_round must be positive")
    main_args = Namespace(
        model=model_name,
        optimization=optimization_name,
        start_sequence=start_sequence,
        positions_to_mutate=(
            None if positions_to_mutate is None else list(positions_to_mutate)
        ),
        output_path=str(Path(output_path)),
        optimization_steps_per_output=1,
        proposals_per_round=proposals_per_round,
        max_seconds=max_seconds,
        max_number_of_rounds=max_number_of_rounds,
        ignore_errors=False,
        ignore_empty_cmd_args=False,
    )
    return parsed_args_type(
        main_args=main_args,
        model_init_args=Namespace(**dict(model_init_args or {})),
        opt_init_args=Namespace(**dict(optimizer_init_args or {})),
    )


def run_official_driver(
    *,
    run_loop: Callable[..., Any],
    model: Callable[[Sequence[str]], Any],
    designer: Any,
    all_args: Any,
    runtime: CampaignRuntime,
    case_id: str,
) -> dict[str, Any]:
    try:
        run_loop(
            model=model,
            opt=designer,
            all_args=all_args,
            ignore_errors=False,
        )
        official_outputs = inventory_official_outputs(
            Path(all_args.main_args.output_path),
            runtime.run_dir,
        )
    except Exception as exc:
        status = json.loads(runtime.status.path.read_text(encoding="utf-8"))
        if status.get("status") != "failed":
            runtime.fail(exc)
        raise

    summary = {
        **designer.summary(),
        "stop_reason": "official_driver_finished",
        "official_output_count": len(official_outputs),
    }
    runtime.finish(summary)
    result = {
        **build_campaign_result(
            runtime.run_dir,
            objective_name="utility",
            direction="maximize",
        ),
        "mode": "official_benchmark",
        "case_id": case_id,
        "official_outputs": official_outputs,
        "artifacts": {
            "summary": "summary.json",
            "official_output_root": Path(all_args.main_args.output_path)
            .resolve()
            .relative_to(runtime.run_dir.resolve())
            .as_posix(),
        },
    }
    atomic_json_write(runtime.run_dir / "result.json", result)
    return result


__all__ = [
    "build_official_runner_args",
    "describe_ldm_task",
    "main",
    "parse_args",
    "run_official_driver",
]
