"""Source-pinned runner invocation and final NucleoBench reporting."""

from __future__ import annotations

import json
from argparse import Namespace
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from ldm_tts.engine.run_store import CampaignRuntime
from tasks.nucleobench.core.benchmark_clock import BenchmarkClock
from tasks.nucleobench.core.designer import NucleoBenchDesigner
from tasks.nucleobench.core.reporting import inventory_official_outputs, write_campaign_reports

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
    designer: NucleoBenchDesigner,
    all_args: Any,
    runtime: CampaignRuntime,
    execution: Mapping[str, Any] | Callable[[int], Mapping[str, Any]],
    oracle_manifest: Mapping[str, Any],
    expected_active_steps: int | None = None,
    benchmark_clock: BenchmarkClock | None = None,
) -> dict[str, Any]:
    try:
        if benchmark_clock is not None:
            # Start just before the unchanged runner, conservatively including entry overhead.
            runtime.record(
                "benchmark_clock_resumed" if benchmark_clock.elapsed_before_start else "benchmark_clock_started",
                benchmark_clock.start(),
            )
        run_loop(model=model, opt=designer, all_args=all_args, ignore_errors=False)
        if (
            expected_active_steps is not None
            and designer.active_steps != expected_active_steps
        ):
            raise RuntimeError(
                "official runner completed a different number of active steps: "
                f"{designer.active_steps} != {expected_active_steps}"
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
    resolved_execution = (
        execution(designer.active_steps) if callable(execution) else execution
    )
    return write_campaign_reports(
        runtime,
        execution=resolved_execution,
        oracle_manifest=oracle_manifest,
        official_outputs=official_outputs,
    )
