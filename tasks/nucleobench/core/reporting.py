"""Publication-facing records for NucleoBench campaigns."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ldm_tts.engine.reporting import (
    build_campaign_result,
    load_successful_observations,
    read_json_object,
    write_trajectory_csv,
)
from ldm_tts.engine.run_store import CampaignRuntime, atomic_json_write

_EXECUTION_PROFILES = frozenset(
    {"qualification", "pilot_evaluation", "official_benchmark"}
)
_PROPOSAL_EVENT_TYPES = frozenset(
    {"reservoir_expanded", "reservoir_built", "candidates_selected"}
)


def inventory_official_outputs(output_dir: Path, run_dir: Path) -> list[dict[str, Any]]:
    output_dir = Path(output_dir).resolve()
    run_dir = Path(run_dir).resolve()
    if not output_dir.is_relative_to(run_dir):
        raise ValueError(
            "official output directory must be inside the campaign directory"
        )
    for marker in ("START.txt", "SUCCESS.txt"):
        if not (output_dir / marker).is_file():
            raise ValueError(f"official runner did not write {marker}")
    files = sorted(path for path in output_dir.rglob("*") if path.is_file())
    if not any(path.suffix == ".parquet" for path in files):
        raise ValueError("official runner did not write a Parquet result")
    return [
        {
            "path": path.relative_to(run_dir).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in files
    ]


def write_campaign_reports(
    runtime: CampaignRuntime,
    *,
    execution: Mapping[str, Any],
    oracle_manifest: Mapping[str, Any],
    official_outputs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Write one profile-labelled result without mixing pilot and official claims."""

    normalized = _validate_execution(execution)
    observations = load_successful_observations(runtime.run_dir / "checkpoint.json")
    rows = _trajectory_rows(runtime, observations, normalized["search_method"])
    completed_rounds = len({int(row["round"]) for row in rows})
    if (
        normalized["termination_kind"] == "rounds"
        and completed_rounds != normalized["total_rounds"]
    ):
        raise ValueError(
            "round-limited result does not contain the declared total rounds"
        )

    write_trajectory_csv(
        runtime.run_dir / "trajectory.csv",
        rows,
        fieldnames=(
            "evaluation",
            "round",
            "elapsed_seconds",
            "oracle_evaluation_index",
            "candidate_id",
            "energy",
            "utility",
            "hamming_distance",
            "best_energy",
            "best_utility",
            "search_method",
        ),
    )
    atomic_json_write(runtime.run_dir / "oracle_manifest.json", dict(oracle_manifest))
    _write_proposal_trace(runtime)

    base = build_campaign_result(
        runtime.run_dir,
        objective_name="utility",
        direction="maximize",
    )
    best = base["best_candidate"]
    result = {
        **base,
        **normalized,
        "completed_rounds": completed_rounds,
        "best_found_utility": None if best is None else best["utility"],
        "best_found_energy": (None if best is None else best["metrics"].get("energy")),
        "trajectory_primary_axis": (
            "elapsed_seconds"
            if normalized["termination_kind"] == "wall_time"
            else "round"
        ),
        "official_outputs": [dict(item) for item in official_outputs],
        "artifacts": {
            "summary": "summary.json",
            "trajectory": "trajectory.csv",
            "oracle_manifest": "oracle_manifest.json",
            "proposal_trace": "proposal_trace.jsonl",
        },
    }
    atomic_json_write(runtime.run_dir / "result.json", result)
    return result


def _validate_execution(execution: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "execution_profile",
        "termination_kind",
        "total_rounds",
        "active_optimization_rounds",
        "initialization_evaluations",
        "benchmark_comparable",
        "case_id",
        "start_index",
        "start_set_digest",
        "optimization_seed",
        "hardware_profile",
        "search_method",
        "method_preset_sha256",
        "max_seconds",
    }
    if set(execution) != required:
        raise ValueError("execution record has unexpected or missing fields")
    result = dict(execution)
    profile = _string(result["execution_profile"], "execution_profile")
    if profile not in _EXECUTION_PROFILES:
        raise ValueError("unknown NucleoBench execution profile")
    termination = _string(result["termination_kind"], "termination_kind")
    if termination not in {"rounds", "wall_time"}:
        raise ValueError("unknown NucleoBench termination kind")
    comparable = result["benchmark_comparable"]
    if not isinstance(comparable, bool):
        raise TypeError("benchmark_comparable must be boolean")
    if profile == "official_benchmark":
        if termination != "wall_time" or not comparable:
            raise ValueError(
                "official benchmark results require wall-time comparability"
            )
        _positive_int(result["max_seconds"], "max_seconds")
    elif termination != "rounds" or comparable or result["max_seconds"] is not None:
        raise ValueError(
            "development results require round termination and no official claim"
        )

    total_rounds = _positive_int(result["total_rounds"], "total_rounds")
    active_rounds = _nonnegative_int(
        result["active_optimization_rounds"], "active_optimization_rounds"
    )
    initialization = _positive_int(
        result["initialization_evaluations"], "initialization_evaluations"
    )
    if total_rounds != active_rounds + 1:
        raise ValueError("total_rounds must contain one initialization round")
    result["total_rounds"] = total_rounds
    result["active_optimization_rounds"] = active_rounds
    result["initialization_evaluations"] = initialization
    result["start_index"] = _nonnegative_int(result["start_index"], "start_index")
    result["optimization_seed"] = _nonnegative_int(
        result["optimization_seed"], "optimization_seed"
    )
    for name in (
        "case_id",
        "start_set_digest",
        "hardware_profile",
        "search_method",
        "method_preset_sha256",
    ):
        result[name] = _string(result[name], name)
    for name in ("start_set_digest", "method_preset_sha256"):
        if len(result[name]) != 64 or any(
            character not in "0123456789abcdef" for character in result[name]
        ):
            raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return result


def _trajectory_rows(
    runtime: CampaignRuntime,
    observations: Sequence[Mapping[str, Any]],
    search_method: str,
) -> list[dict[str, Any]]:
    events = runtime.events()
    event_by_candidate = {
        str(event.get("candidate_id")): event
        for event in events
        if event.get("event_type") in {"baseline_evaluated", "candidate_evaluated"}
        and event.get("candidate_id")
    }
    model_calls = [
        event
        for event in events
        if event.get("event_type") == "official_model_called"
        and isinstance(event.get("payload"), Mapping)
        and event["payload"].get("status") == "succeeded"
    ]
    started_at = float(
        read_json_object(runtime.run_dir / "status.json")["started_at_unix"]
    )
    best_energy: float | None = None
    best_utility: float | None = None
    rows: list[dict[str, Any]] = []
    for index, observation in enumerate(observations, start=1):
        candidate = observation["candidate"]
        evaluation = observation["evaluation"]
        metrics = evaluation["metrics"]
        candidate_id = str(candidate["candidate_id"])
        event = event_by_candidate.get(candidate_id)
        if event is None:
            raise ValueError(f"missing evaluation event for {candidate_id}")
        timestamp = float(event["timestamp_unix"])
        energy = _finite(metrics.get("energy"), "energy")
        utility = _finite(metrics.get("utility"), "utility")
        hamming = _finite(metrics.get("hamming_distance"), "hamming_distance")
        best_energy = energy if best_energy is None else min(best_energy, energy)
        best_utility = utility if best_utility is None else max(best_utility, utility)
        oracle_index = index
        for call in model_calls:
            if float(call["timestamp_unix"]) > timestamp:
                break
            oracle_index = int(call["payload"]["cumulative_sequence_count"])
        rows.append(
            {
                "evaluation": index,
                "round": int(observation["round_idx"]),
                "elapsed_seconds": max(0.0, timestamp - started_at),
                "oracle_evaluation_index": oracle_index,
                "candidate_id": candidate_id,
                "energy": energy,
                "utility": utility,
                "hamming_distance": hamming,
                "best_energy": best_energy,
                "best_utility": best_utility,
                "search_method": search_method,
            }
        )
    return rows


def _write_proposal_trace(runtime: CampaignRuntime) -> None:
    path = runtime.run_dir / "proposal_trace.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for event in runtime.events():
            if event.get("event_type") in _PROPOSAL_EVENT_TYPES:
                handle.write(json.dumps(event, sort_keys=True) + chr(10))


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _positive_int(value: Any, name: str) -> int:
    result = _nonnegative_int(value, name)
    if result < 1:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ["inventory_official_outputs", "write_campaign_reports"]
