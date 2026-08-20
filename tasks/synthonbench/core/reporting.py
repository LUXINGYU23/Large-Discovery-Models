"""Durable campaign reports and exact official audit integration."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from ldm_tts.engine import LDMEngineResult
from ldm_tts.engine.run_store import CampaignRuntime, atomic_json_write
from tasks.synthonbench.core.audit import (
    run_official_submission_audit,
    write_submission_csv,
)
from tasks.synthonbench.core.constants import OBJECTIVE_NAME
from tasks.synthonbench.core.data import LoadedSynthonBenchmark
from tasks.synthonbench.core.evaluator import OfficialSynthonEvaluator


def write_campaign_reports(
    runtime: CampaignRuntime,
    result: LDMEngineResult,
    evaluator: OfficialSynthonEvaluator,
    benchmark: LoadedSynthonBenchmark,
    *,
    audit_timeout_seconds: float,
) -> dict[str, Any]:
    """Write result.json, trajectory.csv, submission.csv, and official metrics."""

    submission = write_submission_csv(evaluator.task.observed_ids, runtime.run_dir / "submission.csv")
    metrics = _official_metrics(benchmark, evaluator, submission, runtime.run_dir, audit_timeout_seconds)
    _write_trajectory(runtime.run_dir / "trajectory.csv", evaluator.task.trace)
    payload = _result_payload(result, evaluator, benchmark, metrics)
    atomic_json_write(runtime.run_dir / "result.json", payload)
    _write_task_manifests(runtime)
    return payload


def _official_metrics(benchmark, evaluator, submission, run_dir, timeout_seconds: float) -> dict[str, Any]:
    if benchmark.mode == "mock":
        return _example_metrics(evaluator.task)
    return run_official_submission_audit(
        source_dir=benchmark.source_dir,
        data_dir=benchmark.data_dir,
        scale=benchmark.scale,
        target=benchmark.target,
        oracle_kind=benchmark.oracle_kind,
        submission_path=submission,
        output_path=run_dir / "official_audit.json",
        timeout_seconds=timeout_seconds,
    )


def _example_metrics(task: object) -> dict[str, Any]:
    from synthonbench.metrics import compute_exact_audit, evaluate_run

    audit = compute_exact_audit(task.space, task.oracle, allowed_reactions=task.allowed_reactions,
                                direction=task.direction, top_ks=(100, 1000))
    return evaluate_run(task.observed_ids, audit, submitted_calls=task.calls,
                        best_found_utility=task.best_utility)


def _write_trajectory(path: Path, trace: list[dict[str, object]]) -> None:
    fields = ("call_idx", "product_id", "reaction_id", "synthon_ids", "raw_score", "utility")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in trace:
            writer.writerow({**item, "synthon_ids": json.dumps(item["synthon_ids"])})


def _write_task_manifests(runtime: CampaignRuntime) -> None:
    _write_manifest(runtime, "search_manifest.json", "rounds", "reservoir_built", {})
    _write_manifest(runtime, "selection_record.json", "selections", "candidates_selected", {})
    _write_manifest(
        runtime,
        "evaluation_manifest.json",
        "evaluations",
        "candidate_evaluated",
        {"result": "result.json", "trajectory": "trajectory.csv", "submission": "submission.csv"},
    )


def _write_manifest(
    runtime: CampaignRuntime,
    filename: str,
    collection_name: str,
    event_type: str,
    artifacts: dict[str, str],
) -> None:
    events = [item for item in runtime.events() if item.get("event_type") == event_type]
    atomic_json_write(
        runtime.run_dir / filename,
        {
            "schema_version": 1,
            "task": "synthonbench",
            "run_id": runtime.run_id,
            collection_name: events,
            "artifacts": artifacts,
        },
    )


def _result_payload(result, evaluator, benchmark, metrics) -> dict[str, Any]:
    task = evaluator.task
    return {
        "task": "synthonbench",
        "mode": benchmark.mode,
        "scale": benchmark.scale,
        "target": benchmark.target,
        "oracle_kind": benchmark.oracle_kind,
        "engine_summary": result.summary,
        "official_calls": task.calls,
        "best_found_utility": task.best_utility,
        "best_candidate": _json_candidate(task.best_candidate),
        "objective": OBJECTIVE_NAME,
        "official_metrics": metrics,
        "artifacts": {"submission": "submission.csv", "trajectory": "trajectory.csv", "result": "result.json"},
    }


def _json_candidate(candidate) -> list[object] | None:
    if candidate is None:
        return None
    return [candidate[0], list(candidate[1])]


__all__ = ["write_campaign_reports"]
