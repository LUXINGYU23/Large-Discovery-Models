"""Immutable ResearchGym Table 3 protocol and result aggregation.

This module deliberately has no dependency on the four-case program-search
adapter.  Table 3 is an *agent* benchmark: one agent owns a complete task
workspace, edits code, executes experiments, and produces an official task
grade.  A result from the program-search sub-protocol must never be admitted
here.

The protocol accepts exactly three independent agent trajectories per task.
``Best@3`` selects one complete trajectory by its task primary score, then
reports every displayed metric from that same trajectory.  It does not take a
per-column maximum.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean, stdev
from typing import Any, Iterable, Mapping


UPSTREAM_COMMIT = "0adc08e93754b606d8a76055ed2f1c9574504312"
SEEDS = (0, 1, 2)
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class BudgetProfile:
    """A complete, fixed Table 3 budget profile."""

    name: str
    api_budget_usd: float
    wall_clock_hours: float
    purpose: str


@dataclass(frozen=True)
class TaskSpec:
    """Displayed and primary metrics for one official ResearchGym task."""

    task_id: str
    upstream_dir: str
    display_name: str
    metrics: tuple[str, ...]
    primary_metrics: tuple[str, ...]
    primary_note: str


PROFILES: dict[str, BudgetProfile] = {
    "table3": BudgetProfile(
        name="table3",
        api_budget_usd=10.0,
        wall_clock_hours=12.0,
        purpose="Table 3 RG-Agent-budget reproduction",
    ),
    "scaffold_comparison": BudgetProfile(
        name="scaffold_comparison",
        api_budget_usd=20.0,
        wall_clock_hours=24.0,
        purpose="matched agent-scaffold comparison",
    ),
}

# Metric keys are intentionally flat.  The task-specific official-grade
# extractor is responsible for deriving these only from its official summary.
TASKS: dict[str, TaskSpec] = {
    "continual_learning": TaskSpec(
        task_id="continual_learning",
        upstream_dir="continual-learning",
        display_name="CL",
        metrics=("acc", "aaa"),
        primary_metrics=("acc", "aaa"),
        primary_note="CIFAR100 final Acc and AAA",
    ),
    "materials_tokenization": TaskSpec(
        task_id="materials_tokenization",
        upstream_dir="materials-tokenization",
        display_name="MDT",
        metrics=("macro_f1", "micro_f1"),
        primary_metrics=("macro_f1", "micro_f1"),
        primary_note="mean test F1 over the primary NER-SOFC and NER-MatScholar subtasks",
    ),
    "cross_modal_retrieval": TaskSpec(
        task_id="cross_modal_retrieval",
        upstream_dir="cross-modal-retrieval",
        display_name="CMR",
        metrics=("i2tr_at_1", "t2ir_at_1"),
        primary_metrics=("i2tr_at_1", "t2ir_at_1"),
        primary_note="Base2Flickr I2TR@1 and T2IR@1",
    ),
    "time_series_explanation": TaskSpec(
        task_id="time_series_explanation",
        upstream_dir="time-series-explanation",
        display_name="TIM",
        metrics=("cpd_a", "cpd_z"),
        primary_metrics=("cpd_a", "cpd_z"),
        primary_note="PAM Average and Zeros CPD",
    ),
    "improving_replay_buffers": TaskSpec(
        task_id="improving_replay_buffers",
        upstream_dir="improving-replay-buffers",
        display_name="IRB",
        metrics=("return",),
        primary_metrics=("return",),
        primary_note="OpenAI Gym Hopper-v2 average return",
    ),
}


class ProtocolError(ValueError):
    """An artifact cannot support an exact Table 3 claim."""


def protocol_manifest(profile_name: str, *, model: str, harness: str,
                      external_access: str = "disabled") -> dict[str, Any]:
    """Create the immutable manifest written before any model call.

    ``external_access`` is explicit because a run without the paper's optional
    search services is a restricted reproduction, not silently an exact one.
    """
    profile = _profile(profile_name)
    if not model.strip() or not harness.strip():
        raise ProtocolError("model and harness are required")
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "benchmark": "ResearchGym",
        "source_commit": UPSTREAM_COMMIT,
        "protocol": "table3_agent_trajectory_v1",
        "profile": asdict(profile),
        "model": model,
        "harness": harness,
        "seeds": list(SEEDS),
        "external_access": external_access,
        "tasks": {
            task_id: {
                "upstream_dir": task.upstream_dir,
                "display_name": task.display_name,
                "metrics": list(task.metrics),
                "primary_metrics": list(task.primary_metrics),
                "primary_note": task.primary_note,
            }
            for task_id, task in TASKS.items()
        },
        "aggregation": {
            "runs_per_task": len(SEEDS),
            "mean": "arithmetic mean over exactly three completed trajectories",
            "std": "sample standard deviation (ddof=1) over exactly three trajectories",
            "best_at_3": "one trajectory maximizing the mean of primary metrics; report all its metrics",
        },
    }


def write_manifest(path: Path, *, profile_name: str, model: str, harness: str,
                   external_access: str = "disabled") -> dict[str, Any]:
    """Write once; refusing overwrite protects configuration provenance."""
    path = Path(path)
    if path.exists():
        raise ProtocolError(f"manifest already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = protocol_manifest(profile_name, model=model, harness=harness,
                                 external_access=external_access)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def validate_run_record(record: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
    """Validate a normalized, completed task result before it reaches a table."""
    task_id = record.get("task_id")
    if task_id not in TASKS:
        raise ProtocolError(f"unknown task_id: {task_id!r}")
    if record.get("status") != "completed":
        raise ProtocolError(f"{task_id}/seed={record.get('seed')} is not completed")
    if record.get("source_commit") != manifest.get("source_commit"):
        raise ProtocolError(f"{task_id}/seed={record.get('seed')} source commit mismatch")
    if record.get("protocol") != manifest.get("protocol"):
        raise ProtocolError(f"{task_id}/seed={record.get('seed')} is not an agent-trajectory result")
    if record.get("profile") != manifest["profile"]["name"]:
        raise ProtocolError(f"{task_id}/seed={record.get('seed')} profile mismatch")
    if record.get("seed") not in manifest["seeds"]:
        raise ProtocolError(f"{task_id}/seed={record.get('seed')} is not an approved seed")
    if record.get("wall_clock_hours", math.inf) > manifest["profile"]["wall_clock_hours"]:
        raise ProtocolError(f"{task_id}/seed={record.get('seed')} exceeded the wall-clock cap")
    if record.get("api_cost_usd", math.inf) > manifest["profile"]["api_budget_usd"]:
        raise ProtocolError(f"{task_id}/seed={record.get('seed')} exceeded the API cap")
    metrics = record.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ProtocolError(f"{task_id}/seed={record.get('seed')} has no metrics object")
    for metric in TASKS[task_id].metrics:
        value = metrics.get(metric)
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ProtocolError(f"{task_id}/seed={record.get('seed')} missing finite {metric}")
    evidence = record.get("official_grade")
    if not isinstance(evidence, Mapping) or not evidence.get("summary_sha256"):
        raise ProtocolError(f"{task_id}/seed={record.get('seed')} lacks official-grader evidence")


def aggregate(manifest: Mapping[str, Any], records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate an all-or-nothing five-task, three-seed Table 3 result."""
    if manifest.get("protocol") != "table3_agent_trajectory_v1":
        raise ProtocolError("wrong manifest protocol")
    by_task: dict[str, dict[int, Mapping[str, Any]]] = {name: {} for name in TASKS}
    for record in records:
        validate_run_record(record, manifest)
        task_id, seed = str(record["task_id"]), int(record["seed"])
        if seed in by_task[task_id]:
            raise ProtocolError(f"duplicate {task_id}/seed={seed}")
        by_task[task_id][seed] = record

    expected = set(manifest["seeds"])
    result: dict[str, Any] = {"status": "completed", "profile": manifest["profile"]["name"],
                              "model": manifest["model"], "harness": manifest["harness"],
                              "tasks": {}}
    for task_id, spec in TASKS.items():
        records_by_seed = by_task[task_id]
        seen = set(records_by_seed)
        if seen != expected:
            raise ProtocolError(f"{task_id} needs seeds {sorted(expected)}; found {sorted(seen)}")
        ordered = [records_by_seed[seed] for seed in manifest["seeds"]]
        mean_std = {
            metric: {"mean": fmean(float(item["metrics"][metric]) for item in ordered),
                     "std": stdev(float(item["metrics"][metric]) for item in ordered)}
            for metric in spec.metrics
        }
        def primary(item: Mapping[str, Any]) -> float:
            return fmean(float(item["metrics"][metric]) for metric in spec.primary_metrics)

        best = max(ordered, key=lambda item: (primary(item), -int(item["seed"])))
        result["tasks"][task_id] = {
            "display_name": spec.display_name,
            "mean_std": mean_std,
            "best_at_3": {metric: float(best["metrics"][metric]) for metric in spec.metrics},
            "best_seed": int(best["seed"]),
            "best_primary_score": primary(best),
            "run_ids": {str(seed): records_by_seed[seed].get("run_id") for seed in manifest["seeds"]},
        }
    return result


def load_json_records(paths: Iterable[Path]) -> list[dict[str, Any]]:
    """Load one normalized record per file, without accepting JSON arrays."""
    records: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ProtocolError(f"record is not an object: {path}")
        records.append(payload)
    return records


def _profile(name: str) -> BudgetProfile:
    try:
        return PROFILES[name]
    except KeyError as exc:
        raise ProtocolError(f"unknown profile {name!r}; choose from {', '.join(PROFILES)}") from exc
