"""Iron Mind artifact exports built on shared campaign reporting helpers."""

from __future__ import annotations

from ldm_tts.engine.reporting import (
    build_campaign_result,
    build_trajectory_rows,
    load_successful_observations,
    write_trajectory_csv,
)
from ldm_tts.engine.run_store import CampaignRuntime, atomic_json_write


def write_campaign_reports(runtime: CampaignRuntime, *, objective_name: str) -> None:
    """Write standard result and trajectory artifacts for one finished campaign."""

    observations = load_successful_observations(runtime.run_dir / "checkpoint.json")
    rows = build_trajectory_rows(
        observations,
        objective_name=objective_name,
        direction="maximize",
    )
    write_trajectory_csv(
        runtime.run_dir / "trajectory.csv",
        rows,
        fieldnames=("evaluation", "round", "candidate_id", objective_name, f"best_{objective_name}"),
    )
    atomic_json_write(
        runtime.run_dir / "result.json",
        build_campaign_result(
            runtime.run_dir,
            objective_name=objective_name,
            direction="maximize",
        ),
    )
    _write_task_manifests(runtime)


def _write_task_manifests(runtime: CampaignRuntime) -> None:
    _write_manifest(
        runtime,
        "search_manifest.json",
        "rounds",
        "reservoir_built",
        artifacts={},
    )
    _write_manifest(
        runtime,
        "selection_record.json",
        "selections",
        "candidates_selected",
        artifacts={},
    )
    _write_manifest(
        runtime,
        "evaluation_manifest.json",
        "evaluations",
        "candidate_evaluated",
        artifacts={"result": "result.json", "trajectory": "trajectory.csv"},
    )


def _write_manifest(
    runtime: CampaignRuntime,
    filename: str,
    collection_name: str,
    event_type: str,
    *,
    artifacts: dict[str, str],
) -> None:
    events = [event for event in runtime.events() if event.get("event_type") == event_type]
    atomic_json_write(
        runtime.run_dir / filename,
        {
            "schema_version": 1,
            "task": "iron_mind",
            "run_id": runtime.run_id,
            collection_name: events,
            "artifacts": artifacts,
        },
    )
