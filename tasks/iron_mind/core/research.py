"""Task-specific measured research records."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ldm_tts.contracts import Observation
from ldm_tts.engine.run_store import atomic_json_write
from tasks.iron_mind.core.constants import OBJECTIVE_NAME

MEASURED_HISTORY_FILE = Path("measured_history/observations.json")


def summarize_measured_observations(
    rows: Sequence[dict[str, Any]]
) -> tuple[dict[str, Any], ...]:
    return tuple(
        {key: row[key] for key in ("candidate_id", "round_index", OBJECTIVE_NAME)}
        for row in rows
    )


def serialize_measured_observations(
    observations: Sequence[Observation],
) -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "candidate_id": observation.candidate_id,
            "round_index": observation.round_idx,
            **observation.candidate.payload,
            OBJECTIVE_NAME: observation.metrics.get(OBJECTIVE_NAME),
            "evaluation_status": observation.evaluation.status,
            "research_annotations": observation.candidate.metadata.get(
                "research_annotations", []
            ),
        }
        for observation in observations
    )


def write_measured_history(
    artifact_root: Path,
    observations: Sequence[Observation],
) -> None:
    atomic_json_write(
        artifact_root / MEASURED_HISTORY_FILE,
        {"observations": serialize_measured_observations(observations)},
    )
