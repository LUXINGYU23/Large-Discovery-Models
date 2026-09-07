"""Measured research records projected from authoritative campaign observations."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ldm_tts.contracts import Observation
from ldm_tts.engine.run_store import atomic_json_write
from tasks.nucleobench.core.candidate import MutationContext, prepare_candidate_payload

MEASURED_HISTORY_FILE = Path("measured_history/observations.json")


def summarize_measured_observations(rows: Sequence[dict[str, Any]]) -> tuple[dict[str, Any], ...]:
    # Full patches stay in the paginated history tool, not every model request.
    return tuple({
        **{key: row[key] for key in ("candidate_id", "round_index", "utility")},
        "hamming_distance": len(row["mutations"]),
    } for row in rows)


def serialize_measured_observations(
    observations: Sequence[Observation], context: MutationContext,
) -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "candidate_id": observation.candidate_id,
            "round_index": observation.round_idx,
            "mutations": prepare_candidate_payload(
                observation.candidate.payload, context, allow_empty=True,
            ).payload["mutations"],
            "utility": observation.metrics["utility"],
            "research_annotations": observation.candidate.metadata.get("research_annotations", []),
        }
        for observation in observations
        if observation.evaluation.succeeded
    )


def write_measured_history(
    artifact_root: Path, observations: Sequence[Observation], context: MutationContext,
) -> None:
    atomic_json_write(
        artifact_root / MEASURED_HISTORY_FILE,
        {"observations": serialize_measured_observations(observations, context)},
    )
