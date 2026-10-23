"""Task-specific measured research records."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ldm_tts.contracts import Observation
from ldm_tts.engine.run_store import atomic_json_write
from tasks.synthonbench.core.constants import OBJECTIVE_NAME
from tasks.synthonbench.core.prompting import serialize_observations

MEASURED_HISTORY_FILE = Path("measured_history/observations.json")


def serialize_measured_observations(
    observations: Sequence[Observation],
    space: Any,
) -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "candidate_id": observation.candidate_id,
            "round_index": observation.round_idx,
            **(
                serialize_observations((observation,), space)[0]
                if observation.evaluation.succeeded
                else {**observation.candidate.payload, OBJECTIVE_NAME: None}
            ),
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
    space: Any,
) -> None:
    atomic_json_write(
        artifact_root / MEASURED_HISTORY_FILE,
        {"observations": serialize_measured_observations(observations, space)},
    )
