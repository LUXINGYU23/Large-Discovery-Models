"""Measured research records and immutable candidate-file admission."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ldm_tts.contracts import Observation
from ldm_tts.engine.run_store import atomic_json_write
from ldm_tts.harness import HarnessSubmittedArtifact
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


def read_candidate_file(
    submission: Mapping[str, Any],
    artifacts: Sequence[HarnessSubmittedArtifact],
    artifact_root: Path,
    candidate_count: int,
) -> list[Any]:
    if dict(submission) != {"artifact_path": "candidates.json"} or len(artifacts) != 1:
        raise ValueError(
            'Write candidates.json and submit only {"artifact_path":"candidates.json"}.'
        )
    artifact = artifacts[0]
    if (
        artifact.path_pointer != "/artifact_path"
        or artifact.relative_path != "candidates.json"
    ):
        raise ValueError(
            "The snapshot must reference candidates.json at /artifact_path."
        )
    root = artifact_root.resolve()
    path = (root / artifact.snapshot_path).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(
            "The submitted snapshot is unavailable inside the artifact root."
        )
    body = path.read_bytes()
    if (
        len(body) != artifact.size_bytes
        or hashlib.sha256(body).hexdigest() != artifact.sha256
    ):
        raise ValueError(
            "The submitted snapshot does not match its recorded digest or size."
        )
    payload = json.loads(body.decode("utf-8"))
    if (
        not isinstance(payload, dict)
        or set(payload) != {"candidates"}
        or not isinstance(payload["candidates"], list)
    ):
        raise ValueError('candidates.json must contain only a "candidates" array.')
    candidates = payload["candidates"]
    if len(candidates) != candidate_count:
        raise ValueError(
            f"Expected exactly {candidate_count} candidates; found {len(candidates)}."
        )
    return candidates
