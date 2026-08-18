"""Strict source-pinned qualification seed input for Iron Mind campaigns."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ldm_tts.contracts import Candidate

from tasks.iron_mind.core.candidate import (
    CandidatePayloadError,
    canonical_candidate_bytes,
    canonical_candidate_key,
    normalize_candidate_payload,
)
from tasks.iron_mind.core.data import FrozenReactionTable, ReactionRow


SEED_SCHEMA_VERSION = 1
SEED_SOURCE = "qualification_seed"
BUCHWALD_DATASET_ID = "buchwald_hartwig"
SEED_FIELDS = {
    "schema_version",
    "task",
    "dataset_id",
    "payload",
    "canonical_payload_sha256",
    "source_row_id",
    "raw_row_sha256",
    "schema_sha256",
    "data_sha256",
}


class QualificationSeedError(ValueError):
    """Raised when a seed input is not reproducible from the pinned table."""


@dataclass(frozen=True)
class QualificationSeed:
    """One verified, non-campaign observation candidate."""

    candidate: Candidate
    source_row_id: int
    raw_row_sha256: str
    schema_sha256: str
    data_sha256: str


def build_qualification_input(
    table: FrozenReactionTable, *, data_sha256: str
) -> dict[str, Any]:
    """Build the lexicographically first canonical Buchwald seed document."""

    _require_buchwald_table(table)
    data_digest = _sha256_text(data_sha256, "data_sha256")
    ranked = sorted(
        ((_payload_for_row(table, row), row) for row in table.rows),
        key=lambda item: canonical_candidate_bytes(item[0]),
    )
    if len(ranked) > 1 and canonical_candidate_bytes(ranked[0][0]) == canonical_candidate_bytes(ranked[1][0]):
        raise QualificationSeedError("Pinned Buchwald table contains duplicate canonical payloads.")
    payload, row = ranked[0]
    return {
        "schema_version": SEED_SCHEMA_VERSION,
        "task": "iron_mind",
        "dataset_id": table.schema.dataset_id,
        "payload": payload,
        "canonical_payload_sha256": canonical_candidate_key(payload),
        "source_row_id": row.row_id,
        "raw_row_sha256": row.raw_row_sha256,
        "schema_sha256": table.schema.schema_sha256,
        "data_sha256": data_digest,
    }


def load_qualification_input(
    path: Path, *, table: FrozenReactionTable, expected_data_sha256: str
) -> QualificationSeed:
    """Load one seed only when every identity field matches the pinned table."""

    _require_buchwald_table(table)
    payload = _read_seed_object(path)
    _require_seed_fields(payload)
    expected_data_digest = _sha256_text(expected_data_sha256, "expected_data_sha256")
    if payload["data_sha256"] != expected_data_digest:
        raise QualificationSeedError("Qualification seed data digest does not match the pinned data.")
    if payload["schema_sha256"] != table.schema.schema_sha256:
        raise QualificationSeedError("Qualification seed schema digest does not match the pinned table.")
    candidate_payload = _normalize_seed_payload(payload, table)
    canonical_key = canonical_candidate_key(candidate_payload)
    if payload["canonical_payload_sha256"] != canonical_key:
        raise QualificationSeedError("Qualification seed canonical payload digest does not match.")
    row = _matching_row(payload, table, candidate_payload)
    candidate = Candidate(
        candidate_id=f"iron-mind-{canonical_key}",
        payload=candidate_payload,
        canonical_key=canonical_key,
        source=SEED_SOURCE,
        metadata={
            "dataset_id": table.schema.dataset_id,
            "schema_sha256": table.schema.schema_sha256,
            "source_row_id": row.row_id,
            "raw_row_sha256": row.raw_row_sha256,
        },
    )
    return QualificationSeed(
        candidate,
        row.row_id,
        row.raw_row_sha256,
        table.schema.schema_sha256,
        expected_data_digest,
    )


def _require_buchwald_table(table: FrozenReactionTable) -> None:
    if table.schema.dataset_id != BUCHWALD_DATASET_ID or not table.rows:
        raise QualificationSeedError("Qualification seed requires a non-empty Buchwald-Hartwig table.")


def _payload_for_row(table: FrozenReactionTable, row: ReactionRow) -> dict[str, Any]:
    return normalize_candidate_payload(
        {"dataset_id": table.schema.dataset_id, "conditions": dict(row.conditions)},
        table.schema,
    )


def _read_seed_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise QualificationSeedError(f"Could not read qualification seed: {path}") from exc
    if not isinstance(payload, dict):
        raise QualificationSeedError("Qualification seed must be a JSON object.")
    return payload


def _require_seed_fields(payload: Mapping[str, Any]) -> None:
    if set(payload) != SEED_FIELDS:
        raise QualificationSeedError("Qualification seed fields do not match the tracked contract.")
    if payload.get("schema_version") != SEED_SCHEMA_VERSION or payload.get("task") != "iron_mind":
        raise QualificationSeedError("Qualification seed identity does not match Iron Mind schema v1.")
    if payload.get("dataset_id") != BUCHWALD_DATASET_ID:
        raise QualificationSeedError("Qualification seed dataset does not match Buchwald-Hartwig.")
    _sha256_text(payload.get("canonical_payload_sha256"), "canonical_payload_sha256")
    _sha256_text(payload.get("raw_row_sha256"), "raw_row_sha256")
    _sha256_text(payload.get("schema_sha256"), "schema_sha256")
    _sha256_text(payload.get("data_sha256"), "data_sha256")


def _normalize_seed_payload(payload: Mapping[str, Any], table: FrozenReactionTable) -> dict[str, Any]:
    try:
        return normalize_candidate_payload(payload["payload"], table.schema)
    except CandidatePayloadError as exc:
        raise QualificationSeedError(str(exc)) from exc


def _matching_row(
    seed: Mapping[str, Any], table: FrozenReactionTable, payload: Mapping[str, Any]
) -> ReactionRow:
    row_id = seed.get("source_row_id")
    if isinstance(row_id, bool) or not isinstance(row_id, int) or row_id < 1:
        raise QualificationSeedError("Qualification seed source_row_id must be a positive integer.")
    matches = table.rows_for_conditions(payload["conditions"])
    rows = [row for row in matches if row.row_id == row_id and row.raw_row_sha256 == seed["raw_row_sha256"]]
    if len(rows) != 1:
        raise QualificationSeedError("Qualification seed row identity does not match the pinned table.")
    return rows[0]


def _sha256_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise QualificationSeedError(f"Qualification seed {label} must be a lowercase SHA-256 digest.")
    return value


__all__ = [
    "QualificationSeed",
    "QualificationSeedError",
    "build_qualification_input",
    "load_qualification_input",
]
