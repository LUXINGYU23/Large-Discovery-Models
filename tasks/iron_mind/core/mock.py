"""Deterministic mock assets for the Iron Mind shared-engine path."""

from __future__ import annotations

import csv
import hashlib
import json
from itertools import product
from pathlib import Path
from types import MappingProxyType

from ldm_tts.transport import ProposalResponse

from tasks.iron_mind.core.data import FrozenReactionTable, ReactionRow
from tasks.iron_mind.core.schema import ReactionDatasetSchema


MOCK_SEED_ROW_COUNT = 4


def load_mock_table(
    schema: ReactionDatasetSchema,
    oracle_path: Path,
    *,
    candidate_count: int = MOCK_SEED_ROW_COUNT,
) -> FrozenReactionTable:
    """Build a deterministic finite mock table sized for one reservoir."""

    _validate_candidate_count(candidate_count)
    seed_rows = _load_seed_rows(schema, oracle_path)
    rows = _expand_mock_rows(schema, seed_rows, candidate_count)
    indexed = {
        tuple(row.conditions[name] for name in schema.factor_names): (row,)
        for row in rows
    }
    return FrozenReactionTable(schema, rows, MappingProxyType(indexed))


def _load_seed_rows(
    schema: ReactionDatasetSchema, oracle_path: Path
) -> tuple[ReactionRow, ...]:
    with oracle_path.open("r", encoding="utf-8", newline="") as handle:
        rows = tuple(
            _mock_row(schema, index, row)
            for index, row in enumerate(csv.DictReader(handle), 1)
        )
    if len(rows) != MOCK_SEED_ROW_COUNT:
        raise ValueError("Mock oracle must contain exactly four rows.")
    return rows


def _expand_mock_rows(
    schema: ReactionDatasetSchema,
    seed_rows: tuple[ReactionRow, ...],
    candidate_count: int,
) -> tuple[ReactionRow, ...]:
    rows = list(seed_rows[:candidate_count])
    known = {
        tuple(row.conditions[name] for name in schema.factor_names) for row in rows
    }
    for values in product(*(factor.options for factor in schema.factors)):
        if len(rows) == candidate_count:
            break
        if values in known:
            continue
        rows.append(_synthetic_mock_row(schema, len(rows) + 1, values))
        known.add(values)
    if len(rows) != candidate_count:
        raise ValueError("Mock reservoir size exceeds the finite reaction domain.")
    return tuple(rows)


def mock_proposal_response(
    table: FrozenReactionTable, *, proposal_index: int
) -> ProposalResponse:
    """Return one deterministic response for one independent proposal request."""

    if proposal_index < 0 or proposal_index >= len(table.rows):
        raise ValueError("Mock proposal index is outside the mock table.")
    row = table.rows[proposal_index]
    text = json.dumps(
        {"dataset_id": table.schema.dataset_id, "conditions": dict(row.conditions)},
        separators=(",", ":"),
    )
    return ProposalResponse(
        text=text,
        metadata={"provider": "mock", "proposal_index": proposal_index},
    )


def _mock_row(
    schema: ReactionDatasetSchema, row_id: int, raw: dict[str, str]
) -> ReactionRow:
    expected = {"dataset_id", *schema.factor_names, "reaction_score"}
    if set(raw) != expected or raw["dataset_id"] != schema.dataset_id:
        raise ValueError("Mock oracle row does not match the tracked Buchwald schema.")
    conditions = {name: raw[name] for name in schema.factor_names}
    score = float(raw["reaction_score"])
    digest = hashlib.sha256(json.dumps(raw, sort_keys=True).encode("utf-8")).hexdigest()
    return ReactionRow(
        row_id,
        MappingProxyType(conditions),
        MappingProxyType({"yield": score}),
        digest,
    )


def _synthetic_mock_row(
    schema: ReactionDatasetSchema, row_id: int, values: tuple[object, ...]
) -> ReactionRow:
    conditions = dict(zip(schema.factor_names, values, strict=True))
    payload = {"dataset_id": schema.dataset_id, "conditions": conditions}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    score = 100.0 * int(digest[:8], 16) / 0xFFFFFFFF
    return ReactionRow(
        row_id=row_id,
        conditions=MappingProxyType(conditions),
        measurements=MappingProxyType({"yield": score}),
        raw_row_sha256=digest,
    )


def _validate_candidate_count(candidate_count: int) -> None:
    if (
        isinstance(candidate_count, bool)
        or not isinstance(candidate_count, int)
        or candidate_count < 1
    ):
        raise ValueError("Mock candidate count must be a positive integer.")


__all__ = ["MOCK_SEED_ROW_COUNT", "load_mock_table", "mock_proposal_response"]
