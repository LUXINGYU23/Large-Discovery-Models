"""Deterministic mock assets for the Iron Mind shared-engine path."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from types import MappingProxyType

from tasks.iron_mind.core.data import FrozenReactionTable, ReactionRow
from tasks.iron_mind.core.schema import ReactionDatasetSchema


def load_mock_table(schema: ReactionDatasetSchema, oracle_path: Path) -> FrozenReactionTable:
    with oracle_path.open("r", encoding="utf-8", newline="") as handle:
        rows = tuple(
            _mock_row(schema, index, row)
            for index, row in enumerate(csv.DictReader(handle), 1)
        )
    if len(rows) != 4:
        raise ValueError("Mock oracle must contain exactly four rows.")
    indexed = {
        tuple(row.conditions[name] for name in schema.factor_names): (row,)
        for row in rows
    }
    return FrozenReactionTable(schema, rows, MappingProxyType(indexed))


def mock_response(table: FrozenReactionTable) -> str:
    candidates = [
        {"dataset_id": table.schema.dataset_id, "conditions": dict(row.conditions)}
        for row in table.rows
    ]
    return json.dumps({"candidates": candidates}, separators=(",", ":"))


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


__all__ = ["load_mock_table", "mock_response"]
