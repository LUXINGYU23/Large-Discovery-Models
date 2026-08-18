"""Deterministic contracts for the source-pinned qualification seed input."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import MappingProxyType

import pytest

from tasks.iron_mind.core.candidate import canonical_candidate_bytes, canonical_candidate_key
from tasks.iron_mind.core.data import FrozenReactionTable, ReactionRow
from tasks.iron_mind.core.schema import load_reaction_schemas
from tasks.iron_mind.core.seed import (
    QualificationSeedError,
    build_qualification_input,
    load_qualification_input,
)


TASK_ROOT = Path(__file__).resolve().parents[1]
DATA_SHA256 = "a" * 64


def test_seed_builder_uses_canonical_payload_order_not_table_row_order() -> None:
    table = _table()
    reversed_table = _table(rows=tuple(reversed(table.rows)))

    seed = build_qualification_input(table, data_sha256=DATA_SHA256)

    assert seed == build_qualification_input(reversed_table, data_sha256=DATA_SHA256)
    expected = min(table.rows, key=lambda row: canonical_candidate_bytes(_payload(table, row)))
    assert seed["source_row_id"] == expected.row_id
    assert seed["raw_row_sha256"] == expected.raw_row_sha256
    assert seed["canonical_payload_sha256"] == canonical_candidate_key(seed["payload"])


def test_seed_loader_rebuilds_one_admitted_candidate(tmp_path: Path) -> None:
    table = _table()
    path = tmp_path / "qualification_input.json"
    path.write_text(json.dumps(build_qualification_input(table, data_sha256=DATA_SHA256)), encoding="utf-8")

    seed = load_qualification_input(path, table=table, expected_data_sha256=DATA_SHA256)

    assert seed.candidate.canonical_key == canonical_candidate_key(seed.candidate.payload)
    assert seed.source_row_id in {row.row_id for row in table.rows}
    assert seed.candidate.source == "qualification_seed"


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("source_row_id", 999),
        ("raw_row_sha256", "b" * 64),
        ("canonical_payload_sha256", "c" * 64),
        ("data_sha256", "d" * 64),
    ),
)
def test_seed_loader_rejects_unpinned_or_inconsistent_input(
    field: str, value: object, tmp_path: Path
) -> None:
    table = _table()
    payload = build_qualification_input(table, data_sha256=DATA_SHA256)
    payload[field] = value
    path = tmp_path / "qualification_input.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(QualificationSeedError):
        load_qualification_input(path, table=table, expected_data_sha256=DATA_SHA256)


def _table(*, rows: tuple[ReactionRow, ...] | None = None) -> FrozenReactionTable:
    schema = load_reaction_schemas(TASK_ROOT / "resources" / "reaction_schemas.json")["buchwald_hartwig"]
    actual_rows = rows or (_row(schema, 2, 0), _row(schema, 1, -1))
    grouped = {
        tuple(row.conditions[name] for name in schema.factor_names): (row,)
        for row in actual_rows
    }
    return FrozenReactionTable(schema, actual_rows, MappingProxyType(grouped))


def _row(schema, row_id: int, first_factor_index: int) -> ReactionRow:
    conditions = {factor.name: factor.categories[0] for factor in schema.factors}
    conditions[schema.factor_names[0]] = schema.factors[0].categories[first_factor_index]
    return ReactionRow(
        row_id=row_id,
        conditions=MappingProxyType(conditions),
        measurements=MappingProxyType({"yield": float(row_id)}),
        raw_row_sha256=hashlib.sha256(f"seed-row-{row_id}".encode("utf-8")).hexdigest(),
    )


def _payload(table: FrozenReactionTable, row: ReactionRow) -> dict[str, object]:
    return {"dataset_id": table.schema.dataset_id, "conditions": dict(row.conditions)}
