"""Tests for frozen-table Iron Mind reaction evaluation."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from types import MappingProxyType

import pytest

from ldm_tts.contracts import Candidate
from tasks.iron_mind.core.data import FrozenReactionTable, ReactionRow
from tasks.iron_mind.core.evaluator import FrozenReactionEvaluator, chan_lam_row_score
from tasks.iron_mind.core.schema import (
    ReactionDatasetSchema,
    ReactionFactor,
    load_reaction_schemas,
)


TASK_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = TASK_ROOT / "resources" / "reaction_schemas.json"


def _schema(dataset_id: str) -> ReactionDatasetSchema:
    return load_reaction_schemas(SCHEMA_PATH)[dataset_id]


def _conditions(schema: ReactionDatasetSchema) -> dict[str, str]:
    return {factor.name: factor.categories[0] for factor in schema.factors}


def _row(
    row_id: int,
    conditions: dict[str, str],
    measurements: dict[str, float],
) -> ReactionRow:
    return ReactionRow(
        row_id=row_id,
        conditions=MappingProxyType(dict(conditions)),
        measurements=MappingProxyType(dict(measurements)),
        raw_row_sha256=hashlib.sha256(f"row-{row_id}".encode("utf-8")).hexdigest(),
    )


def _table(schema: ReactionDatasetSchema, rows: tuple[ReactionRow, ...]) -> FrozenReactionTable:
    grouped: dict[tuple[str, ...], list[ReactionRow]] = {}
    for row in rows:
        key = tuple(row.conditions[name] for name in schema.factor_names)
        grouped.setdefault(key, []).append(row)
    return FrozenReactionTable(
        schema=schema,
        rows=rows,
        rows_by_conditions=MappingProxyType(
            {key: tuple(group) for key, group in grouped.items()}
        ),
    )


def _candidate(schema: ReactionDatasetSchema, conditions: dict[str, str]) -> Candidate:
    return Candidate(
        candidate_id=f"candidate-{schema.dataset_id}",
        payload={"dataset_id": schema.dataset_id, "conditions": dict(conditions)},
        canonical_key=f"key-{schema.dataset_id}",
    )


def test_buchwald_returns_the_raw_yield_without_normalization() -> None:
    schema = _schema("buchwald_hartwig")
    conditions = _conditions(schema)
    row = _row(1, conditions, {"yield": 26.8886154})

    result = FrozenReactionEvaluator(_table(schema, (row,))).evaluate(_candidate(schema, conditions))

    assert result.status == "succeeded"
    assert result.metrics == {"reaction_score": 26.8886154}
    assert result.resource_usage == {"benchmark_jobs": 1.0}


@pytest.mark.parametrize(
    ("dataset_id", "measurement"),
    [
        ("alkylation_deprotection", "yield"),
        ("amide_coupling_hte", "yield"),
        ("buchwald_hartwig", "yield"),
        ("reductive_amination", "percent_conversion"),
        ("suzuki_cernak", "conversion"),
        ("suzuki_doyle", "yield"),
    ],
)
def test_all_single_row_official_datasets_report_the_raw_measurement(
    dataset_id: str, measurement: str
) -> None:
    schema = ReactionDatasetSchema(
        dataset_id=dataset_id,
        factors=(ReactionFactor("condition", ("A",)),),
        measurements=(measurement,),
        objective="reaction_score",
        direction="maximize",
        observation_policy="single_row",
        schema_sha256="0" * 64,
    )
    conditions = {"condition": "A"}
    row = _row(1, conditions, {measurement: 42.5})

    result = FrozenReactionEvaluator(_table(schema, (row,))).evaluate(
        _candidate(schema, conditions)
    )

    assert result.status == "succeeded"
    assert result.metrics == {"reaction_score": 42.5}
    assert result.metadata["raw_measurements"] == [{"row_id": 1, measurement: 42.5}]


def test_chan_lam_scores_single_zero_denominator_and_replicates_exactly() -> None:
    schema = _schema("chan_lam_full")
    conditions = _conditions(schema)
    single = _row(1, conditions, {"desired_yield": 78.8, "undesired_yield": 1.04})
    zero = _row(2, conditions, {"desired_yield": 0.0, "undesired_yield": 0.0})
    replicate_rows = (
        _row(3, conditions, {"desired_yield": 10.0, "undesired_yield": 0.0}),
        _row(4, conditions, {"desired_yield": 20.0, "undesired_yield": 0.0}),
        _row(5, conditions, {"desired_yield": 8.0, "undesired_yield": 8.0}),
    )

    assert chan_lam_row_score(single) == 78.8 / (78.8 + 1.04) * 78.8
    assert chan_lam_row_score(zero) == 0.0
    result = FrozenReactionEvaluator(_table(schema, replicate_rows)).evaluate(
        _candidate(schema, conditions)
    )
    assert result.metrics == {"reaction_score": 4.0}
    assert result.metadata["replicate_scores"] == [10.0, 20.0, 4.0]


def test_successful_result_preserves_diagnostics_outside_objective_metrics() -> None:
    schema = _schema("chan_lam_full")
    conditions = _conditions(schema)
    rows = (
        _row(1, conditions, {"desired_yield": 12.0, "undesired_yield": 3.0}),
        _row(2, conditions, {"desired_yield": 15.0, "undesired_yield": 5.0}),
    )

    result = FrozenReactionEvaluator(_table(schema, rows)).evaluate(_candidate(schema, conditions))

    assert result.metrics["reaction_score"] == pytest.approx(9.6)
    assert result.metadata == {
        "dataset_id": "chan_lam_full",
        "schema_sha256": schema.schema_sha256,
        "raw_measurements": [
            {"row_id": 1, "desired_yield": 12.0, "undesired_yield": 3.0},
            {"row_id": 2, "desired_yield": 15.0, "undesired_yield": 5.0},
        ],
        "replicate_scores": pytest.approx([9.6, 11.25]),
        "replicate_count": 2,
        "row_ids": [1, 2],
    }
    assert all(math.isfinite(value) for value in result.metrics.values())


def test_unknown_missing_or_non_finite_data_returns_an_explicit_failed_evaluation() -> None:
    schema = _schema("chan_lam_full")
    conditions = _conditions(schema)
    missing = _row(1, conditions, {"desired_yield": 12.0})
    non_finite = _row(2, conditions, {"desired_yield": float("nan"), "undesired_yield": 1.0})
    unknown_conditions = {**conditions, schema.factor_names[0]: schema.factors[0].categories[-1]}

    missing_result = FrozenReactionEvaluator(_table(schema, (missing,))).evaluate(
        _candidate(schema, conditions)
    )
    non_finite_result = FrozenReactionEvaluator(_table(schema, (non_finite,))).evaluate(
        _candidate(schema, conditions)
    )
    unknown_result = FrozenReactionEvaluator(_table(schema, (missing,))).evaluate(
        _candidate(schema, unknown_conditions)
    )

    for result in (missing_result, non_finite_result, unknown_result):
        assert result.status == "failed"
        assert result.metrics == {}
        assert result.resource_usage == {}
        assert result.error.isascii() and result.error


def test_non_finite_derived_score_returns_an_explicit_failed_evaluation() -> None:
    schema = _schema("chan_lam_full")
    conditions = _conditions(schema)
    row = _row(
        1,
        conditions,
        {"desired_yield": 1e308, "undesired_yield": -1e308 + 1e292},
    )

    result = FrozenReactionEvaluator(_table(schema, (row,))).evaluate(
        _candidate(schema, conditions)
    )

    assert result.status == "failed"
    assert result.metrics == {}
    assert result.resource_usage == {}
    assert result.error.isascii() and result.error
