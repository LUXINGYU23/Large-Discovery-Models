"""Contracts for a non-campaign qualification seed evaluation record."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import MappingProxyType

import pytest

from ldm_tts.contracts import Candidate, EvaluationResult
from tasks.iron_mind.core.data import FrozenReactionTable, ReactionRow
from tasks.iron_mind.core.schema import load_reaction_schemas
from tasks.iron_mind.core.seed import (
    QualificationSeed,
    build_qualification_input,
    load_qualification_input,
)
from tasks.iron_mind.core.seed_evaluation import (
    SeedEvaluationRecordError,
    build_seed_evaluation_record,
    load_qualification_seed_prior,
)
from tasks.iron_mind.core.surrogate import ReactionOneHotEncoder


TASK_ROOT = Path(__file__).resolve().parents[1]
DATA_SHA256 = "a" * 64


def test_seed_record_requires_one_successful_non_campaign_evaluation() -> None:
    record = build_seed_evaluation_record(seed=_seed(), result=_result())

    assert record["status"] == "passed"
    assert record["excluded_from_campaign_budget"] is True
    assert record["evaluation"] == {"reaction_score": 12.5, "benchmark_jobs": 1}
    assert (
        record["qualification_input"]["path"]
        == "tasks/iron_mind/resources/qualification_input.json"
    )
    assert record["candidate"] == {"candidate_id": "seed-candidate", "canonical_key": "seed-key"}
    assert record["source_row_id"] == 7
    assert record["raw_row_sha256"] == "a" * 64


@pytest.mark.parametrize(
    "result",
    (
        EvaluationResult("seed-candidate", "failed", error="fixture failure"),
        EvaluationResult(
            "seed-candidate",
            "succeeded",
            metrics={"reaction_score": float("nan")},
        ),
        EvaluationResult("seed-candidate", "succeeded", metrics={"reaction_score": 12.5}),
        EvaluationResult(
            "seed-candidate",
            "succeeded",
            metrics={"reaction_score": 12.5},
            resource_usage={"benchmark_jobs": 2},
        ),
    ),
)
def test_seed_record_rejects_incomplete_or_nonexact_evaluation(result: EvaluationResult) -> None:
    with pytest.raises(SeedEvaluationRecordError):
        build_seed_evaluation_record(seed=_seed(), result=result)


def test_seed_record_rejects_evaluation_for_another_candidate() -> None:
    result = EvaluationResult(
        "another-candidate",
        "succeeded",
        metrics={"reaction_score": 12.5},
        resource_usage={"benchmark_jobs": 1},
    )

    with pytest.raises(SeedEvaluationRecordError):
        build_seed_evaluation_record(seed=_seed(), result=result)


def test_seed_prior_reuses_verified_record_outside_campaign_budget(tmp_path: Path) -> None:
    table = _prior_table()
    input_path = tmp_path / "qualification_input.json"
    input_path.write_text(
        json.dumps(build_qualification_input(table, data_sha256=DATA_SHA256)),
        encoding="utf-8",
    )
    seed = load_qualification_input(
        input_path,
        table=table,
        expected_data_sha256=DATA_SHA256,
    )
    record_path = tmp_path / "seed_evaluation_record.json"
    _write_seed_prior_record(record_path, input_path, seed, table)
    encoder = ReactionOneHotEncoder(table.schema)

    prior = load_qualification_seed_prior(
        input_path=input_path,
        record_path=record_path,
        table=table,
        expected_data_sha256=DATA_SHA256,
        encoder=encoder,
    )

    assert prior.seed == seed
    assert prior.observation.candidate_id == seed.candidate.candidate_id
    assert prior.observation.objectives == (10.0,)
    assert prior.observation.feature == encoder.encode(seed.candidate)
    assert prior.observation.metadata == {
        "excluded_from_campaign_budget": True,
        "source": "qualification_seed",
    }
    assert prior.blocked_canonical_keys == (seed.candidate.canonical_key,)


def test_seed_prior_rejects_a_record_with_a_changed_source_score(tmp_path: Path) -> None:
    table = _prior_table()
    input_path = tmp_path / "qualification_input.json"
    input_path.write_text(
        json.dumps(build_qualification_input(table, data_sha256=DATA_SHA256)),
        encoding="utf-8",
    )
    seed = load_qualification_input(
        input_path,
        table=table,
        expected_data_sha256=DATA_SHA256,
    )
    record_path = tmp_path / "seed_evaluation_record.json"
    _write_seed_prior_record(record_path, input_path, seed, table, score_offset=1.0)

    with pytest.raises(SeedEvaluationRecordError, match="raw Buchwald score"):
        load_qualification_seed_prior(
            input_path=input_path,
            record_path=record_path,
            table=table,
            expected_data_sha256=DATA_SHA256,
            encoder=ReactionOneHotEncoder(table.schema),
        )


def _seed() -> QualificationSeed:
    candidate = Candidate(
        candidate_id="seed-candidate",
        payload={"dataset_id": "buchwald_hartwig", "conditions": {}},
        canonical_key="seed-key",
        source="qualification_seed",
    )
    return QualificationSeed(candidate, 7, "a" * 64, "b" * 64, "c" * 64)


def _result() -> EvaluationResult:
    return EvaluationResult(
        "seed-candidate",
        "succeeded",
        metrics={"reaction_score": 12.5},
        resource_usage={"benchmark_jobs": 1},
    )


def _prior_table() -> FrozenReactionTable:
    schema = load_reaction_schemas(TASK_ROOT / "resources" / "reaction_schemas.json")[
        "buchwald_hartwig"
    ]
    rows = tuple(_prior_row(schema, row_id, index) for row_id, index in ((1, 0), (2, 1)))
    rows_by_conditions = {
        tuple(row.conditions[name] for name in schema.factor_names): (row,)
        for row in rows
    }
    return FrozenReactionTable(schema, rows, MappingProxyType(rows_by_conditions))


def _prior_row(schema, row_id: int, base_index: int) -> ReactionRow:
    conditions = {factor.name: factor.categories[0] for factor in schema.factors}
    conditions["base"] = schema.factors[0].categories[base_index]
    return ReactionRow(
        row_id=row_id,
        conditions=MappingProxyType(conditions),
        measurements=MappingProxyType({"yield": float(row_id * 10)}),
        raw_row_sha256=hashlib.sha256(f"prior-row-{row_id}".encode("utf-8")).hexdigest(),
    )


def _write_seed_prior_record(
    record_path: Path,
    input_path: Path,
    seed: QualificationSeed,
    table: FrozenReactionTable,
    *,
    score_offset: float = 0.0,
) -> None:
    row = table.rows_for_conditions(seed.candidate.payload["conditions"])[0]
    record = {
        "schema_version": 1,
        "record_type": "seed_evaluation",
        "task": "iron_mind",
        "status": "passed",
        "excluded_from_campaign_budget": True,
        "qualification_input": {
            "path": "tasks/iron_mind/resources/qualification_input.json",
            "sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        },
        "experiment_contract_at_evaluation": {
            "path": "tasks/iron_mind/experiment.json",
            "sha256": "b" * 64,
        },
        "candidate": {
            "candidate_id": seed.candidate.candidate_id,
            "canonical_key": seed.candidate.canonical_key,
        },
        "dataset_id": table.schema.dataset_id,
        "source_row_id": row.row_id,
        "raw_row_sha256": row.raw_row_sha256,
        "schema_sha256": table.schema.schema_sha256,
        "data_sha256": DATA_SHA256,
        "evaluation": {
            "reaction_score": row.measurements["yield"] + score_offset,
            "benchmark_jobs": 1,
        },
    }
    record_path.write_text(json.dumps(record), encoding="utf-8")
