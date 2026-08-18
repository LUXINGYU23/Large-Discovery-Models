"""Contracts for a non-campaign qualification seed evaluation record."""

from __future__ import annotations

import pytest

from ldm_tts.contracts import Candidate, EvaluationResult
from tasks.iron_mind.core.seed import QualificationSeed
from tasks.iron_mind.core.seed_evaluation import (
    SeedEvaluationRecordError,
    build_seed_evaluation_record,
)

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
