"""Qualification evidence for one frozen-table seed evaluation."""

from __future__ import annotations

import math
from typing import Any

from ldm_tts.contracts import EvaluationResult

from tasks.iron_mind.core.qualification_support import (
    TASK_ID,
    TASK_ROOT,
    QualificationRecordError,
    contract_reference,
    file_reference,
    load_iron_mind_contract,
)
from tasks.iron_mind.core.seed import QualificationSeed, SEED_SOURCE


OBJECTIVE_NAME = "reaction_score"
BENCHMARK_JOBS = 1
QUALIFICATION_INPUT_PATH = TASK_ROOT / "resources" / "qualification_input.json"


class SeedEvaluationRecordError(QualificationRecordError):
    """Raised when one result cannot qualify the pinned seed."""


def build_seed_evaluation_record(
    *, seed: QualificationSeed, result: EvaluationResult
) -> dict[str, Any]:
    """Return passed evidence for exactly one successful seed evaluation."""

    _require_seed_identity(seed)
    score = _require_exact_seed_result(seed, result)
    contract = load_iron_mind_contract()
    return {
        "schema_version": 1,
        "record_type": "seed_evaluation",
        "task": TASK_ID,
        "status": "passed",
        "excluded_from_campaign_budget": True,
        "qualification_input": file_reference(QUALIFICATION_INPUT_PATH),
        "experiment_contract_at_evaluation": contract_reference(contract),
        "candidate": {
            "candidate_id": seed.candidate.candidate_id,
            "canonical_key": seed.candidate.canonical_key,
        },
        "dataset_id": seed.candidate.payload["dataset_id"],
        "source_row_id": seed.source_row_id,
        "raw_row_sha256": seed.raw_row_sha256,
        "schema_sha256": seed.schema_sha256,
        "data_sha256": seed.data_sha256,
        "evaluation": {OBJECTIVE_NAME: score, "benchmark_jobs": BENCHMARK_JOBS},
    }


def _require_seed_identity(seed: QualificationSeed) -> None:
    if seed.candidate.source != SEED_SOURCE:
        raise SeedEvaluationRecordError("Qualification evidence requires a qualification seed candidate.")
    if (
        not isinstance(seed.source_row_id, int)
        or isinstance(seed.source_row_id, bool)
        or seed.source_row_id < 1
    ):
        raise SeedEvaluationRecordError("Qualification seed row identity must be a positive integer.")
    for label, digest in {
        "raw row": seed.raw_row_sha256,
        "schema": seed.schema_sha256,
        "data": seed.data_sha256,
    }.items():
        _require_sha256(digest, label)


def _require_exact_seed_result(seed: QualificationSeed, result: EvaluationResult) -> float:
    if result.candidate_id != seed.candidate.candidate_id:
        raise SeedEvaluationRecordError("Seed evaluation candidate does not match the qualification seed.")
    if not result.succeeded:
        raise SeedEvaluationRecordError("Qualification seed evaluation must succeed.")
    if set(result.metrics) != {OBJECTIVE_NAME}:
        raise SeedEvaluationRecordError("Qualification seed evaluation must report only reaction_score.")
    score = result.metrics[OBJECTIVE_NAME]
    if not math.isfinite(score):
        raise SeedEvaluationRecordError("Qualification seed reaction_score must be finite.")
    if result.resource_usage != {"benchmark_jobs": float(BENCHMARK_JOBS)}:
        raise SeedEvaluationRecordError("Qualification seed evaluation must consume exactly one benchmark job.")
    return score


def _require_sha256(value: object, label: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SeedEvaluationRecordError(
            f"Qualification seed {label} digest must be a lowercase SHA-256 value."
        )


__all__ = ["SeedEvaluationRecordError", "build_seed_evaluation_record"]
