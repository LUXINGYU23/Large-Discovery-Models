"""Qualification evidence for one frozen-table seed evaluation."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ldm_tts.contracts import EvaluationResult
from ldm_tts.optimization import BOObservation

from tasks.iron_mind.core.data import FrozenReactionTable
from tasks.iron_mind.core.qualification_support import (
    TASK_ID,
    TASK_ROOT,
    UPSTREAM_CONTRACT_PATH,
    QualificationRecordError,
    contract_reference,
    file_reference,
    load_iron_mind_contract,
    read_json_object,
    sha256_file,
)
from tasks.iron_mind.core.seed import (
    QualificationSeed,
    SEED_SOURCE,
    load_qualification_input,
)
from tasks.iron_mind.core.surrogate import ReactionOneHotEncoder


OBJECTIVE_NAME = "reaction_score"
BENCHMARK_JOBS = 1
QUALIFICATION_INPUT_PATH = TASK_ROOT / "resources" / "qualification_input.json"
SEED_EVALUATION_RECORD_PATH = TASK_ROOT / "resources" / "seed_evaluation_record.json"
QUALIFICATION_INPUT_REFERENCE = "tasks/iron_mind/resources/qualification_input.json"
EXPERIMENT_CONTRACT_REFERENCE = "tasks/iron_mind/experiment.json"
SEED_RECORD_FIELDS = frozenset(
    (
        "schema_version record_type task status excluded_from_campaign_budget "
        "qualification_input experiment_contract_at_evaluation candidate dataset_id "
        "source_row_id raw_row_sha256 schema_sha256 data_sha256 evaluation"
    ).split()
)


class SeedEvaluationRecordError(QualificationRecordError):
    """Raised when one result cannot qualify the pinned seed."""


@dataclass(frozen=True)
class QualificationSeedPrior:
    """One verified warm-start observation kept outside campaign state."""

    seed: QualificationSeed
    observation: BOObservation

    @property
    def blocked_canonical_keys(self) -> tuple[str, ...]:
        """Return the seed key reserved from proposal admission."""

        return (self.seed.candidate.canonical_key,)


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


def load_qualification_seed_prior(
    *,
    input_path: Path,
    record_path: Path,
    table: FrozenReactionTable,
    expected_data_sha256: str,
    encoder: ReactionOneHotEncoder,
) -> QualificationSeedPrior:
    """Rebuild one seed prior only when its tracked record matches the table."""

    if encoder.schema != table.schema:
        raise SeedEvaluationRecordError("Qualification seed encoder schema does not match the pinned table.")
    seed = load_qualification_input(
        input_path,
        table=table,
        expected_data_sha256=expected_data_sha256,
    )
    record = _read_seed_evaluation_record(record_path)
    score = _require_seed_record(record, seed=seed, table=table, input_path=input_path)
    observation = BOObservation(
        candidate_id=seed.candidate.candidate_id,
        objectives=(score,),
        feature=encoder.encode(seed.candidate),
        metadata={"excluded_from_campaign_budget": True, "source": SEED_SOURCE},
    )
    return QualificationSeedPrior(seed, observation)


def load_tracked_qualification_seed_prior(
    *, input_path: Path, table: FrozenReactionTable
) -> QualificationSeedPrior:
    """Load the sole tracked seed input and record for a real campaign."""

    return load_qualification_seed_prior(
        input_path=input_path,
        record_path=SEED_EVALUATION_RECORD_PATH,
        table=table,
        expected_data_sha256=_buchwald_data_sha256(),
        encoder=ReactionOneHotEncoder(table.schema),
    )


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


def _read_seed_evaluation_record(path: Path) -> dict[str, Any]:
    try:
        return read_json_object(path, "seed evaluation record")
    except QualificationRecordError as exc:
        raise SeedEvaluationRecordError(str(exc)) from exc


def _buchwald_data_sha256() -> str:
    upstream = read_json_object(UPSTREAM_CONTRACT_PATH, "upstream contract")
    try:
        digest = upstream["datasets"]["buchwald_hartwig"]["artifacts"]["data"]["sha256"]
    except (KeyError, TypeError) as exc:
        raise SeedEvaluationRecordError("Upstream contract is missing the Buchwald data digest.") from exc
    if not isinstance(digest, str):
        raise SeedEvaluationRecordError("Upstream contract Buchwald data digest must be a string.")
    return digest


def _require_seed_record(
    record: Mapping[str, Any],
    *,
    seed: QualificationSeed,
    table: FrozenReactionTable,
    input_path: Path,
) -> float:
    if set(record) != SEED_RECORD_FIELDS:
        raise SeedEvaluationRecordError("Seed evaluation record fields do not match the compact contract.")
    _require_record_status(record)
    _require_input_reference(record, input_path)
    _require_historical_contract_reference(record)
    _require_record_identity(record, seed)
    score = _require_record_evaluation(record)
    _require_raw_buchwald_score(table, seed, score)
    return score


def _require_record_status(record: Mapping[str, Any]) -> None:
    if record.get("schema_version") != 1 or record.get("record_type") != "seed_evaluation":
        raise SeedEvaluationRecordError("Seed evaluation record does not identify schema v1 evidence.")
    if record.get("task") != TASK_ID or record.get("status") != "passed":
        raise SeedEvaluationRecordError("Seed evaluation record is not a passed Iron Mind record.")
    if record.get("excluded_from_campaign_budget") is not True:
        raise SeedEvaluationRecordError("Seed evaluation record must exclude the campaign budget.")


def _require_input_reference(record: Mapping[str, Any], input_path: Path) -> None:
    try:
        digest = sha256_file(input_path)
    except QualificationRecordError as exc:
        raise SeedEvaluationRecordError(str(exc)) from exc
    expected = {"path": QUALIFICATION_INPUT_REFERENCE, "sha256": digest}
    if record.get("qualification_input") != expected:
        raise SeedEvaluationRecordError("Seed evaluation record does not match the qualification input artifact.")


def _require_historical_contract_reference(record: Mapping[str, Any]) -> None:
    reference = record.get("experiment_contract_at_evaluation")
    if not isinstance(reference, Mapping) or set(reference) != {"path", "sha256"}:
        raise SeedEvaluationRecordError("Seed record must retain one historical experiment contract reference.")
    if reference.get("path") != EXPERIMENT_CONTRACT_REFERENCE:
        raise SeedEvaluationRecordError("Seed record references an unexpected experiment contract.")
    _require_sha256(reference.get("sha256"), "historical contract")


def _require_record_identity(record: Mapping[str, Any], seed: QualificationSeed) -> None:
    expected = {
        "candidate": {
            "candidate_id": seed.candidate.candidate_id,
            "canonical_key": seed.candidate.canonical_key,
        },
        "dataset_id": seed.candidate.payload["dataset_id"],
        "source_row_id": seed.source_row_id,
        "raw_row_sha256": seed.raw_row_sha256,
        "schema_sha256": seed.schema_sha256,
        "data_sha256": seed.data_sha256,
    }
    if any(record.get(field) != value for field, value in expected.items()):
        raise SeedEvaluationRecordError("Seed evaluation record identity does not match the pinned seed.")


def _require_record_evaluation(record: Mapping[str, Any]) -> float:
    evaluation = record.get("evaluation")
    if not isinstance(evaluation, Mapping) or set(evaluation) != {OBJECTIVE_NAME, "benchmark_jobs"}:
        raise SeedEvaluationRecordError("Seed evaluation record must contain one reaction_score and benchmark_jobs.")
    score = evaluation.get(OBJECTIVE_NAME)
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
        raise SeedEvaluationRecordError("Seed evaluation record reaction_score must be finite.")
    if evaluation.get("benchmark_jobs") != BENCHMARK_JOBS:
        raise SeedEvaluationRecordError("Seed evaluation record must retain exactly one benchmark job.")
    return float(score)


def _require_raw_buchwald_score(
    table: FrozenReactionTable, seed: QualificationSeed, score: float
) -> None:
    rows = table.rows_for_conditions(seed.candidate.payload["conditions"])
    if len(rows) != 1 or rows[0].row_id != seed.source_row_id:
        raise SeedEvaluationRecordError("Pinned seed row no longer resolves uniquely.")
    raw_score = rows[0].measurements.get("yield")
    if isinstance(raw_score, bool) or not isinstance(raw_score, (int, float)) or not math.isfinite(raw_score):
        raise SeedEvaluationRecordError("Pinned seed row has no finite Buchwald yield.")
    if float(raw_score) != score:
        raise SeedEvaluationRecordError("Seed evaluation record does not match the raw Buchwald score.")


def _require_sha256(value: object, label: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SeedEvaluationRecordError(
            f"Qualification seed {label} digest must be a lowercase SHA-256 value."
        )


__all__ = [
    "QualificationSeedPrior",
    "SEED_EVALUATION_RECORD_PATH",
    "SeedEvaluationRecordError",
    "build_seed_evaluation_record",
    "load_qualification_seed_prior",
    "load_tracked_qualification_seed_prior",
]
