"""Deterministic evaluation against source-pinned Iron Mind reaction tables."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

from ldm_tts.contracts import Candidate, EvaluationResult

from tasks.iron_mind.core.candidate import CandidatePayloadError, normalize_candidate_payload
from tasks.iron_mind.core.data import FrozenReactionTable, ReactionRow


CHAN_LAM_DATASET = "chan_lam_full"
RAW_MEASUREMENT_BY_DATASET = {
    "alkylation_deprotection": "yield",
    "amide_coupling_hte": "yield",
    "buchwald_hartwig": "yield",
    "reductive_amination": "percent_conversion",
    "suzuki_cernak": "conversion",
    "suzuki_doyle": "yield",
}


@dataclass(frozen=True)
class FrozenReactionEvaluator:
    """Score admitted reaction conditions with an injected frozen reaction table."""

    table: FrozenReactionTable

    def evaluate(self, candidate: Candidate) -> EvaluationResult:
        """Return one exact frozen-table score or an explicit failed evaluation."""

        try:
            conditions = _candidate_conditions(candidate, self.table)
            rows = self.table.rows_for_conditions(conditions)
            if not rows:
                raise ValueError("Candidate conditions are not present in the frozen reaction table.")
            score, replicate_scores = _score_rows(rows, self.table)
        except (CandidatePayloadError, ValueError) as exc:
            return EvaluationResult(candidate.candidate_id, "failed", error=str(exc))

        return EvaluationResult(
            candidate.candidate_id,
            "succeeded",
            metrics={"reaction_score": score},
            resource_usage={"benchmark_jobs": 1},
            metadata=_evaluation_metadata(self.table, rows, replicate_scores),
        )


def chan_lam_row_score(row: ReactionRow) -> float:
    """Score one Chan-Lam replicate using the pinned selectivity-weighted yield."""

    desired = _finite_measurement(row, "desired_yield")
    undesired = _finite_measurement(row, "undesired_yield")
    denominator = desired + undesired
    return 0.0 if denominator == 0.0 else desired / denominator * desired


def _candidate_conditions(candidate: Candidate, table: FrozenReactionTable) -> Mapping[str, str]:
    payload = normalize_candidate_payload(candidate.payload, table.schema)
    return payload["conditions"]


def _score_rows(
    rows: tuple[ReactionRow, ...], table: FrozenReactionTable
) -> tuple[float, list[float]]:
    dataset_id = table.schema.dataset_id
    measurement = RAW_MEASUREMENT_BY_DATASET.get(dataset_id)
    if measurement is not None:
        if len(rows) != 1:
            raise ValueError("Single-row Iron Mind candidates must map to exactly one frozen row.")
        score = _finite_measurement(rows[0], measurement)
        return score, [score]
    if dataset_id == CHAN_LAM_DATASET:
        scores = [chan_lam_row_score(row) for row in rows]
        if any(not math.isfinite(score) for score in scores):
            raise ValueError("Chan-Lam replicate score must be finite.")
        return min(scores), scores
    raise ValueError(f"Unsupported Iron Mind dataset: {dataset_id!r}.")


def _finite_measurement(row: ReactionRow, name: str) -> float:
    value = row.measurements.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Row {row.row_id} is missing numeric measurement {name!r}.")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"Row {row.row_id} has non-finite measurement {name!r}.")
    return numeric


def _evaluation_metadata(
    table: FrozenReactionTable,
    rows: tuple[ReactionRow, ...],
    replicate_scores: list[float],
) -> dict[str, Any]:
    return {
        "dataset_id": table.schema.dataset_id,
        "schema_sha256": table.schema.schema_sha256,
        "raw_measurements": [
            {"row_id": row.row_id, **dict(row.measurements)} for row in rows
        ],
        "replicate_scores": replicate_scores,
        "replicate_count": len(rows),
        "row_ids": [row.row_id for row in rows],
    }


__all__ = ["FrozenReactionEvaluator", "chan_lam_row_score"]
