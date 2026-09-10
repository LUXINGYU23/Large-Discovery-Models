"""Task-local feature and validation boundary for compiled Iron Mind policies."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ldm_tts.contracts import Candidate
from ldm_tts.harness import (
    HarnessProfile,
    HarnessSubmissionError,
    PolicyCapabilityContract,
    PolicyExecutionResult,
    PolicyRoundInput,
    directory_sha256,
    file_sha256,
)
from ldm_tts.optimization import BOObservation, BOPrediction, SurrogateVector
from tasks.iron_mind.core.constants import (
    FORBIDDEN_QUERY_TERMS,
    OBJECTIVE_NAME,
    TASK_ID,
)
from tasks.iron_mind.core.history import condition_evidence
from tasks.iron_mind.core.policy_diagnostics import with_feedback
from tasks.iron_mind.core.reaction_gp import (
    DEFAULT_MODEL_MISMATCH_VARIANCE,
    PRIOR_MEAN_CLIP,
    TARGET_STD_FLOOR,
)
from tasks.iron_mind.core.research import summarize_measured_observations
from tasks.iron_mind.core.schema import ReactionDatasetSchema
from tasks.iron_mind.core.surrogate import decode_reaction_one_hot

_POLICY_PROFILE_ID = "policy_architect"
_HARNESS_RESOURCE_ROOT = Path(__file__).resolve().parents[1] / "resources" / "harness"
_LOCAL_PROFILE_PATH = (
    _HARNESS_RESOURCE_ROOT
    / "profiles"
    / _POLICY_PROFILE_ID
    / "AGENTS.md"
)
_LOCAL_SKILL_ROOT = _HARNESS_RESOURCE_ROOT / "skills" / "compile-ldm-policy"


def policy_harness_profile() -> tuple[HarnessProfile, ...]:
    return (
        HarnessProfile(
            _POLICY_PROFILE_ID,
            Path("/resources/profiles/policy_architect/AGENTS.md"),
            skill_dirs=(Path("/resources/skills/compile-ldm-policy"),),
            agents_sha256=file_sha256(_LOCAL_PROFILE_PATH),
            skill_dir_sha256=(directory_sha256(_LOCAL_SKILL_ROOT),),
        ),
    )


class IronMindOptimizationPolicyAdapter:
    def __init__(
        self,
        schema: ReactionDatasetSchema,
        *,
        seed: int,
        acquisition_beta: float,
        default_alpha: float,
        default_eta: float,
        proposal_sampling: Mapping[str, Any],
        enabled_capabilities: Sequence[str] = ("prior_mean@1", "ldm_weights@1"),
    ) -> None:
        if seed < 0:
            raise ValueError("Iron Mind policy seed must be non-negative")
        if not math.isfinite(acquisition_beta) or acquisition_beta < 0:
            raise ValueError("Iron Mind policy acquisition beta must be non-negative")
        self.schema = schema
        self.seed = seed
        self.proposal_sampling = dict(proposal_sampling)
        self.acquisition_beta = float(acquisition_beta)
        self._contract = PolicyCapabilityContract(
            task_id=TASK_ID,
            api_version=1,
            enabled_capabilities=tuple(enabled_capabilities),
            feature_names=_feature_names(schema),
            feature_groups=_feature_groups(schema),
            mean_clip=PRIOR_MEAN_CLIP,
            default_alpha=default_alpha,
            default_eta=default_eta,
        )

    def capability_contract(self) -> PolicyCapabilityContract:
        return self._contract

    def build_selection_round(
        self,
        *,
        round_index: int,
        history: Sequence[BOObservation],
        candidates: Sequence[Candidate],
        representations: Mapping[str, SurrogateVector],
        q0: np.ndarray,
        baseline_predictions: Sequence[BOPrediction],
        valid_proposal_occurrences: int,
    ) -> PolicyRoundInput:
        if not history:
            raise ValueError("compiled Iron Mind policy requires measured history")
        history_features = np.asarray(
            [item.feature_vector for item in history], dtype=float
        )
        query_features = np.asarray(
            [representations[item.candidate_id].values for item in candidates],
            dtype=float,
        )
        history_utilities = np.asarray(
            [item.scalar_score for item in history], dtype=float
        )
        target_location = float(history_utilities.mean())
        target_scale = max(float(history_utilities.std()), TARGET_STD_FLOOR)
        acquisition = np.asarray(
            [_prediction_acquisition(item) for item in baseline_predictions],
            dtype=float,
        )
        measured = _serialized_history(history, self.schema)
        research_snapshot = {
            "task": TASK_ID,
            "dataset": {
                "dataset_id": self.schema.dataset_id,
                "schema_sha256": self.schema.schema_sha256,
                "objective": OBJECTIVE_NAME,
                "direction": "maximize",
                "factors": [
                    {
                        "name": factor.name,
                        "type": factor.parameter_type,
                        "options": list(factor.options),
                    }
                    for factor in self.schema.factors
                ],
            },
            "feature_contract": self._contract.to_dict(),
            "fixed_optimization": {
                "surrogate": "factor-aware categorical ARD residual GP",
                "model_mismatch_variance": DEFAULT_MODEL_MISMATCH_VARIANCE,
                "target_std_floor": TARGET_STD_FLOOR,
                "acquisition": "GP-UCB",
                "acquisition_beta": self.acquisition_beta,
                "selection": (
                    "q0^alpha * exp(eta * robust_z(acquisition)), "
                    "then Gumbel top-k"
                ),
                "editable_components": ["prior_mean", "alpha", "eta"],
            },
            "condition_evidence": condition_evidence(measured, self.schema),
            "proposal_pool": {
                "unique_candidate_count": len(candidates),
                "valid_proposal_occurrences": valid_proposal_occurrences,
                "factor_distribution": _factor_distribution(
                    query_features, q0, self.schema
                ),
            },
            "q0_summary": _probability_summary(q0),
            "baseline_acquisition_summary": _numeric_summary(acquisition),
            "forbidden_query_terms": list(FORBIDDEN_QUERY_TERMS),
        }
        execution_context = {
            "mean_context": {
                "round_index": round_index,
                "seed": self.seed,
                "dataset_id": self.schema.dataset_id,
                "schema_sha256": self.schema.schema_sha256,
                "objective": OBJECTIVE_NAME,
                "direction": "maximize",
                "feature_names": list(self._contract.feature_names),
                "feature_groups": {
                    name: list(bounds)
                    for name, bounds in self._contract.feature_groups.items()
                },
                "target_location": target_location,
                "target_scale": target_scale,
            },
            "weight_context": {
                "proposal_sampling": self.proposal_sampling,
                "seed": self.seed,
                "history_size": len(history),
                "unique_candidate_count": len(candidates),
                "valid_proposal_occurrences": valid_proposal_occurrences,
                "q0_summary": _probability_summary(q0),
                "baseline_acquisition_summary": _numeric_summary(acquisition),
                "default_alpha": self._contract.default_alpha,
                "default_eta": self._contract.default_eta,
            },
        }
        return PolicyRoundInput(
            round_index=round_index,
            history_features=history_features,
            history_utilities=history_utilities,
            query_features=query_features,
            history_candidate_ids=tuple(item.candidate_id for item in history),
            history_rounds=tuple(item.metadata["round_idx"] for item in history),
            measured_observations=summarize_measured_observations(measured),
            research_snapshot=research_snapshot,
            execution_context=execution_context,
        )

    def with_feedback(
        self, round_input: PolicyRoundInput, records: Sequence[Mapping[str, Any]],
    ) -> PolicyRoundInput:
        return with_feedback(round_input, records)

    def validate_task_execution(
        self,
        execution: PolicyExecutionResult,
        round_input: PolicyRoundInput,
    ) -> Sequence[HarnessSubmissionError]:
        errors = []
        for path, values, expected in (
            (
                "/outputs/history_prior_mean",
                execution.history_prior_mean,
                len(round_input.history_features),
            ),
            (
                "/outputs/query_prior_mean",
                execution.query_prior_mean,
                len(round_input.query_features),
            ),
        ):
            if values.shape != (expected,) or not np.isfinite(values).all():
                errors.append(
                    HarnessSubmissionError(
                        path,
                        "iron_mind_prior_alignment",
                        (
                            "The prior mean does not align with the authoritative "
                            "Iron Mind rows."
                        ),
                        (
                            "Return one finite standardized mean for every supplied "
                            "feature row."
                        ),
                    )
                )
        return tuple(errors)


def _feature_names(schema: ReactionDatasetSchema) -> tuple[str, ...]:
    return tuple(
        f"{factor.name}={json.dumps(option, ensure_ascii=False, separators=(',', ':'))}"
        for factor in schema.factors
        for option in factor.options
    )


def _feature_groups(schema: ReactionDatasetSchema) -> dict[str, tuple[int, int]]:
    groups: dict[str, tuple[int, int]] = {}
    offset = 0
    for factor in schema.factors:
        end = offset + len(factor.options)
        groups[factor.name] = (offset, end)
        offset = end
    return groups


def _serialized_history(
    history: Sequence[BOObservation], schema: ReactionDatasetSchema
) -> list[dict[str, Any]]:
    return [
        {
            "candidate_id": item.candidate_id,
            "round_index": item.metadata["round_idx"],
            "conditions": _conditions(item.feature_vector, schema),
            OBJECTIVE_NAME: item.scalar_score,
        }
        for item in history
    ]


def _conditions(
    values: Sequence[float], schema: ReactionDatasetSchema
) -> dict[str, Any]:
    codes = decode_reaction_one_hot(values, schema)
    return {
        factor.name: factor.options[code]
        for factor, code in zip(schema.factors, codes, strict=True)
    }


def _factor_distribution(
    features: np.ndarray,
    q0: np.ndarray,
    schema: ReactionDatasetSchema,
) -> dict[str, list[dict[str, Any]]]:
    if q0.shape != (len(features),) or not np.isfinite(q0).all():
        raise ValueError("Iron Mind q0 does not align with the maintained BO pool")
    codes = [decode_reaction_one_hot(row, schema) for row in features]
    distributions: dict[str, list[dict[str, Any]]] = {}
    for factor_index, factor in enumerate(schema.factors):
        distributions[factor.name] = [
            {
                "option": option,
                "candidate_count": sum(
                    code[factor_index] == option_index for code in codes
                ),
                "q0_mass": float(
                    sum(
                        q0[row_index]
                        for row_index, code in enumerate(codes)
                        if code[factor_index] == option_index
                    )
                ),
            }
            for option_index, option in enumerate(factor.options)
        ]
    return distributions


def _prediction_acquisition(prediction: BOPrediction) -> float:
    value = prediction.acquisition_score
    if value is None or not math.isfinite(value):
        raise ValueError("Iron Mind baseline acquisition must be finite")
    return float(value)


def _numeric_summary(values: np.ndarray) -> dict[str, float | int | None]:
    if not len(values):
        return {"count": 0, "min": None, "max": None, "mean": None, "std": None}
    return {
        "count": len(values),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std()),
    }


def _probability_summary(q0: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(q0, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Iron Mind q0 must be a finite non-empty vector")
    if np.any(values < 0) or not np.isclose(values.sum(), 1.0):
        raise ValueError("Iron Mind q0 must be a normalized probability vector")
    positive = values[values > 0]
    return {
        "count": len(values),
        "min": float(values.min()),
        "max": float(values.max()),
        "entropy": float(-np.sum(positive * np.log(positive))),
        "effective_sample_size": float(1.0 / np.sum(values * values)),
    }


__all__ = [
    "IronMindOptimizationPolicyAdapter",
    "policy_harness_profile",
]
