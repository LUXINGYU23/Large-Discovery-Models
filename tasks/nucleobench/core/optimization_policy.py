"""Compiled-policy boundary for the NucleoBench task."""

from __future__ import annotations

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
from tasks.nucleobench.core.constants import TASK_ID
from tasks.nucleobench.core.hamming_gp import HammingGPUCBConfig, PRIOR_MEAN_CLIP
from tasks.nucleobench.core.policy_features import NucleoPolicyFeatureEncoder

_POLICY_PROFILE_ID = "policy_architect"
_HARNESS_RESOURCE_ROOT = Path(__file__).resolve().parents[1] / "resources" / "harness"
_LOCAL_PROFILE_PATH = _HARNESS_RESOURCE_ROOT / "profiles" / _POLICY_PROFILE_ID / "AGENTS.md"
_LOCAL_SKILL_ROOT = _HARNESS_RESOURCE_ROOT / "skills" / "compile-ldm-policy"


def policy_harness_profile(
    *,
    profile_root: Path = Path("/resources/profiles"),
    skill_root: Path = Path("/resources/skills/compile-ldm-policy"),
) -> tuple[HarnessProfile, ...]:
    return (
        HarnessProfile(
            _POLICY_PROFILE_ID,
            profile_root / _POLICY_PROFILE_ID / "AGENTS.md",
            skill_dirs=(skill_root,),
            agents_sha256=file_sha256(_LOCAL_PROFILE_PATH),
            skill_dir_sha256=(directory_sha256(_LOCAL_SKILL_ROOT),),
        ),
    )


class NucleoOptimizationPolicyAdapter:
    def __init__(
        self,
        feature_encoder: NucleoPolicyFeatureEncoder,
        *,
        seed: int,
        gp_config: HammingGPUCBConfig,
        default_alpha: float,
        default_eta: float,
        enabled_capabilities: Sequence[str] = ("prior_mean@1", "ldm_weights@1"),
    ) -> None:
        if seed < 0:
            raise ValueError("Nucleo policy seed must be non-negative")
        self.features = feature_encoder
        self.seed = seed
        self.gp_config = gp_config
        self._contract = PolicyCapabilityContract(
            task_id=TASK_ID,
            api_version=1,
            enabled_capabilities=tuple(enabled_capabilities),
            feature_names=feature_encoder.feature_names,
            feature_groups=feature_encoder.feature_groups,
            mean_clip=PRIOR_MEAN_CLIP,
            default_alpha=default_alpha,
            default_eta=default_eta,
        )

    def capability_contract(self) -> PolicyCapabilityContract:
        return self._contract

    def build_selection_round(
        self,
        *,
        history: Sequence[BOObservation],
        candidates: Sequence[Candidate],
        representations: Mapping[str, SurrogateVector],
        q0: np.ndarray,
        baseline_predictions: Sequence[BOPrediction],
        valid_proposal_occurrences: int,
    ) -> PolicyRoundInput:
        if not history:
            raise ValueError("compiled Nucleo policy requires measured history")
        history_features = np.asarray(
            [self.features.encode_vector(item.feature_vector) for item in history],
            dtype=float,
        )
        query_features = np.asarray(
            [
                self.features.encode_vector(
                    representations[item.candidate_id].values
                )
                for item in candidates
            ],
            dtype=float,
        )
        history_utilities = np.asarray(
            [item.scalar_score for item in history], dtype=float
        )
        round_index = _next_round_index(history)
        target_location = float(history_utilities.mean())
        target_scale = max(
            float(history_utilities.std()), self.gp_config.target_std_floor
        )
        acquisition = np.asarray(
            [_prediction_acquisition(item) for item in baseline_predictions],
            dtype=float,
        )
        research_snapshot = {
            "task": TASK_ID,
            "case": {
                "case_id": self.features.context.case.case_id,
                "target": self.features.context.case.target,
                "model_family": self.features.context.case.model_family,
                "sequence_length": self.features.context.case.sequence_length,
                "editable_position_count": len(self.features.positions),
                "start_set_digest": self.features.context.start_set_digest,
                "start_index": self.features.context.start_index,
                "policy_feature_version": self.features.version,
            },
            "feature_contract": self._contract.to_dict(),
            "fixed_optimization": {
                "surrogate": "exact normalized-Hamming residual GP",
                "noise_variance": self.gp_config.noise_variance,
                "jitter": self.gp_config.jitter,
                "target_std_floor": self.gp_config.target_std_floor,
                "length_scale_grid": list(self.gp_config.length_scale_grid),
                "max_observations": self.gp_config.max_observations,
                "acquisition": "GP-UCB",
                "acquisition_beta": self.gp_config.beta,
                "selection": "q0^alpha * exp(eta * robust_z(acquisition)), then Gumbel top-k",
                "editable_components": ["prior_mean", "alpha", "eta"],
            },
            "measured_observations": _serialized_history(history, self.features),
            "proposal_pool": {
                "unique_candidate_count": len(candidates),
                "valid_proposal_occurrences": valid_proposal_occurrences,
                "feature_summary": _feature_summary(query_features, q0),
            },
            "q0_summary": _probability_summary(q0),
            "baseline_acquisition_summary": _numeric_summary(acquisition),
        }
        execution_context = {
            "mean_context": {
                "round_index": round_index,
                "seed": self.seed,
                "case_id": self.features.context.case.case_id,
                "target": self.features.context.case.target,
                "feature_version": self.features.version,
                "feature_names": list(self._contract.feature_names),
                "feature_groups": {
                    name: list(bounds)
                    for name, bounds in self._contract.feature_groups.items()
                },
                "target_location": target_location,
                "target_scale": target_scale,
            },
            "weight_context": {
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
            research_snapshot=research_snapshot,
            execution_context=execution_context,
        )

    def validate_task_execution(
        self,
        execution: PolicyExecutionResult,
        round_input: PolicyRoundInput,
    ) -> Sequence[HarnessSubmissionError]:
        errors = []
        for path, values, expected in (
            ("/outputs/history_prior_mean", execution.history_prior_mean, len(round_input.history_features)),
            ("/outputs/query_prior_mean", execution.query_prior_mean, len(round_input.query_features)),
        ):
            if values.shape != (expected,) or not np.isfinite(values).all():
                errors.append(
                    HarnessSubmissionError(
                        path,
                        "nucleobench_prior_alignment",
                        "The prior mean does not align with the authoritative sequence rows.",
                        "Return one finite standardized mean for every supplied feature row.",
                    )
                )
        return tuple(errors)


def _next_round_index(history: Sequence[BOObservation]) -> int:
    rounds = [item.metadata.get("round_idx") for item in history]
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in rounds):
        raise ValueError("Nucleo BO history is missing authoritative round_idx metadata")
    return 1 + max(rounds)


def _serialized_history(
    history: Sequence[BOObservation],
    features: NucleoPolicyFeatureEncoder,
) -> list[dict[str, Any]]:
    return [
        {
            "round_index": item.metadata["round_idx"],
            **features.describe_vector(item.feature_vector),
            "utility": item.scalar_score,
        }
        for item in history
    ]


def _prediction_acquisition(prediction: BOPrediction) -> float:
    value = prediction.acquisition_score
    if value is None or not math.isfinite(value):
        raise ValueError("Nucleo baseline acquisition must be finite")
    return float(value)


def _feature_summary(features: np.ndarray, q0: np.ndarray) -> dict[str, Any]:
    if features.ndim != 2 or q0.shape != (len(features),):
        raise ValueError("Nucleo policy pool features and q0 do not align")
    return {
        "mean": features.mean(axis=0).tolist(),
        "std": features.std(axis=0).tolist(),
        "q0_weighted_mean": (q0 @ features).tolist(),
    }


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
        raise ValueError("Nucleo q0 must be a finite non-empty vector")
    if np.any(values < 0) or not np.isclose(values.sum(), 1.0):
        raise ValueError("Nucleo q0 must be a normalized probability vector")
    positive = values[values > 0]
    return {
        "count": len(values),
        "min": float(values.min()),
        "max": float(values.max()),
        "entropy": float(-np.sum(positive * np.log(positive))),
        "effective_sample_size": float(1.0 / np.sum(values * values)),
    }


__all__ = [
    "NucleoOptimizationPolicyAdapter",
    "policy_harness_profile",
]
