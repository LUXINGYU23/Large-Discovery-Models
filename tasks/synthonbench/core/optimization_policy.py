"""Compiled-policy boundary for the SynthonBench task."""

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
from ldm_tts.optimization import BOObservation, BOPrediction
from tasks.synthonbench.core.constants import OBJECTIVE_NAME, TASK_ID
from tasks.synthonbench.core.policy_diagnostics import with_feedback
from tasks.synthonbench.core.policy_features import SynthonPolicyFeatureEncoder
from tasks.synthonbench.core.proposal_pool import empirical_base_masses
from tasks.synthonbench.core.tanimoto_gp import PRIOR_MEAN_CLIP, TARGET_STD_FLOOR

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


class SynthonOptimizationPolicyAdapter:
    def __init__(
        self,
        feature_encoder: SynthonPolicyFeatureEncoder,
        *,
        target: str,
        seed: int,
        acquisition_beta: float,
        gp_signal_std: float,
        gp_mean_std: float,
        gp_observation_noise_std: float,
        gp_reaction_weight: float,
        fingerprint_bits: int,
        gp_landmarks: int,
        gp_kernel_jitter: float,
        default_alpha: float,
        default_eta: float,
        proposal_sampling: Mapping[str, Any],
        enabled_capabilities: Sequence[str] = ("prior_mean@1", "ldm_weights@1"),
    ) -> None:
        if seed < 0:
            raise ValueError("Synthon policy seed must be non-negative")
        for name, value in (
            ("acquisition_beta", acquisition_beta),
            ("gp_signal_std", gp_signal_std),
            ("gp_mean_std", gp_mean_std),
            ("gp_observation_noise_std", gp_observation_noise_std),
            ("gp_reaction_weight", gp_reaction_weight),
            ("gp_kernel_jitter", gp_kernel_jitter),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"Synthon policy {name} must be finite and non-negative")
        if gp_signal_std == 0 or gp_observation_noise_std == 0:
            raise ValueError("Synthon policy GP signal and noise scales must be positive")
        if fingerprint_bits < 1 or gp_landmarks < 1 or gp_kernel_jitter <= 0:
            raise ValueError(
                "Synthon policy fingerprint, landmark, and jitter values must be positive"
            )
        self.features = feature_encoder
        self.target = target
        self.seed = seed
        self.proposal_sampling = dict(proposal_sampling)
        self.acquisition_beta = float(acquisition_beta)
        self.gp_parameters = {
            "signal_std": float(gp_signal_std),
            "mean_std": float(gp_mean_std),
            "observation_noise_std": float(gp_observation_noise_std),
            "reaction_weight": float(gp_reaction_weight),
            "fingerprint_bits": int(fingerprint_bits),
            "landmark_count": int(gp_landmarks),
            "kernel_jitter": float(gp_kernel_jitter),
        }
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
        round_index: int,
        history: Sequence[BOObservation],
        candidates: Sequence[Candidate],
        baseline_predictions: Sequence[BOPrediction],
        valid_proposal_occurrences: int,
    ) -> PolicyRoundInput:
        if not history:
            raise ValueError("compiled Synthon policy requires measured history")
        history_features = np.asarray(
            [self.features.encode_candidate_id(item.candidate_id) for item in history],
            dtype=float,
        )
        query_features = np.asarray(
            [self.features.encode_candidate(item) for item in candidates],
            dtype=float,
        )
        history_utilities = np.asarray(
            [item.scalar_score for item in history],
            dtype=float,
        )
        target_location = float(history_utilities.mean())
        target_scale = max(float(history_utilities.std()), TARGET_STD_FLOOR)
        q0 = empirical_base_masses(candidates)
        acquisition = np.asarray(
            [_prediction_acquisition(item) for item in baseline_predictions],
            dtype=float,
        )
        research_snapshot = {
            "task": TASK_ID,
            "target": self.target,
            "public_space": {
                "space_sha256": self.features.space_digest,
                "reaction_order": list(self.features.reactions),
                "max_slot_count": self.features.max_slots,
                "policy_feature_version": self.features.version,
            },
            "feature_contract": self._contract.to_dict(),
            "descriptor_standardization": self.features.descriptor_statistics(),
            "fixed_optimization": {
                "surrogate": "online Nyström/FITC count-Tanimoto residual GP",
                **self.gp_parameters,
                "target_std_floor": TARGET_STD_FLOOR,
                "acquisition": "GP-UCB",
                "acquisition_beta": self.acquisition_beta,
                "selection": (
                    "q0^alpha * exp(eta * robust_z(acquisition)), "
                    "then Gumbel top-k"
                ),
                "editable_components": ["prior_mean", "alpha", "eta"],
            },
            "proposal_pool": {
                "unique_candidate_count": len(candidates),
                "valid_proposal_occurrences": valid_proposal_occurrences,
                "reaction_summary": _reaction_summary(
                    candidates,
                    q0,
                    acquisition,
                ),
                "feature_summary": _feature_summary(query_features, q0),
            },
            "q0_summary": _probability_summary(q0),
            "baseline_acquisition_summary": _numeric_summary(acquisition),
        }
        execution_context = {
            "mean_context": {
                "round_index": round_index,
                "seed": self.seed,
                "target": self.target,
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
            measured_observations=tuple(
                {
                    "candidate_id": item.candidate_id,
                    "round_index": item.metadata["round_idx"],
                    OBJECTIVE_NAME: item.scalar_score,
                }
                for item in history
            ),
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
                        "synthon_prior_alignment",
                        "The prior mean does not align with the authoritative Synthon rows.",
                        "Return one finite standardized mean for every supplied feature row.",
                    )
                )
        return tuple(errors)


def _reaction_summary(
    candidates: Sequence[Candidate],
    q0: np.ndarray,
    acquisition: np.ndarray,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[int]] = {}
    for index, candidate in enumerate(candidates):
        payload = candidate.payload
        if not isinstance(payload, Mapping) or not isinstance(
            payload.get("reaction_id"), str
        ):
            raise ValueError("Synthon policy pool candidate has no reaction_id")
        grouped.setdefault(payload["reaction_id"], []).append(index)
    return [
        {
            "reaction_id": reaction_id,
            "candidate_count": len(indices),
            "q0_mass": float(q0[indices].sum()),
            "acquisition_mean": float(acquisition[indices].mean()),
            "acquisition_max": float(acquisition[indices].max()),
        }
        for reaction_id, indices in sorted(grouped.items())
    ]


def _feature_summary(features: np.ndarray, q0: np.ndarray) -> dict[str, Any]:
    if features.ndim != 2 or q0.shape != (len(features),):
        raise ValueError("Synthon policy pool features and q0 do not align")
    return {
        "mean": features.mean(axis=0).tolist(),
        "std": features.std(axis=0).tolist(),
        "q0_weighted_mean": (q0 @ features).tolist(),
    }


def _prediction_acquisition(prediction: BOPrediction) -> float:
    value = prediction.acquisition_score
    if value is None or not math.isfinite(value):
        raise ValueError("Synthon baseline acquisition must be finite")
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
        raise ValueError("Synthon q0 must be a finite non-empty vector")
    if np.any(values < 0) or not np.isclose(values.sum(), 1.0):
        raise ValueError("Synthon q0 must be a normalized probability vector")
    positive = values[values > 0]
    return {
        "count": len(values),
        "min": float(values.min()),
        "max": float(values.max()),
        "entropy": float(-np.sum(positive * np.log(positive))),
        "effective_sample_size": float(1.0 / np.sum(values * values)),
    }


__all__ = [
    "SynthonOptimizationPolicyAdapter",
    "policy_harness_profile",
]
