"""Factor-aware exact GP-UCB selection for finite reaction-condition reservoirs."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ldm_tts.contracts import AcquisitionSpec, Candidate
from ldm_tts.optimization import (
    BOObservation,
    BOPrediction,
    BOSelectionResult,
    SurrogateVector,
)
from ldm_tts.optimization.acquisition import make_acquisition
from tasks.iron_mind.core.reaction_kernel import (
    ReactionKernelParameters,
    default_kernel_parameters,
    learn_kernel_parameters,
    posterior_terms,
    reaction_ard_kernel,
)
from tasks.iron_mind.core.schema import ReactionDatasetSchema
from tasks.iron_mind.core.surrogate import decode_reaction_one_hot

MIN_HISTORY_FOR_ARD = 8
TARGET_STD_FLOOR = 1.0
DEFAULT_MODEL_MISMATCH_VARIANCE = 0.04
PRIOR_MEAN_CLIP = 5.0


@dataclass(frozen=True)
class ReactionGPUCBConfig:
    """Fixed, campaign-local settings for categorical GP-UCB selection."""

    base_beta: float = 1.0
    noise: float = DEFAULT_MODEL_MISMATCH_VARIANCE
    target_std_floor: float = TARGET_STD_FLOOR

    def __post_init__(self) -> None:
        if not math.isfinite(self.base_beta) or self.base_beta < 0:
            raise ValueError("base_beta must be finite and non-negative")
        if not math.isfinite(self.noise) or self.noise <= 0:
            raise ValueError("noise must be finite and positive")
        if not math.isfinite(self.target_std_floor) or self.target_std_floor <= 0:
            raise ValueError("target_std_floor must be finite and positive")


class ReactionCategoricalGPUCBSelector:
    """Rank an LDM reaction reservoir with a schema-aware categorical GP."""

    def __init__(
        self,
        *,
        schema: ReactionDatasetSchema,
        objective_name: str,
        beta: float = 1.0,
        feature_version: str = "",
    ) -> None:
        self.schema = schema
        self.objective_name = str(objective_name)
        self.config = ReactionGPUCBConfig(
            base_beta=float(beta),
        )
        self.feature_version = str(feature_version)
        self.history: list[BOObservation] = []
        self.surrogate = _ReactionCategoricalGPSurrogate(
            schema,
            (),
            self.config,
            history_prior_mean=None,
            mean_source="zero",
            artifact_digest=None,
        )

    def describe(self) -> AcquisitionSpec:
        return AcquisitionSpec(
            name="ucb",
            objective_names=(self.objective_name,),
            score_direction="maximize",
            selection_rule="highest factor-aware categorical ARD GP upper confidence bound",
            parameters={
                "base_beta": self.config.base_beta,
                "beta_schedule": "constant",
                "kernel": "factor_ard_categorical_rbf",
                "model_mismatch_variance": self.config.noise,
            },
        )

    def fit(
        self,
        history: Sequence[BOObservation],
        *,
        history_prior_mean: Sequence[float] | None = None,
        mean_source: str = "zero",
        artifact_digest: str | None = None,
    ) -> None:
        if any(len(item.objectives) != 1 for item in history):
            raise ValueError("ReactionCategoricalGPUCBSelector requires one objective")
        if not mean_source.strip():
            raise ValueError("reaction GP mean_source must not be empty")
        _validate_history_features(history, self.feature_version)
        self.history = list(history)
        self.surrogate = _ReactionCategoricalGPSurrogate(
            self.schema,
            self.history,
            self.config,
            history_prior_mean=history_prior_mean,
            mean_source=mean_source,
            artifact_digest=artifact_digest,
        )

    def select(
        self,
        candidates: Sequence[Candidate],
        representations: Mapping[str, SurrogateVector],
        *,
        count: int = 1,
        query_prior_mean: Sequence[float] | None = None,
    ) -> BOSelectionResult:
        if count < 1:
            raise ValueError("selection count must be positive")
        _validate_representations(candidates, representations, self.feature_version)
        prior, clip_count = _prior_vector(
            query_prior_mean,
            len(candidates),
            "query prior mean",
        )
        self.surrogate.query_prior_clip_count = clip_count
        effective_beta = self.config.base_beta
        predictions = tuple(
            self.surrogate.predict(
                candidate.candidate_id,
                representations[candidate.candidate_id],
                effective_beta,
                float(prior[index]),
            )
            for index, candidate in enumerate(candidates)
        )
        ranked = sorted(
            predictions,
            key=lambda item: (float(item.acquisition_score), item.candidate_id),
            reverse=True,
        )
        return BOSelectionResult(
            selected_candidate_ids=tuple(item.candidate_id for item in ranked[:count]),
            predictions=predictions,
            metadata={
                "surrogate": self.surrogate.summary(),
                "base_beta": self.config.base_beta,
                "effective_beta": effective_beta,
                "beta_schedule": "constant",
                "tie_breaker": "descending_candidate_id",
            },
        )


class _ReactionCategoricalGPSurrogate:
    """Small exact GP using reaction-factor distances instead of one-hot geometry."""

    def __init__(
        self,
        schema: ReactionDatasetSchema,
        observations: Sequence[BOObservation],
        config: ReactionGPUCBConfig,
        *,
        history_prior_mean: Sequence[float] | None,
        mean_source: str,
        artifact_digest: str | None,
    ) -> None:
        self.schema = schema
        self.observations = list(observations)
        self.config = config
        self.mean_source = mean_source
        self.artifact_digest = artifact_digest
        self.parameters = default_kernel_parameters(schema)
        self.ard_history_threshold = max(
            MIN_HISTORY_FOR_ARD,
            2 * len(schema.factors),
        )
        self.fit_status = "prior"
        self.y_mean = 0.0
        self.y_scale = config.target_std_floor
        self.codes: np.ndarray | None = None
        self.cholesky: np.ndarray | None = None
        self.alpha: np.ndarray | None = None
        self.history_prior_mean, self.history_prior_clip_count = _prior_vector(
            history_prior_mean,
            len(self.observations),
            "history prior mean",
        )
        self.query_prior_clip_count = 0
        self.residual_target_mean = 0.0
        self.residual_target_std = 0.0
        self._fit()

    def _fit(self) -> None:
        if not self.observations:
            return
        scores = np.asarray([item.scalar_score for item in self.observations], dtype=float)
        if not np.all(np.isfinite(scores)):
            raise ValueError("reaction GP observations must be finite")
        self.y_mean = float(scores.mean())
        self.y_scale = max(float(scores.std()), self.config.target_std_floor)
        self.codes = np.asarray(
            [decode_reaction_one_hot(item.feature_vector, self.schema) for item in self.observations],
            dtype=int,
        )
        targets = (scores - self.y_mean) / self.y_scale - self.history_prior_mean
        self.residual_target_mean = float(targets.mean())
        self.residual_target_std = float(targets.std())
        if len(self.observations) >= self.ard_history_threshold:
            self.parameters = learn_kernel_parameters(
                self.codes,
                targets,
                self.schema,
                noise=self.config.noise,
            )
            self.fit_status = "fitted_ard_marginal_likelihood"
        else:
            self.fit_status = "fitted_default_hyperparameters"
        self.cholesky, self.alpha = posterior_terms(
            self.codes,
            targets,
            self.schema,
            self.parameters,
            noise=self.config.noise,
        )

    def predict(
        self,
        candidate_id: str,
        feature: SurrogateVector,
        beta: float,
        prior_mean: float,
    ) -> BOPrediction:
        if not self.observations:
            residual_mean_z, std_z = 0.0, 1.0
        else:
            residual_mean_z, std_z = self._posterior(
                decode_reaction_one_hot(feature.values, self.schema)
            )
        mean = self.y_mean + self.y_scale * (prior_mean + residual_mean_z)
        std = self.y_scale * std_z
        acquisition = make_acquisition("ucb", minimize=(False,), beta=beta)
        return BOPrediction.scalar(
            candidate_id,
            mean=mean,
            std=std,
            acquisition_score=float(acquisition.score(mean, std)),
            metadata={
                "surrogate": "reaction_categorical_ard_gp",
                "fit_status": self.fit_status,
                "mean_source": self.mean_source,
                "prior_mean_standardized": prior_mean,
                "residual_mean_standardized": residual_mean_z,
            },
        )

    def _posterior(self, code: tuple[int, ...]) -> tuple[float, float]:
        assert self.cholesky is not None and self.alpha is not None and self.codes is not None
        cross = reaction_ard_kernel(np.asarray([code]), self.codes, self.schema, self.parameters)
        mean_z = float((cross @ self.alpha)[0])
        projected = np.linalg.solve(self.cholesky, cross.T)
        variance_z = max(
            self.parameters.signal_variance - float(np.sum(projected * projected)),
            1.0e-12,
        )
        return mean_z, math.sqrt(variance_z)

    def summary(self) -> dict[str, Any]:
        return {
            "name": "reaction_categorical_ard_gp",
            "fit_status": self.fit_status,
            "history_size": len(self.observations),
            "ard_history_threshold": self.ard_history_threshold,
            "target_mean": self.y_mean,
            "target_scale": self.y_scale,
            "mean_source": self.mean_source,
            "mean_artifact_sha256": self.artifact_digest,
            "prior_mean_clip_count": (
                self.history_prior_clip_count + self.query_prior_clip_count
            ),
            "residual_target_mean": self.residual_target_mean,
            "residual_target_std": self.residual_target_std,
            "kernel": self.parameters.to_dict(self.schema),
            "noise": self.config.noise,
        }


def _prior_vector(
    values: Sequence[float] | None,
    expected_size: int,
    label: str,
) -> tuple[np.ndarray, int]:
    raw = np.zeros(expected_size, dtype=float) if values is None else np.asarray(values, dtype=float)
    if raw.shape != (expected_size,) or not np.all(np.isfinite(raw)):
        raise ValueError(f"reaction GP {label} must be a finite aligned vector")
    clip_count = int(np.count_nonzero(np.abs(raw) > PRIOR_MEAN_CLIP))
    return np.clip(raw, -PRIOR_MEAN_CLIP, PRIOR_MEAN_CLIP), clip_count


def _validate_history_features(history: Sequence[BOObservation], feature_version: str) -> None:
    if not feature_version:
        return
    if any(item.feature is None or item.feature.version != feature_version for item in history):
        raise ValueError("reaction GP history representation version does not match the selector")


def _validate_representations(
    candidates: Sequence[Candidate],
    representations: Mapping[str, SurrogateVector],
    feature_version: str,
) -> None:
    missing = [item.candidate_id for item in candidates if item.candidate_id not in representations]
    if missing:
        raise ValueError("missing surrogate representation for candidate(s): " + ", ".join(missing))
    if feature_version and any(
        representations[item.candidate_id].version != feature_version for item in candidates
    ):
        raise ValueError("reaction GP representation version does not match the selector")


__all__ = [
    "PRIOR_MEAN_CLIP",
    "ReactionCategoricalGPUCBSelector",
    "ReactionGPUCBConfig",
    "ReactionKernelParameters",
    "reaction_ard_kernel",
]
