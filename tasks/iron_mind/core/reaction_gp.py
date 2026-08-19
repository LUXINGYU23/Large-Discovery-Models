"""Factor-aware exact GP-UCB selection for finite reaction-condition reservoirs."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from ldm_tts.contracts import AcquisitionSpec, Candidate
from ldm_tts.optimization import BOObservation, BOPrediction, BOSelectionResult, SurrogateVector
from ldm_tts.optimization.acquisition import make_acquisition
from tasks.iron_mind.core.reaction_kernel import (
    ReactionKernelParameters,
    default_kernel_parameters,
    learn_kernel_parameters,
    posterior_terms,
    reaction_ard_kernel,
)
from tasks.iron_mind.core.schema import ReactionDatasetSchema


MIN_HISTORY_FOR_ARD = 3
TARGET_STD_FLOOR = 1.0


@dataclass(frozen=True)
class ReactionGPUCBConfig:
    """Fixed, campaign-local settings for categorical GP-UCB selection."""

    base_beta: float = 1.0
    confidence_delta: float = 0.1
    noise: float = 1.0e-6
    target_std_floor: float = TARGET_STD_FLOOR

    def __post_init__(self) -> None:
        if not math.isfinite(self.base_beta) or self.base_beta < 0:
            raise ValueError("base_beta must be finite and non-negative")
        if not math.isfinite(self.confidence_delta) or not 0 < self.confidence_delta < 1:
            raise ValueError("confidence_delta must be between zero and one")
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
        confidence_delta: float = 0.1,
        feature_version: str = "",
    ) -> None:
        self.schema = schema
        self.objective_name = str(objective_name)
        self.config = ReactionGPUCBConfig(
            base_beta=float(beta),
            confidence_delta=float(confidence_delta),
        )
        self.feature_version = str(feature_version)
        self.history: list[BOObservation] = []
        self.surrogate = _ReactionCategoricalGPSurrogate(schema, (), self.config)

    def describe(self) -> AcquisitionSpec:
        return AcquisitionSpec(
            name="ucb",
            objective_names=(self.objective_name,),
            score_direction="maximize",
            selection_rule="highest factor-aware categorical ARD GP upper confidence bound",
            parameters={
                "base_beta": self.config.base_beta,
                "confidence_delta": self.config.confidence_delta,
                "kernel": "factor_ard_categorical_rbf",
            },
        )

    def fit(self, history: Sequence[BOObservation]) -> None:
        if any(len(item.objectives) != 1 for item in history):
            raise ValueError("ReactionCategoricalGPUCBSelector requires one objective")
        _validate_history_features(history, self.feature_version)
        self.history = list(history)
        self.surrogate = _ReactionCategoricalGPSurrogate(self.schema, self.history, self.config)

    def select(
        self,
        candidates: Sequence[Candidate],
        representations: Mapping[str, SurrogateVector],
        *,
        count: int = 1,
    ) -> BOSelectionResult:
        if count < 1:
            raise ValueError("selection count must be positive")
        _validate_representations(candidates, representations, self.feature_version)
        effective_beta = _finite_reservoir_beta(
            self.config,
            history_size=len(self.history),
            candidate_count=len(candidates),
        )
        predictions = tuple(
            self.surrogate.predict(
                candidate.candidate_id,
                representations[candidate.candidate_id],
                effective_beta,
            )
            for candidate in candidates
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
                "confidence_delta": self.config.confidence_delta,
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
    ) -> None:
        self.schema = schema
        self.observations = list(observations)
        self.config = config
        self.parameters = default_kernel_parameters(schema)
        self.fit_status = "prior"
        self.y_mean = 0.0
        self.y_scale = config.target_std_floor
        self.codes: np.ndarray | None = None
        self.cholesky: np.ndarray | None = None
        self.alpha: np.ndarray | None = None
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
            [_decode_one_hot(item.feature_vector, self.schema) for item in self.observations],
            dtype=int,
        )
        targets = (scores - self.y_mean) / self.y_scale
        if len(self.observations) >= MIN_HISTORY_FOR_ARD:
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
    ) -> BOPrediction:
        if not self.observations:
            mean, std = 0.0, self.config.target_std_floor
        else:
            mean, std = self._posterior(_decode_one_hot(feature.values, self.schema))
        acquisition = make_acquisition("ucb", minimize=(False,), beta=beta)
        return BOPrediction.scalar(
            candidate_id,
            mean=mean,
            std=std,
            acquisition_score=float(acquisition.score(mean, std)),
            metadata={"surrogate": "reaction_categorical_ard_gp", "fit_status": self.fit_status},
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
        return self.y_mean + mean_z * self.y_scale, math.sqrt(variance_z) * self.y_scale

    def summary(self) -> dict[str, Any]:
        return {
            "name": "reaction_categorical_ard_gp",
            "fit_status": self.fit_status,
            "history_size": len(self.observations),
            "target_mean": self.y_mean,
            "target_scale": self.y_scale,
            "kernel": self.parameters.to_dict(self.schema),
            "noise": self.config.noise,
        }


def _decode_one_hot(values: Sequence[float], schema: ReactionDatasetSchema) -> tuple[int, ...]:
    vector = np.asarray(values, dtype=float)
    if vector.shape != (schema.one_hot_dimension,) or not np.all(np.isfinite(vector)):
        raise ValueError("reaction GP feature vector does not match the schema")
    codes, offset = [], 0
    for factor in schema.factors:
        segment = vector[offset : offset + len(factor.options)]
        active = np.flatnonzero(np.isclose(segment, 1.0))
        if len(active) != 1 or not np.isclose(segment.sum(), 1.0):
            raise ValueError("reaction GP requires one active option per factor")
        codes.append(int(active[0]))
        offset += len(factor.options)
    return tuple(codes)


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


def _finite_reservoir_beta(
    config: ReactionGPUCBConfig,
    *,
    history_size: int,
    candidate_count: int,
) -> float:
    if config.base_beta == 0:
        return 0.0
    round_index = max(1, history_size + 1)
    reservoir_size = max(1, candidate_count)
    argument = reservoir_size * math.pi**2 * round_index**2 / (6.0 * config.confidence_delta)
    return config.base_beta * math.sqrt(2.0 * math.log(max(argument, 1.0)))


__all__ = [
    "ReactionCategoricalGPUCBSelector",
    "ReactionGPUCBConfig",
    "ReactionKernelParameters",
    "reaction_ard_kernel",
]
