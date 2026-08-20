"""Task-local online Nyström/FITC count-Tanimoto GP-UCB selector."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from ldm_tts.contracts import AcquisitionSpec, Candidate
from ldm_tts.optimization import BOObservation, BOPrediction, BOSelectionResult, SurrogateVector
from ldm_tts.optimization.acquisition import make_acquisition


MIN_VARIANCE_TOLERANCE = 1.0e-10


@dataclass(frozen=True)
class TanimotoGPUCBConfig:
    """Numerical settings for the task-local sparse Tanimoto GP posterior."""

    beta: float = 1.0
    confidence_delta: float = 0.1
    signal_std: float = 1.0
    mean_std: float = 1.0
    observation_noise_std: float = 1.0

    def __post_init__(self) -> None:
        _positive("confidence_delta", self.confidence_delta)
        if self.confidence_delta >= 1.0:
            raise ValueError("confidence_delta must be smaller than one")
        _positive("signal_std", self.signal_std)
        _nonnegative("mean_std", self.mean_std)
        _positive("observation_noise_std", self.observation_noise_std)
        _nonnegative("beta", self.beta)


class SynthonTanimotoGPUCBSelector:
    """Fit a fixed-basis FITC posterior and rank a finite proposal reservoir."""

    def __init__(
        self,
        *,
        objective_name: str,
        feature_dimension: int,
        feature_version: str,
        config: TanimotoGPUCBConfig,
    ) -> None:
        if feature_dimension < 2:
            raise ValueError("feature_dimension must include at least one landmark and one FITC residual")
        self.objective_name = str(objective_name)
        self.feature_dimension = int(feature_dimension)
        self.feature_version = str(feature_version)
        self.config = config
        self._posterior = _OnlinePosterior(self.feature_dimension)
        self._signature: tuple[tuple[str, float], ...] = ()

    def describe(self) -> AcquisitionSpec:
        return AcquisitionSpec(
            name="ucb",
            objective_names=(self.objective_name,),
            score_direction="maximize",
            selection_rule="highest task-local Nyström/FITC count-Tanimoto GP upper confidence bound",
            parameters={
                "base_beta": self.config.beta,
                "confidence_delta": self.config.confidence_delta,
                "surrogate": "online_nystrom_fitc_count_tanimoto_gaussian_process",
                "signal_std": self.config.signal_std,
                "mean_std": self.config.mean_std,
                "observation_noise_std": self.config.observation_noise_std,
            },
        )

    def fit(self, history: Sequence[BOObservation]) -> None:
        _validate_history(history, self.feature_dimension, self.feature_version)
        signature = tuple((item.candidate_id, item.scalar_score) for item in history)
        if signature[: len(self._signature)] != self._signature:
            self._posterior = _OnlinePosterior(self.feature_dimension)
            self._signature = ()
        for observation in history[len(self._signature) :]:
            feature, residual = self._latent_feature(observation.feature_vector)
            self._posterior.update(feature, observation.scalar_score, self._observation_variance(residual))
        self._signature = signature

    def select(
        self,
        candidates: Sequence[Candidate],
        representations: Mapping[str, SurrogateVector],
        *,
        count: int = 1,
    ) -> BOSelectionResult:
        if count < 1:
            raise ValueError("selection count must be positive")
        _validate_representations(candidates, representations, self.feature_dimension, self.feature_version)
        predictions = self._predictions(candidates, representations)
        ranked = sorted(predictions, key=_rank_key, reverse=True)
        return BOSelectionResult(
            selected_candidate_ids=tuple(item.candidate_id for item in ranked[:count]),
            predictions=predictions,
            metadata={
                "surrogate": self._posterior.summary(),
                "effective_beta": self._effective_beta(len(candidates)),
            },
        )

    def _predictions(
        self,
        candidates: Sequence[Candidate],
        representations: Mapping[str, SurrogateVector],
    ) -> tuple[BOPrediction, ...]:
        if not candidates:
            return ()
        features_and_residuals = [self._latent_feature(representations[item.candidate_id].values) for item in candidates]
        features = np.asarray([item[0] for item in features_and_residuals], dtype=float)
        residuals = np.asarray([item[1] for item in features_and_residuals], dtype=float)
        means, variances = self._posterior.predict(features)
        total_variances = variances + self.config.signal_std**2 * residuals
        _validate_variances(total_variances)
        stds = np.sqrt(np.maximum(total_variances, 0.0))
        beta = self._effective_beta(len(candidates))
        acquisition = make_acquisition("ucb", minimize=(False,), beta=beta)
        return tuple(
            BOPrediction.scalar(
                candidate.candidate_id,
                mean=float(mean),
                std=float(std),
                acquisition_score=float(acquisition.score(float(mean), float(std))),
                metadata={
                    "surrogate": "synthon_nystrom_fitc_tanimoto_gp",
                    "history_size": len(self._signature),
                    "fitc_residual": float(residual),
                },
            )
            for candidate, mean, std, residual in zip(candidates, means, stds, residuals, strict=True)
        )

    def _latent_feature(self, values: Sequence[float]) -> tuple[np.ndarray, float]:
        vector = np.asarray(values, dtype=float)
        if len(vector) != self.feature_dimension or not np.all(np.isfinite(vector)):
            raise ValueError("Synthon Tanimoto GP representation does not match the configured encoder")
        residual = float(vector[-1])
        if residual < -MIN_VARIANCE_TOLERANCE or residual > 1.0 + MIN_VARIANCE_TOLERANCE:
            raise ValueError("Synthon Tanimoto GP FITC residual must lie in [0, 1]")
        return np.concatenate(((self.config.mean_std,), self.config.signal_std * vector[:-1])), max(0.0, residual)

    def _observation_variance(self, residual: float) -> float:
        return self.config.observation_noise_std**2 + self.config.signal_std**2 * residual

    def _effective_beta(self, candidate_count: int) -> float:
        if self.config.beta == 0.0:
            return 0.0
        step = max(1, len(self._signature) + 1)
        argument = max(1.0, candidate_count * math.pi**2 * step**2 / (6.0 * self.config.confidence_delta))
        return self.config.beta * math.sqrt(2.0 * math.log(argument))


class _OnlinePosterior:
    """Rank-one Bayesian linear updates over fixed Nyström coordinates."""

    def __init__(self, dimension: int) -> None:
        self.mean = np.zeros(dimension, dtype=float)
        self.covariance = np.eye(dimension, dtype=float)
        self.count = 0

    def update(self, feature: np.ndarray, target: float, observation_variance: float) -> None:
        if not np.isfinite(target) or not np.all(np.isfinite(feature)):
            raise ValueError("Tanimoto GP observations must be finite")
        if not math.isfinite(observation_variance) or observation_variance <= 0.0:
            raise ValueError("Tanimoto GP observation variance must be finite and positive")
        projected = self.covariance @ feature
        denominator = observation_variance + float(feature @ projected)
        if denominator <= 0.0 or not math.isfinite(denominator):
            raise ValueError("Tanimoto GP posterior update has non-positive variance")
        gain = projected / denominator
        self.mean += gain * (target - float(feature @ self.mean))
        self.covariance -= np.outer(gain, projected)
        self.covariance = 0.5 * (self.covariance + self.covariance.T)
        self.count += 1

    def predict(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        means = features @ self.mean
        variances = np.einsum("ij,jk,ik->i", features, self.covariance, features)
        _validate_variances(variances)
        return means, np.maximum(variances, 0.0)

    def summary(self) -> dict[str, object]:
        return {
            "name": "online_nystrom_fitc_tanimoto_gp",
            "fit_status": "prior" if not self.count else "posterior",
            "history_size": self.count,
        }


def _validate_history(history: Sequence[BOObservation], dimension: int, version: str) -> None:
    for observation in history:
        if len(observation.objectives) != 1 or observation.feature is None:
            raise ValueError("Synthon Tanimoto GP requires one objective and one feature per observation")
        _validate_vector(observation.feature, dimension, version)


def _validate_representations(
    candidates: Sequence[Candidate],
    representations: Mapping[str, SurrogateVector],
    dimension: int,
    version: str,
) -> None:
    for candidate in candidates:
        if candidate.candidate_id not in representations:
            raise ValueError(f"missing surrogate representation for {candidate.candidate_id}")
        _validate_vector(representations[candidate.candidate_id], dimension, version)


def _validate_vector(vector: SurrogateVector, dimension: int, version: str) -> None:
    if vector.version != version or len(vector.values) != dimension:
        raise ValueError("Synthon Tanimoto GP representation does not match the configured encoder")
    if not np.all(np.isfinite(np.asarray(vector.values, dtype=float))):
        raise ValueError("Synthon Tanimoto GP representation must be finite")


def _validate_variances(variances: np.ndarray) -> None:
    if not np.all(np.isfinite(variances)) or np.any(variances < -MIN_VARIANCE_TOLERANCE):
        raise ValueError("Tanimoto GP predictive variance must be finite and non-negative")


def _rank_key(prediction: BOPrediction) -> tuple[float, str]:
    if prediction.acquisition_score is None:
        raise ValueError("Tanimoto GP prediction is missing UCB acquisition")
    return float(prediction.acquisition_score), prediction.candidate_id


def _positive(name: str, value: float) -> None:
    if not math.isfinite(float(value)) or float(value) <= 0.0:
        raise ValueError(f"{name} must be finite and positive")


def _nonnegative(name: str, value: float) -> None:
    if not math.isfinite(float(value)) or float(value) < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")


__all__ = ["SynthonTanimotoGPUCBSelector", "TanimotoGPUCBConfig"]
