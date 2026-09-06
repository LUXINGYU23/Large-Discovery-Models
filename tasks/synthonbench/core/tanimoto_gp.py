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
TARGET_STD_FLOOR = 0.1
PRIOR_MEAN_CLIP = 5.0


@dataclass(frozen=True)
class TanimotoGPUCBConfig:
    """Numerical settings for the task-local sparse Tanimoto GP posterior."""

    beta: float = 1.0
    signal_std: float = 1.0
    mean_std: float = 1.0
    observation_noise_std: float = 1.0

    def __post_init__(self) -> None:
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
        self._signature: tuple[tuple[str, float, float], ...] = ()
        self._target_mean = 0.0
        self._target_scale = 1.0
        self._mean_source = "zero"
        self._artifact_digest: str | None = None
        self._history_prior_clip_count = 0
        self._query_prior_clip_count = 0
        self._residual_target_mean = 0.0
        self._residual_target_std = 0.0

    def describe(self) -> AcquisitionSpec:
        return AcquisitionSpec(
            name="ucb",
            objective_names=(self.objective_name,),
            score_direction="maximize",
            selection_rule="highest task-local Nyström/FITC count-Tanimoto GP upper confidence bound",
            parameters={
                "base_beta": self.config.beta,
                "beta_schedule": "constant",
                "surrogate": "online_nystrom_fitc_count_tanimoto_gaussian_process",
                "signal_std": self.config.signal_std,
                "mean_std": self.config.mean_std,
                "observation_noise_std": self.config.observation_noise_std,
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
        _validate_history(history, self.feature_dimension, self.feature_version)
        if not mean_source.strip():
            raise ValueError("Synthon Tanimoto GP mean_source must not be empty")
        prior, clip_count = _prior_vector(
            history_prior_mean,
            len(history),
            "history prior mean",
        )
        signature = tuple(
            (item.candidate_id, item.scalar_score, float(prior[index]))
            for index, item in enumerate(history)
        )
        self._mean_source = mean_source
        self._artifact_digest = artifact_digest
        self._history_prior_clip_count = clip_count
        if signature == self._signature:
            return
        self._posterior = _OnlinePosterior(self.feature_dimension)
        scores = np.asarray([item.scalar_score for item in history], dtype=float)
        self._target_mean, self._target_scale = _target_transform(scores)
        targets = (
            (scores - self._target_mean) / self._target_scale - prior
            if len(scores)
            else np.asarray((), dtype=float)
        )
        self._residual_target_mean = float(targets.mean()) if len(targets) else 0.0
        self._residual_target_std = float(targets.std()) if len(targets) else 0.0
        for observation, target in zip(history, targets, strict=True):
            feature, residual = self._latent_feature(observation.feature_vector)
            self._posterior.update(
                feature,
                float(target),
                self._observation_variance(residual),
            )
        self._signature = signature

    def posterior_projection(self, history, features):
        training = [self._latent_feature(item.feature_vector) for item in history]
        query = [self._latent_feature(row) for row in features]
        train_features = np.asarray([item[0] for item in training])
        query_features = np.asarray([item[0] for item in query])
        noise = np.asarray([self._observation_variance(item[1]) for item in training])
        weights = (query_features @ self._posterior.covariance @ train_features.T) / noise
        _, variance = self._posterior.predict(query_features)
        variance += self.config.signal_std**2 * np.asarray([item[1] for item in query])
        return {"weights": weights, "std": np.sqrt(np.maximum(variance, 0.0)),
                "location_scale": np.asarray([self._target_mean, self._target_scale])}

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
        _validate_representations(candidates, representations, self.feature_dimension, self.feature_version)
        prior, self._query_prior_clip_count = _prior_vector(
            query_prior_mean,
            len(candidates),
            "query prior mean",
        )
        predictions = self._predictions(candidates, representations, prior)
        ranked = sorted(predictions, key=_rank_key, reverse=True)
        return BOSelectionResult(
            selected_candidate_ids=tuple(item.candidate_id for item in ranked[:count]),
            predictions=predictions,
            metadata={
                "surrogate": self._summary(),
                "effective_beta": self.config.beta,
                "beta_schedule": "constant",
            },
        )

    def _predictions(
        self,
        candidates: Sequence[Candidate],
        representations: Mapping[str, SurrogateVector],
        prior_mean: np.ndarray,
    ) -> tuple[BOPrediction, ...]:
        if not candidates:
            return ()
        features_and_residuals = [self._latent_feature(representations[item.candidate_id].values) for item in candidates]
        features = np.asarray([item[0] for item in features_and_residuals], dtype=float)
        residuals = np.asarray([item[1] for item in features_and_residuals], dtype=float)
        normalized_means, normalized_variances = self._posterior.predict(features)
        total_variances = normalized_variances + self.config.signal_std**2 * residuals
        _validate_variances(total_variances)
        means = self._target_mean + self._target_scale * (
            prior_mean + normalized_means
        )
        stds = self._target_scale * np.sqrt(np.maximum(total_variances, 0.0))
        beta = self.config.beta
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
                    "mean_source": self._mean_source,
                    "prior_mean_standardized": float(prior),
                    "residual_mean_standardized": float(residual_mean),
                },
            )
            for candidate, mean, std, residual, prior, residual_mean in zip(
                candidates,
                means,
                stds,
                residuals,
                prior_mean,
                normalized_means,
                strict=True,
            )
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

    def _summary(self) -> dict[str, object]:
        return {
            **self._posterior.summary(),
            "target_standardization": "z_score",
            "target_mean": self._target_mean,
            "target_scale": self._target_scale,
            "mean_source": self._mean_source,
            "mean_artifact_sha256": self._artifact_digest,
            "prior_mean_clip_count": (
                self._history_prior_clip_count + self._query_prior_clip_count
            ),
            "residual_target_mean": self._residual_target_mean,
            "residual_target_std": self._residual_target_std,
        }


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


def _prior_vector(
    values: Sequence[float] | None,
    expected_size: int,
    label: str,
) -> tuple[np.ndarray, int]:
    raw = (
        np.zeros(expected_size, dtype=float)
        if values is None
        else np.asarray(values, dtype=float)
    )
    if raw.shape != (expected_size,) or not np.all(np.isfinite(raw)):
        raise ValueError(f"Synthon Tanimoto GP {label} must be a finite aligned vector")
    clip_count = int(np.count_nonzero(np.abs(raw) > PRIOR_MEAN_CLIP))
    return np.clip(raw, -PRIOR_MEAN_CLIP, PRIOR_MEAN_CLIP), clip_count


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


def _target_transform(scores: np.ndarray) -> tuple[float, float]:
    if not len(scores):
        return 0.0, 1.0
    return float(scores.mean()), max(float(scores.std()), TARGET_STD_FLOOR)


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


__all__ = [
    "PRIOR_MEAN_CLIP",
    "TARGET_STD_FLOOR",
    "SynthonTanimotoGPUCBSelector",
    "TanimotoGPUCBConfig",
]
