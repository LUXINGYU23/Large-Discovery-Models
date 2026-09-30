"""Task-local categorical Hamming representation and exact GP-UCB."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from ldm_tts.contracts import AcquisitionSpec, Candidate, SurrogateSpaceSpec
from ldm_tts.optimization import (
    BOObservation,
    BOPrediction,
    BOSelectionResult,
    SurrogateVector,
)
from ldm_tts.optimization.acquisition import make_acquisition
from tasks.nucleobench.core.candidate import MutationContext, prepare_candidate_payload
from tasks.nucleobench.core.digests import canonical_json_sha256
from tasks.nucleobench.core.hamming_posterior import (
    MIN_VARIANCE, code_matrix, normalized_hamming_kernel, residual_moments,
)

ENCODER_ALGORITHM = "editable_categorical_hamming_v1"
ENCODER_PATH = "tasks.nucleobench.core.hamming_gp:NucleotideHammingEncoder"
BASE_CODE = {"A": 0.0, "C": 1.0, "G": 2.0, "T": 3.0}
PRIOR_MEAN_CLIP = 8.0


class NucleotideHammingEncoder:
    """Encode only official editable positions in their fixed mask order."""

    def __init__(self, context: MutationContext) -> None:
        self.context = context
        self.dimension = len(context.editable_positions)
        identity = {
            "algorithm": ENCODER_ALGORITHM,
            "case_id": context.case.case_id,
            "start_set_digest": context.start_set_digest,
            "start_index": context.start_index,
            "editable_positions": list(context.editable_positions),
        }
        digest = canonical_json_sha256(identity)
        self.version = f"{ENCODER_ALGORITHM}:{digest[:16]}"

    def describe(self) -> SurrogateSpaceSpec:
        return SurrogateSpaceSpec(
            kind="vector",
            representation=(
                "Position-ordered categorical DNA codes over the official editable mask; "
                "the surrogate compares codes only by equality."
            ),
            dimension_policy="fixed",
            dimension=self.dimension,
            encoder=ENCODER_PATH,
            version=self.version,
            metadata={
                "algorithm": ENCODER_ALGORITHM,
                "case_id": self.context.case.case_id,
                "start_set_digest": self.context.start_set_digest,
                "start_index": self.context.start_index,
                "editable_position_count": self.dimension,
            },
        )

    def encode(self, candidate: Candidate) -> SurrogateVector:
        prepared = prepare_candidate_payload(
            candidate.payload,
            self.context,
            allow_empty=True,
        )
        if prepared.canonical_key != candidate.canonical_key:
            raise ValueError(
                "candidate identity does not match the configured paired start"
            )
        mutations = {
            int(item["position"]): str(item["base"])
            for item in prepared.payload["mutations"]
        }
        values = tuple(
            BASE_CODE[mutations.get(position, self.context.start_sequence[position])]
            for position in self.context.editable_positions
        )
        return SurrogateVector(
            values,
            self.version,
            candidate.candidate_id,
            metadata={"encoder": ENCODER_ALGORITHM},
        )


@dataclass(frozen=True)
class HammingGPUCBConfig:
    """Numerical and bounded-history settings for the exact local GP."""

    beta: float = 1.0
    noise_variance: float = 1.0e-4
    jitter: float = 1.0e-8
    target_std_floor: float = 0.1
    min_history_for_fit: int = 4
    length_scale_grid: tuple[float, ...] = (0.02, 0.05, 0.1, 0.2, 0.4)
    max_observations: int = 256
    global_best_observations: int = 64

    def __post_init__(self) -> None:
        for name, value in (
            ("noise_variance", self.noise_variance),
            ("jitter", self.jitter),
            ("target_std_floor", self.target_std_floor),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.beta) or self.beta < 0.0:
            raise ValueError("beta must be finite and non-negative")
        if self.min_history_for_fit < 2:
            raise ValueError("min_history_for_fit must be at least 2")
        if self.max_observations < self.min_history_for_fit:
            raise ValueError("max_observations must cover the GP fit threshold")
        if not 1 <= self.global_best_observations <= self.max_observations:
            raise ValueError(
                "global_best_observations must lie within the working-set cap"
            )
        if not self.length_scale_grid or any(
            not math.isfinite(value) or value <= 0.0 for value in self.length_scale_grid
        ):
            raise ValueError("length_scale_grid must contain finite positive values")


class HammingGPUCBSelector:
    """Fit an exact normalized-Hamming GP over a bounded campaign history."""

    def __init__(
        self,
        *,
        objective_name: str,
        feature_dimension: int,
        feature_version: str,
        config: HammingGPUCBConfig | None = None,
    ) -> None:
        if feature_dimension < 1:
            raise ValueError("feature_dimension must be positive")
        self.objective_name = str(objective_name)
        self.feature_dimension = int(feature_dimension)
        self.feature_version = str(feature_version)
        self.config = config or HammingGPUCBConfig()
        self._history_size = 0
        self._working_ids: tuple[str, ...] = ()
        self._target_mean = 0.0
        self._target_scale = 1.0
        self._length_scale: float | None = None
        self._codes: np.ndarray | None = None
        self._cholesky: np.ndarray | None = None
        self._alpha: np.ndarray | None = None
        self._fit_status = "neutral_prior"
        self._mean_source = "zero"
        self._artifact_digest: str | None = None
        self._history_prior_clip_count = 0
        self._query_prior_clip_count = 0

    def describe(self) -> AcquisitionSpec:
        return AcquisitionSpec(
            name="ucb",
            objective_names=(self.objective_name,),
            score_direction="maximize",
            selection_rule="highest exact normalized-Hamming GP upper confidence bound",
            parameters={
                "beta": self.config.beta,
                "kernel": "normalized_hamming_exponential",
                "length_scale_grid": list(self.config.length_scale_grid),
                "min_history_for_fit": self.config.min_history_for_fit,
                "max_observations": self.config.max_observations,
                "global_best_observations": self.config.global_best_observations,
                "target_standardization": "z_score",
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
            raise ValueError("Hamming GP mean_source must not be empty")
        full_prior, clip_count = _prior_vector(
            history_prior_mean, len(history), "history prior mean"
        )
        prior_by_id = {
            observation.candidate_id: float(full_prior[index])
            for index, observation in enumerate(history)
        }
        working = _working_set(history, self.config)
        working_prior = np.asarray(
            [prior_by_id[item.candidate_id] for item in working], dtype=float
        )
        self._history_size = len(history)
        self._working_ids = tuple(item.candidate_id for item in working)
        scores = np.asarray([item.scalar_score for item in working], dtype=float)
        self._target_mean = float(scores.mean()) if len(scores) else 0.0
        self._target_scale = (
            max(float(scores.std()), self.config.target_std_floor)
            if len(scores)
            else 1.0
        )
        self._codes = self._cholesky = self._alpha = None
        self._length_scale = None
        self._fit_status = "neutral_prior"
        self._mean_source = mean_source
        self._artifact_digest = artifact_digest
        self._history_prior_clip_count = clip_count
        if len(working) < self.config.min_history_for_fit:
            return

        self._codes = np.asarray([item.feature_vector for item in working], dtype=float)
        targets = (scores - self._target_mean) / self._target_scale - working_prior
        self._length_scale, self._cholesky, self._alpha = _fit_grid_gp(
            self._codes,
            targets,
            self.config,
        )
        self._fit_status = "fitted_grid_length_scale"

    def posterior_projection(self, history, features):
        weights = np.zeros((len(features), len(history)))
        std = np.ones(len(features))
        if self._fit_status != "neutral_prior":
            cross = normalized_hamming_kernel(features, self._codes, self._length_scale)
            projected = np.linalg.solve(self._cholesky, cross.T)
            working_weights = np.linalg.solve(self._cholesky.T, projected).T
            indices = {item.candidate_id: index for index, item in enumerate(history)}
            weights[:, [indices[key] for key in self._working_ids]] = working_weights
            std = np.sqrt(np.maximum(1.0 - np.sum(projected**2, axis=0), MIN_VARIANCE))
        return {"weights": weights, "std": std,
                "location_scale": np.asarray([self._target_mean, self._target_scale])}

    def select(
        self,
        candidates: Sequence[Candidate],
        representations: Mapping[str, SurrogateVector],
        *,
        count: int = 1,
        round_idx: int = 0,
        query_prior_mean: Sequence[float] | None = None,
    ) -> BOSelectionResult:
        if count < 1:
            raise ValueError("selection count must be positive")
        _validate_representations(
            candidates,
            representations,
            self.feature_dimension,
            self.feature_version,
        )
        prior, self._query_prior_clip_count = _prior_vector(
            query_prior_mean, len(candidates), "query prior mean"
        )
        residuals, deviations = residual_moments(
            np.asarray([
                representations[candidate.candidate_id].values for candidate in candidates
            ]).reshape(len(candidates), self.feature_dimension),
            self._codes, self._cholesky, self._alpha, self._length_scale,
        )
        means = self._target_mean + self._target_scale * (prior + residuals)
        deviations = self._target_scale * deviations
        acquisition = make_acquisition("ucb", minimize=(False,), beta=self.config.beta)
        scores = acquisition.score(means, deviations)
        predictions = tuple(
            BOPrediction.scalar(
                candidate.candidate_id,
                mean=float(means[index]),
                std=float(deviations[index]),
                acquisition_score=float(scores[index]),
                metadata={
                    "surrogate": "nucleobench_exact_hamming_gp",
                    "fit_status": self._fit_status,
                    "mean_source": self._mean_source,
                    "prior_mean_standardized": float(prior[index]),
                    "residual_mean_standardized": float(residuals[index]),
                },
            )
            for index, candidate in enumerate(candidates)
        )
        ranked = sorted(
            predictions,
            key=lambda item: (float(item.acquisition_score), item.candidate_id),
            reverse=True,
        )
        return BOSelectionResult(
            tuple(item.candidate_id for item in ranked[:count]),
            predictions,
            metadata={"surrogate": self._summary(), "effective_beta": self.config.beta},
        )

    def posterior_snapshot(self) -> dict[str, object]:
        return {
            **self._summary(),
            "objective_name": self.objective_name,
            "feature_version": self.feature_version,
            "beta": self.config.beta,
            "training_codes": None if self._codes is None else self._codes.tolist(),
            "cholesky": None if self._cholesky is None else self._cholesky.tolist(),
            "alpha": None if self._alpha is None else self._alpha.tolist(),
        }

    def _summary(self) -> dict[str, object]:
        return {
            "name": "nucleobench_exact_hamming_gp",
            "fit_status": self._fit_status,
            "history_size": self._history_size,
            "working_set_size": len(self._working_ids),
            "working_set_candidate_ids": list(self._working_ids),
            "length_scale": self._length_scale,
            "target_mean": self._target_mean,
            "target_scale": self._target_scale,
            "mean_source": self._mean_source,
            "mean_artifact_sha256": self._artifact_digest,
            "prior_mean_clip_count": (
                self._history_prior_clip_count + self._query_prior_clip_count
            ),
        }


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
        raise ValueError(f"Hamming GP {label} must be a finite aligned vector")
    clip_count = int(np.count_nonzero(np.abs(raw) > PRIOR_MEAN_CLIP))
    return np.clip(raw, -PRIOR_MEAN_CLIP, PRIOR_MEAN_CLIP), clip_count


def _working_set(
    history: Sequence[BOObservation],
    config: HammingGPUCBConfig,
) -> tuple[BOObservation, ...]:
    latest_by_id: dict[str, tuple[int, BOObservation]] = {}
    for index, observation in enumerate(history):
        latest_by_id[observation.candidate_id] = (index, observation)
    unique = tuple(item for _, item in sorted(latest_by_id.values()))
    if len(unique) <= config.max_observations:
        return unique

    best = sorted(
        unique,
        key=lambda item: (-item.scalar_score, item.candidate_id),
    )[: config.global_best_observations]
    selected = {item.candidate_id for item in best}
    remaining = config.max_observations - len(best)
    recent = [item for item in reversed(unique) if item.candidate_id not in selected][
        :remaining
    ]
    selected.update(item.candidate_id for item in recent)
    return tuple(item for item in unique if item.candidate_id in selected)


def _fit_grid_gp(
    codes: np.ndarray,
    targets: np.ndarray,
    config: HammingGPUCBConfig,
) -> tuple[float, np.ndarray, np.ndarray]:
    best: tuple[float, float, np.ndarray, np.ndarray] | None = None
    identity = np.eye(len(codes), dtype=float)
    for length_scale in config.length_scale_grid:
        covariance = normalized_hamming_kernel(codes, codes, length_scale)
        covariance += config.noise_variance * identity
        cholesky = _cholesky(covariance, config.jitter)
        alpha = np.linalg.solve(cholesky.T, np.linalg.solve(cholesky, targets))
        log_likelihood = (
            -0.5 * float(targets @ alpha)
            - float(np.log(np.diag(cholesky)).sum())
            - 0.5 * len(targets) * math.log(2.0 * math.pi)
        )
        candidate = (log_likelihood, -length_scale, cholesky, alpha)
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    assert best is not None
    return -best[1], best[2], best[3]


def _cholesky(covariance: np.ndarray, jitter: float) -> np.ndarray:
    try:
        return np.linalg.cholesky(covariance + jitter * np.eye(len(covariance)))
    except np.linalg.LinAlgError as exc:
        raise ValueError("Hamming GP covariance is not positive definite") from exc


def _validate_history(
    history: Sequence[BOObservation],
    dimension: int,
    version: str,
) -> None:
    for observation in history:
        if len(observation.objectives) != 1 or observation.feature is None:
            raise ValueError("Hamming GP requires one objective and one feature")
        if not math.isfinite(observation.scalar_score):
            raise ValueError("Hamming GP observations must be finite")
        _validate_vector(observation.feature, dimension, version)


def _validate_representations(
    candidates: Sequence[Candidate],
    representations: Mapping[str, SurrogateVector],
    dimension: int,
    version: str,
) -> None:
    for candidate in candidates:
        feature = representations.get(candidate.candidate_id)
        if feature is None:
            raise ValueError(
                f"missing surrogate representation for {candidate.candidate_id}"
            )
        _validate_vector(feature, dimension, version)


def _validate_vector(vector: SurrogateVector, dimension: int, version: str) -> None:
    if vector.version != version or len(vector.values) != dimension:
        raise ValueError(
            "Hamming GP representation does not match the configured encoder"
        )
    code_matrix(vector.values)


__all__ = [
    "HammingGPUCBConfig",
    "HammingGPUCBSelector",
    "NucleotideHammingEncoder",
    "PRIOR_MEAN_CLIP",
    "normalized_hamming_kernel",
]
