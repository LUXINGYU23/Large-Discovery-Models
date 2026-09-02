"""Task-local categorical Hamming representation and exact GP-UCB."""

from __future__ import annotations

import hashlib
import json
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

ENCODER_ALGORITHM = "editable_categorical_hamming_v1"
ENCODER_PATH = "tasks.nucleobench.core.hamming_gp:NucleotideHammingEncoder"
BASE_CODE = {"A": 0.0, "C": 1.0, "G": 2.0, "T": 3.0}
MIN_VARIANCE = 1.0e-12


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
        digest = hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
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

    def fit(self, history: Sequence[BOObservation]) -> None:
        _validate_history(history, self.feature_dimension, self.feature_version)
        working = _working_set(history, self.config)
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
        if len(working) < self.config.min_history_for_fit:
            return

        self._codes = np.asarray([item.feature_vector for item in working], dtype=float)
        targets = (scores - self._target_mean) / self._target_scale
        self._length_scale, self._cholesky, self._alpha = _fit_grid_gp(
            self._codes,
            targets,
            self.config,
        )
        self._fit_status = "fitted_grid_length_scale"

    def select(
        self,
        candidates: Sequence[Candidate],
        representations: Mapping[str, SurrogateVector],
        *,
        count: int = 1,
    ) -> BOSelectionResult:
        if count < 1:
            raise ValueError("selection count must be positive")
        _validate_representations(
            candidates,
            representations,
            self.feature_dimension,
            self.feature_version,
        )
        predictions = tuple(
            self._predict(candidate, representations[candidate.candidate_id])
            for candidate in candidates
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

    def _predict(self, candidate: Candidate, feature: SurrogateVector) -> BOPrediction:
        if self._fit_status == "neutral_prior":
            mean = self._target_mean
            std = self._target_scale
        else:
            assert self._codes is not None
            assert self._cholesky is not None
            assert self._alpha is not None
            assert self._length_scale is not None
            code = np.asarray(feature.values, dtype=float)
            cross = normalized_hamming_kernel(code, self._codes, self._length_scale)
            mean_z = float((cross @ self._alpha)[0])
            projected = np.linalg.solve(self._cholesky, cross.T)
            variance_z = max(1.0 - float(np.sum(projected * projected)), MIN_VARIANCE)
            mean = self._target_mean + self._target_scale * mean_z
            std = self._target_scale * math.sqrt(variance_z)
        acquisition = make_acquisition("ucb", minimize=(False,), beta=self.config.beta)
        return BOPrediction.scalar(
            candidate.candidate_id,
            mean=mean,
            std=std,
            acquisition_score=float(acquisition.score(mean, std)),
            metadata={
                "surrogate": "nucleobench_exact_hamming_gp",
                "fit_status": self._fit_status,
            },
        )

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
        }


def normalized_hamming_kernel(
    left: Sequence[float] | np.ndarray,
    right: Sequence[float] | np.ndarray,
    length_scale: float,
) -> np.ndarray:
    """Return exp(-normalized Hamming distance / length_scale)."""

    if not math.isfinite(length_scale) or length_scale <= 0.0:
        raise ValueError("length_scale must be finite and positive")
    left_array = _code_matrix(left)
    right_array = _code_matrix(right)
    if left_array.shape[1] != right_array.shape[1]:
        raise ValueError("Hamming kernel inputs must have the same dimension")
    distances = np.mean(
        left_array[:, None, :] != right_array[None, :, :],
        axis=2,
    )
    return np.exp(-distances / length_scale)


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
    _code_matrix(vector.values)


def _code_matrix(values: Sequence[float] | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim != 2 or not array.shape[1] or not np.all(np.isfinite(array)):
        raise ValueError("categorical Hamming codes must be a finite non-empty matrix")
    if (
        np.any(array < 0.0)
        or np.any(array > 3.0)
        or not np.allclose(array, np.rint(array))
    ):
        raise ValueError("categorical Hamming codes must be integers in [0, 3]")
    return array


__all__ = [
    "HammingGPUCBConfig",
    "HammingGPUCBSelector",
    "NucleotideHammingEncoder",
    "normalized_hamming_kernel",
]
