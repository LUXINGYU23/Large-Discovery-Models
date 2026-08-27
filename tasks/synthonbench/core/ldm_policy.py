"""Finite-reservoir LDM policy mathematics for SynthonBench."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from tasks.synthonbench.core.constants import DEFAULT_ACQUISITION_ETA

MAD_SCALE = 1.4826
DEFAULT_Z_CLIP = 5.0
EPSILON = 1.0e-12


@dataclass(frozen=True)
class AcquisitionTiltConfig:
    """The q0 and acquisition weights of the LDM sampling policy."""

    alpha: float = 1.0
    eta: float = DEFAULT_ACQUISITION_ETA
    z_clip: float = DEFAULT_Z_CLIP
    seed: int = 0
    pool_size: int | None = None
    proposal_sample_count: int | None = None

    def __post_init__(self) -> None:
        _nonnegative("alpha", self.alpha)
        _nonnegative("eta", self.eta)
        _positive("z_clip", self.z_clip)
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        _optional_positive("pool_size", self.pool_size)
        _optional_positive("proposal_sample_count", self.proposal_sample_count)
        if (
            self.pool_size is not None
            and self.proposal_sample_count is not None
            and self.proposal_sample_count <= self.pool_size
        ):
            raise ValueError("proposal_sample_count must exceed pool_size")


def robust_z(values: np.ndarray, *, clip: float = DEFAULT_Z_CLIP) -> np.ndarray:
    """Robustly normalize finite acquisition values for exponential tilting."""

    array = _finite_vector(values, "acquisition values")
    if not len(array):
        return array
    median = float(np.median(array))
    scale = MAD_SCALE * float(np.median(np.abs(array - median)))
    if scale <= EPSILON:
        scale = float(np.std(array))
    if scale <= EPSILON:
        return np.zeros_like(array)
    return np.clip((array - median) / (scale + EPSILON), -clip, clip)


def tilted_logits(q0: np.ndarray, acquisition: np.ndarray, *, config: AcquisitionTiltConfig) -> np.ndarray:
    base = normalize_probability(q0)
    normalized = robust_z(acquisition, clip=config.z_clip)
    if base.shape != normalized.shape:
        raise ValueError("q0 and acquisition shapes must match")
    return config.alpha * np.log(base + EPSILON) + config.eta * normalized


def gumbel_top_k(probabilities: np.ndarray, count: int, rng: np.random.Generator) -> list[int]:
    probability = normalize_probability(probabilities)
    if not len(probability) or count <= 0:
        return []
    log_probability = np.full(probability.shape, float("-inf"), dtype=float)
    np.log(probability, out=log_probability, where=probability > 0)
    scores = log_probability + rng.gumbel(size=len(probability))
    return [int(index) for index in np.argsort(scores)[::-1][: min(count, len(scores))]]


def normalize_probability(values: np.ndarray) -> np.ndarray:
    array = _finite_vector(values, "probabilities")
    if np.any(array < 0):
        raise ValueError("probabilities must be non-negative")
    total = float(array.sum())
    if len(array) and total <= 0:
        raise ValueError("probabilities must contain positive mass")
    return array if not len(array) else array / total


def softmax_probabilities(logits: np.ndarray) -> np.ndarray:
    values = _finite_vector(logits, "tilted logits")
    if not len(values):
        return values
    exponentials = np.exp(values - float(np.max(values)))
    return exponentials / float(exponentials.sum())


def probability_entropy(probabilities: np.ndarray) -> float:
    probability = normalize_probability(probabilities)
    return float(-np.sum(probability * np.log(probability + EPSILON)))


def effective_sample_size(probabilities: np.ndarray) -> float:
    probability = normalize_probability(probabilities)
    return float(1.0 / np.sum(probability * probability))


def _finite_vector(values: np.ndarray, label: str) -> np.ndarray:
    array = np.asarray(values, dtype=float).ravel()
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must be finite")
    return array


def _nonnegative(name: str, value: float) -> None:
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative")


def _positive(name: str, value: float) -> None:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


def _optional_positive(name: str, value: int | None) -> None:
    if value is not None and value < 1:
        raise ValueError(f"{name} must be positive when provided")


__all__ = [
    "DEFAULT_Z_CLIP",
    "AcquisitionTiltConfig",
    "effective_sample_size",
    "gumbel_top_k",
    "probability_entropy",
    "robust_z",
    "softmax_probabilities",
    "tilted_logits",
]
