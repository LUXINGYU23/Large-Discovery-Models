"""Iron Mind acquisition-tilted LDM policy math."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

MAD_SCALE = 1.4826
DEFAULT_Z_CLIP = 5.0
DEFAULT_ETA = 1.0
NUMERICAL_EPSILON = 1.0e-12


@dataclass(frozen=True)
class AcquisitionTiltConfig:
    """Weights and reproducibility settings for the finite LDM policy."""

    alpha: float = 1.0
    eta: float = DEFAULT_ETA
    z_clip: float = DEFAULT_Z_CLIP
    seed: int = 0
    pool_size: int | None = None
    proposal_sample_count: int | None = None
    eps: float = NUMERICAL_EPSILON

    def __post_init__(self) -> None:
        for name, value in (("alpha", self.alpha), ("eta", self.eta)):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not math.isfinite(self.z_clip) or self.z_clip <= 0:
            raise ValueError("z_clip must be finite and positive")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.pool_size is not None and self.pool_size < 1:
            raise ValueError("pool_size must be positive when provided")
        if self.proposal_sample_count is not None and self.proposal_sample_count < 1:
            raise ValueError("proposal_sample_count must be positive when provided")
        if (
            self.pool_size is not None
            and self.proposal_sample_count is not None
            and self.proposal_sample_count <= self.pool_size
        ):
            raise ValueError("proposal_sample_count must exceed pool_size")
        if not math.isfinite(self.eps) or self.eps <= 0:
            raise ValueError("eps must be finite and positive")


def robust_z(
    values: np.ndarray,
    *,
    clip: float = DEFAULT_Z_CLIP,
    eps: float = NUMERICAL_EPSILON,
) -> np.ndarray:
    arr = _finite_vector(values, "acquisition scores")
    if arr.size == 0:
        return arr
    median = float(np.median(arr))
    scale = MAD_SCALE * float(np.median(np.abs(arr - median)))
    if scale <= eps:
        scale = float(np.std(arr))
    if scale <= eps:
        return np.zeros_like(arr)
    return np.clip((arr - median) / (scale + eps), -clip, clip)


def tilted_logits(
    q0: np.ndarray,
    acquisition: np.ndarray,
    *,
    config: AcquisitionTiltConfig,
) -> np.ndarray:
    base = normalize_probability(q0)
    normalized = robust_z(acquisition, clip=config.z_clip)
    if base.shape != normalized.shape:
        raise ValueError("base measure and acquisition score shapes must match")
    return config.alpha * np.log(base + config.eps) + config.eta * normalized


def tilted_probabilities(
    q0: np.ndarray,
    acquisition: np.ndarray,
    *,
    alpha: float = 1.0,
    eta: float = DEFAULT_ETA,
    z_clip: float = DEFAULT_Z_CLIP,
) -> np.ndarray:
    logits = tilted_logits(
        q0,
        acquisition,
        config=AcquisitionTiltConfig(alpha=alpha, eta=eta, z_clip=z_clip),
    )
    return softmax_probabilities(logits)


def gumbel_top_k(
    probabilities: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> list[int]:
    probability = normalize_probability(probabilities)
    if probability.size == 0 or count <= 0:
        return []
    log_probability = np.full(probability.shape, float("-inf"), dtype=float)
    np.log(probability, out=log_probability, where=probability > 0)
    scores = log_probability + rng.gumbel(size=len(probability))
    return [int(index) for index in np.argsort(scores)[::-1][: min(count, len(scores))]]


def normalize_probability(values: np.ndarray) -> np.ndarray:
    arr = _finite_vector(values, "probabilities")
    if np.any(arr < 0):
        raise ValueError("probabilities must be non-negative")
    total = float(arr.sum())
    if arr.size and total <= 0:
        raise ValueError("probabilities must contain positive mass")
    return arr if arr.size == 0 else arr / total


def softmax_probabilities(logits: np.ndarray) -> np.ndarray:
    """Normalize finite log weights without numerical overflow."""

    values = _finite_vector(logits, "tilted logits")
    if values.size == 0:
        return values
    exponentials = np.exp(values - float(np.max(values)))
    return exponentials / float(exponentials.sum())


def _finite_vector(values: np.ndarray, label: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float).ravel()
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{label} must be finite")
    return arr


def probability_entropy(
    probabilities: np.ndarray,
    *,
    eps: float = NUMERICAL_EPSILON,
) -> float:
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    probability = normalize_probability(probabilities)
    return float(-np.sum(probability * np.log(probability + eps)))


def effective_sample_size(probabilities: np.ndarray) -> float:
    probability = normalize_probability(probabilities)
    return float(1.0 / np.sum(probability * probability))


__all__ = [
    "AcquisitionTiltConfig",
    "DEFAULT_ETA",
    "effective_sample_size",
    "gumbel_top_k",
    "normalize_probability",
    "probability_entropy",
    "robust_z",
    "softmax_probabilities",
    "tilted_logits",
    "tilted_probabilities",
]
