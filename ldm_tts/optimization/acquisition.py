"""Task-independent posterior acquisition functions for LDM-TTS.

The public :class:`PosteriorAcquisition` module is the seam used by task
adapters.  Surrogate fitting and candidate encoding remain task-specific, but
posterior scoring does not.  Every acquisition score returned here follows one
invariant: larger is better.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from math import erf
from typing import Any, Sequence

import numpy as np


SINGLE_OBJECTIVE_ACQUISITIONS = frozenset({"mean", "ei", "lcb", "ucb"})
MULTI_OBJECTIVE_ACQUISITIONS = frozenset({"mean", "ehvi"})
SUPPORTED_ACQUISITIONS = SINGLE_OBJECTIVE_ACQUISITIONS | MULTI_OBJECTIVE_ACQUISITIONS


@dataclass(frozen=True)
class AcquisitionConfig:
    """Validated, task-independent acquisition configuration."""

    name: str
    beta: float = 1.0
    xi: float = 0.001
    weights: tuple[float, ...] = ()
    n_samples: int = 128

    def __post_init__(self) -> None:
        normalized = str(self.name).strip().lower()
        object.__setattr__(self, "name", normalized)
        if normalized not in SUPPORTED_ACQUISITIONS:
            raise ValueError(
                f"Unsupported acquisition {normalized!r}; "
                f"choose one of {sorted(SUPPORTED_ACQUISITIONS)}."
            )
        if self.beta < 0:
            raise ValueError(f"beta must be non-negative, got {self.beta}")
        if self.xi < 0:
            raise ValueError(f"xi must be non-negative, got {self.xi}")
        if self.n_samples < 1:
            raise ValueError(f"n_samples must be >= 1, got {self.n_samples}")
        if any(weight < 0 for weight in self.weights):
            raise ValueError("acquisition weights must be non-negative")
        if self.weights and sum(self.weights) <= 0:
            raise ValueError("at least one acquisition weight must be positive")


@dataclass(frozen=True)
class PosteriorAcquisition:
    """Score surrogate posterior predictions through one shared interface.

    ``mean``, ``ei``, ``lcb`` and ``ucb`` support one objective. ``mean`` also
    supports multiple objectives using a configurable weighted mean after
    converting each objective to a larger-is-better orientation. ``ehvi``
    supports exactly two objectives.
    """

    config: AcquisitionConfig
    minimize: tuple[bool, ...]

    def __post_init__(self) -> None:
        if not self.minimize:
            raise ValueError("at least one objective direction is required")
        n_obj = len(self.minimize)
        allowed = SINGLE_OBJECTIVE_ACQUISITIONS if n_obj == 1 else MULTI_OBJECTIVE_ACQUISITIONS
        if self.config.name not in allowed:
            raise ValueError(
                f"Acquisition {self.config.name!r} does not support {n_obj} objectives; "
                f"choose one of {sorted(allowed)}."
            )
        if self.config.name == "ehvi" and n_obj != 2:
            raise ValueError("EHVI requires exactly two objectives")
        if self.config.weights and len(self.config.weights) != n_obj:
            raise ValueError(
                f"acquisition weights length ({len(self.config.weights)}) "
                f"does not match objective count ({n_obj})"
            )

    @property
    def name(self) -> str:
        return self.config.name

    def score(
        self,
        mean: Any,
        std: Any,
        *,
        best: float | Any | None = None,
        pareto_points: Sequence[Sequence[float]] = (),
        ref_point: Sequence[float] | None = None,
        rng: Any = None,
    ) -> Any:
        if self.name == "mean":
            return posterior_mean_score(mean, minimize=self.minimize, weights=self.config.weights)
        if len(self.minimize) != 1:
            if ref_point is None:
                raise ValueError("EHVI requires ref_point")
            return expected_hypervolume_improvement(
                mean,
                std,
                pareto_points,
                ref_point,
                minimize=self.minimize,
                n_samples=self.config.n_samples,
                rng=rng,
            )
        minimize = self.minimize[0]
        if self.name == "ei":
            if best is None:
                raise ValueError("EI requires the best observed objective value")
            return expected_improvement(
                mean,
                std,
                best,
                xi=self.config.xi,
                minimize=minimize,
            )
        return confidence_bound(
            mean,
            std,
            kind=self.name,
            beta=self.config.beta,
            minimize=minimize,
        )


def make_acquisition(
    name: str,
    *,
    minimize: Sequence[bool],
    beta: float = 1.0,
    xi: float = 0.001,
    weights: Sequence[float] = (),
    n_samples: int = 128,
) -> PosteriorAcquisition:
    """Construct and validate a shared posterior acquisition module."""

    return PosteriorAcquisition(
        AcquisitionConfig(
            name=name,
            beta=float(beta),
            xi=float(xi),
            weights=tuple(float(weight) for weight in weights),
            n_samples=int(n_samples),
        ),
        tuple(bool(value) for value in minimize),
    )


def posterior_mean_score(
    mean: Any,
    *,
    minimize: Sequence[bool],
    weights: Sequence[float] = (),
) -> Any:
    """Return an oriented posterior mean score; larger is always better."""

    directions = tuple(bool(value) for value in minimize)
    if len(directions) == 1:
        return -mean if directions[0] else mean
    means = _objective_arrays(mean, len(directions), "mean")
    normalized_weights = _normalized_weights(weights, len(directions))
    score = np.zeros_like(means[0], dtype=float)
    for values, is_minimize, weight in zip(means, directions, normalized_weights):
        score += weight * (-values if is_minimize else values)
    return score


def expected_improvement(
    mean: Any,
    std: Any,
    best: float | Any,
    *,
    xi: float = 0.001,
    minimize: bool = True,
) -> Any:
    """Expected improvement for one objective; larger is better."""

    if xi < 0:
        raise ValueError(f"xi must be non-negative, got {xi}")
    if _is_torch_tensor(mean):
        import torch

        sigma = std.clamp_min(0)
        improvement = best - mean - xi if minimize else mean - best - xi
        safe_sigma = sigma.clamp_min(torch.finfo(sigma.dtype).eps)
        z = improvement / safe_sigma
        cdf = 0.5 * (1.0 + torch.erf(z / np.sqrt(2.0)))
        pdf = torch.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)
        value = improvement * cdf + sigma * pdf
        return torch.where(sigma > 0, value.clamp_min(0), torch.zeros_like(value))

    mu = np.asarray(mean, dtype=float)
    sigma = np.asarray(std, dtype=float)
    mu, sigma = np.broadcast_arrays(mu, sigma)
    out = np.zeros_like(mu, dtype=float)
    finite = np.isfinite(mu) & np.isfinite(sigma) & (sigma > 0)
    if np.any(finite):
        improvement = (
            float(best) - mu[finite] - xi
            if minimize
            else mu[finite] - float(best) - xi
        )
        z = improvement / sigma[finite]
        out[finite] = improvement * _normal_cdf(z) + sigma[finite] * _normal_pdf(z)
    return np.clip(out, 0.0, None)


def confidence_bound(
    mean: Any,
    std: Any,
    *,
    kind: str | None = None,
    beta: float | None = None,
    minimize: bool = True,
) -> Any:
    """LCB or UCB score for one objective; larger is always better.

    When ``kind`` is omitted, use LCB for minimization and UCB for maximization.
    """

    exploration = float(1.0 if beta is None else beta)
    if exploration < 0:
        raise ValueError(f"beta must be non-negative, got {exploration}")
    resolved_kind = (kind or ("lcb" if minimize else "ucb")).strip().lower()
    if resolved_kind not in {"lcb", "ucb"}:
        raise ValueError("confidence-bound kind must be 'lcb' or 'ucb'")
    bound = mean - exploration * std if resolved_kind == "lcb" else mean + exploration * std
    return -bound if minimize else bound


def dominates(a: Sequence[float], b: Sequence[float], minimize: Sequence[bool]) -> bool:
    """Return whether ``a`` Pareto-dominates ``b``."""

    if not (len(a) == len(b) == len(minimize)):
        raise ValueError("a, b, and minimize must have equal lengths")
    no_worse = all(x <= y if is_min else x >= y for x, y, is_min in zip(a, b, minimize))
    strictly_better = any(x < y if is_min else x > y for x, y, is_min in zip(a, b, minimize))
    return no_worse and strictly_better


def pareto_front(
    points: Sequence[Sequence[float]],
    minimize: Sequence[bool],
) -> list[tuple[float, ...]]:
    """Return non-dominated points in first-seen order."""

    result: list[tuple[float, ...]] = []
    for index, point in enumerate(points):
        value = tuple(float(item) for item in point)
        if len(value) != len(minimize):
            raise ValueError(f"point length {len(value)} does not match objective count {len(minimize)}")
        if any(dominates(other, value, minimize) for j, other in enumerate(points) if j != index):
            continue
        if value not in result:
            result.append(value)
    return result


def hypervolume(
    points: Sequence[Sequence[float]],
    ref: Sequence[float],
    *,
    minimize: Sequence[bool] | None = None,
) -> float:
    """Return exact one- or two-objective dominated hypervolume."""

    n_obj = len(ref)
    if n_obj not in {1, 2}:
        raise NotImplementedError(f"hypervolume supports one or two objectives; got {n_obj}")
    directions = tuple(True for _ in ref) if minimize is None else tuple(minimize)
    if len(directions) != n_obj:
        raise ValueError("minimize length does not match reference-point length")
    converted_ref = tuple(float(value) if is_min else -float(value) for value, is_min in zip(ref, directions))
    converted: list[tuple[float, ...]] = []
    for point in points:
        if len(point) != n_obj:
            raise ValueError(f"expected {n_obj}D point, got length {len(point)}")
        value = tuple(float(item) if is_min else -float(item) for item, is_min in zip(point, directions))
        if all(item < limit for item, limit in zip(value, converted_ref)):
            converted.append(value)
    if not converted:
        return 0.0
    if n_obj == 1:
        return max(0.0, converted_ref[0] - min(point[0] for point in converted))
    front = sorted(pareto_front(converted, (True, True)), key=lambda point: point[0])
    volume = 0.0
    for index, (x_value, y_value) in enumerate(front):
        x_next = front[index + 1][0] if index + 1 < len(front) else converted_ref[0]
        volume += (x_next - x_value) * (converted_ref[1] - y_value)
    return max(0.0, float(volume))


def expected_hypervolume_improvement(
    mu_per_obj: Sequence[Any],
    sigma_per_obj: Sequence[Any],
    pareto_points: Sequence[Sequence[float]],
    ref: Sequence[float],
    *,
    minimize: Sequence[bool],
    n_samples: int = 128,
    rng: Any = None,
) -> np.ndarray:
    """Monte Carlo expected hypervolume improvement for two objectives."""

    if len(ref) != 2:
        raise NotImplementedError(f"EHVI supports exactly two objectives; got {len(ref)}")
    if len(mu_per_obj) != 2 or len(sigma_per_obj) != 2 or len(minimize) != 2:
        raise ValueError("EHVI mean, std, ref, and minimize inputs must each have length 2")
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}")
    means = _objective_arrays(mu_per_obj, 2, "mean")
    stds = _objective_arrays(sigma_per_obj, 2, "std")
    if not (means[0].shape == means[1].shape == stds[0].shape == stds[1].shape):
        raise ValueError("EHVI posterior arrays must have identical shapes")
    base_hv = hypervolume(pareto_points, ref, minimize=minimize)
    result = np.zeros(means[0].shape, dtype=float)
    for index in np.ndindex(means[0].shape):
        samples_by_objective = [
            _normal_samples(rng, means[obj][index], abs(stds[obj][index]), n_samples)
            for obj in range(2)
        ]
        increments = np.empty(n_samples, dtype=float)
        for sample_idx in range(n_samples):
            sample = (samples_by_objective[0][sample_idx], samples_by_objective[1][sample_idx])
            increments[sample_idx] = max(
                0.0,
                hypervolume([*pareto_points, sample], ref, minimize=minimize) - base_hv,
            )
        result[index] = float(increments.mean())
    return result


def chebyshev_scalarize(
    point: Sequence[float],
    weights: Sequence[float],
    ideal: Sequence[float],
    minimize: Sequence[bool],
) -> float:
    """ParEGO-style Chebyshev scalarization; smaller is better."""

    if not (len(point) == len(weights) == len(ideal) == len(minimize)):
        raise ValueError("point, weights, ideal, and minimize must have equal lengths")
    gaps = []
    for value, weight, target, is_min in zip(point, weights, ideal, minimize):
        if weight < 0:
            raise ValueError(f"weights must be non-negative, got {weight}")
        gap = weight * (value - target if is_min else target - value)
        gaps.append(max(0.0, gap))
    return float(max(gaps))


def sample_simplex_weights(rng: Any, n: int, alpha: float = 1.0) -> np.ndarray:
    """Sample non-negative weights that sum to one."""

    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if alpha <= 0:
        raise ValueError(f"alpha must be > 0, got {alpha}")
    if hasattr(rng, "beta") and not isinstance(rng, np.random.Generator):
        raw = np.asarray(rng.beta(alpha, size=n), dtype=float)
    else:
        raw = _numpy_generator(rng).beta(alpha, 1.0, size=n)
    total = float(raw.sum())
    return raw / total if total > 0 else np.full(n, 1.0 / n)


def _objective_arrays(values: Any, n_obj: int, label: str) -> list[np.ndarray]:
    if not isinstance(values, (list, tuple)) or len(values) != n_obj:
        raise ValueError(f"{label} must contain one array per objective ({n_obj})")
    return [np.asarray(value, dtype=float) for value in values]


def _normalized_weights(weights: Sequence[float], n_obj: int) -> tuple[float, ...]:
    if not weights:
        return tuple(1.0 / n_obj for _ in range(n_obj))
    if len(weights) != n_obj:
        raise ValueError(f"weights length ({len(weights)}) does not match objective count ({n_obj})")
    total = float(sum(weights))
    if total <= 0 or any(weight < 0 for weight in weights):
        raise ValueError("weights must be non-negative with a positive sum")
    return tuple(float(weight) / total for weight in weights)


def _normal_pdf(value: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * value * value) / np.sqrt(2.0 * np.pi)


def _normal_cdf(value: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.vectorize(erf)(value / np.sqrt(2.0)))


def _is_torch_tensor(value: Any) -> bool:
    return value.__class__.__module__.startswith("torch") and hasattr(value, "dtype")


def _normal_samples(rng: Any, mean: float, std: float, size: int) -> np.ndarray:
    if hasattr(rng, "normal") and not isinstance(rng, (random.Random, np.random.Generator)):
        return np.asarray(rng.normal(float(mean), float(std), size), dtype=float)
    return _numpy_generator(rng).normal(float(mean), float(std), size=size)


def _numpy_generator(rng: Any) -> np.random.Generator:
    if isinstance(rng, np.random.Generator):
        return rng
    if hasattr(rng, "numpy") and isinstance(rng.numpy, np.random.Generator):
        return rng.numpy
    if isinstance(rng, random.Random):
        state = rng.getstate()
        seed = hash((state[0], state[1], state[2])) & 0x7FFFFFFF
        return np.random.default_rng(seed)
    if isinstance(rng, (int, np.integer)):
        return np.random.default_rng(int(rng))
    return np.random.default_rng()


__all__ = [
    "AcquisitionConfig",
    "MULTI_OBJECTIVE_ACQUISITIONS",
    "PosteriorAcquisition",
    "SINGLE_OBJECTIVE_ACQUISITIONS",
    "SUPPORTED_ACQUISITIONS",
    "chebyshev_scalarize",
    "confidence_bound",
    "dominates",
    "expected_hypervolume_improvement",
    "expected_improvement",
    "hypervolume",
    "make_acquisition",
    "pareto_front",
    "posterior_mean_score",
    "sample_simplex_weights",
]
