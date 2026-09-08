"""Schema-aware covariance primitives for the Iron Mind reaction GP."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from tasks.iron_mind.core.schema import ReactionDatasetSchema


HYPERPARAMETER_GRID = (0.25, 0.5, 1.0, 2.0, 4.0)
HYPERPARAMETER_SWEEPS = 2
HYPERPARAMETER_PENALTY = 0.02


@dataclass(frozen=True)
class ReactionKernelParameters:
    """Signal scale and one positive distance weight per reaction factor."""

    signal_variance: float
    factor_weights: tuple[float, ...]

    def __post_init__(self) -> None:
        values = (self.signal_variance, *self.factor_weights)
        if any(not math.isfinite(value) or value <= 0 for value in values):
            raise ValueError("reaction kernel parameters must be finite and positive")

    def to_dict(self, schema: ReactionDatasetSchema) -> dict[str, Any]:
        return {
            "signal_variance": self.signal_variance,
            "factor_weights": dict(zip(schema.factor_names, self.factor_weights, strict=True)),
        }


def reaction_ard_kernel(
    left: np.ndarray,
    right: np.ndarray,
    schema: ReactionDatasetSchema,
    parameters: ReactionKernelParameters,
) -> np.ndarray:
    """Return the factor-product RBF kernel for decoded finite reaction conditions."""

    if len(parameters.factor_weights) != len(schema.factors):
        raise ValueError("reaction kernel weight count does not match the schema")
    distance = np.zeros((len(left), len(right)), dtype=float)
    for index, factor in enumerate(schema.factors):
        distance += parameters.factor_weights[index] * _factor_distance(
            left[:, index], right[:, index], factor
        )
    return parameters.signal_variance * np.exp(-distance)


def default_kernel_parameters(schema: ReactionDatasetSchema) -> ReactionKernelParameters:
    """Return the neutral ARD kernel before enough history exists to fit it."""

    return ReactionKernelParameters(1.0, (1.0,) * len(schema.factors))


def learn_kernel_parameters(
    codes: np.ndarray,
    targets: np.ndarray,
    schema: ReactionDatasetSchema,
    *,
    noise: float,
) -> ReactionKernelParameters:
    """Fit finite-grid ARD distance weights by regularized marginal likelihood."""

    best = default_kernel_parameters(schema)
    for _ in range(HYPERPARAMETER_SWEEPS):
        for index in range(len(schema.factors) + 1):
            proposals = [_with_parameter(best, index, value) for value in HYPERPARAMETER_GRID]
            best = min(
                proposals,
                key=lambda item: negative_log_marginal_likelihood(
                    codes, targets, schema, item, noise
                ),
            )
    return best


def posterior_terms(
    codes: np.ndarray,
    targets: np.ndarray,
    schema: ReactionDatasetSchema,
    parameters: ReactionKernelParameters,
    *,
    noise: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute exact GP Cholesky and posterior coefficients for one campaign."""

    kernel = reaction_ard_kernel(codes, codes, schema, parameters)
    cholesky = np.linalg.cholesky(kernel + noise * np.eye(len(codes)))
    alpha = np.linalg.solve(cholesky.T, np.linalg.solve(cholesky, targets))
    return cholesky, alpha


def negative_log_marginal_likelihood(
    codes: np.ndarray,
    targets: np.ndarray,
    schema: ReactionDatasetSchema,
    parameters: ReactionKernelParameters,
    noise: float,
) -> float:
    """Evaluate the regularized exact-GP marginal likelihood for one parameter set."""

    cholesky, alpha = posterior_terms(codes, targets, schema, parameters, noise=noise)
    squared_logs = (math.log(parameters.signal_variance) ** 2,) + tuple(
        math.log(value) ** 2 for value in parameters.factor_weights
    )
    normalization = 0.5 * len(targets) * math.log(2.0 * math.pi)
    return (
        0.5 * float(targets @ alpha)
        + float(np.log(np.diag(cholesky)).sum())
        + normalization
        + HYPERPARAMETER_PENALTY * sum(squared_logs)
    )


def _factor_distance(left: np.ndarray, right: np.ndarray, factor: Any) -> np.ndarray:
    if factor.parameter_type == "categorical":
        return (left[:, None] != right[None, :]).astype(float)
    options = np.asarray(factor.options, dtype=float)
    span = max(float(options.max() - options.min()), 1.0)
    return ((options[left.astype(int)][:, None] - options[right.astype(int)][None, :]) / span) ** 2


def _with_parameter(
    parameters: ReactionKernelParameters, index: int, value: float
) -> ReactionKernelParameters:
    if index == len(parameters.factor_weights):
        return ReactionKernelParameters(value, parameters.factor_weights)
    weights = list(parameters.factor_weights)
    weights[index] = value
    return ReactionKernelParameters(parameters.signal_variance, tuple(weights))


__all__ = [
    "ReactionKernelParameters",
    "default_kernel_parameters",
    "learn_kernel_parameters",
    "negative_log_marginal_likelihood",
    "posterior_terms",
    "reaction_ard_kernel",
]
