"""Task-local count-Tanimoto kernel primitives for SynthonBench."""

from __future__ import annotations

import math

import numpy as np


KERNEL_BLOCK_SIZE = 16


def count_tanimoto(left: np.ndarray, right: np.ndarray) -> float:
    """Return the min/max Tanimoto similarity of two count fingerprints."""

    left_values = _count_vector(left, "left")
    right_values = _count_vector(right, "right")
    if left_values.shape != right_values.shape:
        raise ValueError("count-Tanimoto vectors must have identical dimensions")
    numerator = float(np.minimum(left_values, right_values).sum())
    denominator = float(np.maximum(left_values, right_values).sum())
    return 1.0 if denominator == 0.0 else numerator / denominator


def count_tanimoto_to_many(query: np.ndarray, references: np.ndarray) -> np.ndarray:
    """Return count-Tanimoto values from one query to each reference row."""

    query_values = _count_vector(query, "query")
    reference_values = _count_matrix(references, "references")
    if reference_values.shape[1] != query_values.shape[0]:
        raise ValueError("count-Tanimoto reference dimension does not match the query")
    numerator = np.minimum(reference_values, query_values).sum(axis=1)
    denominator = np.maximum(reference_values, query_values).sum(axis=1)
    return np.divide(numerator, denominator, out=np.ones_like(numerator), where=denominator > 0.0)


def count_tanimoto_matrix(left: np.ndarray, right: np.ndarray | None = None) -> np.ndarray:
    """Build a count-Tanimoto matrix without allocating a cubic tensor."""

    left_values = _count_matrix(left, "left")
    right_values = left_values if right is None else _count_matrix(right, "right")
    if left_values.shape[1] != right_values.shape[1]:
        raise ValueError("count-Tanimoto matrices must have identical feature dimensions")
    output = np.empty((len(left_values), len(right_values)), dtype=float)
    for start in range(0, len(right_values), KERNEL_BLOCK_SIZE):
        stop = min(start + KERNEL_BLOCK_SIZE, len(right_values))
        block = right_values[start:stop]
        numerator = np.minimum(left_values[:, None, :], block[None, :, :]).sum(axis=2)
        denominator = np.maximum(left_values[:, None, :], block[None, :, :]).sum(axis=2)
        output[:, start:stop] = np.divide(
            numerator,
            denominator,
            out=np.ones_like(numerator),
            where=denominator > 0.0,
        )
    return output


def composite_count_tanimoto_to_many(
    query_counts: np.ndarray,
    query_reaction: str,
    reference_counts: np.ndarray,
    reference_reactions: np.ndarray,
    *,
    reaction_weight: float,
) -> np.ndarray:
    """Add a declared reaction-delta ablation term to count Tanimoto."""

    chemical = count_tanimoto_to_many(query_counts, reference_counts)
    reactions = np.asarray(reference_reactions, dtype=str)
    if reactions.ndim != 1 or len(reactions) != len(chemical):
        raise ValueError("reference reactions must align with reference count fingerprints")
    weight = _reaction_weight(reaction_weight)
    if weight == 0.0:
        return chemical
    return (chemical + weight * (reactions == str(query_reaction))) / (1.0 + weight)


def composite_count_tanimoto_matrix(
    counts: np.ndarray,
    reactions: np.ndarray,
    *,
    reaction_weight: float,
) -> np.ndarray:
    """Build the declared prior kernel over a common set of tuple proxies."""

    chemical = count_tanimoto_matrix(counts)
    labels = np.asarray(reactions, dtype=str)
    if labels.ndim != 1 or len(labels) != len(chemical):
        raise ValueError("reaction labels must align with count fingerprints")
    weight = _reaction_weight(reaction_weight)
    if weight == 0.0:
        return chemical
    same_reaction = labels[:, None] == labels[None, :]
    return (chemical + weight * same_reaction) / (1.0 + weight)


def _count_vector(values: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.ndim != 1:
        raise ValueError(f"{name} count fingerprint must be one-dimensional")
    _validate_count_values(result, name)
    return result


def _count_matrix(values: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.ndim != 2:
        raise ValueError(f"{name} count fingerprints must be two-dimensional")
    _validate_count_values(result, name)
    return result


def _validate_count_values(values: np.ndarray, name: str) -> None:
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError(f"{name} count fingerprints must be finite and non-negative")


def _reaction_weight(value: float) -> float:
    weight = float(value)
    if not math.isfinite(weight) or weight < 0.0:
        raise ValueError("reaction_weight must be finite and non-negative")
    return weight


__all__ = [
    "KERNEL_BLOCK_SIZE",
    "composite_count_tanimoto_matrix",
    "composite_count_tanimoto_to_many",
    "count_tanimoto",
    "count_tanimoto_matrix",
    "count_tanimoto_to_many",
]
