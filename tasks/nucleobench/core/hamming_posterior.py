"""NumPy-only Hamming posterior shared by selection and read-only research queries."""

from __future__ import annotations

import math

import numpy as np

MIN_VARIANCE = 1.0e-12


def code_matrix(values) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim != 2 or not array.shape[1] or not np.all(np.isfinite(array)):
        raise ValueError("categorical Hamming codes must be a finite non-empty matrix")
    if np.any(array < 0.0) or np.any(array > 3.0) or not np.allclose(array, np.rint(array)):
        raise ValueError("categorical Hamming codes must be integers in [0, 3]")
    return array


def normalized_hamming_kernel(left, right, length_scale: float) -> np.ndarray:
    if not math.isfinite(length_scale) or length_scale <= 0.0:
        raise ValueError("length_scale must be finite and positive")
    left_array, right_array = code_matrix(left), code_matrix(right)
    if left_array.shape[1] != right_array.shape[1]:
        raise ValueError("Hamming kernel inputs must have the same dimension")
    distances = np.mean(left_array[:, None, :] != right_array[None, :, :], axis=2)
    return np.exp(-distances / length_scale)


def residual_moments(query, training, cholesky, alpha, length_scale):
    query = code_matrix(query)
    if training is None:
        return np.zeros(len(query)), np.ones(len(query))
    cross = normalized_hamming_kernel(query, training, length_scale)
    projected = np.linalg.solve(np.asarray(cholesky), cross.T)
    means = cross @ np.asarray(alpha)
    std = np.sqrt(np.maximum(1.0 - np.sum(projected * projected, axis=0), MIN_VARIANCE))
    return means, std
