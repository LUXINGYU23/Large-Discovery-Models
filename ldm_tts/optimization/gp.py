"""Reusable exact RBF Gaussian process for finite LDM candidate reservoirs."""

from __future__ import annotations

import math
from dataclasses import dataclass
from collections.abc import Mapping
from typing import Any, Sequence

import numpy as np

from ldm_tts.optimization.acquisition import make_acquisition
from ldm_tts.optimization.records import (
    BOObservation,
    BOPrediction,
    BOSelectionResult,
    SurrogateVector,
)
from ldm_tts.contracts import AcquisitionSpec, Candidate


@dataclass(frozen=True)
class SearchObservation:
    """Compatibility scalar observation; new callers should use BOObservation."""

    candidate_id: str
    feature_vector: tuple[float, ...]
    score: float

    def to_bo(self, *, feature_version: str = "") -> BOObservation:
        return BOObservation.scalar(
            self.candidate_id,
            self.score,
            self.feature_vector,
            feature_version=feature_version,
        )


@dataclass(frozen=True)
class GPPrediction:
    """Compatibility scalar prediction; new callers should use BOPrediction."""

    mean: float
    std: float
    acquisition_score: float

    def to_bo(self, candidate_id: str) -> BOPrediction:
        return BOPrediction.scalar(
            candidate_id,
            mean=self.mean,
            std=self.std,
            acquisition_score=self.acquisition_score,
        )


class RBFGPSurrogate:
    """Standardized exact GP with a stable prior/fallback for sparse history."""

    def __init__(
        self,
        observations: Sequence[BOObservation | SearchObservation],
        *,
        lengthscale: float = 1.5,
        noise: float = 1.0e-4,
        prior_mean: float = 0.0,
        prior_std: float = 0.25,
        feature_version: str = "",
    ) -> None:
        self.observations = [
            item.to_bo(feature_version=feature_version)
            if isinstance(item, SearchObservation)
            else item
            for item in observations
        ]
        for item in self.observations:
            if len(item.objectives) != 1:
                raise ValueError("RBFGPSurrogate requires scalar BO observations")
            if item.feature is None:
                raise ValueError("RBFGPSurrogate observations require surrogate representations")
        self.lengthscale = max(1.0e-6, float(lengthscale))
        self.noise = max(1.0e-9, float(noise))
        self.prior_mean = float(prior_mean)
        self.prior_std = max(1.0e-9, float(prior_std))
        self.feature_version = str(feature_version)
        self.ready = False
        self.fit_status = "prior" if not observations else "fallback"
        self._fit()

    def _fit(self) -> None:
        if len(self.observations) < 2:
            return
        dimensions = {len(item.feature_vector) for item in self.observations}
        if len(dimensions) != 1:
            raise ValueError("GP observations have inconsistent feature dimensions")
        self.X = np.asarray([item.feature_vector for item in self.observations], dtype=float)
        self.y = np.asarray([item.scalar_score for item in self.observations], dtype=float)
        self.x_mean = self.X.mean(axis=0)
        self.x_std = self.X.std(axis=0) + 1.0e-8
        self.Xz = (self.X - self.x_mean) / self.x_std
        self.y_mean = float(self.y.mean())
        self.y_std = float(self.y.std() + 1.0e-9)
        yz = (self.y - self.y_mean) / self.y_std
        kernel = _rbf_kernel(self.Xz, self.Xz, self.lengthscale)
        kernel = kernel + self.noise * np.eye(len(self.Xz))
        try:
            self.L = np.linalg.cholesky(kernel + 1.0e-8 * np.eye(len(self.Xz)))
            self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, yz))
        except np.linalg.LinAlgError:
            return
        self.ready = True
        self.fit_status = "fitted"

    def predict(self, vector: Sequence[float], *, beta: float = 1.0) -> GPPrediction:
        """Return the legacy scalar prediction shape."""

        prediction = self.predict_record("candidate", vector, beta=beta)
        return GPPrediction(
            prediction.scalar_mean,
            prediction.scalar_std,
            float(prediction.acquisition_score),
        )

    def predict_record(
        self,
        candidate_id: str,
        vector: Sequence[float],
        *,
        beta: float = 1.0,
    ) -> BOPrediction:
        """Return the canonical task-neutral prediction record."""

        if self.observations and len(vector) != len(self.observations[0].feature_vector):
            raise ValueError("candidate feature dimension does not match GP observations")
        if not self.observations:
            mean, std = self.prior_mean, self.prior_std
        elif not self.ready:
            scores = np.asarray([item.scalar_score for item in self.observations], dtype=float)
            mean = float(scores.mean())
            std = float(max(scores.std(), self.prior_std))
        else:
            x = np.asarray(vector, dtype=float)[None, :]
            xz = (x - self.x_mean) / self.x_std
            cross = _rbf_kernel(xz, self.Xz, self.lengthscale)
            mean_z = float(np.asarray(cross @ self.alpha).ravel()[0])
            projected = np.linalg.solve(self.L, cross.T)
            variance_z = max(1.0 - float((projected * projected).sum()), 1.0e-9)
            mean = self.y_mean + mean_z * self.y_std
            std = math.sqrt(variance_z) * self.y_std
        acquisition = make_acquisition("ucb", minimize=(False,), beta=max(0.0, beta))
        acquisition_score = float(acquisition.score(mean, std))
        return BOPrediction.scalar(
            candidate_id,
            mean=float(mean),
            std=float(std),
            acquisition_score=acquisition_score,
            metadata={"surrogate": "exact_rbf_gp", "fit_status": self.fit_status},
        )

    def summary(self) -> dict[str, Any]:
        scores = [item.scalar_score for item in self.observations]
        return {
            "feature_version": self.feature_version,
            "fit_status": self.fit_status,
            "ready": self.ready,
            "history_size": len(scores),
            "best_observed": max(scores) if scores else None,
            "mean_observed": float(np.mean(scores)) if scores else None,
        }


class RBFGPUCBSelector:
    """Canonical finite-reservoir selector backed by the shared exact RBF GP."""

    def __init__(
        self,
        *,
        objective_name: str,
        beta: float = 1.0,
        lengthscale: float = 1.5,
        noise: float = 1.0e-4,
        prior_mean: float = 0.0,
        prior_std: float = 0.25,
        feature_version: str = "",
    ) -> None:
        self.objective_name = str(objective_name)
        self.beta = float(beta)
        self.lengthscale = float(lengthscale)
        self.noise = float(noise)
        self.prior_mean = float(prior_mean)
        self.prior_std = float(prior_std)
        self.feature_version = str(feature_version)
        self.history: list[BOObservation] = []
        self.surrogate = self._make_surrogate()

    def describe(self) -> AcquisitionSpec:
        return AcquisitionSpec(
            name="ucb",
            objective_names=(self.objective_name,),
            score_direction="maximize",
            selection_rule="highest shared RBF GP upper confidence bound",
            parameters={"beta": self.beta},
        )

    def fit(self, history: Sequence[BOObservation]) -> None:
        if any(len(item.objectives) != 1 for item in history):
            raise ValueError("RBFGPUCBSelector requires one objective")
        self.history = list(history)
        self.surrogate = self._make_surrogate()

    def select(
        self,
        candidates: Sequence[Candidate],
        representations: Mapping[str, SurrogateVector],
        *,
        count: int = 1,
        round_idx: int = 0,
    ) -> BOSelectionResult:
        if count < 1:
            raise ValueError("selection count must be positive")
        missing = [item.candidate_id for item in candidates if item.candidate_id not in representations]
        if missing:
            raise ValueError(
                "missing surrogate representation for candidate(s): " + ", ".join(missing)
            )
        predictions = tuple(
            self.surrogate.predict_record(
                item.candidate_id,
                representations[item.candidate_id].values,
                beta=self.beta,
            )
            for item in candidates
        )
        ranked = sorted(
            predictions,
            key=lambda item: (float(item.acquisition_score), item.candidate_id),
            reverse=True,
        )
        return BOSelectionResult(
            selected_candidate_ids=tuple(item.candidate_id for item in ranked[:count]),
            predictions=predictions,
            metadata={"surrogate": self.surrogate.summary(), "beta": self.beta},
        )

    def _make_surrogate(self) -> RBFGPSurrogate:
        return RBFGPSurrogate(
            self.history,
            lengthscale=self.lengthscale,
            noise=self.noise,
            prior_mean=self.prior_mean,
            prior_std=self.prior_std,
            feature_version=self.feature_version,
        )


def select_max_ucb(
    candidates: Sequence[tuple[str, Sequence[float]]],
    surrogate: RBFGPSurrogate,
    *,
    beta: float = 1.0,
) -> tuple[str, GPPrediction]:
    """Return a deterministic highest-UCB candidate from a finite pool."""

    if not candidates:
        raise ValueError("cannot select from an empty candidate reservoir")
    predictions = [
        (candidate_id, surrogate.predict(vector, beta=beta))
        for candidate_id, vector in candidates
    ]
    return max(
        predictions,
        key=lambda item: (item[1].acquisition_score, item[0]),
    )


def select_max_ucb_record(
    candidates: Sequence[tuple[str, Sequence[float]]],
    surrogate: RBFGPSurrogate,
    *,
    beta: float = 1.0,
) -> tuple[str, BOPrediction]:
    """Return a canonical BOPrediction for the deterministic best UCB item."""

    if not candidates:
        raise ValueError("cannot select from an empty candidate reservoir")
    predictions = [
        (candidate_id, surrogate.predict_record(candidate_id, vector, beta=beta))
        for candidate_id, vector in candidates
    ]
    return max(
        predictions,
        key=lambda item: (float(item[1].acquisition_score), item[0]),
    )


def _rbf_kernel(left: np.ndarray, right: np.ndarray, lengthscale: float) -> np.ndarray:
    left_sq = np.sum(left * left, axis=1)[:, None]
    right_sq = np.sum(right * right, axis=1)[None, :]
    distance_sq = np.maximum(left_sq + right_sq - 2.0 * left @ right.T, 0.0)
    return np.exp(-0.5 * distance_sq / (lengthscale * lengthscale))


__all__ = [
    "GPPrediction",
    "RBFGPUCBSelector",
    "RBFGPSurrogate",
    "SearchObservation",
    "select_max_ucb",
    "select_max_ucb_record",
]
