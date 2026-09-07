"""Molecular Tanimoto GP using the shared posterior-acquisition contract."""

from __future__ import annotations
import numpy as np
from ldm_tts.contracts import AcquisitionSpec
from ldm_tts.optimization.acquisition import make_acquisition
from ldm_tts.optimization.records import BOSelectionResult, BOPrediction


def tanimoto_matrix(x, z):
    dot = x @ z.T
    denominator = (x * x).sum(axis=1)[:, None] + (z * z).sum(axis=1)[None, :] - dot
    return np.divide(dot, denominator, out=np.ones_like(dot), where=denominator > 0)


class TanimotoGPSelector:
    def __init__(
        self, *, objective_name, beta=1.0, history_limit=256, feature_version=""
    ):
        self.objective_name = objective_name
        self.beta = beta
        self.history_limit = history_limit
        self.feature_version = feature_version
        self.history = []
        self.acquisition = make_acquisition("ucb", minimize=(False,), beta=beta)

    def describe(self):
        return AcquisitionSpec(
            name="tanimoto_gp_ucb",
            objective_names=(self.objective_name,),
            score_direction="maximize",
            selection_rule="Highest shared UCB of exact Tanimoto GP posterior",
            parameters={"beta": self.beta, "history_limit": self.history_limit},
        )

    def fit(self, history):
        self.history = list(history[-self.history_limit :])
        if not self.history:
            return
        if any(h.feature.version != self.feature_version for h in self.history):
            raise ValueError("surrogate feature version mismatch")
        self.x = np.array([h.feature.values for h in self.history], dtype=float)
        y = np.array([h.scalar_score for h in self.history], dtype=float)
        self.y_mean = float(y.mean())
        self.y_scale = max(float(y.std()), 0.1)
        kernel = tanimoto_matrix(self.x, self.x) + np.eye(len(y)) * 1e-5
        self.chol = np.linalg.cholesky(kernel)
        self.alpha = np.linalg.solve(
            self.chol.T, np.linalg.solve(self.chol, (y - self.y_mean) / self.y_scale)
        )

    def select(self, candidates, representations, *, count=1):
        x = np.array(
            [representations[c.candidate_id].values for c in candidates], dtype=float
        )
        if self.history:
            kernel = tanimoto_matrix(x, self.x)
            mean = self.y_mean + self.y_scale * (kernel @ self.alpha)
            v = np.linalg.solve(self.chol, kernel.T)
            std = self.y_scale * np.sqrt(np.maximum(1.0 - (v * v).sum(axis=0), 1e-9))
        else:
            mean = np.zeros(len(candidates))
            std = np.ones(len(candidates)) * 0.25
        scores = np.asarray(self.acquisition.score(mean, std))
        predictions = tuple(
            BOPrediction.scalar(
                c.candidate_id,
                mean=float(mean[i]),
                std=float(std[i]),
                acquisition_score=float(scores[i]),
            )
            for i, c in enumerate(candidates)
        )
        ranked = sorted(range(len(candidates)), key=lambda i: (-scores[i], i))
        return BOSelectionResult(
            tuple(candidates[i].candidate_id for i in ranked[:count]),
            predictions,
            metadata={
                "kernel": "tanimoto",
                "training_observations": len(self.history),
                "history_limit": self.history_limit,
            },
        )
