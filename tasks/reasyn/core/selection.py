"""Molecular Tanimoto GP using the shared posterior-acquisition contract."""

from __future__ import annotations
import numpy as np
from ldm_tts.contracts import AcquisitionSpec
from ldm_tts.optimization.acquisition import make_acquisition
from ldm_tts.optimization.records import BOSelectionResult, BOPrediction
from .sampling import (
    Q0_METADATA_KEY,
    candidate_set_seed,
    empirical_base_masses,
    gumbel_top_k,
    tilted_distribution,
)


def tanimoto_matrix(x, z):
    dot = x @ z.T
    denominator = (x * x).sum(axis=1)[:, None] + (z * z).sum(axis=1)[None, :] - dot
    return np.divide(dot, denominator, out=np.ones_like(dot), where=denominator > 0)


class TanimotoGPSelector:
    def __init__(
        self, *, objective_name, beta=1.0, history_limit=256, feature_version=""
    ):
        if history_limit < 1 or not np.isfinite(beta) or beta < 0:
            raise ValueError("GP history limit must be positive and beta finite/nonnegative")
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

    def fit(self, history, *, history_prior_mean=None):
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
        prior = np.zeros(len(y))
        if history_prior_mean is not None:
            supplied = np.asarray(history_prior_mean, dtype=float)
            if supplied.shape != (len(history),) or not np.isfinite(supplied).all():
                raise ValueError("compiled history prior must align with finite GP history")
            prior = supplied[-self.history_limit:]
        self.alpha = np.linalg.solve(
            self.chol.T, np.linalg.solve(self.chol, (y - self.y_mean) / self.y_scale - prior)
        )

    def select(self, candidates, representations, *, count=1, round_idx=0, query_prior_mean=None):
        if not candidates:
            return BOSelectionResult(())
        if any(representations[c.candidate_id].version != self.feature_version for c in candidates):
            raise ValueError("surrogate feature version mismatch")
        x = np.array(
            [representations[c.candidate_id].values for c in candidates], dtype=float
        )
        prior = np.zeros(len(candidates))
        if query_prior_mean is not None:
            prior = np.asarray(query_prior_mean, dtype=float)
            if prior.shape != (len(candidates),) or not np.isfinite(prior).all():
                raise ValueError("compiled query prior must align with finite GP queries")
        if self.history:
            kernel = tanimoto_matrix(x, self.x)
            mean = self.y_mean + self.y_scale * (prior + kernel @ self.alpha)
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

    def posterior_projection(self, history, query_vectors):
        """Frozen linear residual-GP operator for task-owned draft diagnostics."""
        self.fit(history)
        query = np.asarray(query_vectors, dtype=float)
        if query.ndim != 2 or not np.isfinite(query).all():
            raise ValueError("GP diagnostic query vectors must be a finite matrix")
        if not history:
            return {"weights": np.zeros((len(query), 0)), "std": np.full(len(query), 0.25),
                    "location_scale": np.asarray([0.0, 1.0])}
        cross = tanimoto_matrix(query, self.x)
        weights = np.zeros((len(query), len(history)))
        weights[:, -len(self.history):] = np.linalg.solve(self.chol.T, np.linalg.solve(self.chol, cross.T)).T
        v = np.linalg.solve(self.chol, cross.T)
        std = self.y_scale * np.sqrt(np.maximum(1.0 - (v * v).sum(axis=0), 1e-9))
        return {"weights": weights, "std": std,
                "location_scale": np.asarray([self.y_mean, self.y_scale])}


class AcquisitionTiltedSelector:
    """Maintain an independent BO pool, then sample the empirical-q0 UCB tilt."""

    def __init__(self, base_selector, *, alpha=1.0, eta=1.0, z_clip=5.0,
                 seed=0, pool_size=None, policy_controller=None, policy_adapter=None):
        if any(not np.isfinite(v) or v < 0 for v in (alpha, eta)):
            raise ValueError("LDM alpha and eta must be finite and nonnegative")
        if not np.isfinite(z_clip) or z_clip <= 0 or seed < 0:
            raise ValueError("LDM z_clip must be positive and seed nonnegative")
        if pool_size is not None and pool_size < 1:
            raise ValueError("BO pool size must be positive")
        if (policy_controller is None) != (policy_adapter is None):
            raise ValueError("compiled policy controller and adapter must be configured together")
        self.base_selector = base_selector
        self.alpha, self.eta, self.z_clip = alpha, eta, z_clip
        self.seed, self.pool_size = seed, pool_size
        self.policy_controller, self.policy_adapter = policy_controller, policy_adapter
        self.history = ()

    def describe(self):
        base = self.base_selector.describe()
        return AcquisitionSpec(
            name="ldm_tanimoto_gp_ucb",
            objective_names=base.objective_names,
            score_direction="sample",
            selection_rule="Sample alpha * log(q0 + epsilon) + eta * robust_z(UCB) without replacement",
            parameters={
                "base_acquisition": base.name, **base.parameters,
                "alpha": self.alpha, "eta": self.eta, "z_clip": self.z_clip,
                "bo_pool_size": self.pool_size, "seed": self.seed,
                "base_measure": "empirical_occurrences_before_canonical_deduplication",
                "pool_maintenance": "q0_gumbel_top_k_without_replacement",
                "sampling": "gumbel_top_k_without_replacement",
                "compiled_policy": self.policy_controller is not None,
            },
        )

    def fit(self, history):
        self.history = tuple(history)
        self.base_selector.fit(self.history)

    def select(self, candidates, representations, *, count=1, round_idx=0):
        if count < 1:
            raise ValueError("selection count must be positive")
        reservoir = tuple(sorted(candidates, key=lambda c: c.candidate_id))
        if not reservoir:
            return BOSelectionResult(())
        if self.pool_size is not None and count > self.pool_size:
            raise ValueError("evaluation count cannot exceed the independent BO pool size")
        reservoir_q0 = empirical_base_masses(reservoir)
        limit = min(self.pool_size or len(reservoir), len(reservoir))
        maintenance_seed = candidate_set_seed(self.seed, round_idx, reservoir, phase="bo_pool")
        if limit < len(reservoir):
            indices = gumbel_top_k(reservoir_q0, limit, seed=maintenance_seed)
            pool = tuple(sorted((reservoir[i] for i in indices), key=lambda c: c.candidate_id))
        else:
            pool = reservoir
        q0 = empirical_base_masses(pool)
        valid_occurrences = reservoir[0].metadata[Q0_METADATA_KEY]["valid_occurrence_count"]
        if self.policy_controller is not None:
            # A previous call may have fitted a compiled residual prior. Always
            # materialize the zero-prior baseline from the authoritative history.
            self.base_selector.fit(self.history)
        baseline = self.base_selector.select(pool, representations, count=len(pool), round_idx=round_idx)
        active = baseline
        alpha, eta = self.alpha, self.eta
        policy = None
        if self.policy_controller is not None and self.history:
            round_input = self.policy_adapter.build_selection_round(
                round_index=round_idx, history=self.history, candidates=pool,
                representations=representations, baseline_predictions=baseline.predictions,
                valid_proposal_occurrences=valid_occurrences, q0=q0,
                requested_evaluation_batch=count,
            )
            policy = self.policy_controller.resolve(round_input)
            self.base_selector.fit(self.history, history_prior_mean=policy.history_prior_mean)
            active = self.base_selector.select(
                pool, representations, count=len(pool), round_idx=round_idx,
                query_prior_mean=policy.query_prior_mean,
            )
            alpha, eta = policy.alpha, policy.eta
        acquisition = np.asarray([p.acquisition_score for p in active.predictions], dtype=float)
        probability, logits, normalized = tilted_distribution(
            q0, acquisition, alpha=alpha, eta=eta, z_clip=self.z_clip,
        )
        selection_seed = candidate_set_seed(self.seed, round_idx, pool, phase="ldm_selection")
        indices = gumbel_top_k(probability, count, seed=selection_seed)
        records = [
            {
                "candidate_id": candidate.candidate_id,
                "q0": float(q0[i]), "normalized_acquisition": float(normalized[i]),
                "logit": float(logits[i]), "selection_probability": float(probability[i]),
                "baseline_mean": baseline.predictions[i].scalar_mean,
                "baseline_std": baseline.predictions[i].scalar_std,
                "active_mean": active.predictions[i].scalar_mean,
                "active_std": active.predictions[i].scalar_std,
                "first_draw_probability": float(probability[i]),
                "q0_relative_to_max": float(q0[i] / q0.max()),
                "pool_size": len(pool), "alpha": alpha, "eta": eta,
                "baseline_acquisition": baseline.predictions[i].acquisition_score,
                "active_acquisition": active.predictions[i].acquisition_score,
            }
            for i, candidate in enumerate(pool)
        ]
        if policy is not None:
            self.policy_controller.record_predictions(round_idx, records)
        metadata = {
            "kernel": "tanimoto", "proposal_reservoir_size": len(reservoir),
            "bo_pool_size": len(pool), "bo_pool_size_requested": self.pool_size,
            "bo_pool_maintenance_seed": maintenance_seed,
            "bo_pool_candidate_ids": [c.candidate_id for c in pool],
            "valid_proposal_occurrences": valid_occurrences,
            "selection_q0_scope": "conditioned_on_maintained_bo_pool",
            "alpha": alpha, "eta": eta, "z_clip": self.z_clip,
            "selection_seed": selection_seed, "distribution": records,
        }
        if policy is not None:
            metadata["compiled_policy"] = {
                **dict(policy.metadata),
                "epoch_id": policy.epoch_id, "artifact_sha256": policy.artifact_digest,
                "source": policy.source, "degraded": policy.degraded,
                "stage": policy.stage, "alpha": alpha, "eta": eta,
            }
        predictions = tuple(
            BOPrediction.scalar(
                p.candidate_id, mean=p.scalar_mean, std=p.scalar_std,
                acquisition_score=p.acquisition_score,
                metadata={**p.metadata, **records[i]},
            ) for i, p in enumerate(active.predictions)
        )
        return BOSelectionResult(
            tuple(pool[i].candidate_id for i in indices), predictions, metadata=metadata,
        )
