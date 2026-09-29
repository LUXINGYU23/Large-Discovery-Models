"""LDM selection: empirical q0, residual RBF GP-UCB, and optional compiled policy."""

from __future__ import annotations

import numpy as np

from ldm_tts.contracts import AcquisitionSpec
from ldm_tts.optimization.gp import RBFGPSurrogate
from ldm_tts.optimization.records import BOPrediction, BOSelectionResult

from .candidate import FEATURE_VERSION
from .sampling import Q0_KEY, candidate_set_seed, empirical_base_masses, gumbel_top_k, tilted_distribution


class GPSettings:
    def __init__(self, *, lengthscale, noise, beta, history_limit, target_scale_floor):
        self.lengthscale, self.noise, self.beta = lengthscale, noise, beta
        self.history_limit, self.target_scale_floor = history_limit, target_scale_floor

    def to_dict(self):
        return dict(self.__dict__)

    def fit(self, history, prior=None) -> RBFGPSurrogate:
        """The fixed task GP; a compiled prior only shifts its residual mean."""
        window = list(history)[-self.history_limit:]
        if prior is not None:
            prior = np.asarray(prior, dtype=float)
            if prior.shape != (len(history),):
                raise ValueError("compiled prior must align with the full measured history")
            prior = prior[-self.history_limit:]
        return RBFGPSurrogate(
            window, lengthscale=self.lengthscale, noise=self.noise, feature_version=FEATURE_VERSION,
            min_training_observations=1, feature_scale_floor=1.0,
            target_scale_floor=self.target_scale_floor, residual_prior_mean=prior,
        )

    def projection(self, history, vectors):
        """Frozen linear residual operator over the full history for draft diagnostics."""
        gp = self.fit(history)
        if not gp.ready:
            return None
        projection = gp.posterior_projection(vectors)
        weights = np.zeros((len(vectors), len(history)))
        weights[:, -len(gp.observations):] = projection["weights"]
        return {**projection, "weights": weights}


class ProgramLDMSelector:
    """Maintain a q0-sampled BO pool, then sample q0^alpha * exp(eta * robust_z(UCB))."""

    def __init__(self, gp: GPSettings, *, objective_name, alpha, eta, z_clip, seed, pool_size,
                 policy_controller=None, policy_adapter=None) -> None:
        if any(not np.isfinite(v) or v < 0 for v in (alpha, eta)) or not np.isfinite(z_clip) or z_clip <= 0:
            raise ValueError("LDM alpha/eta must be finite nonnegative and z_clip positive")
        if pool_size < 1:
            raise ValueError("BO pool size must be positive")
        if (policy_controller is None) != (policy_adapter is None):
            raise ValueError("compiled policy controller and adapter must be configured together")
        self.gp, self.objective_name = gp, objective_name
        self.alpha, self.eta, self.z_clip, self.seed, self.pool_size = alpha, eta, z_clip, seed, pool_size
        self.policy_controller, self.policy_adapter = policy_controller, policy_adapter
        # Optional (before, after) callables around a policy turn, used for spending checks.
        self.policy_hooks = None
        self.history = ()

    def describe(self) -> AcquisitionSpec:
        return AcquisitionSpec(
            name="ldm_rbf_gp_ucb", objective_names=(self.objective_name,), score_direction="sample",
            selection_rule="q0^alpha * exp(eta * robust_z(UCB)); Gumbel top-k without replacement",
            parameters={"alpha": self.alpha, "eta": self.eta, "z_clip": self.z_clip, "seed": self.seed,
                        "bo_pool_size": self.pool_size, "gp": self.gp.to_dict(),
                        "base_measure": "empirical_occurrences_before_canonical_deduplication",
                        "pool_maintenance": "q0_gumbel_top_k_without_replacement",
                        "compiled_policy": self.policy_controller is not None},
        )

    def fit(self, history) -> None:
        if any(item.feature is None or item.feature.version != FEATURE_VERSION for item in history):
            raise ValueError("measured history feature version mismatch")
        self.history = tuple(history)

    def select(self, candidates, representations, *, count=1, round_idx=0) -> BOSelectionResult:
        if count < 1:
            raise ValueError("selection count must be positive")
        reservoir = tuple(sorted(candidates, key=lambda c: c.candidate_id))
        if not reservoir:
            return BOSelectionResult(())
        if count > self.pool_size:
            raise ValueError("evaluation batch cannot exceed the BO pool size")
        if any(representations[c.candidate_id].version != FEATURE_VERSION for c in reservoir):
            raise ValueError("query feature version mismatch")
        maintenance_seed = candidate_set_seed(self.seed, round_idx, reservoir, phase="bo_pool")
        reservoir_q0 = empirical_base_masses(reservoir)
        if self.pool_size < len(reservoir):
            kept = gumbel_top_k(reservoir_q0, self.pool_size, seed=maintenance_seed)
            pool = tuple(sorted((reservoir[i] for i in kept), key=lambda c: c.candidate_id))
        else:
            pool = reservoir
        q0 = empirical_base_masses(pool)
        vectors = [representations[c.candidate_id].values for c in pool]
        baseline_gp = self.gp.fit(self.history)
        baseline = tuple(baseline_gp.predict_record(c.candidate_id, v, beta=self.gp.beta) for c, v in zip(pool, vectors))
        active, alpha, eta, policy = baseline, self.alpha, self.eta, None
        occurrences = reservoir[0].metadata[Q0_KEY]["valid_occurrence_count"]
        if self.policy_controller is not None and self.history:
            round_input = self.policy_adapter.build_selection_round(
                round_index=round_idx, history=self.history, candidates=pool, representations=representations,
                baseline_predictions=baseline, q0=q0, valid_proposal_occurrences=occurrences,
                requested_evaluation_batch=count, gp=self.gp)
            if self.policy_hooks:
                self.policy_hooks[0]()
            try:
                policy = self.policy_controller.resolve(round_input)
            finally:
                if self.policy_hooks:
                    self.policy_hooks[1]()
            compiled_gp = self.gp.fit(self.history, policy.history_prior_mean)
            query_prior = np.asarray(policy.query_prior_mean, dtype=float)
            active = tuple(compiled_gp.predict_record(c.candidate_id, v, beta=self.gp.beta,
                                                      query_prior_mean=float(query_prior[i]))
                           for i, (c, v) in enumerate(zip(pool, vectors)))
            alpha, eta = policy.alpha, policy.eta
        acquisition = np.asarray([p.acquisition_score for p in active], dtype=float)
        probability, logits, normalized = tilted_distribution(q0, acquisition, alpha=alpha, eta=eta, z_clip=self.z_clip)
        selection_seed = candidate_set_seed(self.seed, round_idx, pool, phase="ldm_selection")
        chosen = gumbel_top_k(probability, count, seed=selection_seed)
        records = [{
            "candidate_id": c.candidate_id, "q0": float(q0[i]), "q0_relative_to_max": float(q0[i] / q0.max()),
            "normalized_acquisition": float(normalized[i]), "logit": float(logits[i]),
            "first_draw_probability": float(probability[i]),
            "baseline_mean": baseline[i].scalar_mean, "baseline_std": baseline[i].scalar_std,
            "baseline_acquisition": baseline[i].acquisition_score,
            "active_mean": active[i].scalar_mean, "active_std": active[i].scalar_std,
            "active_acquisition": active[i].acquisition_score,
            "pool_size": len(pool), "alpha": alpha, "eta": eta,
        } for i, c in enumerate(pool)]
        if policy is not None:
            self.policy_controller.record_predictions(round_idx, records)
        k = max(1, min(count, len(pool)))
        metadata = {
            "surrogate": baseline_gp.summary(), "proposal_reservoir_size": len(reservoir),
            "bo_pool_size": len(pool), "bo_pool_size_requested": self.pool_size,
            "bo_pool_candidate_ids": [c.candidate_id for c in pool], "bo_pool_maintenance_seed": maintenance_seed,
            "valid_proposal_occurrences": occurrences, "selection_seed": selection_seed,
            "alpha": alpha, "eta": eta, "z_clip": self.z_clip, "distribution": records,
            "base_probability_entropy": float(-np.sum(q0 * np.log(q0))),
            "probability_entropy": float(-np.sum(probability * np.log(np.maximum(probability, 1e-300)))),
            "probability_effective_sample_size": float(1 / np.sum(probability ** 2)),
            "tilted_kl_from_q0": float(np.sum(probability * np.log(np.maximum(probability, 1e-300) / q0))),
            "q0_tilted_topk_overlap": len(set(np.argsort(q0)[-k:]) & set(np.argsort(probability)[-k:])) / k,
        }
        if policy is not None:
            metadata["compiled_policy"] = {
                **dict(policy.metadata), "epoch_id": policy.epoch_id, "artifact_sha256": policy.artifact_digest,
                "source": policy.source, "degraded": policy.degraded, "stage": policy.stage,
                "alpha": alpha, "eta": eta,
                "prediction_mean_abs_change": float(np.mean([abs(b.scalar_mean - a.scalar_mean)
                                                             for b, a in zip(baseline, active)])),
                **{f"{name}_prediction_diagnostics": {
                    "mean_mean": float(np.mean([p.scalar_mean for p in rows])),
                    "acquisition_mean": float(np.mean([p.acquisition_score for p in rows]))}
                   for name, rows in (("baseline", baseline), ("compiled", active))},
            }
        predictions = tuple(BOPrediction.scalar(p.candidate_id, mean=p.scalar_mean, std=p.scalar_std,
                                                acquisition_score=p.acquisition_score,
                                                metadata={**p.metadata, **records[i]})
                            for i, p in enumerate(active))
        return BOSelectionResult(tuple(pool[i].candidate_id for i in chosen), predictions, metadata=metadata)
