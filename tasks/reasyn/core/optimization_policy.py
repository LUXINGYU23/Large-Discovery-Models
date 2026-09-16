"""Authoritative measured-only policy inputs for the fixed molecular GP."""

from dataclasses import replace

import numpy as np

from ldm_tts.harness import PolicyCapabilityContract, PolicyRoundInput, HarnessSubmissionError
from .sampling import tilted_distribution, empirical_group_sizes
from .selection import TanimotoGPSelector


class ReaSynPolicyAdapter:
    def __init__(self, encoder, *, alpha, eta, seed, history_limit=256, beta=1.0, z_clip=5.0, benchmark="tdc"):
        self.encoder, self.seed = encoder, seed
        self.history_limit, self.beta, self.z_clip = history_limit, beta, z_clip
        self.diagnostic_gp = TanimotoGPSelector(
            objective_name="utility", history_limit=history_limit, beta=beta,
            feature_version=encoder.version,
        )
        width = encoder.describe().dimension
        self.contract = PolicyCapabilityContract(
            task_id="reasyn", api_version=1,
            enabled_capabilities=("prior_mean@1", "ldm_weights@1"),
            feature_names=tuple(f"fingerprint_{i}" for i in range(width)),
            feature_groups={"molecular_fingerprint": (0, width)}, mean_clip=3.0,
            default_alpha=alpha, default_eta=eta,
        )

    def capability_contract(self):
        return self.contract

    def build_selection_round(self, *, round_index, history, candidates,
                              representations, baseline_predictions, valid_proposal_occurrences,
                              q0, requested_evaluation_batch):
        width = len(self.contract.feature_names)
        if any(h.feature is None or h.feature.version != self.encoder.version for h in history):
            raise ValueError("policy history feature version does not match the molecular encoder")
        if any(representations[c.candidate_id].version != self.encoder.version for c in candidates):
            raise ValueError("policy query feature version does not match the molecular encoder")
        if (not candidates or len(candidates) != len(baseline_predictions)
                or len({c.candidate_id for c in candidates}) != len(candidates)
                or any(c.candidate_id != p.candidate_id for c, p in zip(candidates, baseline_predictions))):
            raise ValueError("policy predictions must exactly align with candidate rows")
        q0 = np.asarray(q0, dtype=float)
        if (q0.shape != (len(candidates),) or not np.isfinite(q0).all()
                or np.any(q0 <= 0) or not np.isclose(q0.sum(), 1.0)):
            raise ValueError("policy q0 must be a normalized positive candidate-aligned vector")
        hx = np.asarray([h.feature.values for h in history], dtype=float).reshape((-1, width))
        qx = np.asarray([representations[c.candidate_id].values for c in candidates], dtype=float)
        if qx.shape != (len(candidates), width):
            raise ValueError("policy query matrix does not match the molecular feature width")
        y = np.asarray([h.scalar_score for h in history], dtype=float)
        acquisition = np.asarray([p.acquisition_score for p in baseline_predictions], dtype=float)
        group_sizes = empirical_group_sizes(candidates)
        probabilities, _, normalized = tilted_distribution(
            q0, acquisition, alpha=self.contract.default_alpha, eta=self.contract.default_eta,
            z_clip=self.z_clip,
            group_sizes=group_sizes,
        )
        projection = self.diagnostic_gp.posterior_projection(history, qx)
        location, scale = projection["location_scale"]
        arrays = {"diagnostic_current_" + name: values for name, values in projection.items()}
        context = {
            "round_index": round_index, "seed": self.seed,
            "feature_version": self.encoder.version, "target_location": float(location),
            "target_scale": float(scale), "feature_names": list(self.contract.feature_names),
            "feature_groups": {"molecular_fingerprint": [0, width]},
        }
        weight_context = {
            "seed": self.seed, "history_size": len(history),
            "unique_candidate_count": len(candidates), "valid_proposal_occurrences": valid_proposal_occurrences,
            "requested_evaluation_batch": requested_evaluation_batch,
            "effective_evaluation_batch": min(requested_evaluation_batch, len(candidates)),
            "default_alpha": self.contract.default_alpha, "default_eta": self.contract.default_eta,
            "q0_summary": {
                "scope": "canonical_query_or_product_groups",
                "group_count": int(round(float(np.sum(1 / group_sizes)))),
                "entropy": float(-np.sum(q0 * np.log(q0 * group_sizes + 1e-12))),
                "max": float(np.max(q0 * group_sizes)),
            },
            "trial_base_summary": {"entropy": float(-np.sum(q0 * np.log(q0 + 1e-12))), "max": float(q0.max())},
            "baseline_acquisition_summary": {"mean": float(acquisition.mean()), "std": float(acquisition.std())},
            "acquisition": {"name": "ucb", "score_direction": "maximize", "beta": self.beta},
            "normalization": {"name": "robust_z", "epsilon": 1e-12, "mad_scale": 1.4826, "z_clip": self.z_clip},
            "candidate_predictions": [
                {"candidate_id": c.candidate_id, "q0": float(q0[i]),
                 "group_trial_count": int(group_sizes[i]), "q0_group_mass": float(q0[i] * group_sizes[i]),
                 "baseline_mean": p.scalar_mean, "baseline_std": p.scalar_std,
                 "baseline_acquisition": p.acquisition_score,
                 "normalized_acquisition": float(normalized[i]),
                 "default_selection_probability": float(probabilities[i])}
                for i, (c, p) in enumerate(zip(candidates, baseline_predictions))
            ],
        }
        rounds = tuple(h.metadata["round_idx"] for h in history)
        folds = []
        for index, held_round in enumerate(sorted(set(rounds))[1:][-3:]):
            train = [i for i, r in enumerate(rounds) if r < held_round]
            test = [i for i, r in enumerate(rounds) if r == held_round]
            projection = self.diagnostic_gp.posterior_projection([history[i] for i in train], hx[test])
            prefix = f"diagnostic_fold_{index}_"
            arrays.update({prefix + name: values for name, values in projection.items()})
            folds.append({"prefix": prefix, "round_index": held_round,
                          "train_indices": train, "test_indices": test})
        measured = tuple({"candidate_id": h.candidate_id, "round_index": r,
                          "utility": h.scalar_score} for h, r in zip(history, rounds))
        return PolicyRoundInput(
            round_index=round_index, history_features=hx, history_utilities=y,
            query_features=qx, history_candidate_ids=tuple(h.candidate_id for h in history),
            history_rounds=rounds, measured_observations=measured,
            research_snapshot={
                "task": "reasyn", "feature_contract": self.contract.to_dict(),
                "fixed_model": "Exact Tanimoto residual GP on the most recent measured training window; utilities standardized on that same window.",
                "feature_interpretation": "Fingerprint indices are hashed and have no fixed chemical names.",
                "gp_history_limit": self.history_limit, "kernel_jitter": 1e-5, "utility_scale_floor": 0.1,
                "sampling": "alpha * log(query_group_q0 + epsilon) - log(group_trial_count) + eta * robust_z(UCB); independent trials sampled without replacement. TDC has one trial per product group.",
                "history_tool": "get_measured_history", "proposal_pool": {
                    "unique_candidate_count": len(candidates), "valid_proposal_occurrences": valid_proposal_occurrences},
                "candidate_catalog": [{"candidate_id": c.candidate_id,
                    "smiles": c.payload.get("smiles", c.payload.get("target_smiles"))} for c in candidates],
                "validation_scope": "Chronological whole-round holdouts use only earlier measured observations; current-pool probabilities have no query labels.",
            },
            execution_context={"mean_context": context, "weight_context": weight_context,
                               "validation_folds": folds},
            diagnostic_arrays=arrays,
        )

    def with_feedback(self, round_input, records):
        measured = {(r, c): float(y) for r, c, y in zip(
            round_input.history_rounds, round_input.history_candidate_ids, round_input.history_utilities)}
        matched = []
        for record in records:
            for prediction in record["predictions"]:
                key = (record["round_index"], prediction["candidate_id"])
                if key in measured:
                    matched.append({"round_index": key[0], **prediction, "measured_utility": measured[key]})
        summary = {
            "count": len(matched), "scope": "Frozen predictions matched to subsequently measured selected candidates",
            "interpretation": "Selected-point predictive error does not establish whole-pool ranking or weight-policy reward. First-draw probability is not batch inclusion probability.",
        }
        if matched:
            for name in ("baseline", "active"):
                errors = np.asarray([row[name + "_mean"] - row["measured_utility"] for row in matched])
                summary[name + "_rmse"] = float(np.sqrt(np.mean(errors ** 2)))
                summary[name + "_mae"] = float(np.mean(np.abs(errors)))
        execution = dict(round_input.execution_context)
        execution["weight_context"] = {
            **execution["weight_context"],
            "prediction_feedback": {"summary": summary, "measurements": matched[-16:],
                                    "omitted_older_measurements": max(0, len(matched) - 16)},
        }
        return replace(round_input, execution_context=execution)

    def validate_task_execution(self, execution, round_input):
        errors = []
        for name, values, count in (
            ("history_prior_mean", execution.history_prior_mean, len(round_input.history_features)),
            ("query_prior_mean", execution.query_prior_mean, len(round_input.query_features)),
        ):
            if values.shape != (count,) or not np.isfinite(values).all():
                errors.append(HarnessSubmissionError(
                    "/outputs/" + name, "reasyn_prior_alignment",
                    "Return one finite standardized mean per authoritative fingerprint row.",
                    "Preserve the supplied row order and shape.",
                ))
        return tuple(errors)
