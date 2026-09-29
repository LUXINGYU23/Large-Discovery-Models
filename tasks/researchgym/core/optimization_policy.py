"""Measured-only compiled-policy inputs for the fixed program GP.

Mean features are the public program features of each row. Proposal
frequencies, acquisition values, and selection probabilities appear only in
the weight context, never in the numeric feature arrays.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from ldm_tts.harness import HarnessSubmissionError, PolicyCapabilityContract, PolicyRoundInput

from .candidate import FEATURE_GROUPS, FEATURE_NAMES, FEATURE_VERSION
from .sampling import tilted_distribution

FEATURE_SEMANTICS = {
    "program_chars": "Program length divided by the case character limit, capped at 1.",
    "function_defs": "Function definitions divided by 10, capped at 1.",
    "loops": "for/while statements divided by 10, capped at 1.",
    "branches": "if statements divided by 10, capped at 1.",
    "calls": "Call expressions divided by 60, capped at 1.",
    "numeric_constants": "Numeric literals divided by 40, capped at 1.",
    "family_confidence": "Called or accessed names matching entropy/softmax/likelihood terms, divided by 6.",
    "family_selection": "Names matching quantile/top-k/sort/masking terms, divided by 6.",
    "family_geometry": "Names matching normalization/statistics/distance terms, divided by 6.",
    "family_memory": "Names matching buffers/copies/EMA/state-dict terms, divided by 6.",
    "family_views": "Names matching dropout/augmentation/noise terms, divided by 6.",
    "family_control": "Names matching optimizer-step/gradient-control terms, divided by 6.",
}


class ResearchGymPolicyAdapter:
    def __init__(self, *, case_id, metric, alpha, eta, seed, beta, z_clip, proposal_facts):
        self.case_id, self.metric, self.seed = case_id, metric, seed
        self.beta, self.z_clip, self.proposal_facts = beta, z_clip, dict(proposal_facts)
        self.contract = PolicyCapabilityContract(
            task_id="researchgym", api_version=1, enabled_capabilities=("prior_mean@1", "ldm_weights@1"),
            feature_names=FEATURE_NAMES, feature_groups=FEATURE_GROUPS, mean_clip=3.0,
            default_alpha=alpha, default_eta=eta,
        )

    def capability_contract(self):
        return self.contract

    def build_selection_round(self, *, round_index, history, candidates, representations, baseline_predictions,
                              q0, valid_proposal_occurrences, requested_evaluation_batch, gp):
        width = len(FEATURE_NAMES)
        if any(h.feature is None or h.feature.version != FEATURE_VERSION for h in history):
            raise ValueError("policy history features do not match the program encoder")
        if [c.candidate_id for c in candidates] != [p.candidate_id for p in baseline_predictions]:
            raise ValueError("policy predictions must align with candidate rows")
        q0 = np.asarray(q0, dtype=float)
        if q0.shape != (len(candidates),) or np.any(q0 <= 0) or not np.isclose(q0.sum(), 1.0):
            raise ValueError("policy q0 must be a normalized positive candidate-aligned vector")
        hx = np.asarray([h.feature.values for h in history], dtype=float).reshape((-1, width))
        qx = np.asarray([representations[c.candidate_id].values for c in candidates], dtype=float)
        y = np.asarray([h.scalar_score for h in history], dtype=float)
        rounds = tuple(int(h.metadata["round_idx"]) for h in history)
        acquisition = np.asarray([p.acquisition_score for p in baseline_predictions], dtype=float)
        default_probability, _, normalized = tilted_distribution(
            q0, acquisition, alpha=self.contract.default_alpha, eta=self.contract.default_eta, z_clip=self.z_clip)
        arrays, location, scale = {}, float(y.mean()), max(float(y.std()), gp.target_scale_floor)
        projection = gp.projection(history, qx)
        if projection is not None:
            arrays.update({"diagnostic_current_" + name: value for name, value in projection.items()})
            location, scale = (float(v) for v in projection["location_scale"])
        folds = []
        for index, held in enumerate(sorted(set(rounds))[1:][-3:]):
            train = [i for i, r in enumerate(rounds) if r < held]
            test = [i for i, r in enumerate(rounds) if r == held]
            fold = gp.projection([history[i] for i in train], hx[test])
            if fold is None:
                continue
            prefix = f"diagnostic_fold_{index}_"
            arrays.update({prefix + name: value for name, value in fold.items()})
            folds.append({"prefix": prefix, "round_index": held, "train_indices": train, "test_indices": test})
        mean_context = {
            "round_index": round_index, "seed": self.seed, "feature_version": FEATURE_VERSION,
            "target_location": location, "target_scale": scale, "objective": self.metric,
            "feature_names": list(FEATURE_NAMES), "feature_groups": {k: list(v) for k, v in FEATURE_GROUPS.items()},
            "feature_semantics": FEATURE_SEMANTICS,
        }
        weight_context = {
            "seed": self.seed, "history_size": len(history), "unique_candidate_count": len(candidates),
            "valid_proposal_occurrences": valid_proposal_occurrences,
            "requested_evaluation_batch": requested_evaluation_batch,
            "effective_evaluation_batch": min(requested_evaluation_batch, len(candidates)),
            "default_alpha": self.contract.default_alpha, "default_eta": self.contract.default_eta,
            "proposal_configuration": self.proposal_facts,
            "q0_summary": {"entropy": float(-np.sum(q0 * np.log(q0))), "max": float(q0.max()),
                           "distinct_programs": len(candidates)},
            "acquisition": {"name": "ucb", "score_direction": "maximize", "beta": self.beta},
            "normalization": {"name": "robust_z", "epsilon": 1e-12, "mad_scale": 1.4826, "z_clip": self.z_clip},
            "candidate_predictions": [
                {"candidate_id": c.candidate_id, "q0": float(q0[i]), "baseline_mean": p.scalar_mean,
                 "baseline_std": p.scalar_std, "baseline_acquisition": p.acquisition_score,
                 "normalized_acquisition": float(normalized[i]),
                 "default_selection_probability": float(default_probability[i])}
                for i, (c, p) in enumerate(zip(candidates, baseline_predictions))
            ],
        }
        measured = tuple({"candidate_id": h.candidate_id, "round_index": r, "utility": float(v)}
                         for h, r, v in zip(history, rounds, y))
        return PolicyRoundInput(
            round_index=round_index, history_features=hx, history_utilities=y, query_features=qx,
            history_candidate_ids=tuple(h.candidate_id for h in history), history_rounds=rounds,
            measured_observations=measured,
            research_snapshot={
                "task": "researchgym", "case_id": self.case_id, "feature_contract": self.contract.to_dict(),
                "fixed_model": ("Exact RBF residual GP on the most recent measured window; features are not "
                                "rescaled (scale floor 1); utilities standardized on that window."),
                "gp": gp.to_dict(),
                "sampling": "p_i proportional to q0_i^alpha * exp(eta * robust_z(UCB_i)); Gumbel top-k without replacement.",
                "history_tool": "get_measured_history",
                "proposal_pool": {"unique_candidate_count": len(candidates),
                                  "valid_proposal_occurrences": valid_proposal_occurrences},
                "validation_scope": "Chronological whole-round holdouts use only earlier measured rounds.",
            },
            execution_context={"mean_context": mean_context, "weight_context": weight_context, "validation_folds": folds},
            diagnostic_arrays=arrays,
        )

    def with_feedback(self, round_input, records):
        measured = {(r, c): float(y) for r, c, y in zip(
            round_input.history_rounds, round_input.history_candidate_ids, round_input.history_utilities)}
        matched = [{"round_index": record["round_index"], **prediction,
                    "measured_utility": measured[(record["round_index"], prediction["candidate_id"])]}
                   for record in records for prediction in record["predictions"]
                   if (record["round_index"], prediction["candidate_id"]) in measured]
        summary = {"count": len(matched),
                   "scope": "Frozen pre-measurement predictions matched to later measured selections",
                   "interpretation": "Selected-point error does not rank the pool or validate alpha/eta."}
        for name in ("baseline", "active"):
            if matched:
                errors = np.asarray([row[name + "_mean"] - row["measured_utility"] for row in matched])
                summary[name + "_rmse"] = float(np.sqrt(np.mean(errors ** 2)))
        execution = dict(round_input.execution_context)
        execution["weight_context"] = {**execution["weight_context"], "prediction_feedback": {
            "summary": summary, "measurements": matched[-16:], "omitted_older": max(0, len(matched) - 16)}}
        return replace(round_input, execution_context=execution)

    def validate_task_execution(self, execution, round_input):
        errors = []
        for name, values, count in (("history_prior_mean", execution.history_prior_mean, len(round_input.history_features)),
                                    ("query_prior_mean", execution.query_prior_mean, len(round_input.query_features))):
            if values.shape != (count,) or not np.isfinite(values).all():
                errors.append(HarnessSubmissionError(
                    "/outputs/" + name, "researchgym_prior_alignment",
                    "Return one finite standardized mean per authoritative program row.",
                    "Preserve the supplied row order and a one-dimensional shape."))
        return tuple(errors)
