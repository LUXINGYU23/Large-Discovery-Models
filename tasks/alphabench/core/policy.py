"""T3 compiled-policy inputs and Host-owned residual-GP selection."""

from dataclasses import replace
import json

import numpy as np

from ldm_tts.harness import HarnessSubmissionError, PolicyCapabilityContract, PolicyRoundInput
from ldm_tts.optimization.gp import RBFGPSurrogate
from ldm_tts.optimization.records import BOSelectionResult, BOPrediction

from .selection import FEATURE_VERSION, LAYOUT, FactorSelector


class T3PolicyAdapter:
    def __init__(self, protocol, initial_candidate_ids=()):
        self.protocol = protocol
        self.initial_candidate_ids = frozenset(initial_candidate_ids)
        self.contract = PolicyCapabilityContract(
            task_id="alphabench", api_version=1,
            enabled_capabilities=tuple(protocol.policy_capabilities),
            feature_names=LAYOUT, feature_groups={"public_formula_ast": (0, len(LAYOUT))},
            mean_clip=3.0, default_alpha=2.0, default_eta=.25)

    def capability_contract(self):
        return self.contract

    def build_selection_round(self, *, round_idx, history, candidates, representations,
                              baseline_predictions, requested_batch, gp):
        width = len(LAYOUT)
        if tuple(item.candidate_id for item in baseline_predictions) != tuple(item.candidate_id for item in candidates):
            raise ValueError("policy baseline predictions do not align with the reservoir")
        hx = np.asarray([item.feature_vector for item in history], dtype=float).reshape((-1, width))
        qx = np.asarray([representations[item.candidate_id].values for item in candidates], dtype=float)
        y = np.asarray([item.scalar_score for item in history], dtype=float)
        rounds = tuple(-1 if item.candidate_id in self.initial_candidate_ids else item.metadata["round_idx"]
                       for item in history)
        if any(value >= round_idx for value in rounds):
            raise ValueError("policy history contains the current or a future search round")
        location = float(gp.y_mean) if gp.ready else float(y.mean()) if len(y) else 0.0
        scale = float(gp.y_std) if gp.ready else max(float(y.std()), .01) if len(y) else .25
        q0 = np.asarray([item.metadata["q0"] for item in candidates], dtype=float)
        if not np.isfinite(q0).all() or np.any(q0 <= 0) or not np.isclose(q0.sum(), 1):
            raise ValueError("policy q0 must be finite positive empirical mass")
        m = sum(len(item.metadata.get("harness_lineage", ())) for item in candidates)
        if m != self.protocol.sessions * self.protocol.candidates_per_session:
            raise ValueError("policy occurrence denominator differs from the frozen Harness protocol")
        measured = tuple({"candidate_id": item.candidate_id, "round_index": source_round,
                          "utility": item.scalar_score} for item, source_round in zip(history, rounds))
        diagnostics = {}
        if gp.ready:
            diagnostics = {"diagnostic_current_" + name: value
                           for name, value in gp.posterior_projection(qx).items()}
        return PolicyRoundInput(
            round_index=round_idx, history_features=hx, history_utilities=y,
            query_features=qx, history_candidate_ids=tuple(item.candidate_id for item in history),
            history_rounds=rounds, measured_observations=measured,
            research_snapshot={
                "task": "alphabench", "backend": self.protocol.backend, "market": self.protocol.market,
                "objective": self.protocol.objective, "capabilities": list(self.contract.enabled_capabilities),
                "feature_contract": self.contract.to_dict(),
                "fixed_model": "Exact RBF residual GP, lengthscale 1.5, noise 1e-4, beta 1.0; standardized measured target with scale floor 0.01.",
                "sampling": "alpha*log(q0)+eta*clip(robust_z(UCB),-5,5), Gumbel top-k without replacement",
                "candidate_catalog": [{"candidate_id": item.candidate_id,
                                       "expression": item.payload["expression"]} for item in candidates],
                "history_mask": [True] * len(history), "target_direction": "maximize_signed_search_correlation",
            },
            execution_context={
                "mean_context": {"feature_version": FEATURE_VERSION, "feature_names": list(LAYOUT),
                                 "target_location": location, "target_scale": scale,
                                 "target_direction": "maximize_signed_search_correlation"},
                "weight_context": {"round_index": round_idx, "sessions": self.protocol.sessions,
                                   "candidates_per_session": self.protocol.candidates_per_session,
                                   "occurrences": m, "unique_candidates": len(candidates),
                                   "requested_evaluation_batch": requested_batch,
                                   "default_alpha": self.contract.default_alpha,
                                   "default_eta": self.contract.default_eta,
                                   "candidate_predictions": [{"candidate_id": item.candidate_id,
                                       "q0": float(q0[index]), "baseline_mean": prediction.scalar_mean,
                                       "baseline_std": prediction.scalar_std,
                                       "baseline_ucb": prediction.acquisition_score}
                                       for index, (item, prediction) in enumerate(zip(candidates, baseline_predictions))]},
            }, diagnostic_arrays=diagnostics)

    def with_feedback(self, round_input, records):
        measured = {(round_idx, candidate_id): float(value) for round_idx, candidate_id, value in
                    zip(round_input.history_rounds, round_input.history_candidate_ids, round_input.history_utilities)}
        matched = [{**prediction, "round_index": record["round_index"],
                    "measured_utility": measured[(record["round_index"], prediction["candidate_id"])]}
                   for record in records for prediction in record["predictions"]
                   if (record["round_index"], prediction["candidate_id"]) in measured]
        context = dict(round_input.execution_context)
        context["weight_context"] = {**context["weight_context"],
            "prediction_feedback": {"count": len(matched), "measurements": matched[-16:]}}
        return replace(round_input, execution_context=context)

    def validate_task_execution(self, execution, round_input):
        return tuple(HarnessSubmissionError("/outputs/" + name, "t3_prior_alignment",
                     "Return one finite standardized prior per authoritative AST feature row.")
                     for name, expected in (("history_prior_mean", len(round_input.history_features)),
                                            ("query_prior_mean", len(round_input.query_features)))
                     if getattr(execution, name).shape != (expected,) or not np.isfinite(getattr(execution, name)).all())


class CompiledFactorSelector(FactorSelector):
    def __init__(self, protocol, *, mock, initial_candidate_ids=()):
        super().__init__(("mock_" if mock else "") + protocol.objective, seed=protocol.random_seed)
        self.protocol = protocol
        self.adapter = T3PolicyAdapter(protocol, initial_candidate_ids)
        self.controller = self.meter = self.gateway = None
        self.history = ()

    def bind(self, controller, meter, gateway):
        if self.gateway is not None:
            raise ValueError("compiled selector is already bound to a campaign")
        self.controller, self.meter, self.gateway = controller, meter, gateway

    def fit(self, history):
        self.history = tuple(history)
        super().fit(history)

    def select(self, candidates, representations, *, count=1, round_idx=0):
        if self.gateway is None:
            raise ValueError("compiled selector requires its policy Harness")
        baseline = tuple(self.gp.predict_record(item.candidate_id, representations[item.candidate_id].values,
                                                beta=self.beta) for item in candidates)
        policy_input = self.adapter.build_selection_round(round_idx=round_idx, history=self.history,
            candidates=candidates, representations=representations, baseline_predictions=baseline,
            requested_batch=count, gp=self.gp)
        self.gateway.runtime.consume_many({"policy_turns": 1}, usage_key=f"task:policy:round:{round_idx}")
        policy = self.gateway.host.run(lambda: self.controller.resolve(policy_input))
        turn = policy.metadata.get("harness_turn")
        if turn:
            accounting = self.meter.reconcile(turn["profile_id"], turn["turn_id"], turn["usage"])
        else:
            manifest = json.loads((self.controller.root / "rounds" / f"round_{round_idx:03d}" / "manifest.json").read_text())
            turn_id = f"alphabench-{self.controller.profile_id}-{round_idx:03d}-{manifest['input_sha256'][:12]}"
            accounting = self.meter.reconcile(self.controller.profile_id, turn_id,
                                              policy.metadata.get("failed_harness_usage", {}))
        if "prior_mean@1" not in self.protocol.policy_capabilities and (np.any(policy.history_prior_mean)
                or np.any(policy.query_prior_mean)):
            raise ValueError("disabled prior capability returned a nonzero mean")
        if "ldm_weights@1" not in self.protocol.policy_capabilities and (policy.alpha != self.alpha or policy.eta != self.eta):
            raise ValueError("disabled weight capability changed LDM weights")
        if self.gp.ready:
            active_gp = RBFGPSurrogate(self.history, lengthscale=1.5, noise=1e-4,
                prior_mean=0, prior_std=.25, min_training_observations=2,
                feature_scale_floor=1.0, target_scale_floor=.01, feature_version=FEATURE_VERSION,
                residual_prior_mean=policy.history_prior_mean)
            if not active_gp.ready:
                raise ValueError("residual GP numerical fit failed")
            active = tuple(active_gp.predict_record(item.candidate_id,
                representations[item.candidate_id].values, beta=self.beta,
                query_prior_mean=float(policy.query_prior_mean[index]))
                for index, item in enumerate(candidates))
            fit = active_gp.summary()
        else:
            scale = policy_input.execution_context["mean_context"]["target_scale"]
            active = tuple(BOPrediction.scalar(item.candidate_id,
                mean=prediction.scalar_mean + scale * float(policy.query_prior_mean[index]),
                std=prediction.scalar_std,
                acquisition_score=prediction.acquisition_score + scale * float(policy.query_prior_mean[index]),
                metadata={"fit_status": self.gp.fit_status, "prior_only_sparse_history": True})
                for index, (item, prediction) in enumerate(zip(candidates, baseline)))
            fit = self.gp.summary()
        result = super().select(candidates, representations, count=count, round_idx=round_idx,
                                predictions=active, alpha=policy.alpha, eta=policy.eta)
        self.controller.record_predictions(round_idx, [{"candidate_id": item.candidate_id,
            "baseline_mean": baseline[index].scalar_mean, "active_mean": active[index].scalar_mean,
            "baseline_std": baseline[index].scalar_std, "active_std": active[index].scalar_std,
            "q0": float(item.metadata["q0"]),
            "selection_probability": result.metadata["probabilities"][index]}
            for index, item in enumerate(candidates)])
        return BOSelectionResult(result.selected_candidate_ids, result.predictions,
            metadata={**result.metadata, "fit": fit, "policy": {"epoch_id": policy.epoch_id,
                "source": policy.source, "degraded": policy.degraded,
                "action": policy.metadata.get("action"), "status": policy.metadata.get("status"),
                "accounting": accounting, "target_location": policy_input.execution_context["mean_context"]["target_location"],
                "target_scale": policy_input.execution_context["mean_context"]["target_scale"],
                "history_mask": policy_input.research_snapshot["history_mask"],
                "target_direction": policy_input.research_snapshot["target_direction"]}})
