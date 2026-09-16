"""Independent compiled prior and LDM weights using past measured correctness."""

from dataclasses import replace
import json

import numpy as np

from ldm_tts.harness import (
    DockerPolicyExecutor, HarnessSubmissionError, PolicyCapabilityContract,
    PolicyResearchController, PolicyRoundInput,
)
from ldm_tts.harness.container import resolve_container_user
from .data import write_json
from .optimization_policy import FEATURE_NAMES
from .selection import FEATURE_VERSION, make_gp, gp_projection, selection_distribution


class AtomWorldPolicyAdapter:
    def __init__(self, args):
        self.args = args

    def capability_contract(self):
        return PolicyCapabilityContract(
            task_id="atomworld", api_version=1,
            enabled_capabilities=("prior_mean@1", "ldm_weights@1"),
            feature_names=FEATURE_NAMES, feature_groups={"public_geometry": (0, len(FEATURE_NAMES))},
            mean_clip=3.0, default_alpha=self.args.acquisition_alpha, default_eta=self.args.acquisition_eta,
        )

    def with_feedback(self, round_input, records):
        measured = {(r, c): float(y) for r, c, y in zip(
            round_input.history_rounds, round_input.history_candidate_ids, round_input.history_utilities)}
        matched = [{**p, "round_index": record["round_index"],
                    "measured_utility": measured[(record["round_index"], p["candidate_id"])]}
                   for record in records for p in record["predictions"]
                   if (record["round_index"], p["candidate_id"]) in measured]
        summary = {"count": len(matched), "scope": "previously frozen selected-point predictions only"}
        if matched:
            for name in ("baseline", "active"):
                error = np.asarray([p[name + "_mean"] - p["measured_utility"] for p in matched])
                summary[name + "_rmse"] = float(np.sqrt(np.mean(error**2)))
        context = dict(round_input.execution_context)
        context["weight_context"] = {**context["weight_context"],
            "prediction_feedback": {"summary": summary, "measurements": matched[-16:]}}
        return replace(round_input, execution_context=context)

    def validate_task_execution(self, execution, round_input):
        return tuple(HarnessSubmissionError("/outputs/" + name, "atomworld_prior_alignment",
                     "Return one finite standardized mean per authoritative geometry row.")
                     for name, n in (("history_prior_mean", len(round_input.history_features)),
                                     ("query_prior_mean", len(round_input.query_features)))
                     if getattr(execution, name).shape != (n,) or not np.isfinite(getattr(execution, name)).all())


def build_policy_input(args, sample, round_idx, history, candidates, representations, predictions, q0, gp):
    if not history:
        raise ValueError("Compiled LDM requires measured history")
    hx = np.asarray([h.feature.values for h in history])
    qx = np.asarray([representations[c.candidate_id].values for c in candidates])
    y = np.asarray([h.scalar_score for h in history])
    rounds = tuple(h.metadata["round_idx"] for h in history)
    projection = gp_projection(gp, len(history), qx)
    arrays = {"diagnostic_current_" + key: value for key, value in projection.items()}
    folds = []
    for i, held_round in enumerate(sorted(set(rounds))[1:][-3:]):
        train = [j for j, r in enumerate(rounds) if r < held_round]
        test = [j for j, r in enumerate(rounds) if r == held_round]
        fold_gp = make_gp([history[j] for j in train], args)
        prefix = f"diagnostic_fold_{i}_"
        arrays.update({prefix + key: value for key, value in gp_projection(fold_gp, len(train), hx[test]).items()})
        folds.append({"prefix": prefix, "round_index": held_round, "train_indices": train, "test_indices": test})
    location, scale = projection["location_scale"]
    probability, _, z = selection_distribution(q0, [p.acquisition_score for p in predictions],
        alpha=args.acquisition_alpha, eta=args.acquisition_eta, z_clip=args.z_clip)
    return PolicyRoundInput(
        round_index=round_idx, history_features=hx, history_utilities=y, query_features=qx,
        history_candidate_ids=tuple(h.candidate_id for h in history), history_rounds=rounds,
        measured_observations=tuple({"candidate_id": h.candidate_id, "round_index": r,
                                     "utility": h.scalar_score} for h, r in zip(history, rounds)),
        research_snapshot={
            "task": "atomworld", "feedback_protocol": "measured_correctness_optimization",
            "public_task": {k: v for k, v in sample.items() if k != "input_cif"},
            "public_task_access": "get_public_task", "draft_access": "get_public_history",
            "feature_contract": AtomWorldPolicyAdapter(args).capability_contract().to_dict(),
            "fixed_model": "Exact RBF residual GP; measured-label standardization with scale floor 0.1; feature scale floor 1.0; fitted separately for each question. Prior changes posterior mean, never covariance.",
            "lengthscale": args.gp_lengthscale, "noise": args.gp_noise, "jitter": 1e-8,
            "gp_history_limit": args.gp_history_limit,
            "sampling": "q0^alpha * exp(eta * robust_z(UCB)); Gumbel top-k without replacement",
            "boundary": "Only already measured scalar correctness is available. No targets, judge distances/errors, future labels or cross-question history. This is the feedback optimization extension, not official blind refinement.",
            "candidate_catalog": [{"candidate_id": c.candidate_id, "rationale": c.metadata.get("rationale", "")}
                                  for c in candidates],
        },
        execution_context={
            "mean_context": {"round_index": round_idx, "feature_version": FEATURE_VERSION,
                "feature_names": list(FEATURE_NAMES), "target_location": float(location), "target_scale": float(scale),
                "feature_semantics": "Each public geometry feature is transformed by sign(x)*log1p(abs(x)); same-order differences compare the public input and draft only, not hidden targets."},
            "weight_context": {"default_alpha": args.acquisition_alpha, "default_eta": args.acquisition_eta,
                "history_size": len(history), "requested_evaluation_batch": 1,
                "acquisition": {"name": "ucb", "score_direction": "maximize", "beta": args.ucb_beta},
                "normalization": {"name": "robust_z", "epsilon": 1e-12, "mad_scale": 1.4826, "z_clip": args.z_clip},
                "candidate_predictions": [{"candidate_id": c.candidate_id, "q0": float(q0[i]),
                    "baseline_mean": p.scalar_mean, "baseline_std": p.scalar_std,
                    "baseline_acquisition": p.acquisition_score, "normalized_acquisition": float(z[i]),
                    "default_selection_probability": float(probability[i])}
                    for i, (c, p) in enumerate(zip(candidates, predictions))]},
            "validation_folds": folds,
        }, diagnostic_arrays=arrays,
    )


def compiled_controller(expander, index, round_idx, candidates):
    client, root = expander._client(index, policy=True)
    pool = json.loads((expander.run_dir / "proposal_pools" / f"{round_idx:06d}.json").read_text())
    write_json(root / "public_history.json", {"drafts": [*pool["history"], *[
        {"draft_id": c.candidate_id, "round_idx": round_idx,
         "generated_output": c.payload["generated_output"], "rationale": c.metadata.get("rationale", "")}
        for c in candidates]]})
    if index not in expander.controllers:
        args = expander.args
        executor = expander.policy_executor or DockerPolicyExecutor(
            image=args.policy_runner_image, docker_host=args.harness_docker_host,
            container_user=resolve_container_user(args.harness_container_user, args.harness_docker_host),
        )
        expander.controllers[index] = PolicyResearchController(
            client=client, adapter=AtomWorldPolicyAdapter(args), executor=executor, root=root,
            account=expander.runtime.consume_many if expander.runtime else None,
            recovery_budget=lambda: float(args.harness_recovery_seconds),
        )
    return expander.controllers[index]
