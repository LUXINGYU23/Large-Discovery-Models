"""Trusted task-owned draft diagnostics executed by the policy research tool."""

import numpy as np


def robust_z_acquisition(values: np.ndarray, z_clip: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("policy acquisition must be a finite non-empty vector")
    if isinstance(z_clip, bool) or not np.isfinite(z_clip) or z_clip <= 0:
        raise ValueError("policy z_clip must be finite and positive")
    median = float(np.median(values))
    scale = 1.4826 * float(np.median(np.abs(values - median)))
    if scale <= 1e-12:
        scale = float(np.std(values))
    if scale <= 1e-12:
        return np.zeros_like(values)
    return np.clip((values - median) / (scale + 1e-12), -z_clip, z_clip)


def evaluate_policy_draft(prior, inputs, outputs):
    arrays = inputs["arrays"]
    execution_context = inputs["execution_context"]
    history_features = arrays["history_features"]
    history_utilities = arrays["history_utilities"]
    history_prior = outputs["history_prior_mean"]
    query_prior = outputs["query_prior_mean"]
    weights = outputs["weights"]
    mean_clip = inputs["contract"]["mean_clip"]
    if history_utilities.ndim != 1:
        raise ValueError("this task's diagnostics require scalar utilities")
    mean_context = execution_context["mean_context"]
    folds = []
    baseline_errors, draft_errors = [], []
    for fold in execution_context.get("validation_folds", []):
        prefix = fold["prefix"]
        train = np.asarray(fold["train_indices"], dtype=int)
        test = np.asarray(fold["test_indices"], dtype=int)
        if not len(train) or not len(test) or train.max() >= test.min():
            raise ValueError("policy holdouts must follow the training prefix")
        location, scale = arrays[prefix + "location_scale"]
        context = {**mean_context, "round_index": fold["round_index"],
                   "target_location": float(location), "target_scale": float(scale)}
        train_prior = np.zeros(len(train))
        test_prior = np.zeros(len(test))
        if prior is not None:
            train_prior = np.clip(prior(history_features[train], history_utilities[train],
                                         history_features[train], context), -mean_clip, mean_clip)
            test_prior = np.clip(prior(history_features[train], history_utilities[train],
                                        history_features[test], context), -mean_clip, mean_clip)
        operator = arrays[prefix + "weights"]
        standardized = (history_utilities[train] - location) / scale
        baseline_mean = location + scale * (operator @ standardized)
        draft_mean = location + scale * (test_prior + operator @ (standardized - train_prior))
        before = baseline_mean - history_utilities[test]
        after = draft_mean - history_utilities[test]
        baseline_errors.extend(before.tolist())
        draft_errors.extend(after.tolist())
        folds.append({
            "round_index": fold["round_index"], "train_count": len(train), "test_count": len(test),
            "baseline_gp_rmse": float(np.sqrt(np.mean(before**2))),
            "draft_gp_rmse": float(np.sqrt(np.mean(after**2))),
            "baseline_prediction": baseline_mean.tolist(), "draft_prediction": draft_mean.tolist(),
            "measured_utility": history_utilities[test].tolist(),
        })
    result = {
        "scope": "chronological_measured_history_fixed_gp",
        "status": "available" if folds else "insufficient_history",
        "hyperparameters": "fitted on each training prefix, frozen during draft comparison",
        "folds": folds,
        "held_out_count": len(baseline_errors),
        "baseline_gp_rmse": float(np.sqrt(np.mean(np.square(baseline_errors)))) if folds else None,
        "draft_gp_rmse": float(np.sqrt(np.mean(np.square(draft_errors)))) if folds else None,
    }
    if "diagnostic_current_weights" in arrays:
        context = execution_context["weight_context"]
        if context["acquisition"]["name"] != "ucb" or context["acquisition"]["score_direction"] != "maximize":
            raise ValueError("this task requires maximization UCB diagnostics")
        rows = context["candidate_predictions"]
        mass = np.asarray([row["q0"] for row in rows])
        sizes = np.asarray([row.get("group_trial_count", 1) for row in rows])
        baseline_acquisition = np.asarray([row["baseline_acquisition"] for row in rows])
        scale = arrays["diagnostic_current_location_scale"][1]
        delta = scale * (query_prior - arrays["diagnostic_current_weights"] @ history_prior)
        acquisition = baseline_acquisition + delta
        normalization = context["normalization"]
        before = selection_probability(mass, baseline_acquisition, context["default_alpha"], context["default_eta"], normalization, sizes)
        after = selection_probability(mass, acquisition, weights["alpha"], weights["eta"], normalization, sizes)
        result["current_pool"] = {
            "scope": "first_draw_probabilities_under_fixed_baseline_gp; no query labels",
            "probability_total_variation": float(np.abs(after - before).sum() / 2),
            "default_effective_sample_size": float(1 / np.sum(before**2)),
            "draft_effective_sample_size": float(1 / np.sum(after**2)),
            "top_candidates": [
                {"candidate_id": rows[i]["candidate_id"], "q0": float(mass[i]),
                 "default_probability": float(before[i]), "draft_probability": float(after[i]),
                 "draft_acquisition": float(acquisition[i])}
                for i in np.argsort(after)[::-1][:5]
            ],
        }
    return result


def selection_probability(mass, acquisition, alpha, eta, normalization, group_sizes=None):
    if normalization["name"] != "robust_z":
        raise ValueError("this task requires robust_z normalization")
    if normalization["epsilon"] != 1e-12 or normalization["mad_scale"] != 1.4826:
        raise ValueError("unsupported robust_z constants")
    z = robust_z_acquisition(acquisition, normalization["z_clip"])
    epsilon = normalization["epsilon"]
    sizes = np.ones_like(mass) if group_sizes is None else np.asarray(group_sizes)
    logits = alpha * np.log(mass * sizes + epsilon) - np.log(sizes) + eta * z
    probability = np.exp(logits - logits.max())
    return probability / probability.sum()
