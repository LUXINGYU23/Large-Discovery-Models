"""UCB policy diagnostics shared by task selectors and the isolated Pi runner."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

# This file is mounted alone in Pi; runtime imports must stay NumPy/stdlib-only.
if TYPE_CHECKING:
    from ldm_tts.contracts import Candidate
    from ldm_tts.harness.policy import PolicyRoundInput
    from ldm_tts.optimization import BOObservation, BOPrediction, SurrogateVector

DIAGNOSTICS_PATH = Path(__file__).resolve()


def prepare_ucb_policy_research(
    round_input: PolicyRoundInput,
    *,
    history: Sequence[BOObservation],
    candidates: Sequence[Candidate],
    representations: Mapping[str, SurrogateVector],
    baseline: Sequence[BOPrediction],
    q0: np.ndarray,
    selector,
    z_clip: float,
) -> PolicyRoundInput:
    spec = selector.describe()
    if spec.name != "ucb" or spec.score_direction != "maximize" or len(spec.objective_names) != 1:
        raise ValueError("policy GP diagnostics require single-objective maximization UCB")
    if (
        tuple(item.candidate_id for item in history) != round_input.history_candidate_ids
        or tuple(item.metadata.get("round_idx") for item in history) != round_input.history_rounds
        or not np.array_equal([item.scalar_score for item in history], round_input.history_utilities)
    ):
        raise ValueError("policy GP history must align with the authoritative round input")
    if (
        len(candidates) != len(round_input.query_features)
        or len(candidates) != len(baseline)
        or len({item.candidate_id for item in candidates}) != len(candidates)
        or any(candidate.candidate_id != prediction.candidate_id for candidate, prediction in zip(candidates, baseline))
    ):
        raise ValueError("policy candidate rows and baseline predictions must be aligned")
    if any(item.score_direction != "maximize" for item in baseline):
        raise ValueError("policy baseline predictions must maximize utility")
    q0 = np.asarray(q0, dtype=float)
    if q0.shape != (len(candidates),) or not np.isfinite(q0).all() or np.any(q0 < 0) or not np.isclose(q0.sum(), 1.0):
        raise ValueError("policy q0 must be a normalized vector aligned with candidate rows")
    means = np.asarray([item.scalar_mean for item in baseline])
    stds = np.asarray([item.scalar_std for item in baseline])
    if not np.isfinite(means).all() or not np.isfinite(stds).all() or np.any(stds < 0):
        raise ValueError("policy baseline means and standard deviations must be finite and valid")
    acquisition = np.asarray([item.acquisition_score for item in baseline], dtype=float)
    normalized = robust_z_acquisition(acquisition, z_clip)
    weight_context = dict(round_input.execution_context["weight_context"])
    alpha, eta = weight_context["default_alpha"], weight_context["default_eta"]
    probability = selection_probability(q0, acquisition, alpha, eta, {
        "name": "robust_z", "mad_scale": 1.4826, "epsilon": 1e-12, "z_clip": z_clip,
    })
    rows = [
        {
            "candidate_id": candidate.candidate_id,
            "q0": float(mass),
            "baseline_mean": prediction.scalar_mean,
            "baseline_std": prediction.scalar_std,
            "baseline_acquisition": prediction.acquisition_score,
            "normalized_acquisition": float(z),
            "default_selection_probability": float(p),
        }
        for candidate, prediction, mass, z, p in zip(
            candidates, baseline, q0, normalized, probability, strict=True
        )
    ]
    weight_context.update({
        "acquisition": {"name": "ucb", "score_direction": "maximize", "objective_names": list(spec.objective_names)},
        "normalization": {"name": "robust_z", "mad_scale": 1.4826, "epsilon": 1e-12, "z_clip": z_clip},
        "candidate_predictions": rows,
        "default_logit_ranges": {
            "proposal": float(np.ptp(alpha * np.log(q0 + 1e-12))),
            "acquisition": float(np.ptp(eta * normalized)),
        },
    })
    rounds = np.asarray(round_input.history_rounds, dtype=int)
    query_vectors = np.asarray([representations[item.candidate_id].values for item in candidates])
    arrays = {
        "diagnostic_current_" + name: values
        for name, values in _projection_arrays(selector, history, query_vectors).items()
    }
    location, scale = arrays["diagnostic_current_location_scale"]
    mean_context = {
        **round_input.execution_context["mean_context"],
        "target_location": float(location), "target_scale": float(scale),
    }
    folds = []
    held_rounds = sorted(set(rounds))[1:][-3:]
    try:
        # Hold out whole evaluation rounds; each GP sees only its training prefix.
        for index, held_round in enumerate(held_rounds):
            train = np.flatnonzero(rounds < held_round)
            test = np.flatnonzero(rounds == held_round)
            training = [history[i] for i in train]
            selector.fit(training)
            projection = _projection_arrays(
                selector, training, np.asarray([history[i].feature_vector for i in test])
            )
            prefix = f"diagnostic_fold_{index}_"
            arrays.update({prefix + name: values for name, values in projection.items()})
            folds.append({
                "prefix": prefix,
                "round_index": int(held_round),
                "train_indices": train.tolist(),
                "test_indices": test.tolist(),
            })
    finally:
        if held_rounds:
            selector.fit(history)
    return replace(
        round_input,
        diagnostic_arrays=arrays,
        research_snapshot={
            **round_input.research_snapshot,
            "candidate_catalog": [
                {"candidate_id": item.candidate_id, "candidate": item.payload}
                for item in candidates
            ],
            "validation_scope": "chronological holdouts with training-prefix GP hyperparameters frozen",
        },
        execution_context={
            **round_input.execution_context,
            "mean_context": mean_context,
            "weight_context": weight_context,
            "validation_folds": folds,
        },
    )


def _projection_arrays(selector, history, features) -> dict[str, np.ndarray]:
    projection = selector.posterior_projection(history, features)
    shapes = {"weights": (len(features), len(history)), "std": (len(features),), "location_scale": (2,)}
    if set(projection) != set(shapes):
        raise ValueError("policy GP projection requires weights, std and location_scale")
    arrays = {name: np.asarray(value, dtype=float) for name, value in projection.items()}
    if any(value.shape != shapes[name] or not np.isfinite(value).all() for name, value in arrays.items()):
        raise ValueError("policy GP projection arrays must be finite and aligned")
    if arrays["location_scale"][1] <= 0 or np.any(arrays["std"] < 0):
        raise ValueError("policy GP projection scale and standard deviations are invalid")
    return arrays


def prediction_feedback(history_ids, history_rounds, utilities, records):
    measured = {(round_index, candidate_id): float(value) for round_index, candidate_id, value
                in zip(history_rounds, history_ids, utilities, strict=True)}
    matched = []
    for record in records:
        for prediction in record["predictions"]:
            key = (record["round_index"], prediction["candidate_id"])
            if key in measured:
                matched.append({"round_index": key[0], **prediction, "measured_utility": measured[key]})
    summary = {
        "count": len(matched),
        "scope": "predictions and pool-relative signals frozen before the matching measurement",
        "interpretation": (
            "Errors describe selected-point prediction, not whole-pool ranking or weight-policy reward. "
            "A positive residual is underprediction, not ranking validation. Use each row's original "
            "pool size, ranks and relative q0; do not compare historical q0 with the current pool maximum. "
            "First-draw probability is not multi-candidate batch inclusion probability."
        ),
    }
    if matched:
        truth = np.asarray([row["measured_utility"] for row in matched])
        for name in ("baseline", "active"):
            error = np.asarray([row[name + "_mean"] for row in matched]) - truth
            summary[name + "_rmse"] = float(np.sqrt(np.mean(error**2)))
            summary[name + "_mae"] = float(np.mean(np.abs(error)))
        summary["active_minus_baseline_rmse"] = summary["active_rmse"] - summary["baseline_rmse"]
    return {"summary": summary, "measurements": matched}


def optimization_progress(rounds, utilities):
    best = None
    records = []
    for round_index in sorted(set(rounds)):
        values = [float(y) for r, y in zip(rounds, utilities, strict=True) if r == round_index]
        current = max(values)
        improvement = None if best is None else max(0.0, current - best)
        best = current if best is None else max(best, current)
        records.append({"round_index": round_index, "count": len(values), "batch_best": current,
                        "batch_mean": float(np.mean(values)), "best_so_far": best, "improvement": improvement})
    return {"rounds": records, "measured_count": len(utilities)}


def prediction_records(baseline, active, q0, *, alpha, eta, z_clip):
    signals = {
        "q0": np.asarray(q0, dtype=float),
        "baseline_acquisition": np.asarray([item.acquisition_score for item in baseline], dtype=float),
        "active_acquisition": np.asarray([item.acquisition_score for item in active], dtype=float),
    }
    ranks = {
        name: 1 + np.searchsorted(np.sort(-values), -values, side="left")
        for name, values in signals.items()
    }
    normalized = robust_z_acquisition(signals["active_acquisition"], z_clip)
    probabilities = selection_probability(q0, signals["active_acquisition"], alpha, eta, {
        "name": "robust_z", "mad_scale": 1.4826, "epsilon": 1e-12, "z_clip": z_clip,
    })
    if (
        signals["q0"].shape != (len(baseline),) or not len(baseline)
        or normalized.shape != signals["q0"].shape or probabilities.shape != signals["q0"].shape
        or not all(np.isfinite(values).all() for values in signals.values())
        or np.any(signals["q0"] < 0) or not np.isclose(signals["q0"].sum(), 1.0)
        or np.any(probabilities < 0) or not np.isclose(probabilities.sum(), 1.0)
        or len({item.candidate_id for item in baseline}) != len(baseline)
    ):
        raise ValueError("policy prediction signals must be aligned finite probabilities and scalar scores")
    max_q0 = signals["q0"].max()
    rows = []
    for index, (first, second, mass) in enumerate(zip(baseline, active, q0, strict=True)):
        if first.candidate_id != second.candidate_id:
            raise ValueError("policy feedback predictions must be aligned")
        rows.append({
            "candidate_id": first.candidate_id,
            "q0": float(mass),
            "q0_relative_to_max": float(mass / max_q0),
            "pool_size": len(q0),
            **{name + "_rank": int(values[index]) for name, values in ranks.items()},
            "baseline_acquisition": first.acquisition_score,
            "active_acquisition": second.acquisition_score,
            "normalized_acquisition": float(normalized[index]),
            "first_draw_probability": float(probabilities[index]),
            "alpha": float(alpha), "eta": float(eta),
            "baseline_mean": first.scalar_mean, "baseline_std": first.scalar_std,
            "active_mean": second.scalar_mean, "active_std": second.scalar_std,
        })
    return rows


def with_feedback(round_input: PolicyRoundInput, records) -> PolicyRoundInput:
    ids = round_input.history_candidate_ids
    rounds = round_input.history_rounds
    context = dict(round_input.execution_context)
    context["weight_context"] = {
        **context["weight_context"],
        "prediction_feedback": prediction_feedback(ids, rounds, round_input.history_utilities, records),
        "optimization_progress": optimization_progress(rounds, round_input.history_utilities),
    }
    return replace(round_input, execution_context=context)


def robust_z_values(values: np.ndarray, z_clip: float, epsilon: float = 1e-12) -> np.ndarray:
    median = float(np.median(values))
    scale = 1.4826 * float(np.median(np.abs(values - median)))
    if scale <= epsilon:
        scale = float(np.std(values))
    if scale <= epsilon:
        return np.zeros_like(values)
    return np.clip((values - median) / (scale + epsilon), -z_clip, z_clip)


def tilt_log_weights(mass, normalized_acquisition, alpha, eta, epsilon=1e-12):
    return alpha * np.log(mass + epsilon) + eta * normalized_acquisition


def softmax_logits(logits):
    exponentials = np.exp(logits - float(np.max(logits)))
    return exponentials / float(exponentials.sum())


def robust_z_acquisition(values: np.ndarray, z_clip: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("policy acquisition must be a finite non-empty vector")
    if isinstance(z_clip, bool) or not np.isfinite(z_clip) or z_clip <= 0:
        raise ValueError("policy z_clip must be finite and positive")
    return robust_z_values(values, z_clip)


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
        baseline_acquisition = np.asarray([row["baseline_acquisition"] for row in rows])
        scale = arrays["diagnostic_current_location_scale"][1]
        delta = scale * (query_prior - arrays["diagnostic_current_weights"] @ history_prior)
        acquisition = baseline_acquisition + delta
        normalization = context["normalization"]
        before = selection_probability(mass, baseline_acquisition, context["default_alpha"], context["default_eta"], normalization)
        after = selection_probability(mass, acquisition, weights["alpha"], weights["eta"], normalization)
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


def selection_probability(mass, acquisition, alpha, eta, normalization):
    if normalization["name"] != "robust_z":
        raise ValueError("this task requires robust_z normalization")
    if normalization["epsilon"] != 1e-12 or normalization["mad_scale"] != 1.4826:
        raise ValueError("unsupported robust_z constants")
    z = robust_z_acquisition(acquisition, normalization["z_clip"])
    epsilon = normalization["epsilon"]
    return softmax_logits(tilt_log_weights(mass, z, alpha, eta, epsilon))
