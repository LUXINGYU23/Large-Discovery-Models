"""Measured-history diagnostics for compiled means and sampling weights."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Protocol

import numpy as np

from ldm_tts.contracts import Candidate
from ldm_tts.optimization import BOObservation, BOPrediction, SurrogateVector

if TYPE_CHECKING:
    from ldm_tts.harness.policy import PolicyRoundInput


class DiagnosticGP(Protocol):
    def fit(self, history: Sequence[BOObservation]) -> None: ...

    def posterior_projection(
        self, history: Sequence[BOObservation], features: np.ndarray,
    ) -> Mapping[str, np.ndarray]: ...


def prepare_policy_research(
    round_input: PolicyRoundInput,
    *,
    history: Sequence[BOObservation],
    candidates: Sequence[Candidate],
    representations: Mapping[str, SurrogateVector],
    baseline: Sequence[BOPrediction],
    q0: np.ndarray,
    selector: DiagnosticGP,
    z_clip: float,
    normalize_acquisition: Callable[[np.ndarray], np.ndarray],
) -> PolicyRoundInput:
    acquisition = np.asarray([item.acquisition_score for item in baseline], dtype=float)
    normalized = normalize_acquisition(acquisition)
    weight_context = dict(round_input.execution_context["weight_context"])
    alpha, eta = weight_context["default_alpha"], weight_context["default_eta"]
    logits = alpha * np.log(q0 + 1e-12) + eta * normalized
    probability = np.exp(logits - logits.max())
    probability /= probability.sum()
    rows = [
        {
            "candidate_id": candidate.candidate_id,
            "q0": float(mass),
            "baseline_mean": prediction.mean[0],
            "baseline_std": prediction.std[0],
            "baseline_acquisition": prediction.acquisition_score,
            "normalized_acquisition": float(z),
            "default_selection_probability": float(p),
        }
        for candidate, prediction, mass, z, p in zip(
            candidates, baseline, q0, normalized, probability, strict=True
        )
    ]
    weight_context.update({
        "normalization": {"name": "robust_z", "mad_scale": 1.4826, "epsilon": 1e-12, "z_clip": z_clip},
        "candidate_predictions": rows,
        "default_logit_ranges": {
            "proposal": float(np.ptp(alpha * np.log(q0 + 1e-12))),
            "acquisition": float(np.ptp(eta * normalized)),
        },
    })
    rounds = np.asarray([item.metadata["round_idx"] for item in history], dtype=int)
    query_vectors = np.asarray([representations[item.candidate_id].values for item in candidates])
    arrays = {
        "diagnostic_current_" + name: values
        for name, values in selector.posterior_projection(history, query_vectors).items()
    }
    arrays["diagnostic_current_baseline_mean"] = np.asarray([item.mean[0] for item in baseline])
    arrays["diagnostic_current_baseline_std"] = np.asarray([item.std[0] for item in baseline])
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
            projection = selector.posterior_projection(
                training, np.asarray([history[i].feature_vector for i in test])
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
            "history_candidate_ids": [item.candidate_id for item in history],
            "history_rounds": rounds.tolist(),
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
