from __future__ import annotations

from importlib import import_module

import numpy as np
import pytest

from ldm_tts.optimization import BOPrediction


@pytest.fixture(params=["iron_mind", "synthonbench", "nucleobench"])
def diagnostics(request):
    return import_module(f"tasks.{request.param}.core.policy_diagnostics")


def test_feedback_matches_the_measurement_round_not_later_labels(diagnostics) -> None:
    predictions = [
        {"round_index": 1, "predictions": [
            {"candidate_id": "a", "baseline_mean": 1.0, "active_mean": 2.0},
            {"candidate_id": "b", "baseline_mean": 50.0, "active_mean": -50.0},
        ]},
        {"round_index": 2, "predictions": [
            {"candidate_id": "b", "baseline_mean": 4.0, "active_mean": 3.0},
        ]},
    ]
    feedback = diagnostics.prediction_feedback(["a", "b"], [1, 2], [2.0, 4.0], predictions)
    assert feedback["summary"]["count"] == 2
    assert feedback["summary"]["baseline_rmse"] == pytest.approx(np.sqrt(0.5))
    assert feedback["summary"]["active_rmse"] == pytest.approx(np.sqrt(0.5))
    assert [row["round_index"] for row in feedback["measurements"]] == [1, 2]
    progress = diagnostics.optimization_progress([0, 1, 1, 2], [-4.0, -5.0, -3.0, -3.5])
    assert [row["improvement"] for row in progress["rounds"]] == [None, 1.0, 0.0]



def test_task_predictions_preserve_ranks_and_actual_sampling_probabilities(diagnostics):
    baseline = tuple(
        BOPrediction.scalar(key, mean=value, std=1.0, acquisition_score=value)
        for key, value in zip(("a", "b", "c"), (1.0, 1.0, 3.0), strict=True)
    )
    active = tuple(
        BOPrediction.scalar(key, mean=value, std=1.0, acquisition_score=value)
        for key, value in zip(("a", "b", "c"), (0.0, 2.0, 2.0), strict=True)
    )
    task = diagnostics.__name__.split(".")[1]
    if task == "nucleobench":
        normalize = import_module(f"tasks.{task}.core.selection")._robust_z
    else:
        normalize = import_module(f"tasks.{task}.core.ldm_policy").robust_z
    q0 = np.asarray((0.5, 0.25, 0.25))
    rows = diagnostics.prediction_records(baseline, active, q0, alpha=0.8, eta=0.3, z_clip=2.0)
    expected = (q0 + 1e-12)**0.8 * np.exp(0.3 * normalize(np.asarray((0.0, 2.0, 2.0)), clip=2.0))
    assert [row["first_draw_probability"] for row in rows] == pytest.approx(expected / expected.sum())
    assert [row["q0_rank"] for row in rows] == [1, 2, 2]
    assert [row["baseline_acquisition_rank"] for row in rows] == [2, 2, 1]
    assert [row["active_acquisition_rank"] for row in rows] == [3, 1, 1]
    record = {"round_index": 1, "predictions": rows}
    feedback = diagnostics.prediction_feedback(["b", "c"], [1, 2], [5.0, 9.0], [record])
    assert len(feedback["measurements"]) == 1
    measured = feedback["measurements"][0]
    assert (measured["alpha"], measured["eta"], measured["pool_size"]) == (0.8, 0.3, 3)
    assert measured["q0_relative_to_max"] == 0.5
    with pytest.raises(ValueError, match="must be aligned"):
        diagnostics.prediction_records(baseline, active[::-1], q0, alpha=0.8, eta=0.3, z_clip=2.0)


def test_task_draft_distribution_matches_frozen_prediction_records(diagnostics):
    task = diagnostics.__name__.split(".")[1]
    hook = import_module(f"tasks.{task}.resources.harness.policy_diagnostics")
    q0 = np.asarray([0.75, 0.25])
    baseline_scores = np.asarray([10.0, 2.0])
    history_prior = np.asarray([0.4, -0.1])
    query_prior = np.asarray([-4.0, 4.0])
    operator = np.asarray([[0.3, 0.2], [0.2, 0.3]])
    active_scores = baseline_scores + 2.0 * (query_prior - operator @ history_prior)
    def predictions(scores):
        return tuple(BOPrediction.scalar(str(i), mean=score, std=1.0, acquisition_score=score)
                     for i, score in enumerate(scores))
    baseline, active = predictions(baseline_scores), predictions(active_scores)
    before = diagnostics.prediction_records(baseline, baseline, q0, alpha=2.0, eta=0.25, z_clip=2.0)
    after = diagnostics.prediction_records(baseline, active, q0, alpha=0.4, eta=2.0, z_clip=2.0)
    result = hook.evaluate_policy_draft(None, {
        "arrays": {
            "history_features": np.eye(2), "history_utilities": np.asarray([4.0, 8.0]),
            "diagnostic_current_weights": operator,
            "diagnostic_current_location_scale": np.asarray([6.0, 2.0]),
        },
        "execution_context": {
            "mean_context": {},
            "weight_context": {
                "acquisition": {"name": "ucb", "score_direction": "maximize"},
                "candidate_predictions": before,
                "normalization": {"name": "robust_z", "epsilon": 1e-12, "mad_scale": 1.4826, "z_clip": 2.0},
                "default_alpha": 2.0, "default_eta": 0.25,
            },
        },
        "contract": {"mean_clip": 5.0},
    }, {
        "history_prior_mean": history_prior, "query_prior_mean": query_prior,
        "weights": {"alpha": 0.4, "eta": 2.0},
    })
    pool = result["current_pool"]
    by_id = {row["candidate_id"]: row for row in pool["top_candidates"]}
    for first, second in zip(before, after, strict=True):
        row = by_id[first["candidate_id"]]
        assert row["default_probability"] == pytest.approx(first["first_draw_probability"])
        assert row["draft_probability"] == pytest.approx(second["first_draw_probability"])
    assert pool["probability_total_variation"] > 0.5
