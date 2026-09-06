from __future__ import annotations

import numpy as np
import pytest

from ldm_tts.harness.policy_diagnostics import prediction_feedback, optimization_progress


def test_feedback_matches_the_measurement_round_not_later_labels() -> None:
    predictions = [
        {"round_index": 1, "predictions": [
            {"candidate_id": "a", "baseline_mean": 1.0, "active_mean": 2.0},
            {"candidate_id": "b", "baseline_mean": 50.0, "active_mean": -50.0},
        ]},
        {"round_index": 2, "predictions": [
            {"candidate_id": "b", "baseline_mean": 4.0, "active_mean": 3.0},
        ]},
    ]
    feedback = prediction_feedback(["a", "b"], [1, 2], [2.0, 4.0], predictions)
    assert feedback["summary"]["count"] == 2
    assert feedback["summary"]["baseline_rmse"] == pytest.approx(np.sqrt(0.5))
    assert feedback["summary"]["active_rmse"] == pytest.approx(np.sqrt(0.5))
    assert [row["round_index"] for row in feedback["measurements"]] == [1, 2]
    progress = optimization_progress([0, 1, 1, 2], [-4.0, -5.0, -3.0, -3.5])
    assert [row["improvement"] for row in progress["rounds"]] == [None, 1.0, 0.0]
