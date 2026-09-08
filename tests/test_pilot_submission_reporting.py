from types import SimpleNamespace
import pytest
from ldm_tts.pilot_evaluation.submission_reporting import submission_trajectory
from ldm_tts.pilot_evaluation.config import _methods


def spec():
    return SimpleNamespace(iterations=3, trajectory=SimpleNamespace(
        step_kind="round", step_column="round", objective_column="correct"))


def test_final_submission_preserves_wrong_last_answer_and_repeated_submissions():
    records = [{"round": 0, "candidate_id": "right", "correct": 1},
        {"round": 1, "candidate_id": "right", "correct": 1},
        {"round": 2, "candidate_id": "wrong", "correct": 0}]
    observations = [{"candidate": {"candidate_id": "right"}, "evaluation": {"metrics": {"correct": 1}}},
        {"candidate": {"candidate_id": "wrong"}, "evaluation": {"metrics": {"correct": 0}}}]
    values = submission_trajectory(spec(), records, observations)
    assert values[-1] == 0
    assert values == [1, 1, 0]
    records[-1]["correct"] = 1
    with pytest.raises(ValueError, match="differs"):
        submission_trajectory(spec(), records, observations)


def test_submission_report_rejects_missing_round_and_unmeasured_candidate():
    with pytest.raises(ValueError, match="exactly one"):
        submission_trajectory(spec(), [{"round": 0, "correct": 1}], [])
    records = [{"round": i, "correct": 0, "candidate_id": "unknown"} for i in range(3)]
    with pytest.raises(ValueError, match="unmeasured"):
        submission_trajectory(spec(), records, [])


def test_submission_methods_do_not_require_inapplicable_optimization_baselines():
    assert _methods(["llm", "harness", "blind_harness_compiled"], require_baselines=False) == ("llm", "harness", "blind_harness_compiled")
    with pytest.raises(ValueError, match="include ldm"):
        _methods(["llm", "harness"])
