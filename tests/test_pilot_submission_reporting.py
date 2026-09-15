import json
from types import SimpleNamespace
import pytest
from ldm_tts.pilot_evaluation.submission_reporting import submission_trajectory
from ldm_tts.pilot_evaluation.config import _methods


def spec():
    return SimpleNamespace(iterations=3, trajectory=SimpleNamespace(
        step_kind="round", step_column="round", objective_column="correct"),
        result_fields={
            "one_shot_accuracy": "one_shot_accuracy",
            "extended_final_accuracy": "extended_final_accuracy",
        })


def observation(candidate_id, canonical_key, correct):
    return {
        "candidate": {"candidate_id": candidate_id, "canonical_key": canonical_key},
        "evaluation": {
            "status": "succeeded",
            "metrics": {"correct": correct},
        },
    }


def write_attempt(root, round_idx, canonical_key, *, sample_id="sample"):
    attempts = root / "attempts"
    attempts.mkdir(exist_ok=True)
    (attempts / f"{round_idx:06d}.json").write_text(
        json.dumps(
            {
                "sample_id": sample_id,
                "action_name": "move",
                "attempt": round_idx,
                "round_idx": round_idx,
                "canonical_key": canonical_key,
            }
        ),
        encoding="utf-8",
    )


def test_final_submission_preserves_wrong_last_answer_and_repeated_submissions():
    records = [{"round": 0, "candidate_id": "right", "correct": 1},
        {"round": 1, "candidate_id": "right", "correct": 1},
        {"round": 2, "candidate_id": "wrong", "correct": 0}]
    observations = [
        observation("right", "right-key", 1),
        observation("wrong", "wrong-key", 0),
    ]
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


def test_submission_rows_must_match_same_round_attempt_receipts(tmp_path):
    for round_idx, key in enumerate(("right-key", "right-key", "wrong-key")):
        write_attempt(tmp_path, round_idx, key)
    records = [
        {"round": 0, "attempt": 1, "sample_id": "sample", "action_name": "move",
         "candidate_id": "right", "correct": 1},
        {"round": 1, "attempt": 2, "sample_id": "sample", "action_name": "move",
         "candidate_id": "right", "correct": 1},
        {"round": 2, "attempt": 3, "sample_id": "sample", "action_name": "move",
         "candidate_id": "right", "correct": 1},
    ]
    observations = [
        observation("right", "right-key", 1),
        observation("wrong", "wrong-key", 0),
    ]

    with pytest.raises(ValueError, match="same-round attempt receipt"):
        submission_trajectory(spec(), records, observations, run_dir=tmp_path)


def test_submission_result_fields_are_cross_checked(tmp_path):
    for round_idx, key in enumerate(("right-key", "right-key", "wrong-key")):
        write_attempt(tmp_path, round_idx, key)
    records = [
        {"round": 0, "candidate_id": "right", "correct": 1},
        {"round": 1, "candidate_id": "right", "correct": 1},
        {"round": 2, "candidate_id": "wrong", "correct": 0},
    ]
    observations = [
        observation("right", "right-key", 1),
        observation("wrong", "wrong-key", 0),
    ]
    result = {
        "one_shot_accuracy": 1,
        "extended_final_accuracy": 0,
        "samples": [{"attempts": [{"correct": True}, {"correct": True}, {"correct": False}]}],
    }

    assert submission_trajectory(
        spec(), records, observations, run_dir=tmp_path, result=result
    ) == [1, 1, 0]
    result["extended_final_accuracy"] = 1
    with pytest.raises(ValueError, match="extended_final_accuracy"):
        submission_trajectory(spec(), records, observations, run_dir=tmp_path, result=result)


def test_submission_methods_do_not_require_inapplicable_optimization_baselines():
    assert _methods(["llm", "harness", "blind_harness_compiled"], require_baselines=False) == ("llm", "harness", "blind_harness_compiled")
    with pytest.raises(ValueError, match="include ldm"):
        _methods(["llm", "harness"])
