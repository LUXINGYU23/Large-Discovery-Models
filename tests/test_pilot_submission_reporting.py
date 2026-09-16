from types import SimpleNamespace

import pytest

from ldm_tts.pilot_evaluation.submission_reporting import submission_trajectory


def test_normalized_submissions_support_arbitrary_objectives_and_missing_penalties():
    spec = SimpleNamespace(iterations=3, trajectory=SimpleNamespace(objective_column="energy"))
    records = [
        {"step": 0, "candidate_id": "a", "canonical_key": "key-a", "objective": 2.5, "status": "succeeded"},
        {"step": 1, "candidate_id": "", "canonical_key": "", "objective": -10, "status": "missing"},
        {"step": 2, "candidate_id": "a", "canonical_key": "key-a", "objective": 2.5, "status": "succeeded"},
    ]
    observations = [{"candidate": {"candidate_id": "a", "canonical_key": "key-a"},
                     "evaluation": {"status": "succeeded", "metrics": {"energy": 2.5}}}]
    assert submission_trajectory(spec, records, observations) == [2.5, -10, 2.5]
    records[-1]["canonical_key"] = "another-answer"
    with pytest.raises(ValueError, match="canonical key"):
        submission_trajectory(spec, records, observations)
    records[-1]["canonical_key"] = "key-a"
    records[-1]["objective"] = 100
    with pytest.raises(ValueError, match="objective differs"):
        submission_trajectory(spec, records, observations)


@pytest.mark.parametrize("steps", [[0, 0, 2], [0, 1], [0, 1, 2.5]])
def test_normalized_schedule_requires_exact_integral_steps(steps):
    spec = SimpleNamespace(iterations=3)
    with pytest.raises(ValueError, match="exactly one"):
        submission_trajectory(spec, [{"step": step} for step in steps], [])
