import json
from tasks.copper_inversion.core.workflow import main


def test_mock_and_resume(tmp_path):
    args = ["--mock", "--iterations", "3", "--out-dir", str(tmp_path / "run")]
    assert main(args) == 0
    root = tmp_path / "run"
    before = json.loads((root / "budget.json").read_text())
    assert main([*args, "--resume"]) == 0
    after = json.loads((root / "budget.json").read_text())
    assert before["counters"] == after["counters"]
    assert json.loads((root / "result.json").read_text())["observations"] == 4
    for name in ("events.jsonl", "checkpoint.json", "summary.json", "trajectory.csv"):
        assert (root / name).exists()


def test_mock_submission_correction(tmp_path, monkeypatch):
    """A rejected provisional artifact is repaired before the turn commits."""
    import hashlib
    from tasks.copper_inversion.core.mock import MockClient
    from ldm_tts.harness import HarnessSubmissionRequest
    from ldm_tts.harness.protocol import HarnessSubmittedArtifact

    original = MockClient.run_turn
    rejected = []

    def repair(self, turns, submission_validator, **kwargs):
        (turn,) = turns
        source = b"# no solve function\n"
        path = self.root / "invalid.py"
        path.write_bytes(source)
        artifact = HarnessSubmittedArtifact(
            path_pointer="/candidates/0/artifact_path",
            relative_path="candidates/0/solver.py",
            snapshot_path="invalid.py",
            sha256=hashlib.sha256(source).hexdigest(),
            size_bytes=len(source),
        )
        submission = dict(
            artifact_path="candidates/0/solver.py",
            parent_sha256=None,
            change_summary="invalid attempt",
            rationale="test",
        )
        submission = {"candidates": [submission]}
        result = submission_validator(
            HarnessSubmissionRequest(
                "researcher", turn.turn_id, 1, submission, (artifact,)
            )
        )
        assert result.decision == "retry"
        assert result.errors[0].path == "/candidates/0/artifact_path"
        rejected.append(result)
        return original(self, turns, submission_validator, **kwargs)

    monkeypatch.setattr(MockClient, "run_turn", repair)
    root = tmp_path / "repair"
    assert main(["--mock", "--iterations", "1", "--out-dir", str(root)]) == 0
    assert len(rejected) == 1
    counters = json.loads((root / "budget.json").read_text())["counters"]
    assert counters["harness_turns"] == 1
    assert counters["external_evaluations"] == 2
