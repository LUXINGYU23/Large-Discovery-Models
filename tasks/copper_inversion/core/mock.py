"""Protocol-faithful fake research with no provider, Docker or source execution."""

import json
from types import SimpleNamespace
from pathlib import Path
import hashlib
from ldm_tts.harness import HarnessSubmissionRequest
from ldm_tts.harness.protocol import HarnessSubmittedArtifact
from ldm_tts.contracts import EvaluationResult


class MockClient:
    def __init__(self, root):
        self.root = Path(root)
        self.config = SimpleNamespace(limits=SimpleNamespace(wall_time_seconds=1))

    def run_turn(self, turns, submission_validator, **kwargs):
        (turn,) = turns
        assert json.loads(turn.message)["research_wall_time_seconds"] == self.config.limits.wall_time_seconds
        history = __import__("json").loads(
            (self.root / "public/history.json").read_text()
        )
        from ldm_tts.harness.protocol import canonical_sha256

        cursor_path = self.root / "cursor.json"
        cursor = json.loads(cursor_path.read_text()) if cursor_path.exists() else 0
        assert turn.history_from_seq == cursor
        assert turn.history_to_seq == len(history)
        assert turn.history_digest == canonical_sha256(history[cursor:])
        source = f"def solve(problem, budget, seed):\n    return problem['initial_level_set'].copy() + {turn.round_index * 0.01!r}\n".encode()
        relative = f"mock/{turn.turn_id}/solver.py"
        p = self.root / relative
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(source)
        sha = hashlib.sha256(source).hexdigest()
        a = HarnessSubmittedArtifact(
            path_pointer="/artifact_path",
            relative_path="candidates/0/solver.py",
            snapshot_path=relative,
            sha256=sha,
            size_bytes=len(source),
        )
        data = dict(
            artifact_path="candidates/0/solver.py",
            parent_sha256=history[-1]["candidate_id"],
            change_summary="Mock geometry shift",
            rationale="Deterministic protocol fixture, not scientific research",
        )
        validation = submission_validator(
            HarnessSubmissionRequest("researcher", turn.turn_id, 1, data, (a,))
        )
        if validation.decision != "accept":
            raise ValueError(validation.errors)
        cursor_path.write_text(json.dumps(turn.history_to_seq))
        return (
            SimpleNamespace(
                submission_status="accepted",
                submission=data,
                submitted_artifacts=(a,),
                submission_id=turn.turn_id,
                usage={},
            ),
        )


class MockEvaluator:
    def evaluation_attempt_usage_key(self, candidate):
        return "mock:" + candidate.canonical_key

    def evaluate(self, candidate):
        from ldm_tts.contracts.evaluation import EVALUATION_ATTEMPT_RECEIPT_KEY

        value = 10.0 - len(candidate.payload.get("source_path", "")) * 0.001
        return EvaluationResult(
            candidate.candidate_id,
            "succeeded",
            metrics={"loss_total": value, "research_utility": -value},
            metadata={
                EVALUATION_ATTEMPT_RECEIPT_KEY: self.evaluation_attempt_usage_key(
                    candidate
                ),
                "mock": True,
            },
        )
