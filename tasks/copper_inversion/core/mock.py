"""Protocol-faithful fake research; no provider or candidate execution."""

import json
import shutil
from dataclasses import asdict, replace
from types import SimpleNamespace
from pathlib import Path
import hashlib
from ldm_tts.harness import HarnessSubmissionRequest
from ldm_tts.harness.protocol import HarnessSubmittedArtifact, HarnessTurnResult, canonical_sha256
from ldm_tts.contracts import EvaluationResult
from ldm_tts.engine.run_store import atomic_json_write


class MockClient:
    def __init__(self, root, *, profiles=("researcher",), fork_sources=None):
        self.root = Path(root)
        self.config = SimpleNamespace(limits=SimpleNamespace(wall_time_seconds=1))
        self.profiles, self.fork_sources = tuple(profiles), fork_sources or {}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def _source(self, turn):
        offset = turn.round_index * .01 + self.profiles.index(turn.profile_id) * .001
        return ("def solve(problem, budget, seed):\n"
                f"    return problem['initial_level_set'].copy() + {offset!r}\n").encode()

    def run_turn(self, turns, submission_validator=None, submission_batch_validator=None, **kwargs):
        results, staged, pending = [], [], []
        decisions = {}
        for turn in turns:
            receipt = self.root / "mock-turns" / (turn.input_digest + ".json")
            if receipt.exists():
                record = json.loads(receipt.read_text())
                record["submitted_artifacts"] = tuple(HarnessSubmittedArtifact(**a) for a in record["submitted_artifacts"])
                record["validation_errors"] = ()
                results.append(replace(HarnessTurnResult(**record), replayed=True))
                continue
            history = json.loads((self.root / "public/history.json").read_text())
            session_root = self.root / "sessions" / turn.profile_id
            workspace = session_root / "workspace"
            cursor_path = session_root / "cursor.json"
            inherited = self.fork_sources.get(turn.profile_id)
            cursor = (json.loads(cursor_path.read_text()) if cursor_path.exists()
                      else inherited.history_cursor if inherited else 0)
            assert turn.history_from_seq == cursor
            assert turn.history_to_seq == len(history)
            assert turn.history_digest == canonical_sha256(history[cursor:])
            workspace.mkdir(parents=True, exist_ok=True)
            if inherited and not cursor_path.exists():
                shutil.copytree(self.root / "fork-sources" / turn.profile_id / "workspace", workspace, dirs_exist_ok=True)
            if turn.profile_id == "main_research":
                data, artifacts = {"brief": "Mock research brief; no scientific claim"}, ()
            else:
                assert json.loads(turn.message)["research_wall_time_seconds"] == self.config.limits.wall_time_seconds
                rows, artifacts = [], []
                for i in range(json.loads(turn.message)["candidates_required"]):
                    relative = f"mock/{turn.profile_id}/{turn.turn_id}/{i}/solver.py"
                    path = self.root / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    source = path.read_bytes() if path.exists() else self._source(turn) + f"# candidate {i}\n".encode()
                    path.write_bytes(source)
                    artifacts.append(HarnessSubmittedArtifact(
                        path_pointer=f"/candidates/{i}/artifact_path", relative_path=f"candidates/{i}/solver.py",
                        snapshot_path=relative, sha256=hashlib.sha256(source).hexdigest(), size_bytes=len(source)))
                    rows.append(dict(artifact_path=f"candidates/{i}/solver.py", parent_sha256=history[-1]["candidate_id"],
                                     change_summary="Mock geometry shift", rationale="Protocol fixture, not scientific research"))
                data, artifacts = {"candidates": rows}, tuple(artifacts)
            submission = HarnessSubmissionRequest(turn.profile_id, turn.turn_id, 1, data, artifacts)
            pending.append(submission)
            staged.append((turn, receipt, session_root, workspace, cursor_path, cursor, history, submission, data, artifacts))
            if submission_batch_validator is None:
                decisions[turn.profile_id] = submission_validator(submission)
            else:
                answered = submission_batch_validator(tuple(r for r in pending if r.profile_id not in decisions))
                decisions.update(answered)
        for turn, receipt, session_root, workspace, cursor_path, cursor, history, submission, data, artifacts in staged:
            validation = decisions.get(turn.profile_id)
            if validation is None or validation.decision != "accept":
                raise ValueError("mock submission barrier not accepted")
            session = session_root / "session.jsonl"
            with session.open("a") as handle:
                handle.write(json.dumps({"mock": True, "turn": turn.turn_id, "digest": turn.input_digest}) + "\n")
            (workspace / "research.md").write_text(f"Mock research through {turn.turn_id}\n")
            result = HarnessTurnResult(
                profile_id=turn.profile_id, session_id=turn.profile_id, turn_id=turn.turn_id,
                round_index=turn.round_index, history_from_seq=cursor, history_to_seq=len(history),
                history_digest=turn.history_digest, input_digest=turn.input_digest,
                replayed=False, submission_status="accepted", submission_id=turn.input_digest,
                submission_digest=submission.digest, submission=data, submitted_artifacts=artifacts,
                validation_errors=(), usage={"providerCalls": 1}, tool_budget={},
                artifacts={"session": str(session.relative_to(self.root))},
            )
            atomic_json_write(cursor_path, turn.history_to_seq)
            atomic_json_write(receipt, asdict(result))
            atomic_json_write(self.root / "manifest.json", {"mock": True, "profiles": list(self.profiles)})
            results.append(result)
        return tuple(results)


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
