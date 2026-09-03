from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ldm_tts.harness.policy import (
    PolicyCapabilityContract,
    PolicyResearchController,
    PolicyRoundInput,
    policy_mcp_server,
    policy_submission_contract,
)
from ldm_tts.harness.policy_execution import (
    PolicyExecutionError,
    PolicyExecutionResult,
)
from ldm_tts.harness.protocol import (
    HarnessSubmissionError,
    HarnessSubmissionRequest,
    HarnessSubmittedArtifact,
    HarnessTurnResult,
    file_sha256,
)


class FakeAdapter:
    def capability_contract(self) -> PolicyCapabilityContract:
        return PolicyCapabilityContract(
            task_id="fixture",
            api_version=1,
            enabled_capabilities=("prior_mean@1", "ldm_weights@1"),
            feature_names=("a", "b"),
            feature_groups={"all": (0, 2)},
            mean_clip=5.0,
            default_alpha=1.0,
            default_eta=2.0,
        )

    def validate_task_execution(self, execution, _round_input):
        if execution.stage == "invalid":
            return (HarnessSubmissionError(
                path="/outputs/prior_mean",
                code="task_validation_failed",
                message="fixture rejected this prior",
                hint="change the policy",
            ),)
        return ()


class FakeExecutor:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def execute(self, artifact_path, input_directory, output_directory):
        source = Path(artifact_path).read_text(encoding="utf-8")
        round_index = json.loads(
            (Path(input_directory) / "input.json").read_text(encoding="utf-8")
        )["round_index"]
        self.calls.append((source, round_index))
        if "CRASH" in source:
            raise PolicyExecutionError((HarnessSubmissionError(
                path="/artifact_path",
                code="runtime_exception",
                message="fixture runtime failure",
                hint="repair the artifact",
            ),))
        with np.load(Path(input_directory) / "arrays.npz", allow_pickle=False) as arrays:
            history_size = len(arrays["history_features"])
            query_size = len(arrays["query_features"])
        stage = "invalid" if "INVALID" in source else f"round_{round_index}"
        return PolicyExecutionResult(
            history_prior_mean=np.full(history_size, float(round_index)),
            query_prior_mean=np.full(query_size, float(round_index)),
            stage=stage,
            alpha=0.5 + round_index,
            eta=1.0 + round_index,
            prior_clip_count=0,
        )


class FakeHarnessClient:
    def __init__(self, root: Path, scripted_turns) -> None:
        self.root = root
        self.scripted_turns = list(scripted_turns)
        self.calls = 0
        self.validation_errors: list[str] = []

    def run_turn(self, turns, *, submission_validator):
        turn = turns[0]
        attempts = self.scripted_turns[self.calls]
        self.calls += 1
        final_request = None
        final_validation = None
        for attempt_index, (submission, source) in enumerate(attempts, start=1):
            artifacts = ()
            if submission.get("action") == "replace":
                snapshot = self.root / "snapshots" / f"{turn.round_index}-{attempt_index}.py"
                snapshot.parent.mkdir(parents=True, exist_ok=True)
                snapshot.write_text(source, encoding="utf-8")
                artifacts = (HarnessSubmittedArtifact(
                    path_pointer="/artifact_path",
                    relative_path="optimization_policy.py",
                    snapshot_path=snapshot.relative_to(self.root).as_posix(),
                    sha256=file_sha256(snapshot),
                    size_bytes=snapshot.stat().st_size,
                ),)
            request = HarnessSubmissionRequest(
                profile_id=turn.profile_id,
                turn_id=turn.turn_id,
                attempt_index=attempt_index,
                submission=submission,
                artifacts=artifacts,
            )
            validation = submission_validator(request)
            final_request = request
            final_validation = validation
            if validation.decision == "accept":
                break
            self.validation_errors.extend(error.code for error in validation.errors)
        assert final_request is not None and final_validation is not None
        accepted = final_validation.decision == "accept"
        errors = () if accepted else final_validation.errors
        return (HarnessTurnResult(
            profile_id=turn.profile_id,
            session_id="session-1",
            turn_id=turn.turn_id,
            round_index=turn.round_index,
            history_from_seq=turn.history_from_seq,
            history_to_seq=turn.history_to_seq,
            history_digest=turn.history_digest,
            input_digest=turn.input_digest,
            replayed=False,
            submission_status="accepted" if accepted else "rejected",
            submission_id=f"submission-{self.calls}",
            submission_digest=final_request.digest,
            submission=final_request.submission,
            submitted_artifacts=final_request.artifacts,
            validation_errors=errors,
            usage={"providerCalls": 2, "toolCalls": {"bash": 1}, "artifactBytes": 10},
            tool_budget={},
            artifacts={"turn": "turn.json", "session": "session.json"},
        ),)


def _round(round_index: int) -> PolicyRoundInput:
    history_size = round_index
    return PolicyRoundInput(
        round_index=round_index,
        history_features=np.arange(history_size * 2, dtype=float).reshape(history_size, 2),
        history_utilities=np.arange(history_size, dtype=float),
        query_features=np.asarray([[1.0, 2.0], [3.0, 4.0]]),
        research_snapshot={
            "new_measured_observations": [{"round": round_index}],
            "task_objective": "fixture objective",
        },
        execution_context={
            "mean_context": {"round_index": round_index},
            "weight_context": {"round_index": round_index},
        },
    )


def _replace(source: str = "VALID = True\n"):
    return ({"action": "replace", "artifact_path": "optimization_policy.py"}, source)


def _controller(tmp_path: Path, scripted_turns):
    client = FakeHarnessClient(tmp_path, scripted_turns)
    executor = FakeExecutor()
    controller = PolicyResearchController(
        client=client,
        adapter=FakeAdapter(),
        executor=executor,
        root=tmp_path,
    )
    return controller, client, executor


def test_policy_controller_replace_keep_disable_and_resume(tmp_path: Path) -> None:
    controller, client, executor = _controller(tmp_path, [
        [_replace()],
        [({"action": "keep"}, "")],
        [({"action": "disable"}, "")],
    ])

    replaced = controller.resolve(_round(1))
    kept = controller.resolve(_round(2))
    disabled = controller.resolve(_round(3))
    replayed = controller.resolve(_round(3))

    assert replaced.source == "artifact" and replaced.epoch_id == "epoch_001"
    assert replaced.metadata["action"] == "replace"
    assert replaced.metadata["harness_turn"]["session_id"] == "session-1"
    assert kept.source == "previous" and kept.epoch_id == replaced.epoch_id
    assert np.all(kept.query_prior_mean == 2.0)
    assert disabled.source == "default" and not disabled.degraded
    assert replayed.source == "default" and client.calls == 3
    assert replayed.metadata["action"] == "disable"
    assert [round_index for _, round_index in executor.calls] == [1, 2]
    assert json.loads((tmp_path / "active_policy.json").read_text())["epoch_id"] is None


def test_policy_controller_returns_no_active_policy_then_repairs(tmp_path: Path) -> None:
    controller, client, _executor = _controller(tmp_path, [[
        ({"action": "keep"}, ""),
        _replace(),
    ]])

    policy = controller.resolve(_round(1))

    assert policy.source == "artifact"
    assert client.validation_errors == ["no_active_policy"]


def test_policy_controller_falls_back_after_three_invalid_submissions(tmp_path: Path) -> None:
    controller, client, _executor = _controller(tmp_path, [[
        _replace("INVALID = True\n"),
        _replace("INVALID = True\n"),
        _replace("INVALID = True\n"),
    ]])

    policy = controller.resolve(_round(1))

    assert policy.source == "default" and policy.degraded
    assert client.validation_errors == ["task_validation_failed"] * 3


def test_policy_controller_uses_previous_epoch_after_runtime_failure(tmp_path: Path) -> None:
    controller, _client, _executor = _controller(tmp_path, [
        [_replace()],
        [
            _replace("CRASH = True\n"),
            _replace("CRASH = True\n"),
            _replace("CRASH = True\n"),
        ],
    ])
    first = controller.resolve(_round(1))

    fallback = controller.resolve(_round(2))

    assert fallback.source == "previous"
    assert fallback.epoch_id == first.epoch_id
    assert fallback.degraded


def test_policy_contract_and_builtin_mcp_are_generic() -> None:
    contract = policy_submission_contract()
    server = policy_mcp_server()

    assert contract.tool_name == "submit_optimization_policy"
    assert contract.max_validation_attempts == 3
    assert server.server_id == "ldm_policy"
    assert server.tools == (
        "inspect_policy_contract",
        "validate_policy_draft",
        "evaluate_policy_draft",
    )
