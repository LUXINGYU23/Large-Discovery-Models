from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ldm_tts.optimization import BOPrediction
from ldm_tts.harness.policy_diagnostics import prediction_feedback

from ldm_tts.harness.client import HarnessError
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
        self.turns = []
        self.validation_errors: list[str] = []

    def run_turn(self, turns, *, submission_validator):
        turn = turns[0]
        self.turns.append(turn)
        attempts = self.scripted_turns[self.calls]
        self.calls += 1
        if isinstance(attempts, Exception):
            raise attempts
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
            "measured_observations": [{"round": index} for index in range(history_size)],
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


def test_prediction_record_freezes_same_round_ranks_probabilities_and_weights(tmp_path: Path) -> None:
    controller, _, _ = _controller(tmp_path, [])
    baseline = tuple(
        BOPrediction.scalar(key, mean=value, std=1.0, acquisition_score=value)
        for key, value in zip(("a", "b", "c"), (1.0, 1.0, 3.0), strict=True)
    )
    active = tuple(
        BOPrediction.scalar(key, mean=value, std=1.0, acquisition_score=value)
        for key, value in zip(("a", "b", "c"), (0.0, 2.0, 2.0), strict=True)
    )
    q0 = np.asarray((0.5, 0.25, 0.25))
    options = {"alpha": 0.8, "eta": 0.3, "normalize_acquisition": lambda values: values}
    controller.record_predictions(1, baseline, active, q0, **options)
    path = tmp_path / "rounds/round_001/predictions.json"
    original = path.read_bytes()
    record = json.loads(original)
    rows = record["predictions"]
    expected = (q0 + 1e-12)**0.8 * np.exp(0.3 * np.asarray((0.0, 2.0, 2.0)))
    assert [row["first_draw_probability"] for row in rows] == pytest.approx(expected / expected.sum())
    assert [row["q0_rank"] for row in rows] == [1, 2, 2]
    assert [row["baseline_acquisition_rank"] for row in rows] == [2, 2, 1]
    assert [row["active_acquisition_rank"] for row in rows] == [3, 1, 1]
    assert [row["q0_relative_to_max"] for row in rows] == [1.0, 0.5, 0.5]
    feedback = prediction_feedback(["b", "c"], [1, 2], [5.0, 9.0], [record])
    assert len(feedback["measurements"]) == 1
    measured = feedback["measurements"][0]
    assert measured["pool_size"] == 3
    assert (measured["alpha"], measured["eta"]) == (0.8, 0.3)
    assert measured["q0_relative_to_max"] == 0.5
    assert measured["measured_utility"] == 5.0
    controller.record_predictions(1, baseline, active, q0, **options)
    with pytest.raises(ValueError, match="cannot be replaced"):
        controller.record_predictions(1, baseline, active, q0, **{**options, "alpha": 1.0})
    with pytest.raises(ValueError, match="must be aligned"):
        controller.record_predictions(1, baseline, active[::-1], q0, **options)
    assert path.read_bytes() == original


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


def test_policy_failure_preserves_committed_cursor_and_delivers_missed_history(tmp_path: Path) -> None:
    controller, client, _executor = _controller(tmp_path, [
        [_replace()], HarnessError("provider 502"), [({"action": "keep"}, "")],
    ])
    first = controller.resolve(_round(1))
    failed = controller.resolve(_round(2))
    recovered = controller.resolve(_round(3))

    assert failed.degraded and not recovered.degraded
    assert recovered.epoch_id == first.epoch_id
    turn = client.turns[-1]
    assert (turn.history_from_seq, turn.history_to_seq) == (1, 3)
    message = json.loads(turn.message)
    assert message["new_measured_observations"] == [{"round": 1}, {"round": 2}]


def test_policy_contract_and_builtin_mcp_are_generic() -> None:
    contract = policy_submission_contract()
    server = policy_mcp_server()

    assert contract.tool_name == "submit_optimization_policy"
    assert contract.max_validation_attempts == 3
    assert contract.payload_schema["allOf"] == [{
        "if": {
            "properties": {"action": {"const": "replace"}},
            "required": ["action"],
        },
        "then": {"required": ["artifact_path"]},
        "else": {"not": {"required": ["artifact_path"]}},
    }]
    assert server.server_id == "ldm_policy"
    assert server.tools == (
        "inspect_policy_contract",
        "validate_policy_draft",
        "evaluate_policy_draft",
    )
