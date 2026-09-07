from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ldm_tts.harness.pi import policy_mcp_server
from ldm_tts.harness.client import HarnessError
from ldm_tts.harness.policy import (
    PolicyCapabilityContract,
    PolicyResearchController,
    PolicyRoundInput,
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

    def with_feedback(self, round_input, records):
        self.feedback_records = records
        return round_input

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
            objective_shape = arrays["history_utilities"].shape[1:]
        stage = "invalid" if "INVALID" in source else f"round_{round_index}"
        return PolicyExecutionResult(
            history_prior_mean=np.full((history_size, *objective_shape), float(round_index)),
            query_prior_mean=np.full((query_size, *objective_shape), float(round_index)),
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

    def run_turn(self, turns, *, submission_validator, recovery_timeout_seconds=0):
        self.recovery_timeout_seconds = recovery_timeout_seconds
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
        history_candidate_ids=tuple(f"candidate-{index}" for index in range(history_size)),
        history_rounds=tuple(range(history_size)),
        measured_observations=tuple({"round": index} for index in range(history_size)),
        research_snapshot={
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


def test_policy_uses_the_current_task_recovery_budget(tmp_path):
    controller, client, _ = _controller(tmp_path, [[({"action": "disable"}, "")]])
    controller.recovery_budget = lambda: 87.0
    result = controller.resolve(_round(1))
    assert result.metadata["status"] == "accepted"
    assert client.recovery_timeout_seconds == 87.0


def test_prediction_records_are_immutable_and_task_feedback_is_delegated(tmp_path: Path) -> None:
    controller, _, _ = _controller(tmp_path, [[({"action": "disable"}, "")]])
    rows = [{"candidate_id": "a", "objectives": [1.0, -2.0], "task_metric": "pareto"}]
    controller.record_predictions(1, rows)
    path = tmp_path / "rounds/round_001/predictions.json"
    original = path.read_bytes()
    controller.record_predictions(1, rows)
    with pytest.raises(ValueError, match="cannot be replaced"):
        controller.record_predictions(1, [{"candidate_id": "b"}])
    with pytest.raises(ValueError, match="JSON serializable"):
        controller.record_predictions(2, [{"value": float("nan")}])
    assert path.read_bytes() == original
    controller.resolve(_round(2))
    assert controller.adapter.feedback_records == [json.loads(original)]


def test_weights_only_multiobjective_policy_uses_task_contract_without_target_assumptions(tmp_path):
    controller, client, _ = _controller(tmp_path, [[({"action": "disable"}, "")]])
    controller.contract = replace(
        controller.contract, enabled_capabilities=("ldm_weights@1",),
    )
    policy_input = replace(
        _round(1),
        history_utilities=np.asarray([[1.0, -4.0]]),
        execution_context={
            "mean_context": {},
            "weight_context": {"objectives": ["yield", "cost"], "progress_metric": "hypervolume"},
        },
    )
    result = controller.resolve(policy_input)
    assert result.query_prior_mean.shape == (2, 2)
    message = json.loads(client.turns[0].message)
    assert message["policy_contract"]["enabled_capabilities"] == ["ldm_weights@1"]
    assert "standardized" not in message["responsibility"]
    assert "only the enabled capabilities" in message["responsibility"]
    assert "inspect_policy_contract" not in message["snapshot_access"]


@pytest.mark.parametrize("objectives", [None, 2])
def test_policy_controller_replace_keep_disable_and_resume(tmp_path: Path, objectives) -> None:
    controller, client, executor = _controller(tmp_path, [
        [_replace()],
        [({"action": "keep"}, "")],
        [({"action": "disable"}, "")],
    ])

    def round_input(index):
        value = _round(index)
        return value if objectives is None else replace(
            value, history_utilities=np.tile(value.history_utilities[:, None], (1, objectives)),
        )

    replaced = controller.resolve(round_input(1))
    kept = controller.resolve(round_input(2))
    disabled = controller.resolve(round_input(3))
    replayed = controller.resolve(round_input(3))
    expected = (2,) if objectives is None else (2, objectives)
    assert replaced.query_prior_mean.shape == kept.query_prior_mean.shape == expected
    assert disabled.query_prior_mean.shape == replayed.query_prior_mean.shape == expected

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


@pytest.mark.parametrize("usage", [
    {"providerCalls": 3, "toolCalls": {"bash": 2}, "artifactBytes": 120, "validationSubmissions": 1},
    {},
])
def test_policy_failure_preserves_cursor_accounting_and_complete_rounds(tmp_path: Path, usage) -> None:
    from ldm_tts.pilot_evaluation.reporting import _integrity

    controller, client, _executor = _controller(tmp_path, [
        [_replace()], HarnessError("provider 502", turn_usage={"policy_architect": usage}),
        [({"action": "keep"}, "")],
    ])
    charges = []
    controller.account = charges.append
    first = controller.resolve(_round(1))
    failed = controller.resolve(_round(2))
    recovered = controller.resolve(_round(3))

    assert failed.degraded and not recovered.degraded
    assert recovered.epoch_id == first.epoch_id
    turn = client.turns[-1]
    assert (turn.history_from_seq, turn.history_to_seq) == (1, 3)
    message = json.loads(turn.message)
    assert message["new_measured_observations"] == [{"round": 1}, {"round": 2}]
    assert sum(cost["policy_harness_turns"] for cost in charges) == 3
    assert failed.metadata["failed_harness_usage"] == usage
    assert "harness_turn" not in failed.metadata
    if usage:
        assert charges[1]["policy_provider_requests"] == 3
        assert charges[1]["policy_tool_calls"] == 2
        assert charges[1]["policy_validation_submissions"] == 1
    else:
        assert "policy_provider_requests" not in charges[1]
        assert "policy_tool_calls" not in charges[1]
    assert controller.resolve(_round(2)).metadata["failed_harness_usage"] == usage
    assert len(charges) == 3 and client.calls == 3
    row = dict(
        case="fixture", seed=0, method="ldm_harness_compiled", initial_candidate_ids=["initial"],
        completed_rounds=4, budget_outer_iterations=4, candidate_ids_unique=True,
        proposal_samples=64, harness_candidates_per_session=16, budget_proposal_attempts=12,
        budget_harness_turns=12, budget_policy_harness_turns=len(charges), policy_rounds=3,
    )
    spec = SimpleNamespace(methods=["ldm_harness_compiled"], iterations=4, optimization_rounds=3)
    assert _integrity(spec, [row], [{}] * 4) == {"valid": True, "errors": []}
    row["policy_rounds"] = 2
    assert not _integrity(spec, [row], [{}] * 4)["valid"]


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


@pytest.mark.parametrize("changes", [
    {"history_candidate_ids": ()},
    {"history_rounds": (1,)},
    {"history_rounds": (True,)},
    {"measured_observations": ()},
    {"research_snapshot": {"measured_observations": []}},
    {"execution_context": {"weight_context": {}}},
    {"history_utilities": np.empty((1, 0))},
])
def test_policy_round_rejects_incomplete_or_ambiguous_history(changes) -> None:
    with pytest.raises(ValueError):
        replace(_round(1), **changes)


def test_policy_round_preserves_replicates_without_task_identity_assumptions() -> None:
    value = replace(_round(2), history_candidate_ids=("same", "same"))
    assert value.history_candidate_ids == ("same", "same")
    assert replace(value, history_rounds=(0, 0)).history_rounds == (0, 0)


def test_policy_mcp_pins_task_owned_diagnostics() -> None:
    server = policy_mcp_server(
        diagnostics_path="/resources/diagnostics.py", diagnostics_sha256="a" * 64,
    )
    env = {key: value.value for key, value in server.env}
    assert env["LDM_POLICY_DIAGNOSTICS"] == "/resources/diagnostics.py"
    assert env["LDM_POLICY_DIAGNOSTICS_SHA256"] == "a" * 64
    assert server.config_sha256 != policy_mcp_server().config_sha256
    with pytest.raises(ValueError, match="path and SHA-256"):
        policy_mcp_server(diagnostics_path="/resources/diagnostics.py")


@pytest.mark.parametrize("name", ["arrays.npz", "research_snapshot.json"])
def test_policy_resume_rejects_changed_authoritative_inputs(tmp_path: Path, name) -> None:
    controller, client, _ = _controller(tmp_path, [[_replace()]])
    controller.resolve(_round(1))
    path = tmp_path / "rounds/round_001" / name
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="input digest mismatch"):
        controller.resolve(_round(1))
    assert client.calls == 1
