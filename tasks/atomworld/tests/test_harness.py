from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from ldm_tts.engine.expansion import ExpansionRequest
from ldm_tts.harness import (
    HarnessError,
    HarnessSubmissionRequest,
    HarnessTurnResult,
)
from tasks.atomworld.core.data import TASK_ROOT
from tasks.atomworld.core.harness import (
    AtomWorldHarnessExpander,
    harness_profiles,
    make_client,
)
from tasks.atomworld.core.optimization_policy import FEATURE_NAMES, public_policy_input
from tasks.atomworld.core.workflow import parse_args, run


def fixture():
    return json.loads((TASK_ROOT / "resources/mock_fixture.json").read_text())


def test_runner_repeated_profile_and_budget_flags_preserve_all_entries():
    common = ["--mock", "--search-method", "harness"]
    repeated = parse_args(common + [
        "--harness-profile", "geometry_research",
        "--harness-profile", "structure_audit",
        "--harness-tool-budget", "bash=4",
        "--harness-tool-budget", "web_search=2",
    ])
    grouped = parse_args(common + [
        "--harness-profile", "geometry_research", "structure_audit",
        "--harness-tool-budget", "bash=4", "web_search=2",
    ])
    assert repeated.harness_profile == grouped.harness_profile == [
        "geometry_research", "structure_audit",
    ]
    assert repeated.harness_tool_budget == grouped.harness_tool_budget == [
        "bash=4", "web_search=2",
    ]
    assert parse_args(common).harness_profile == grouped.harness_profile


class Poison:
    def __getattribute__(self, name):
        raise AssertionError("Read hidden judge state")


class ResearchClient:
    """Protocol double for deterministic boundary tests; never a live-model claim."""

    def __init__(self, args, root, sample, *, policy=False):
        self.config = SimpleNamespace(profiles=harness_profiles(args.harness_profile))
        self.root, self.sample, self.calls, self.closed = root, sample, [], False
        self.outputs = fixture()["mock_outputs"][fixture()["public"][0]["sample_id"]]

    def run_turn(self, turns, *, submission_validator, recovery_timeout_seconds=0):
        self.calls.append(turns)
        results = []
        for turn in turns:
            invalid = HarnessSubmissionRequest(
                turn.profile_id,
                turn.turn_id,
                1,
                {"generated_output": "invalid", "rationale": "draft"},
            )
            rejection = submission_validator(invalid)
            assert rejection.decision == "retry"
            assert rejection.errors[0].code == "invalid_cif"
            payload = {
                "generated_output": self.outputs[min(len(self.calls) - 1, 1)],
                "rationale": "Checked the public Cartesian transformation.",
            }
            request = HarnessSubmissionRequest(
                turn.profile_id, turn.turn_id, 2, payload
            )
            assert submission_validator(request).decision == "accept"
            results.append(
                HarnessTurnResult(
                    profile_id=turn.profile_id,
                    session_id=f"persistent_{self.sample['sample_id']}_{turn.profile_id}",
                    turn_id=turn.turn_id,
                    round_index=turn.round_index,
                    history_from_seq=turn.history_from_seq,
                    history_to_seq=turn.history_to_seq,
                    history_digest=turn.history_digest,
                    input_digest=turn.input_digest,
                    replayed=False,
                    submission_status="accepted",
                    submission_id=turn.turn_id,
                    submission_digest=request.digest,
                    submission=payload,
                    submitted_artifacts=(),
                    validation_errors=(),
                    usage={
                        "providerCalls": 2,
                        "toolCalls": {"get_public_task": 1, "bash": 1},
                        "validationSubmissions": 2,
                        "artifactBytes": 0,
                    },
                    tool_budget={},
                    artifacts={},
                )
            )
        return tuple(results)

    def close(self):
        self.closed = True


def test_persistent_repair_history_and_full_campaign_resume(tmp_path):
    clients = []

    def factory(*args, **kwargs):
        client = ResearchClient(*args, **kwargs)
        clients.append(client)
        return client

    args = parse_args(
        [
            "--mock",
            "--search-method",
            "harness",
            "--iterations",
            "2",
            "--out-dir",
            str(tmp_path / "run"),
        ]
    )
    first = run(args, harness_client_factory=factory)
    assert first.projected["one_shot_accuracy"] == 0
    assert first.projected["extended_final_accuracy"] == 1
    assert len(clients) == 1 and len(clients[0].calls) == 2 and clients[0].closed
    first_turn, second_turn = clients[0].calls[0][0], clients[0].calls[1][0]
    assert first_turn.profile_id == second_turn.profile_id
    assert first_turn.history_to_seq == 0 and second_turn.history_to_seq == 1
    assert json.loads(second_turn.message)["message_type"] == "public_history_delta"
    assert (
        len(json.loads((clients[0].root / "public_history.json").read_text())["drafts"])
        == 1
    )
    assert first.runtime.budget.counters["harness_turns"] == 4
    assert first.runtime.budget.counters["llm_requests"] == 8
    assert first.runtime.budget.counters["harness_validation_submissions"] == 8
    args.resume, args.out_dir = True, first.runtime.run_dir
    second = run(
        args,
        harness_client_factory=lambda *a, **k: pytest.fail(
            "Completed session was repeated"
        ),
    )
    assert second.projected["extended_final_accuracy"] == 1


def test_harness_never_reads_judge_and_isolates_questions(tmp_path):
    public = fixture()["public"][0]
    second = {**public, "sample_id": "independent_sample"}
    args = parse_args(["--mock", "--search-method", "harness", "--iterations", "2"])
    clients = []

    def factory(*args, **kwargs):
        client = ResearchClient(*args, **kwargs)
        clients.append(client)
        return client

    expander = AtomWorldHarnessExpander(
        [public, second], args, tmp_path, client_factory=factory
    )
    for index in range(4):
        expander.expand(
            ExpansionRequest(
                index,
                1,
                observations=(Poison(),),
                parent=Poison(),
                acquisition_feedback={"judge_secret": "LEAK"},
            )
        )
    assert len(clients) == 2 and len(expander.clients) == 1 and clients[0].closed
    for index, client in enumerate(clients):
        encoded = json.dumps([json.loads(turns[0].message) for turns in client.calls])
        assert "LEAK" not in encoded and "target_cif" not in encoded
        for turns in client.calls:
            assert (
                json.loads(turns[0].message)["sample"]["sample_id"]
                == [public, second][index]["sample_id"]
            )
        assert json.loads(client.calls[0][0].message)["new_public_drafts"] == []


def test_rejected_or_missing_session_does_not_silently_reduce_batch(tmp_path):
    class Missing(ResearchClient):
        def run_turn(self, *args, **kwargs):
            return super().run_turn(*args, **kwargs)[:1]

    args = parse_args(["--mock", "--search-method", "harness"])
    expander = AtomWorldHarnessExpander(
        fixture()["public"], args, tmp_path, client_factory=Missing
    )
    with pytest.raises(RuntimeError, match="exactly one"):
        expander.expand(ExpansionRequest(0, 1))
    assert not (tmp_path / "attempts/000000.json").exists()


def test_policy_inputs_have_no_labels_and_remain_public(tmp_path):
    data = fixture()
    answers = [
        SimpleNamespace(
            submission={"generated_output": text, "rationale": "Public hypothesis"}
        )
        for text in data["mock_outputs"][data["public"][0]["sample_id"]]
    ]
    value = public_policy_input(1, data["public"][0], answers, [], mock=True)
    assert value.history_utilities.size == 0 and value.history_features.shape == (
        0,
        len(FEATURE_NAMES),
    )
    assert value.measured_observations == () and value.query_features.shape == (
        len(answers),
        len(FEATURE_NAMES),
    )
    assert np.isfinite(value.query_features).all()
    assert "target_cif" not in json.dumps(value.research_snapshot)


def test_factory_mounts_only_public_artifacts_and_task_resources(tmp_path, monkeypatch):
    captured = {}

    class Capture:
        def __init__(self, command, **kwargs):
            captured.update(command=command, **kwargs)

        def start(self):
            pass

    monkeypatch.setattr("tasks.atomworld.core.harness.HarnessClient", Capture)
    monkeypatch.setenv("LLM_API_KEY", "test-secret")
    args = parse_args(
        [
            "--mock",
            "--search-method",
            "harness",
            "--llm-url",
            "https://provider.example/v1",
            "--llm-model-name",
            "model",
            "--harness-cache-dir",
            str(tmp_path / "cache"),
        ]
    )
    root = tmp_path / "campaign/harness/sample_000000"
    make_client(args, root, fixture()["public"][0])
    command = captured["command"]
    mounts = [command[i + 1] for i, part in enumerate(command) if part == "--mount"]
    assert len(mounts) == 3
    assert not any(
        "dst=/artifacts" in item and f"src={tmp_path / 'campaign'}," in item
        for item in mounts
    )
    assert "test-secret" not in " ".join(command)
    payload = captured["config"].initialize_payload()
    assert payload["toolExtensions"][0]["toolNames"] == [
        "get_public_task",
        "get_public_history",
        "get_geometry_contract",
    ]
    assert payload["profiles"][0]["skillDirs"] == ["/resources/skills/crystal_geometry"]
    assert payload["limits"]["toolCallBudgets"]["bash"] == 64
    assert set(json.loads((root / "public_task.json").read_text())) == {
        "sample_id",
        "action_name",
        "input_cif",
        "action_prompt",
    }


def test_pilot_launch_and_resume_cli(tmp_path):
    args = parse_args(
        [
            "--proposal-mode",
            "mock",
            "--search-method",
            "harness",
            "--iterations",
            "4",
            "--campaign-index",
            "2",
            "--initialization-mode",
            "shared_start",
            "--out-dir",
            str(tmp_path),
            "--run-name",
            "child",
        ]
    )
    assert (
        args.mock
        and args.attempts_per_sample == 4
        and args.out_dir == tmp_path / "child"
    )
    resumed = parse_args(["--mock", "--resume-from", str(tmp_path / "child")])
    assert resumed.resume and resumed.out_dir == tmp_path / "child"


def test_interrupted_harness_resumes_same_turn_without_losing_answer_budget(tmp_path):
    interrupted_turns = []

    class Outage(ResearchClient):
        def run_turn(self, turns, **kwargs):
            interrupted_turns.extend(turns)
            raise HarnessError("temporary fixture outage", retryable=True)

    args = parse_args(
        [
            "--mock",
            "--search-method",
            "harness",
            "--iterations",
            "2",
            "--out-dir",
            str(tmp_path / "run"),
        ]
    )
    with pytest.raises(HarnessError, match="temporary"):
        run(args, harness_client_factory=Outage)
    assert (
        json.loads((args.out_dir / "status.json").read_text())["status"]
        == "paused_harness"
    )
    resumed_clients = []

    def factory(*args, **kwargs):
        client = ResearchClient(*args, **kwargs)
        resumed_clients.append(client)
        return client

    args.resume = True
    result = run(args, harness_client_factory=factory)
    assert interrupted_turns[0].turn_id == resumed_clients[0].calls[0][0].turn_id
    assert result.runtime.budget.counters["outer_iterations"] == 2
    assert result.runtime.budget.counters["harness_turns"] == 6
    assert len(result.projected["samples"][0]["attempts"]) == 2
    assert result.projected["extended_final_accuracy"] == 1
