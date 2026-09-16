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
    HarnessSubmittedArtifact, file_sha256,
)
from ldm_tts.transport.openai import EndpointRequestError
from tasks.atomworld.core.data import TASK_ROOT
from tasks.atomworld.core.harness import (
    AtomWorldHarnessExpander,
    harness_profiles,
    make_client,
)
from tasks.atomworld.core.optimization_policy import FEATURE_NAMES, public_policy_input
from tasks.atomworld.core.workflow import parse_args, run
from tasks.atomworld.core.proposals import extract_cif


def fixture():
    return json.loads((TASK_ROOT / "resources/mock_fixture.json").read_text())


def real_fixture_payload():
    data = fixture()
    sample = data["public"][0]
    return [sample], {sample["sample_id"]: data["private"][0]["target_cif"]}, {
        "dataset_kind": "unit_real_fixture",
        "paper_split_verified": False,
    }


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

    def artifact(self, turn, attempt, text):
        path = self.root / "snapshots" / f"{turn.turn_id}_{attempt}.cif"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return HarnessSubmittedArtifact("/artifact_path", "answer.cif", str(path.relative_to(self.root)),
                                        file_sha256(path), path.stat().st_size)

    def run_turn(self, turns, *, submission_validator, recovery_timeout_seconds=0):
        self.calls.append(turns)
        results = []
        for turn in turns:
            invalid = HarnessSubmissionRequest(
                turn.profile_id,
                turn.turn_id,
                1,
                {"artifact_path": "answer.cif", "rationale": "draft"},
                (self.artifact(turn, 1, "invalid"),),
            )
            rejection = submission_validator(invalid)
            assert rejection.decision == "retry"
            assert rejection.errors[0].code == "invalid_cif"
            payload = {
                "artifact_path": "answer.cif",
                "rationale": "Checked the public Cartesian transformation.",
            }
            request = HarnessSubmissionRequest(
                turn.profile_id, turn.turn_id, 2, payload,
                (self.artifact(turn, 2, extract_cif(self.outputs[min(len(self.calls) - 1, 1)])),),
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
                    submitted_artifacts=request.artifacts,
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


def test_journal_replay_after_receipt_interruption_counts_turn_once(tmp_path, monkeypatch):
    from tasks.atomworld.core import harness
    saved, provider_barriers = {}, []

    class JournalClient(ResearchClient):
        def run_turn(self, turns, **kwargs):
            key = tuple(turn.turn_id for turn in turns)
            if key not in saved:
                provider_barriers.append(key)
                saved[key] = super().run_turn(turns, **kwargs)
            return saved[key]

    write = harness.write_json
    interrupted = False

    def crash(path, value):
        nonlocal interrupted
        if path.parent.name == "attempts" and not interrupted:
            interrupted = True
            raise HarnessError("interrupted before attempt receipt")
        write(path, value)

    monkeypatch.setattr(harness, "write_json", crash)
    args = parse_args(["--mock", "--search-method", "harness", "--iterations", "1",
                       "--out-dir", str(tmp_path / "campaign")])
    with pytest.raises(HarnessError, match="before attempt"):
        run(args, harness_client_factory=JournalClient)
    before = json.loads((args.out_dir / "budget.json").read_text())["counters"]
    assert before["harness_turns"] == 2 and before["llm_requests"] == 4
    args.resume = True
    result = run(args, harness_client_factory=JournalClient)
    assert len(provider_barriers) == 1
    assert result.runtime.budget.counters["harness_turns"] == 2
    assert result.runtime.budget.counters["llm_requests"] == 4
    attempt = json.loads((args.out_dir / "attempts/000000.json").read_text())
    assert attempt["rationale"]
    assert all("generated_output" not in item["submission"] for item in attempt["sessions"])


def test_cif_submission_verifies_immutable_snapshot(tmp_path):
    from tasks.atomworld.core.harness import validate_answer
    client = ResearchClient(parse_args(["--mock"]), tmp_path, fixture()["public"][0])
    artifact = client.artifact(SimpleNamespace(turn_id="test"), 1, extract_cif(client.outputs[0]))
    request = HarnessSubmissionRequest("geometry_research", "test", 1,
        {"artifact_path": "answer.cif", "rationale": "Public geometry"}, (artifact,))
    assert validate_answer(request, root=tmp_path, mock=True).decision == "accept"
    (tmp_path / artifact.snapshot_path).write_text("data_tampered")
    validation = validate_answer(request, root=tmp_path, mock=True)
    assert validation.decision == "retry"
    assert validation.errors[0].code == "invalid_artifact"


def test_real_harness_preflight_runs_before_first_sidecar_turn(tmp_path, monkeypatch):
    calls = []

    class PreflightClient:
        def __init__(self, **kwargs):
            calls.append(("preflight_client", kwargs["wire_api"], kwargs["api_key"]))

        def preflight(self):
            calls.append("preflight")
            return {"status": "ok", "request_model": "fixture", "response_model": "fixture"}

    def factory(*args, **kwargs):
        calls.append("factory")
        return ResearchClient(*args, **kwargs)

    monkeypatch.setattr("tasks.atomworld.core.workflow.load_prepared", lambda data_dir: real_fixture_payload())
    monkeypatch.setattr("tasks.atomworld.core.workflow.load_official_evaluator",
                        lambda upstream: lambda *args, **kwargs: SimpleNamespace(
                            correct=True, wrong_type=None, rmsd=None, max_dist=None))
    monkeypatch.setattr("tasks.atomworld.core.workflow.OpenAICompatibleProposalClient", PreflightClient)
    monkeypatch.setattr("tasks.atomworld.core.harness.public_validation",
                            lambda text, mock=False: {"parseable": "invalid" not in text})
    monkeypatch.setenv("LDM_LLM_API_KEY", "must-not-appear-in-artifacts")
    args = parse_args([
        "--data-dir", str(tmp_path / "data"),
        "--search-method", "harness",
        "--llm-url", "http://fixture.invalid/v1",
        "--llm-model-name", "fixture",
        "--out-dir", str(tmp_path / "run"),
    ])

    result = run(args, harness_client_factory=factory)

    assert calls[:3] == [
        ("preflight_client", "responses", "must-not-appear-in-artifacts"),
        "preflight",
        "factory",
    ]
    preflight = json.loads((result.runtime.run_dir / "endpoint_preflight.json").read_text())
    assert preflight["backend"] == "harness"
    assert preflight["wire_api"] == "responses"
    assert result.runtime.budget.counters["endpoint_preflights"] == 1
    events = [
        json.loads(line)
        for line in (result.runtime.run_dir / "events.jsonl").read_text().splitlines()
    ]
    assert any(event["event_type"] == "endpoint_preflight_succeeded" for event in events)
    for file in result.runtime.run_dir.rglob("*"):
        if file.is_file():
            assert "must-not-appear-in-artifacts" not in file.read_text()


def test_mock_default_harness_preflights_provider_before_sidecar(tmp_path, monkeypatch):
    calls = []

    class PreflightClient:
        def __init__(self, **kwargs):
            calls.append(("preflight_client", kwargs["wire_api"], kwargs["api_key"]))

        def preflight(self):
            calls.append("preflight")
            return {"status": "ok", "request_model": "fixture", "response_model": "fixture"}

    def factory(*args, **kwargs):
        calls.append("factory")
        return ResearchClient(*args, **kwargs)

    monkeypatch.setattr("tasks.atomworld.core.workflow.OpenAICompatibleProposalClient", PreflightClient)
    monkeypatch.setattr("tasks.atomworld.core.harness.make_client", factory)
    monkeypatch.setenv("LLM_API_KEY", "mock-harness-secret")
    args = parse_args([
        "--mock",
        "--search-method", "harness",
        "--llm-url", "http://fixture.invalid/v1",
        "--llm-model-name", "fixture",
        "--out-dir", str(tmp_path / "run"),
    ])

    result = run(args)

    assert calls[:3] == [
        ("preflight_client", "responses", "mock-harness-secret"),
        "preflight",
        "factory",
    ]
    assert result.runtime.budget.counters["endpoint_preflights"] == 1
    assert json.loads((result.runtime.run_dir / "endpoint_preflight.json").read_text())[
        "backend"
    ] == "harness"


def test_mock_default_harness_preflight_failure_happens_before_sidecar(tmp_path, monkeypatch):
    calls = []

    def factory(*args, **kwargs):
        pytest.fail("Harness sidecar should not start without provider settings")

    for name in (
        "LLM_BASE_URL",
        "LDM_LLM_URL",
        "OPENAI_BASE_URL",
        "LLM_MODEL_NAME",
        "LDM_LLM_MODEL",
        "OPENAI_MODEL",
        "LLM_API_KEY",
        "LDM_LLM_API_KEY",
        "OPENAI_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("tasks.atomworld.core.harness.make_client", factory)
    args = parse_args([
        "--mock",
        "--search-method", "harness",
        "--out-dir", str(tmp_path / "run"),
    ])

    with pytest.raises(EndpointRequestError):
        run(args)

    assert calls == []
    status = json.loads((args.out_dir / "status.json").read_text())
    assert status["status"] == "paused_endpoint"
    assert status["phase"] == "preflight"
    assert json.loads((args.out_dir / "budget.json").read_text())["counters"][
        "endpoint_preflights"
    ] == 1
    assert not (args.out_dir / "harness").exists()


def test_real_harness_preflight_failure_happens_before_sidecar(tmp_path, monkeypatch):
    calls = []

    class PreflightClient:
        def __init__(self, **kwargs):
            pass

        def preflight(self):
            calls.append("preflight")
            raise EndpointRequestError("fixture endpoint rejected key")

    def factory(*args, **kwargs):
        pytest.fail("Harness sidecar should not start before provider preflight passes")

    monkeypatch.setattr("tasks.atomworld.core.workflow.load_prepared", lambda data_dir: real_fixture_payload())
    monkeypatch.setattr("tasks.atomworld.core.workflow.load_official_evaluator",
                        lambda upstream: lambda *args, **kwargs: SimpleNamespace(
                            correct=True, wrong_type=None, rmsd=None, max_dist=None))
    monkeypatch.setattr("tasks.atomworld.core.workflow.OpenAICompatibleProposalClient", PreflightClient)
    monkeypatch.setenv("LLM_API_KEY", "fixture-secret")
    args = parse_args([
        "--data-dir", str(tmp_path / "data"),
        "--search-method", "harness",
        "--llm-url", "http://fixture.invalid/v1",
        "--llm-model-name", "fixture",
        "--out-dir", str(tmp_path / "run"),
    ])

    with pytest.raises(EndpointRequestError):
        run(args, harness_client_factory=factory)

    assert calls == ["preflight"]
    status = json.loads((args.out_dir / "status.json").read_text())
    assert status["status"] == "paused_endpoint"
    assert status["phase"] == "preflight"
    assert json.loads((args.out_dir / "budget.json").read_text())["counters"]["endpoint_preflights"] == 1
    assert not (args.out_dir / "harness").exists()


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
    assert payload["providerRequestBody"] == {"temperature": 0.0, "max_output_tokens": 8192}
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
    assert result.runtime.budget.counters["harness_turns"] == 4
    assert len(result.projected["samples"][0]["attempts"]) == 2
    assert result.projected["extended_final_accuracy"] == 1
