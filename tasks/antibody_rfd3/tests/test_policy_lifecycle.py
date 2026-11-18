"""Policy selection, decision feedback and recovery against a simulated experiment service."""
from __future__ import annotations

from collections import Counter
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from experiment_service import SimulatedExecutor, VirtualClock
from experiment_service.testing import running_service
from ldm_tts.async_runtime import (CampaignConfig, CrashPlan, DeterministicProposer, ESClient,
    InlineExecutor, PacedController, PacingConfig, PendingItem, RuntimeOptions, SimulatedCrash, campaign_extra)
from ldm_tts.async_runtime.selection import SelectionContext
from ldm_tts.async_runtime.selectors import decision_choice
from ldm_tts.contracts import Candidate, EvaluationResult, ObjectiveSet, ObjectiveSpec, Observation
from ldm_tts.harness import PolicyResearchController
from ldm_tts.harness.policy_execution import PolicyExecutionResult
from ldm_tts.harness.protocol import (HarnessError, HarnessSubmissionRequest, HarnessSubmittedArtifact,
                                       HarnessTurnResult, file_sha256)
from ldm_tts.harness.ucb_diagnostics import prediction_feedback
from ldm_tts.optimization.pending import PriorTable
from ldm_tts.optimization.records import SelectionSeed
from tasks.antibody_rfd3.core.experiments import AntibodyCodec, ENERGY_UTILITY, KIND
from tasks.antibody_rfd3.core.policy import (AMINO_ACIDS, AntibodyPolicyAdapter, AntibodyPolicyPrior,
                                             make_selector, prepare_policy_turn)


def candidate(sequence):
    return Candidate(sequence, {"sequence": sequence}, sequence)


def measured(sequence, energy, decision):
    item = candidate(sequence)
    result = EvaluationResult(item.candidate_id, "succeeded",
        {"mock_energy": energy, ENERGY_UTILITY: -energy})
    return Observation(item, result, decision_id=decision)


def test_active_prior_conditions_real_selector_on_inflight_candidate():
    history = (measured("AAAAAAAAAAA", -9.0, "d000001"),
               measured("CAAAAAAAAAA", -3.0, "d000002"))
    query = (candidate("DAAAAAAAAAA"), candidate("ECCCCCCCCCC"))
    in_flight = candidate("DAAAAAAAAAC")
    application = PriorTable("a000003", {history[0].canonical_key: 9.0, history[1].canonical_key: 3.0,
        query[0].canonical_key: 4.0, query[1].canonical_key: 3.0, in_flight.canonical_key: 4.0})
    application.weights = (1.5, 1.25)
    application.prior_free = False
    selector = make_selector()
    selector.prior = lambda: application
    pending = PendingItem("exp-pending", 3, in_flight.canonical_key, in_flight, "RUNNING", False)
    context = SelectionContext(None, query, {item.canonical_key: 0.5 for item in query}, (pending,),
                               SelectionSeed(42, "d000004"), 1, history, 2)
    active = selector.choose(context)
    no_pending = selector.choose(replace(context, pending=()))
    step, baseline = active.record["selection"]["steps"][0], no_pending.record["selection"]["steps"][0]
    assert active.inputs["pending"] == [["exp-pending", in_flight.canonical_key]]
    assert active.inputs["prior_application_id"] == "a000003"
    assert active.record["selection"]["metadata"]["gp_ready"]
    assert active.record["selection"]["metadata"]["prior_applied"]
    assert step["metadata"]["pending_handling"] == "kriging_believer"
    assert step["pending_keys"] == [in_flight.canonical_key]
    np.testing.assert_allclose(step["mean"], baseline["mean"], atol=1e-10)
    assert np.all(np.asarray(step["std"]) <= np.asarray(baseline["std"]) + 1e-10)
    assert step["std"][0] < baseline["std"][0] - 1e-6
    assert step["candidate_ids"] == baseline["candidate_ids"]
    assert active.record["selection"]["parameters"]["alpha"] == 1.5
    assert active.record["selection"]["parameters"]["eta"] == 1.25


def test_feedback_uses_decision_and_candidate_not_duplicate_or_completion_order():
    row = {"baseline_mean": 1.0, "active_mean": 2.0}
    records = [
        {"decision_id": "d000001", "application_id": "a000001", "predictions": [{"candidate_id": "a", **row}]},
        {"decision_id": "d000004", "application_id": "a000002", "predictions": [{"candidate_id": "a", **row}]},
        {"decision_id": "d000002", "application_id": "a000001", "predictions": [{"candidate_id": "b", **row}]},
    ]
    # The first attempt at "a" was not measured; "b" finished before the replacement attempt at "a".
    result = prediction_feedback(("b", "a"), (1, 1), (7.0, 9.0), records, ("d000002", "d000004"))
    found = {(item["decision_id"], item["candidate_id"]): item for item in result["measurements"]}
    assert set(found) == {("d000002", "b"), ("d000004", "a")}
    assert found[("d000004", "a")]["application_id"] == "a000002"
    assert found[("d000004", "a")]["measured_utility"] == 9.0


POLICY_SOURCE = '''import numpy as np
POLICY_API_VERSION = 1
CAPABILITIES = {"prior_mean": 1, "ldm_weights": 1}

def compute_prior_mean(history_features, history_utilities, query_features, context):
    return query_features[:, 0] * 0.5

def choose_ldm_weights(context):
    count = context["optimization_progress"]["measured_count"]
    feedback_count = context["prediction_feedback"]["summary"]["count"]
    return {"stage": f"measured-{count}-feedback-{feedback_count}",
            "alpha": 1.0 + count * 0.25, "eta": 1.0 + feedback_count * 0.5}
'''


class PolicyExecutor:
    """Test-only in-process runner for a fixed trusted artifact; production uses Docker."""

    def __init__(self):
        self.calls = []
        path = Path(__file__).resolve().parents[3] / "harnesses/pi/policy_runner.py"
        spec = importlib.util.spec_from_file_location("antibody_test_policy_runner", path)
        self.runner = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = self.runner
        spec.loader.exec_module(self.runner)

    def execute(self, artifact_path, input_directory, output_directory):
        input_directory = Path(input_directory)
        context = json.loads((input_directory / "input.json").read_text())
        self.calls.append((context.get("application", "validation"), context["round_index"]))
        metadata = self.runner.execute_artifact(Path(artifact_path), input_directory, Path(output_directory))
        with np.load(Path(output_directory) / "arrays.npz", allow_pickle=False) as arrays:
            history = arrays["history_prior_mean"].copy()
            query = arrays["query_prior_mean"].copy()
        return PolicyExecutionResult(history, query, metadata["stage"], metadata["alpha"], metadata["eta"],
                                     metadata["prior_clip_count"])


class PolicyClient:
    """Submit one fixed test artifact, then keep its committed epoch."""

    def __init__(self, root, calls):
        self.root, self.calls = Path(root), calls

    def run_turn(self, turns, *, submission_validator, recovery_timeout_seconds=0):
        turn, = turns
        self.calls.append(turn.round_index)
        if turn.round_index == 1:
            path = self.root / "snapshots" / f"{turn.turn_id}.py"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(POLICY_SOURCE, encoding="utf-8")
            descriptor = HarnessSubmittedArtifact(path_pointer="/artifact_path",
                relative_path="optimization_policy.py", snapshot_path=path.relative_to(self.root).as_posix(),
                sha256=file_sha256(path), size_bytes=path.stat().st_size)
            action, artifacts = {"action": "replace", "artifact_path": "optimization_policy.py"}, (descriptor,)
        else:
            action, artifacts = {"action": "keep"}, ()
        request = HarnessSubmissionRequest(turn.profile_id, turn.turn_id, 1, action, artifacts)
        validation = submission_validator(request)
        assert validation.decision == "accept", validation.errors
        return (HarnessTurnResult(profile_id=turn.profile_id, session_id="test-session", turn_id=turn.turn_id,
            round_index=turn.round_index, history_from_seq=turn.history_from_seq, history_to_seq=turn.history_to_seq,
            history_digest=turn.history_digest, input_digest=turn.input_digest, replayed=False,
            submission_status="accepted", submission_id=f"test-{turn.round_index}", submission_digest=request.digest,
            submission=action, submitted_artifacts=artifacts, validation_errors=(),
            usage={"providerCalls": 1}, tool_budget={}, artifacts={}),)


class CountingPanel:
    name = "test-antibody-panel"
    version = "1"

    def __init__(self):
        pool = [candidate(a + b + "A" * 9) for a in AMINO_ACIDS[:3] for b in AMINO_ACIDS]
        self.source = DeterministicProposer(pool, per_generation=6, seed=42)
        self.calls = []

    def research(self, round_input, history):
        self.calls.append(round_input["generation"])
        return self.source.research(round_input, history)


class Lifecycle:
    def __init__(self, root, clock, service):
        self.root, self.clock, self.service = Path(root), clock, service
        self.panel, self.executor, self.turns = CountingPanel(), PolicyExecutor(), []
        self.pacing = PacingConfig.balanced(concurrency=2, reservoir_size=6, research_budget=10,
            dispatch_during_research="continue", research_timeout_seconds=100)
        self.config = CampaignConfig("policy-lifecycle", budget=8, concurrency=2, seed=42,
                                     extra=campaign_extra(self.pacing))
        self.research_seconds = 2.0
        self.ctrl = None

    def open(self, faults=None):
        selector = make_selector()
        root = self.root / "policy_harness"
        research = PolicyResearchController(client=PolicyClient(root, self.turns), adapter=AntibodyPolicyAdapter(),
                                             executor=self.executor, root=root)
        policy = AntibodyPolicyPrior(research,
            prepare_turn=lambda round_input, info: prepare_policy_turn(selector, round_input, info))
        self.ctrl = PacedController(self.root / "controller.sqlite3", self.config,
            codec=AntibodyCodec("test", "mock", utility=True), client=ESClient(self.service.base_url),
            clock=self.clock, research=self.panel, lane_executor=InlineExecutor(self.clock, self.research_seconds),
            selector=selector, policy=policy, faults=faults,
            objectives=ObjectiveSet.from_specs([ObjectiveSpec("mock_energy", "minimize")]),
            options=RuntimeOptions(anti_entropy_interval=20, long_poll_seconds=0))
        return self.ctrl

    def close(self):
        if self.ctrl is not None:
            self.ctrl.close()
            self.ctrl = None

    def drive(self):
        for _ in range(3000):
            progressed = False
            for _ in range(10000):
                if not self.ctrl.step():
                    break
                progressed = True
            else:
                raise AssertionError("controller failed to settle")
            if self.service.service.run_until_idle():
                continue
            if progressed:
                continue
            if self.ctrl.finished:
                return
            due = [time for time in (self.service.service.next_event_time(), self.ctrl.next_due()) if time is not None]
            assert due, "campaign stalled without a scheduled event"
            target = min(due)
            assert target > self.clock.now()
            self.clock.advance_to(target)
        raise AssertionError("campaign did not finish")


@pytest.fixture
def lifecycle(tmp_path):
    clock = VirtualClock()
    simulator = SimulatedExecutor(clock, kind=KIND, duration=10, seed=42,
        objective=lambda spec: {"mock_energy": -float(AMINO_ACIDS.index(spec["sequence"][0]) * 20
                                                       + AMINO_ACIDS.index(spec["sequence"][1]) + 1)})
    with running_service([simulator], pools={"default": 2}, clock=clock) as service:
        rig = Lifecycle(tmp_path, clock, service)
        rig.simulator = simulator
        try:
            yield rig
        finally:
            rig.close()


def test_real_policy_prior_collects_decision_feedback_for_later_turns(lifecycle):
    rig = lifecycle
    ctrl = rig.open()
    rig.drive()
    observed = {item.decision_id: item for item in ctrl.observations()}
    assert len(observed) == 8
    checked = set()
    for generation in ctrl.generations():
        index = generation["generation"]
        root = rig.root / "policy_harness" / "rounds" / f"round_{index:03d}"
        snapshot = json.loads((root / "research_snapshot.json").read_text())
        context = json.loads((root / "input.json").read_text())["execution_context"]["weight_context"]
        feedback = context["prediction_feedback"]
        assert feedback["summary"]["count"] == len(snapshot["measured_observations"])
        for item in feedback["measurements"]:
            observation = observed[item["decision_id"]]
            record, inputs, step = decision_choice(ctrl.store, int(item["decision_id"][1:]))
            prediction = record["predictions"][step]
            assert item["candidate_id"] == observation.candidate_id == prediction["candidate_id"]
            assert item["application_id"] == inputs["prior_application_id"]
            assert item["active_mean"] == prediction["active_mean"]
            assert item["measured_utility"] == observation.metrics[ENERGY_UTILITY]
            assert item["measured_utility"] == -observation.metrics["mock_energy"]
            checked.add(item["decision_id"])
    assert len(ctrl.generations()) >= 3
    assert len(checked) >= 2
    assert ctrl.policy.check() == []
    assert all(ctrl.audit(d["decision_id"]) == [] for d in ctrl.decisions())


def test_application_weights_use_frozen_history_arriving_during_research(lifecycle):
    rig = lifecycle
    rig.research_seconds = 15.0
    rig.pacing = PacingConfig.balanced(concurrency=2, reservoir_size=6, research_budget=10,
        theta=1.0, dispatch_during_research="continue", research_timeout_seconds=100)
    rig.config = CampaignConfig("policy-lifecycle", budget=8, concurrency=2, seed=42,
                                extra=campaign_extra(rig.pacing))
    ctrl = rig.open()
    assert ctrl.policy.describe()["version"] == "2"
    rig.drive()
    generations = {g["generation"]: g for g in ctrl.generations()}
    applications = ctrl.policy.applications()
    initial = next(app for app in applications if app["generation"] == 1 and app["trigger"] == "generation")
    assert initial["head_ingest_seq"] == 0
    assert (initial["stage"], initial["alpha"], initial["eta"]) == ("measured-0-feedback-0", 1.0, 1.0)
    advanced = [app for app in applications if app["trigger"] == "generation" and app["epoch_id"]
                and app["head_ingest_seq"] > generations[app["generation"]]["base_ingest_seq"]]
    assert advanced, "an evaluation must finish after a policy turn's frozen research baseline"
    app = advanced[0]
    index, head = app["generation"], app["head_ingest_seq"]
    round_root = rig.root / "policy_harness" / "rounds" / f"round_{index:03d}"
    round_input = json.loads((round_root / "input.json").read_text())
    research_weights = round_input["execution_context"]["weight_context"]
    assert research_weights["optimization_progress"]["measured_count"] < head
    application_root = (rig.root / "controller.sqlite3.policy" / "applications" /
                        app["input_digest"].split(":", 1)[1][:16])
    app_input = json.loads((application_root / "input/input.json").read_text())
    weights = app_input["execution_context"]["weight_context"]
    frozen = {item.decision_id: item for item in ctrl.observations()
              if item.metadata["ingest_seq"] <= head and item.evaluation.succeeded}
    assert any(item.metadata["ingest_seq"] > head for item in ctrl.observations())
    feedback = weights["prediction_feedback"]
    assert weights["optimization_progress"]["measured_count"] == len(frozen) == head
    assert feedback["summary"]["count"] == len(frozen)
    assert {item["decision_id"] for item in feedback["measurements"]} == set(frozen)
    for item in feedback["measurements"]:
        observation = frozen[item["decision_id"]]
        record, inputs, step = decision_choice(ctrl.store, int(item["decision_id"][1:]))
        assert item["candidate_id"] == observation.candidate_id == record["predictions"][step]["candidate_id"]
        assert item["application_id"] == inputs["prior_application_id"]
        assert item["measured_utility"] == observation.metrics[ENERGY_UTILITY]
    assert app["stage"] == f"measured-{len(frozen)}-feedback-{len(frozen)}"
    assert app["alpha"] == 1.0 + len(frozen) * 0.25
    assert app["eta"] == 1.0 + len(frozen) * 0.5
    used = [decision for decision in ctrl.decisions()
            if decision["snapshot"]["inputs"]["prior_application_id"] == app["application_id"]]
    assert used, "a decision must use the updated application"
    for decision in used:
        parameters = decision["snapshot"]["record"]["selection"]["parameters"]
        assert (parameters["alpha"], parameters["eta"]) == (app["alpha"], app["eta"])
        assert ctrl.audit(decision["decision_id"]) == []


@pytest.mark.parametrize("point", ["policy_turn.after_panel", "generation_end.after_commit"])
def test_policy_and_application_recover_without_duplicate_turn_or_commit(lifecycle, point):
    rig = lifecycle
    fault = CrashPlan(point)
    rig.open(faults=fault)
    with pytest.raises(SimulatedCrash):
        rig.drive()
    assert fault.fired
    rig.close()
    ctrl = rig.open()
    rig.drive()
    assert ctrl.finished
    assert Counter(rig.panel.calls)[1] == 1
    assert Counter(rig.turns)[1] == 1
    assert Counter(rig.executor.calls)[("generation", 1)] == 1
    generation_apps = [a for a in ctrl.policy.applications() if a["generation"] == 1 and a["trigger"] == "generation"]
    assert len(generation_apps) == 1 and generation_apps[0]["epoch_id"] == "epoch_001"
    assert len(rig.simulator.launches) == len(set(rig.simulator.launches)) == 8
    assert ctrl.policy.check() == []
    assert all(ctrl.audit(d["decision_id"]) == [] for d in ctrl.decisions())


@pytest.mark.parametrize("interruption", [None, "transaction", "outcome.json", "result.json", "compiled.npz"])
def test_failed_policy_turn_resumes_from_persisted_panel_after_retry_interruption(lifecycle, monkeypatch, interruption):
    from ldm_tts.async_runtime import policy as policy_module

    rig = lifecycle
    original_turn = PolicyClient.run_turn
    seen_turns = []

    def fail_once(client, turns, **kwargs):
        turn, = turns
        seen_turns.append((turn.round_index, turn.turn_id, turn.input_digest))
        if turn.round_index == 2 and sum(index == 2 for index, _, _ in seen_turns) == 1:
            raise HarnessError("provider 502", turn_usage={"policy_architect": {"providerCalls": 1}}, retryable=True)
        return original_turn(client, turns, **kwargs)

    monkeypatch.setattr(PolicyClient, "run_turn", fail_once)
    ctrl = rig.open()
    for _ in range(3000):
        progressed = ctrl.step()
        serviced = rig.service.service.run_until_idle()
        if ctrl.view().lane.state == "blocked":
            break
        if not progressed and not serviced:
            due = [value for value in (rig.service.service.next_event_time(), ctrl.next_due())
                   if value is not None and value > rig.clock.now()]
            assert due
            rig.clock.advance_to(min(due))
    else:
        raise AssertionError("policy failure did not block the lane")

    assert ctrl.generations()[-1]["generation"] == 2
    old_ids = [row["experiment_id"] for row in ctrl.decisions()]
    assert old_ids and not ctrl.policy.applications()[-1]["generation"] == 2
    panel_path = rig.root / "controller.sqlite3.policy/panel/g000002.json"
    panel = panel_path.read_bytes()
    round_root = rig.root / "policy_harness/rounds/round_002"
    failed_result = (round_root / "result.json").read_bytes()
    failed_arrays = (round_root / "compiled.npz").read_bytes()
    outcome_path = rig.root / "controller.sqlite3.research/g000002.outcome.json"
    failed_outcome = outcome_path.read_bytes()
    assert json.loads(failed_result)["status"] == "runtime_fallback"

    if interruption == "transaction":
        ctrl.store.faults = CrashPlan("lane.after_commit")
    elif interruption:
        original_write = policy_module._write_atomic
        def interrupted_write(path, data):
            original_write(path, data)
            if path.name == interruption and "retries" in path.parts:
                raise SimulatedCrash("policy_retry.archive")
        monkeypatch.setattr(policy_module, "_write_atomic", interrupted_write)
    if interruption:
        with pytest.raises(SimulatedCrash):
            ctrl.resolve_lane("retry", "test")
        if interruption != "transaction":
            monkeypatch.setattr(policy_module, "_write_atomic", original_write)
        rig.close()
        ctrl = rig.open()
    else:
        assert ctrl.resolve_lane("retry", "test")

    rig.drive()
    assert ctrl.finished
    assert Counter(rig.panel.calls)[2] == 1 and panel_path.read_bytes() == panel
    turns = [turn for turn in seen_turns if turn[0] == 2]
    assert len(turns) == 2 and turns[0] == turns[1]
    assert [row["experiment_id"] for row in ctrl.decisions()][:len(old_ids)] == old_ids
    assert len(rig.simulator.launches) == len(set(rig.simulator.launches)) == 8
    archives = list((round_root / "retries").glob("event-*"))
    assert len(archives) == 1
    archive = archives[0]
    assert (archive / "outcome.json").read_bytes() == failed_outcome
    assert (archive / "result.json").read_bytes() == failed_result
    assert (archive / "compiled.npz").read_bytes() == failed_arrays
    manifest = json.loads((archive / "prepared.json").read_text())
    assert manifest["generation"] == 2
    assert all(file_sha256(archive / name) == sha for name, sha in manifest["files"].items())
    result = json.loads((round_root / "result.json").read_text())
    assert result["status"] == "accepted" and not result["degraded"]
    assert ctrl.policy.check() == []
