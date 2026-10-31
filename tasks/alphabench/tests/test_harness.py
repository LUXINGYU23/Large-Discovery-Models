import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from ldm_tts.contracts import RawProposal
from ldm_tts.contracts.evaluation import EvaluationPaused, EvaluationResult, Observation
from ldm_tts.engine.expansion import ExpansionRequest
from ldm_tts.engine.run_store import CampaignRuntime
from ldm_tts.harness import (HarnessError, HarnessLimits, HarnessPoolConfig, HarnessProfile,
    HarnessSubmissionRequest, HarnessSubmittedArtifact)
from tasks.alphabench.core.candidate import FactorDomain
from tasks.alphabench.core.collection import AcceptedActions
from tasks.alphabench.core.gateway import OracleGateway
from tasks.alphabench.core.harness import HarnessExpander, submission_contract
from tasks.alphabench.core.protocol import T3Protocol
from tasks.alphabench.core.quality import audit_quality
from tasks.alphabench.core.reporting import generation_costs


class FakeHarnessClient:
    def __init__(self, root, runtime, batches):
        self.root = root
        self.batches = batches
        self.committed = None
        self.validations = []
        self.config = HarnessPoolConfig(artifact_root=root,
            profiles=tuple(HarnessProfile(name, root / name / "AGENTS.md", agents_sha256="a" * 64)
                           for name in batches),
            campaign_id=runtime.run_id, task_id="alphabench", case_id="csi300", seed=42,
            submission_contract=submission_contract(max(map(len, batches.values()))),
            limits=HarnessLimits(wall_time_seconds=1))

    def run_turn(self, turns, *, submission_validator, provider_authorizer, recovery_timeout_seconds):
        if self.committed is not None:
            assert tuple(turn.turn_id for turn in turns) == tuple(result.turn_id for result in self.committed)
            return self.committed
        results = []
        for turn in turns:
            body = json.dumps({"candidates": self.batches[turn.profile_id]}).encode()
            path = self.root / (turn.profile_id + ".json")
            path.write_bytes(body)
            artifact = HarnessSubmittedArtifact("/artifact_path", "candidates.json", path.name,
                                                hashlib.sha256(body).hexdigest(), len(body))
            submission = {"artifact_path": "candidates.json"}
            submission_digest = hashlib.sha256((turn.turn_id + body.decode()).encode()).hexdigest()
            request = HarnessSubmissionRequest(turn.profile_id, turn.turn_id, 1, submission,
                                               (artifact,), submission_digest)
            validation = submission_validator(request)
            self.validations.append(validation)
            if validation.decision != "accept":
                raise HarnessError("fixture submission rejected")
            results.append(SimpleNamespace(profile_id=turn.profile_id, session_id=turn.profile_id,
                turn_id=turn.turn_id, submission_status="accepted", submission_id="fixture-" + turn.profile_id,
                submission_digest=submission_digest, submission=submission, submitted_artifacts=(artifact,),
                usage={"providerCalls": 0}))
        self.committed = tuple(results)
        return self.committed


def fixture(tmp_path, method, batches):
    protocol = T3Protocol(method=method, cold_seed_count=0, sessions=len(batches),
                          candidates_per_session=len(next(iter(batches.values()))))
    runtime = CampaignRuntime.open(tmp_path / "run", task="alphabench", budget_limits={
        "model_requests": 0, "harness_turns": 4, "proposal_attempts": 4,
        "dynamic_checks": 20, "oracle_job_slots": 40, "benchmark_jobs": 20})
    gateway = OracleGateway(protocol, runtime, mock=True)
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    client = FakeHarnessClient(artifact_root, runtime, batches)
    return protocol, gateway, client, HarnessExpander(client, protocol, gateway, artifact_root)


def factor(name, window):
    return {"name": name, "expression": f"Mean($close,{window})"}


def test_direct_harness_uses_effective_tail_batch_and_replays_without_paid_checks(tmp_path):
    protocol, gateway, client, expander = fixture(tmp_path, "harness", {"research": [factor("a", 5)]})
    expander.collection = AcceptedActions(gateway.runtime.run_dir)
    request = ExpansionRequest(0, protocol.batch_size, context={"evaluation_budget": {"effective": 1}})
    first = gateway.host.run(lambda: expander.expand(request))
    assert first.selection_mode == "reservoir_order"
    assert len(first.proposals) == 1 and first.proposals[0].metadata["q0"] == 1
    assert gateway.runtime.budget.counters["dynamic_checks"] == 1
    second = gateway.host.run(lambda: expander.expand(request))
    assert second == first
    assert gateway.runtime.budget.counters["dynamic_checks"] == 1
    assert gateway.runtime.budget.counters["harness_turns"] == 1
    assert len(list(expander.collection.journal.root.glob("*.json"))) == 1
    audit = audit_quality(protocol, gateway.runtime, gateway)
    assert audit["raw_occurrences"] == 1 and audit["complete"]
    generations = [json.loads(path.read_text()) for path in (gateway.runtime.run_dir / "generation").glob("*.json")]
    cost = generation_costs(generations)
    assert cost["unit"] == "sidecar provider calls per committed Harness turn"
    assert cost["submission_attempts_per_step"] == [1]


def test_ldm_harness_preserves_cross_session_consensus_without_random_fill(tmp_path):
    protocol, gateway, _, expander = fixture(tmp_path, "ldm_harness", {
        "research_a": [factor("first", 5), factor("second", 6)],
        "research_b": [factor("agreement", 5), factor("third", 7)],
    })
    expander.collection = AcceptedActions(gateway.runtime.run_dir)
    request = ExpansionRequest(0, protocol.sessions * protocol.candidates_per_session,
                               context={"evaluation_budget": {"effective": 2}})
    result = gateway.host.run(lambda: expander.expand(request))
    assert result.selection_mode == "acquisition"
    assert result.metadata["occurrences"] == 4 and result.metadata["unique"] == 3
    assert [item.metadata["q0"] for item in result.proposals] == [.5, .25, .25]
    assert len(result.proposals[0].metadata["harness_lineage"]) == 2
    assert gateway.runtime.budget.counters["dynamic_checks"] == 3
    assert gateway.runtime.budget.counters["harness_turns"] == 2
    assert len(list(expander.collection.journal.root.glob("*.json"))) == 2
    audit = audit_quality(protocol, gateway.runtime, gateway)
    assert audit["raw_occurrences"] == 4 and audit["complete"]
    generations = [json.loads(path.read_text()) for path in (gateway.runtime.run_dir / "generation").glob("*.json")]
    assert generation_costs(generations)["submission_attempts_per_step"] == [1, 1]


def test_harness_rejects_evaluated_failures_and_same_session_duplicates(tmp_path):
    protocol, gateway, client, expander = fixture(tmp_path, "harness", {
        "research": [factor("failed_before", 5), factor("new", 6)],
    })
    candidate = FactorDomain().admit(RawProposal(factor("old", 5), "fixture"))
    failed = Observation(candidate, EvaluationResult(candidate.candidate_id, "invalid"))
    request = ExpansionRequest(0, protocol.batch_size, observations=(failed,),
                               context={"evaluation_budget": {"effective": 2}})
    with pytest.raises(EvaluationPaused, match="requires recovery"):
        gateway.host.run(lambda: expander.expand(request))
    assert client.validations[0].errors[0].code == "historical_duplicate"
    assert gateway.runtime.budget.counters["dynamic_checks"] == 1
    client.batches["research"] = [factor("new", 6), factor("same", 6)]
    with pytest.raises(EvaluationPaused, match="requires recovery"):
        gateway.host.run(lambda: expander.expand(request))
    assert client.validations[1].errors[0].code == "same_session_duplicate"
    assert gateway.runtime.budget.counters["dynamic_checks"] == 1


def test_harness_attempt_ledger_replays_after_accept_receipt_interruption(tmp_path, monkeypatch):
    protocol, gateway, _, expander = fixture(tmp_path, "harness", {"research": [factor("a", 5)]})
    body = json.dumps({"candidates": [factor("a", 5)]}).encode()
    path = expander.artifact_root / "research.json"
    path.write_bytes(body)
    artifact = HarnessSubmittedArtifact("/artifact_path", "candidates.json", path.name,
        hashlib.sha256(body).hexdigest(), len(body))
    submission = HarnessSubmissionRequest("research", "turn-1", 1,
        {"artifact_path": "candidates.json"}, (artifact,))
    original = expander.accepted.accept

    def interrupt(*_args):
        raise RuntimeError("interrupted after attempt ledger")

    monkeypatch.setattr(expander.accepted, "accept", interrupt)
    with pytest.raises(RuntimeError, match="interrupted"):
        gateway.host.run(lambda: expander._validate(submission, 1, set(), 0))
    assert expander.attempts.load(["turn-1", 1, submission.digest])["occurrences"][0]["status"] == "accepted"
    monkeypatch.setattr(expander.accepted, "accept", original)
    replay = gateway.host.run(lambda: expander._validate(submission, 1, set(), 0))
    assert replay.decision == "accept"
    assert gateway.runtime.budget.counters["dynamic_checks"] == 1
    assert expander.accepted.load(["turn-1", submission.digest]) is not None


def test_host_process_death_after_sidecar_commit_replays_without_provider_charge(tmp_path):
    fixture = Path(__file__).parents[3] / "tests/fixtures/fake_harness_sidecar.py"
    script = r'''
import json, os, sys
from pathlib import Path
from ldm_tts.engine.run_store import CampaignRuntime
from ldm_tts.harness import (HarnessClient, HarnessPoolConfig, HarnessProfile, HarnessSubmissionValidation, HarnessTurn)
from tasks.alphabench.core.harness import HarnessMeter, submission_contract
from tasks.alphabench.core.host import HostDispatcher

run, fixture = Path(sys.argv[1]), Path(sys.argv[2])
limits = {"model_requests": 1, "harness_turns": 1, "proposal_attempts": 1}
runtime = CampaignRuntime.open(run, task="alphabench", budget_limits=limits,
    resume=(run / "campaign.json").exists())
artifact_root = run / "harness"
artifact_root.mkdir(exist_ok=True)
config = HarnessPoolConfig(artifact_root=artifact_root,
    profiles=(HarnessProfile("research", Path("/resources/AGENTS.md"), agents_sha256="a" * 64),),
    campaign_id=runtime.run_id, task_id="alphabench", case_id="recovery", seed=0,
    submission_contract=submission_contract(1))
turn = HarnessTurn("research", "round-0-research", 0, 0, 0, "c" * 64, "research")
host = HostDispatcher()
meter = HarnessMeter(runtime, host)
os.environ.update(HARNESS_TEST_RELEASE="0.2.0", HARNESS_TEST_PROVIDER_AUTH="1",
    HARNESS_TEST_PROCESS_FAILURE="exit")
client = HarnessClient((sys.executable, "-u", str(fixture)), api_key="fixture",
    config=config, response_timeout_seconds=5)
original_request = client._request
marker = run / "host-crashed.marker"

def request(*args, **kwargs):
    result = original_request(*args, **kwargs)
    if args[0] == "run_turn" and not marker.exists():
        marker.write_text("after-sidecar-commit")
        os._exit(96)
    return result

client._request = request

def stage():
    meter.reserve_turns([turn.turn_id])
    with client:
        result, = client.run_turn((turn,),
            submission_validator=lambda _: HarnessSubmissionValidation(),
            provider_authorizer=meter.authorize, recovery_timeout_seconds=10)
        return {"replayed": result.replayed, "submission": result.submission,
            "usage": meter.reconcile(result.profile_id, result.turn_id, result.usage)}

print(json.dumps(host.run(stage)))
'''
    command = [sys.executable, "-c", script, str(tmp_path / "run"), str(fixture)]
    killed = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert killed.returncode == 96, killed.stderr
    resumed = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert resumed.returncode == 0, resumed.stderr
    result = json.loads(resumed.stdout)
    replayed = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert replayed.returncode == 0, replayed.stderr
    assert json.loads(replayed.stdout) == result
    assert result["replayed"] is True
    assert result["submission"] == {"candidates": [{"value": "research"}]}
    assert result["usage"] == {"host_authorizations": 1, "sidecar_provider_calls": 1}
    budget = json.loads((tmp_path / "run/budget.json").read_text())["counters"]
    assert budget == {"model_requests": 1, "harness_turns": 1, "proposal_attempts": 1}
    assert len(list((tmp_path / "run/private/harness/provider").glob("*.json"))) == 1


def test_harness_schema_rejection_stays_rejected_in_quality_audit(tmp_path):
    protocol, gateway, _, expander = fixture(tmp_path, "harness", {"research": [factor("a", 5)]})
    body = json.dumps({"candidates": [{"name": 12, "expression": "Mean($close,5)"}]}).encode()
    path = expander.artifact_root / "bad.json"
    path.write_bytes(body)
    artifact = HarnessSubmittedArtifact("/artifact_path", "candidates.json", path.name,
        hashlib.sha256(body).hexdigest(), len(body))
    submission = HarnessSubmissionRequest("research", "turn-invalid", 1,
        {"artifact_path": "candidates.json"}, (artifact,))
    result = gateway.host.run(lambda: expander._validate(submission, 1, set(), 0))
    assert result.errors[0].code == "invalid_candidate"
    audit = audit_quality(protocol, gateway.runtime, gateway)
    assert audit["raw_occurrences"] == 1
    assert audit["static_success_rate"] == 0
    assert audit["qlib_dynamic_success_rate"] == 0
    assert gateway.runtime.budget.counters.get("dynamic_checks", 0) == 0
