"""Batch panel contracts, durable generation budgets, catalogs and recovery."""
import http.client
import json
import socket
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from ldm_tts.engine.run_store import atomic_json_write
from ldm_tts.harness import HarnessHistoryConfig
from ldm_tts.harness.history import HistoryLog, HistoryProjection, HistoryRecord
from ldm_tts.harness.history_host import HistoryService
from ldm_tts.harness.pi_research import proposal_validator
from ldm_tts.harness.protocol import HarnessSubmissionRequest
from tasks.antibody_rfd3.core.harness import (AntibodyPiResearch, AntibodyResearchPanel, PANEL,
                                               RecordingClient, worker_socket)
from tasks.antibody_rfd3.core import harness
from tasks.antibody_rfd3.core.worker import DesignWorker
from tasks.antibody_rfd3.core.workflow import parse_args
from tasks.antibody_rfd3.tests.test_multi_research import FakeClient, Result, round_input


def batch_args():
    return parse_args(["--method", "ldm_harness", "--mock", "--iterations", "64", "--research-rounds", "2",
                       "--batch-size", "32", "--pool-size", "64", "--research-candidates-per-role", "32",
                       "--backbones-per-round", "16", "--sequences-per-round", "512", "--samples-per-call", "32"])


class BatchClient(FakeClient):
    """Exercise the real proposal validator and history without a provider or model."""

    def run_turn(self, turns, *, submission_validator, **kwargs):
        turn, = turns
        if turn.turn_id in self.commits:
            return super().run_turn(turns, submission_validator=submission_validator, **kwargs)
        self.sent_messages.append(turn.message)
        settings = json.loads(turn.message.rsplit("\n\nTask settings: ", 1)[1])
        self.history.handle(self.role, turn.turn_id, "query", {"view": "summary"}, turn.history_to_seq)
        if self.role == "evidence_reviewer":
            path = self.workspace / Path(settings["peer_catalog"]["path"]).relative_to("/workspace")
            rows = json.loads(path.read_text())["candidates"]
            ids = [rows[0]["design_id"]]
        else:
            rows, backbone = [], None
            while len(self.worker.eligible(rows)) < settings["candidate_count"]["minimum"]:
                left = self.worker.role_remaining(self.role)
                if left["sequences"] <= 0 or (backbone is None and left["backbones"] <= 0):
                    break
                count = min(self.worker.args.samples_per_call, left["sequences"])
                call_id = f"call-{len([j for j in self.worker.records() if j['round'] == self.worker.round and j.get('profile_id') == self.role])}"
                new = self.worker.execute("design_h3" if backbone is None else "sequence_h3", count, 0.5,
                                          backbone_id=backbone, call_id=call_id, profile_id=self.role)
                backbone = new[0]["backbone_id"]
                rows.extend(new)
            ids = [row["design_id"] for row in self.worker.eligible(rows)][:settings["candidate_count"]["maximum"]]
        submission = {"design_ids": ids}
        if self.role == "evidence_reviewer":
            submission.update(evidence="The catalog contains unmeasured proposals.", hypotheses="none")
        admission = self.history.admit(self.role, turn.turn_id, 1, self.history.log.head, 0)
        request = HarnessSubmissionRequest(self.role, turn.turn_id, 1, submission, history_seq=admission.history_seq)
        validation = submission_validator(request)
        accepted = validation.decision == "accept"
        self.history.record(self.role, turn.turn_id, "decision", attempt=1,
                            decision="accept" if accepted else "reject_turn", submission_digest=request.digest)
        result = Result(self.role, turn.turn_id, turn.round_index, turn.history_from_seq, turn.history_to_seq,
                        self.history.log.head, submission, submission_status="accepted" if accepted else "rejected",
                        validation_errors=validation.errors)
        self.commits[turn.turn_id] = (turn, result, admission.history_seq)
        return (result,)


def make_batch_panel(tmp_path):
    args = batch_args()
    worker = DesignWorker(tmp_path, args)
    service = HistoryService(HistoryLog(), HistoryProjection(()), record_path=tmp_path / "visibility.jsonl")
    root = tmp_path / "harness"
    sources, clients = {}, {}
    for role in PANEL:
        workspace = root / role / "sessions" / role / "workspace"
        workspace.mkdir(parents=True)
        client = BatchClient(role, service, worker, workspace)
        clients[role] = client
        sources[role] = AntibodyPiResearch(RecordingClient(client), service, worker, args, workspace=workspace)
    return AntibodyResearchPanel(sources, worker, args, root), clients, service


def test_batch_profile_schema_and_timeouts(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "https://example.invalid/v1/responses")
    monkeypatch.setenv("LLM_MODEL_NAME", "test-model")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    clients = []

    class Probe:
        def __init__(self, **_kwargs):
            pass

        def preflight(self):
            pass

    class Client:
        def __init__(self, _cmd, *, config, history, response_timeout_seconds, **_kwargs):
            self.config, self.history, self.timeout = config, history, response_timeout_seconds
            clients.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

    @contextmanager
    def fake_socket(_worker, *, workspaces):
        assert set(workspaces) == set(PANEL)
        yield tmp_path

    monkeypatch.setattr(harness, "OpenAICompatibleProposalClient", Probe)
    monkeypatch.setattr(harness, "HarnessClient", Client)
    monkeypatch.setattr(harness, "worker_socket", fake_socket)
    monkeypatch.setattr(harness, "pi_sidecar_command", lambda *args, **kwargs: ("sidecar",))
    service = SimpleNamespace(config=HarnessHistoryConfig(), lineage_of=lambda role: role)
    bridge = SimpleNamespace(service=service, add_client=lambda _client: None)
    args = batch_args()
    worker = DesignWorker(tmp_path, args)
    with harness.open_agent(args, tmp_path, worker, bridge, "batch"):
        for client in clients:
            role = client.config.profiles[0].profile_id
            schema = client.config.submission_contract.payload_schema["properties"]["design_ids"]
            if role == "evidence_reviewer":
                assert schema["minItems"] == 1 and schema["maxItems"] == 64
                assert client.config.limits.wall_time_seconds == 1800 and client.timeout == 2100
            else:
                # A partial list can report exhaustion, but the host requires exactly 32 to accept.
                assert schema["minItems"] == 0 and schema["maxItems"] == 32
                assert client.config.limits.wall_time_seconds == 9000 and client.timeout == 9300


def test_first_batch_has_two_exact_lineages_and_reviewer_is_advice_only(tmp_path):
    panel, clients, _service = make_batch_panel(tmp_path)
    frozen = round_input(1)
    result = panel.research(frozen, [])
    assert panel.version == "3" and all(source.version == "3" for source in panel.sources.values())
    assert result.status == "committed" and not result.metadata["bootstrap"]
    assert result.metadata["candidate_counts"] == {"explorer": 32, "refiner": 32}
    assert len(result.proposals) == len({p.candidate.canonical_key for p in result.proposals}) == 64
    assert {role: sum(p.lineage == role for p in result.proposals) for role in PANEL} == {
        "explorer": 32, "refiner": 32, "evidence_reviewer": 0}
    reviewer = json.loads((tmp_path / result.metadata["peer_catalogs"]["evidence_reviewer"]["record"]).read_text())
    assert len(reviewer["candidates"]) == 64
    assert len(result.metadata["review"]["design_ids"]) == 1
    assert {row["generation_job"]["profile_id"] for row in reviewer["candidates"]} == {"explorer", "refiner"}
    for role, client in clients.items():
        settings = json.loads(client.sent_messages[-1].rsplit("\n\nTask settings: ", 1)[1])
        expected = {"minimum": 1, "maximum": 64} if role == "evidence_reviewer" else {"minimum": 32, "maximum": 32}
        assert settings["candidate_count"] == expected
    jobs = list(panel.worker.records())
    before = [clients[role].sent_messages[-1] for role in PANEL]
    replayed = panel.research(frozen, [])
    assert replayed.status == "committed" and len(replayed.proposals) == 64
    assert list(panel.worker.records()) == jobs
    assert [clients[role].sent_messages[-1] for role in PANEL] == before
    assert all(session["replayed"] for session in replayed.metadata["sessions"].values())


def test_recovery_rejects_a_modified_peer_catalog(tmp_path):
    panel, _clients, _service = make_batch_panel(tmp_path)
    result = panel.research(round_input(1), [])
    source = panel.sources["refiner"]
    info = result.metadata["peer_catalogs"]["refiner"]
    assert len(source._catalog_ids(info)) == 32
    path = source.workspace / Path(info["path"]).relative_to("/workspace")
    path.write_text('{"candidates": []}\n')
    with pytest.raises(ValueError, match="Peer catalog differs"):
        source._catalog_ids(info)


def test_exact_contract_enforces_own_current_issuance_and_sequence_collisions(tmp_path):
    args = batch_args()
    worker = DesignWorker(tmp_path, args)
    explorer = worker.execute("design_h3", 128, 0.5, profile_id="explorer")
    peer = worker.eligible(explorer)[:32]
    assert len(peer) == 32
    refiner = worker.execute("design_h3", 128, 0.5, profile_id="refiner")
    receipt_path = worker.jobs / refiner[0]["backbone_id"] / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["candidates"][0]["sequence"] = peer[0]["sequence"]
    receipt["candidates"][0]["violations"] = []
    atomic_json_write(receipt_path, receipt)
    worker.evaluated = {row["sequence"] for row in peer}
    own = worker.eligible(receipt["candidates"])[:32]
    assert len(own) == 32
    source = object.__new__(AntibodyPiResearch)
    source.worker, source.args, source.required = worker, args, 32
    source.role, source.peer_ids = "refiner", {row["design_id"] for row in peer}
    log = HistoryLog()
    contract = source.make_contract()
    assert contract.count == (32, 32)
    validate = proposal_validator(contract, log, generation=1)

    def decision(ids, seq=0):
        return validate(HarnessSubmissionRequest("refiner", "g000001-refiner", 1,
                                                  {"design_ids": ids}, history_seq=seq))

    ids = [row["design_id"] for row in own]
    assert decision(ids).decision == "accept"
    assert decision(ids[:-1]).decision == "retry"
    assert decision(ids + [ids[0]]).decision == "retry"
    assert decision([ids[0]] * 32).decision == "retry"
    assert decision([peer[0]["design_id"], *ids[1:]]).decision == "retry"
    assert decision([refiner[0]["design_id"], *ids[1:]]).decision == "retry"  # different ID, peer sequence
    alias = next(row for row in receipt["candidates"] if row["design_id"] not in ids
                 and row["design_id"] != refiner[0]["design_id"])
    alias.update(sequence=own[0]["sequence"], violations=[])
    atomic_json_write(receipt_path, receipt)
    assert decision([ids[0], alias["design_id"], *ids[2:]]).decision == "retry"
    log.append([HistoryRecord("selected", "pending", own[0]["sequence"], "pending", "selected"),
                HistoryRecord("result", "measured", own[1]["sequence"], "measured", "succeeded")])
    assert {error.code for error in decision(ids, 2).errors} >= {"already_pending", "already_measured"}
    # An old ID cannot stand in for a new call, but a re-generated, unmeasured old sequence is eligible.
    worker.round = 1
    new = worker.execute("design_h3", 1, 0.5, profile_id="refiner")
    path = worker.jobs / new[0]["backbone_id"] / "receipt.json"
    receipt = json.loads(path.read_text())
    receipt["candidates"][0].update(sequence=own[2]["sequence"], violations=[])
    atomic_json_write(path, receipt)
    current = source.make_contract()
    assert current.admit(own[2]["design_id"]) is None
    assert current.admit(new[0]["design_id"]).canonical_key == own[2]["sequence"]


def test_budget_and_call_id_replay_are_durable_and_role_scoped(tmp_path):
    args = batch_args()
    worker = DesignWorker(tmp_path, args)
    assert worker.role_budget("explorer") == worker.role_budget("refiner") == {"backbones": 8, "sequences": 256}
    assert worker.role_budget("evidence_reviewer") == {"backbones": 0, "sequences": 0}
    first = worker.execute("design_h3", 256, 0.5, call_id="call", profile_id="explorer")
    second = worker.execute("design_h3", 256, 0.5, call_id="call", profile_id="refiner")
    assert first != second and worker.remaining()["sequences"] == 0
    recovered = DesignWorker(tmp_path, args)
    assert recovered.execute("design_h3", 256, 0.5, call_id="call", profile_id="explorer") == first
    assert len(list(recovered.records())) == 2
    with pytest.raises(RuntimeError, match="different parameters"):
        recovered.execute("design_h3", 255, 0.5, call_id="call", profile_id="explorer")
    with pytest.raises(ValueError, match="budget"):
        recovered.execute("design_h3", 1, 0.5, call_id="other", profile_id="explorer")
    with pytest.raises(ValueError, match="budget"):
        recovered.execute("design_h3", 1, 0.5, profile_id="evidence_reviewer")


class UnixConnection(http.client.HTTPConnection):
    def __init__(self, path):
        super().__init__("localhost")
        self.path = str(path)

    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.connect(self.path)


def test_tool_catalog_rebuilds_from_receipts_and_filters_peer_sequences(tmp_path, monkeypatch):
    args = batch_args()
    args.mock = False
    worker = DesignWorker(tmp_path, args)
    a, b, c = "AGSPDTYAGSP", "GSPDTYAGSPA", "PDTYAGSPTGA"
    batches = iter(([a, a, "AAAAAAAAAAA"], [a, b, c]))
    monkeypatch.setattr(worker, "run_models", lambda *_: next(batches))
    workspaces = {role: tmp_path / role for role in ("explorer", "refiner")}

    def request(path, role, call_id):
        connection = UnixConnection(path / "design.sock")
        connection.request("POST", "/", json.dumps({"action": "design_h3", "count": 3, "temperature": 0.5,
                                                      "profile_id": role, "call_id": call_id}))
        response = connection.getresponse()
        payload = json.loads(response.read())
        connection.close()
        assert response.status == 200, payload
        return payload

    with worker_socket(worker, workspaces=workspaces) as path:
        first = request(path, "explorer", "first")
        catalog = workspaces["explorer"] / "generated/g000001.json"
        snapshot = json.loads(catalog.read_text())
        assert first["candidate_catalog"] == "/workspace/generated/g000001.json"
        assert "candidates" not in first and first["total_eligible_unique_new"] == 1
        assert [r["sequence"] for r in snapshot["candidates"]] == [a]
        catalog.unlink()
        replay = request(path, "explorer", "first")
        assert json.loads(catalog.read_text()) == snapshot and replay == first
        assert len(list(worker.records())) == 1
        worker.evaluated.add(a)
        result = request(path, "refiner", "first")
        rows = json.loads((workspaces["refiner"] / "generated/g000001.json").read_text())["candidates"]
        assert {row["sequence"] for row in rows} == {b, c}
        assert result["total_eligible_unique_new"] == 2 and result["role_remaining"]["sequences"] == 253


def test_exhausted_generation_rejects_panel_and_records_shortfall(tmp_path, monkeypatch):
    panel, _clients, _service = make_batch_panel(tmp_path)
    monkeypatch.setattr(panel.worker, "eligible", lambda rows: [])
    outcome = panel.research(round_input(1), [])
    assert outcome.status == "rejected" and outcome.proposals == ()
    assert "0 eligible unique new candidates available" in outcome.reason
    assert "32 required" in outcome.reason and "explorer" in outcome.reason
    assert sum(job["count"] for job in panel.worker.records()) == 256


def test_abandoned_batch_settles_each_role_from_its_frozen_catalog(tmp_path, monkeypatch):
    panel, clients, _service = make_batch_panel(tmp_path)
    panel.research(round_input(1), [])
    reviewer = clients["evidence_reviewer"]
    original = reviewer.run_turn

    def interrupted(*_args, **_kwargs):
        raise RuntimeError("interrupted before reviewer submission")

    monkeypatch.setattr(reviewer, "run_turn", interrupted)
    with pytest.raises(RuntimeError, match="interrupted"):
        panel.research(round_input(2), [])
    monkeypatch.setattr(reviewer, "run_turn", original)
    recovered = panel.research(round_input(3, unsettled=(1, 2)), [])
    settled = recovered.metadata["settled"]
    assert recovered.status == "committed" and len(recovered.proposals) == 64
    assert settled["explorer"]["generation"] == settled["refiner"]["generation"] == 2
    assert settled["evidence_reviewer"]["generation"] == 1
    assert all(value["replayed"] for value in settled.values())
