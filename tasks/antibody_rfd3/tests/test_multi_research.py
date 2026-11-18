"""Exercise the panel, durable budgets and replay without a provider or model."""
import json
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from ldm_tts.harness.history import HistoryLog, HistoryProjection, HistoryRecord
from ldm_tts.harness.history_host import HistoryService
from ldm_tts.harness.pi_research import proposal_validator
from ldm_tts.harness.protocol import HarnessSubmissionRequest
from tasks.antibody_rfd3.core.harness import (AntibodyPiResearch, AntibodyResearchPanel, PANEL, RecordingClient,
                                               profile, submission_contract)
from tasks.antibody_rfd3.core.worker import DesignWorker
from tasks.antibody_rfd3.core.workflow import parse_args


def panel_args():
    return parse_args(["--method", "ldm_harness", "--mock", "--pool-size", "6"])


def test_panel_budget_is_shared_and_replay_is_per_role(tmp_path):
    worker = DesignWorker(tmp_path, panel_args())
    assert worker.role_budget("explorer") == {"backbones": 1, "sequences": 16}
    assert worker.role_budget("refiner") == worker.role_budget("evidence_reviewer") == {
        "backbones": 0, "sequences": 0}
    with pytest.raises(ValueError, match="Role generation budget"):
        worker.execute("design_h3", 16, 0.5, profile_id="refiner")
    initial = worker.execute("design_h3", 16, 0.5, call_id="same", profile_id="explorer")
    assert worker.execute("design_h3", 16, 0.5, call_id="same", profile_id="explorer") == initial
    worker.round = 1
    first = worker.execute("design_h3", 16, 0.5, call_id="same", profile_id="explorer")
    second = worker.execute("design_h3", 16, 0.5, call_id="same", profile_id="refiner")
    assert first != second
    assert worker.remaining() == {"backbones": 0, "sequences": 0}
    assert worker.role_remaining("explorer") == worker.role_remaining("refiner") == {
        "backbones": 0, "sequences": 0}
    with pytest.raises(ValueError, match="budget"):
        worker.execute("sequence_h3", 1, 0.5, backbone_id=first[0]["backbone_id"],
                       profile_id="evidence_reviewer")
    assert [j["profile_id"] for j in worker.records()] == ["explorer", "explorer", "refiner"]


def test_own_candidates_and_catalog_are_enforced(tmp_path):
    args = panel_args()
    worker = DesignWorker(tmp_path, args)
    worker.round = 1
    explorer_rows = worker.eligible(worker.execute("design_h3", 16, 0.5, profile_id="explorer"))
    own = explorer_rows[0]
    other = next(r for r in worker.execute("design_h3", 8, 0.5, profile_id="refiner") if not r["violations"])
    log = HistoryLog()
    source = object.__new__(AntibodyPiResearch)
    source.worker, source.args, source.required = worker, args, 3
    source.role, source.peer_ids = "refiner", None
    validator = proposal_validator(source.make_contract(), log, generation=2)
    def validate(*ids):
        return validator(HarnessSubmissionRequest("refiner", "g000002-refiner", 1,
                                                  {"design_ids": list(ids)}, history_seq=0)).decision
    assert validate(other["design_id"]) == "accept"
    assert validate(own["design_id"]) == "retry"
    source.role, source.peer_ids = "evidence_reviewer", {row["design_id"] for row in explorer_rows[:4]}
    contract = source.make_contract()
    assert contract.count == (1, 3)
    validator = proposal_validator(contract, log, generation=2)
    assert validate(own["design_id"]) == "accept"
    assert len(source.peer_ids) == 4
    assert validate(*(row["design_id"] for row in explorer_rows[:3])) == "accept"
    assert validate(*(row["design_id"] for row in explorer_rows[:4])) == "retry"
    assert validate(other["design_id"]) == "retry"
    schema = submission_contract(6, review=True).payload_schema
    assert set(schema["required"]) == {"design_ids", "evidence", "hypotheses"}


class FakeClient:
    """A provider/sidecar substitute that still calls the real host proposal validator."""

    def __init__(self, role, service, worker, workspace):
        self.role, self.history, self.worker, self.workspace = role, service, worker, workspace
        self.config = SimpleNamespace(profiles=(profile(role),))
        self.commits = {}
        self.sent_messages = []

    def run_turn(self, turns, *, submission_validator, **_kwargs):
        turn, = turns
        self.sent_messages.append(turn.message)
        if turn.turn_id in self.commits:
            original_turn, result, seq = self.commits[turn.turn_id]
            assert turn.input_digest == original_turn.input_digest
            request = HarnessSubmissionRequest(self.role, turn.turn_id, 1, result.submission, history_seq=seq)
            assert submission_validator(request).decision == "accept"
            return (replace(result, replayed=True),)
        settings = json.loads(turn.message.rsplit("\n\nTask settings: ", 1)[1])
        self.history.handle(self.role, turn.turn_id, "query", {"view": "summary"}, turn.history_to_seq)
        if self.role == "explorer" or (self.role == "refiner" and turn.round_index > 1):
            count = 16
            rows = self.worker.execute("design_h3", count, 0.5, call_id="same-call", profile_id=self.role)
            ids = [row["design_id"] for row in self.worker.eligible(rows)][:settings["candidate_count"]["maximum"]]
        else:
            path = self.workspace / Path(settings["peer_catalog"]["path"]).relative_to("/workspace")
            catalog = json.loads(path.read_text())
            ids = [row["design_id"] for row in catalog["candidates"]][:settings["candidate_count"]["maximum"]]
        submission = {"design_ids": ids}
        if self.role == "evidence_reviewer":
            submission.update(evidence="Catalog proposals and history, with no invented energy.", hypotheses="none")
        admission = self.history.admit(self.role, turn.turn_id, 1, self.history.log.head, 0)
        request = HarnessSubmissionRequest(self.role, turn.turn_id, 1, submission, history_seq=admission.history_seq)
        validation = submission_validator(request)
        assert validation.decision == "accept", validation.errors
        self.history.record(self.role, turn.turn_id, "decision", attempt=1, decision="accept",
                            submission_digest=request.digest)
        result = Result(self.role, turn.turn_id, turn.round_index, turn.history_from_seq, turn.history_to_seq,
                        self.history.log.head, submission)
        self.commits[turn.turn_id] = (turn, result, admission.history_seq)
        return (result,)


@dataclass(frozen=True)
class Result:
    profile_id: str
    turn_id: str
    round_index: int
    history_from_seq: int
    history_to_seq: int
    visible_seq: int
    submission: dict
    submission_status: str = "accepted"
    replayed: bool = False
    usage: dict = None
    task_messages: tuple = ()
    validation_errors: tuple = ()

    def __post_init__(self):
        object.__setattr__(self, "usage", {"providerCalls": 1})


def make_panel(tmp_path):
    args = panel_args()
    worker = DesignWorker(tmp_path, args)
    log = HistoryLog()
    service = HistoryService(log, HistoryProjection(()), record_path=tmp_path / "visibility.jsonl")
    root = tmp_path / "harness"
    sources, clients = {}, {}
    for role in PANEL:
        workspace = root / role / "sessions" / role / "workspace"
        workspace.mkdir(parents=True)
        client = FakeClient(role, service, worker, workspace)
        clients[role] = client
        sources[role] = AntibodyPiResearch(RecordingClient(client), service, worker, args, workspace=workspace)
    return AntibodyResearchPanel(sources, worker, args, root), clients, service


def round_input(generation, *, base=0, sessions=None, unsettled=()):
    return {"generation": generation, "watermarks": {"base_history_seq": base},
            "sessions": sessions or {}, "unsettled": unsettled}


def test_baseline_settings_and_contract_require_one_candidate(tmp_path):
    args = parse_args(["--method", "harness", "--mock", "--pool-size", "6"])
    worker = DesignWorker(tmp_path, args)
    service = HistoryService(HistoryLog(), HistoryProjection(()))
    client = SimpleNamespace(config=SimpleNamespace(profiles=(profile(),)), history=service)
    source = AntibodyPiResearch(client, service, worker, args)
    assert source.version == "3"
    settings = json.loads(source.message(round_input(1), {}).rsplit("\n\nTask settings: ", 1)[1])
    assert "required_candidates" not in settings
    assert settings["candidate_count"] == {"minimum": 1, "maximum": 1}
    assert source.contract.count == (1, 1)
    assert source.contract.submission.payload_schema["properties"]["design_ids"]["maxItems"] == 1


def test_panel_bootstrap_review_and_exact_generation_replay(tmp_path):
    panel, clients, service = make_panel(tmp_path)
    assert panel.version == "3"
    first = panel.research(round_input(1), [])
    assert first.status == "committed" and first.metadata["bootstrap"]
    for client in clients.values():
        settings = json.loads(client.sent_messages[-1].rsplit("\n\nTask settings: ", 1)[1])
        assert "required_candidates" not in settings
        assert settings["candidate_count"] == {"minimum": 1, "maximum": 1}
    assert set(first.metadata["sessions"]) == set(PANEL)
    assert len({p.candidate.canonical_key for p in first.proposals}) == 1
    assert {p.lineage for p in first.proposals} == set(PANEL)
    assert [j["profile_id"] for j in panel.worker.records()] == ["explorer"]
    reviewed = json.loads((tmp_path / first.metadata["peer_catalogs"]["evidence_reviewer"]["record"]).read_text())
    assert reviewed["candidates"][0]["proposed_by"] == ["explorer", "refiner"]
    assert first.metadata["review"]["design_ids"] == [reviewed["candidates"][0]["design_id"]]

    sequence = first.proposals[0].candidate.canonical_key
    service.log.append([HistoryRecord("selected", "initial", sequence, "explorer", "selected", source="event:1"),
                        HistoryRecord("result", "initial", sequence, "explorer", "succeeded", source="event:2")])
    observed = [SimpleNamespace(canonical_key=sequence)]
    frozen = round_input(2, base=service.log.head, sessions=first.metadata["sessions"])
    second = panel.research(frozen, observed)
    assert set(second.metadata["sessions"]) == set(PANEL)
    assert [j["profile_id"] for j in panel.worker.records()] == ["explorer", "explorer", "refiner"]
    reviewer_input = json.loads((tmp_path / second.metadata["peer_catalogs"]["evidence_reviewer"]["record"]).read_text())
    for role, client in clients.items():
        settings = json.loads(client.sent_messages[-1].rsplit("\n\nTask settings: ", 1)[1])
        assert "required_candidates" not in settings
        maximum = min(6, len(reviewer_input["candidates"])) if role == "evidence_reviewer" else 3
        assert settings["candidate_count"] == {"minimum": 1, "maximum": maximum}
    assert {row["generation_job"]["profile_id"] for row in reviewer_input["candidates"]} == {
        "explorer", "refiner"}
    assert set(second.metadata["review"]["design_ids"]) <= {r["design_id"] for r in reviewer_input["candidates"]}
    before = [client.sent_messages[-1] for client in clients.values()]
    replayed = panel.research(frozen, observed)
    assert len(list(panel.worker.records())) == 3
    assert [client.sent_messages[-1] for client in clients.values()] == before
    assert all(s["replayed"] for s in replayed.metadata["sessions"].values())


def test_rejected_review_rejects_the_entire_panel(tmp_path, monkeypatch):
    panel, clients, _service = make_panel(tmp_path)
    reviewer = clients["evidence_reviewer"]
    run_turn = reviewer.run_turn

    def reject_review(*args, **kwargs):
        result, = run_turn(*args, **kwargs)
        return (replace(result, submission_status="rejected",
                        validation_errors=(SimpleNamespace(code="review_rejected"),)),)

    monkeypatch.setattr(reviewer, "run_turn", reject_review)
    outcome = panel.research(round_input(1), [])
    assert outcome.status == "rejected" and not outcome.proposals
    assert "evidence_reviewer" in outcome.reason and "review_rejected" in outcome.reason


def test_settle_each_role_from_its_own_latest_terminal_turn(tmp_path, monkeypatch):
    panel, clients, service = make_panel(tmp_path)
    panel.research(round_input(1), [])
    reviewer = clients["evidence_reviewer"]
    run_turn = reviewer.run_turn
    def fail_before_decision(*_args, **_kwargs):
        raise RuntimeError("simulated interruption before reviewer decision")
    monkeypatch.setattr(reviewer, "run_turn", fail_before_decision)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        panel.research(round_input(2), [])
    monkeypatch.setattr(reviewer, "run_turn", run_turn)
    third = panel.research(round_input(3, unsettled=(1, 2)), [])
    settled = third.metadata["settled"]
    assert settled["explorer"]["generation"] == settled["refiner"]["generation"] == 2
    assert settled["evidence_reviewer"]["generation"] == 1
    assert all(value["replayed"] for value in settled.values())
