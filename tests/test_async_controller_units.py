"""Controller building blocks: wire digest, store, run lock, frozen config, ES client, codec mapping."""
import ast
import json
from pathlib import Path

import pytest

from async_controller_support import CAMPAIGN, KIND, Harness, ToyCodec
from experiment_service import SimulatedExecutor, VirtualClock, digest as es_digest
from experiment_service.testing import running_service
from ldm_tts.async_runtime import (CampaignConfig, CampaignConfigMismatch, CrashPlan, DedupOnlySelector, ESBackpressure,
                                   ESClient, ESRejected, ESUnavailable, FaultyTransport, RunLockHeld, SimulatedCrash,
                                   Store, UrllibTransport, expected_status)
from ldm_tts.async_runtime.es_client import parse_retry_after
from ldm_tts.async_runtime.store import append_event
from ldm_tts.async_runtime.wire import canonical, digest, experiment_id_for, revision_number
from ldm_tts.contracts import Candidate

ROOT = Path(__file__).resolve().parents[1]


def test_controller_never_imports_experiment_service():
    for path in sorted((ROOT / "ldm_tts" / "async_runtime").glob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else \
                [node.module or ""] if isinstance(node, ast.ImportFrom) else []
            assert not any(name.split(".")[0] == "experiment_service" for name in names), path


def test_digest_matches_the_es_algorithm():
    values = [{"x": 42}, {"b": [1, 2.5, None, True], "a": {"z": "é", "y": "日本"}}, {"x": 1e-12, "y": -0.0},
              {"nested": {"k": [{"q": 1}, {"p": " "}]}}]
    for value in values:
        assert digest(value) == es_digest(value)
    assert canonical({"b": 1, "a": 2}) == '{"a":2,"b":1}'
    assert experiment_id_for("c7f3", 412) == "c7f3.d000412.s0"
    assert revision_number("0000000000001873") == 1873
    with pytest.raises(ValueError):
        revision_number("12a")


def test_store_transactions_roll_back_at_crash_points_and_events_are_append_only(tmp_path):
    store = Store(tmp_path / "s.sqlite3", faults=CrashPlan("t.before_commit"))
    with pytest.raises(SimulatedCrash):
        with store.tx("t") as db:
            append_event(db, 0.0, "x")
    assert store.value("SELECT COUNT(*) FROM events") == 0
    store.faults = CrashPlan("t.after_commit")
    with pytest.raises(SimulatedCrash):
        with store.tx("t") as db:
            append_event(db, 0.0, "x")
    assert store.value("SELECT COUNT(*) FROM events") == 1
    store.faults = None
    with pytest.raises(Exception, match="append-only"):
        with store.tx("t") as db:
            db.execute("UPDATE events SET type='y'")
    assert store.value("SELECT journal_mode FROM pragma_journal_mode") == "wal"
    store.close()


def test_run_lock_and_frozen_configuration(tmp_path):
    with Harness(tmp_path) as h:
        h.start_es()
        h.open()
        with pytest.raises(RunLockHeld):
            h2 = Harness(tmp_path)
            h2.open()
        h.crash()
        changed = CampaignConfig(CAMPAIGN, budget=6, concurrency=3)
        with pytest.raises(CampaignConfigMismatch, match="campaign.concurrency"):
            h.open(config=changed)

        class OtherCodec(ToyCodec):
            version = "2"
        with pytest.raises(CampaignConfigMismatch, match="plugins.codec"):
            h.open(codec=OtherCodec())

        class OtherSelector(DedupOnlySelector):
            name = "other"
        with pytest.raises(CampaignConfigMismatch, match="plugins.selector"):
            h.open(selector=OtherSelector())
        h.open(selector=OtherSelector(), collect_only=True)  # collect-only does not decide: selector not compared
        h.crash()
        h.open()  # the lock was released by every failed open
        assert h.ctrl.view().status == "active"


def test_es_client_classifies_every_answer(tmp_path):
    clock = VirtualClock()
    sim = SimulatedExecutor(clock, kind=KIND)
    with running_service([sim], pools={"default": 1}, clock=clock, max_queued_per_campaign=1,
                         retry_after_seconds=7) as es:
        client = ESClient(es.base_url)
        body = lambda x, **extra: json.dumps({"campaign_id": "c", "kind": KIND, "kind_version": "1",
                                               "spec": {"x": x}, **extra}).encode()
        assert client.put_experiment("c.d000001.s0", body(1))[0] == 201
        status, resource = client.put_experiment("c.d000001.s0", body(1))
        assert status == 200 and resource["state"] == "QUEUED"
        with pytest.raises(ESRejected) as conflict:
            client.put_experiment("c.d000001.s0", body(2))
        assert conflict.value.status == 409
        with pytest.raises(ESRejected) as bad:
            client.put_experiment("c.d000002.s0", body(2, spec_digest="sha256:00"))
        assert bad.value.status == 422
        with pytest.raises(ESBackpressure) as busy:
            client.put_experiment("c.d000003.s0", body(3))
        assert busy.value.retry_after == 7.0
        assert client.get_experiment("c.d000404.s0") is None
        status, tombstone = client.cancel_experiment("c.d000009.s0", "c")
        assert status == 200 and tombstone["tombstone"] and tombstone["campaign_id"] == "c"
        page = client.changes("c", "")
        assert [i["experiment_id"] for i in page["items"]] == ["c.d000001.s0", "c.d000009.s0"]
        assert client.changes("c", page["cursor"])["items"] == []
        assert client.kind(KIND)["kind_version"] == "1"
        faulty = FaultyTransport(UrllibTransport())
        faulty.add("status", status=503)
        with pytest.raises(ESUnavailable):
            ESClient(es.base_url, transport=faulty).get_experiment("c.d000001.s0")
    with pytest.raises(ESUnavailable):
        ESClient("http://127.0.0.1:9", timeout=1.0).get_experiment("x")
    assert parse_retry_after("3", 5.0) == 3.0
    assert parse_retry_after("soon", 5.0) == 5.0
    assert parse_retry_after("Wed, 21 Oct 2015 07:28:00 GMT", 5.0, now=1445412470.0) == 10.0


def test_status_mapping_follows_design_9_1():
    assert expected_status({"state": "SUCCEEDED"}) == "succeeded"
    assert expected_status({"state": "FAILED", "failure": {"class": "infeasible"}}) == "invalid"
    assert expected_status({"state": "FAILED", "failure": {"class": "timeout"}}) == "timed_out"
    assert expected_status({"state": "FAILED", "failure": {"class": "execution_error"}}) == "failed"
    assert expected_status({"state": "FAILED", "failure": {"class": "infrastructure"}}) == "failed"
    assert expected_status({"state": "CANCELED"}) is None
    candidate = Candidate("c", 3, "int:3")
    result = ToyCodec().decode(candidate, {"state": "SUCCEEDED", "result": {"value": 0.5},
                                           "usage": {"runtime_seconds": 4.0, "unit_seconds": 4.0, "pool": "default"}})
    assert result.status == "succeeded" and result.metrics == {"objective": 0.5}
    assert result.resource_usage == {"runtime_seconds": 4.0, "unit_seconds": 4.0}
    failed = ToyCodec().decode(candidate, {"state": "FAILED", "failure": {"class": "timeout", "message": "late"}})
    assert failed.status == "timed_out" and failed.error == "late" and failed.metrics == {}
    with pytest.raises(ValueError):
        ToyCodec().decode(candidate, {"state": "CANCELED"})
