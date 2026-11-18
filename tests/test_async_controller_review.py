"""Regression tests for the controller-core review findings fixed in controller-pacing (items 1-7, 9; item 8 is
covered by the flaky hook in test_async_controller_faults)."""
import threading

import pytest

from async_controller_support import CAMPAIGN, KIND, Harness
from experiment_service.testing import request
from ldm_tts.async_runtime import EffectHook, ESClient, ESUnavailable, FixedProposer
from ldm_tts.async_runtime.es_client import EXPERIMENT_NOT_FOUND
from ldm_tts.contracts import Candidate


def test_item1_a_cancel_settling_in_the_same_step_does_not_stop_the_campaign(tmp_path):
    with Harness(tmp_path, budget=3, candidates=1, concurrency=1, pools={"default": 1}, sim={"duration": 500.0}) as h:
        h.es_check = False
        h.start_es()
        status, _, _ = request(h.es.base_url, "PUT", "/v1/experiments/blocker.1",
                               {"campaign_id": "other", "kind": KIND, "kind_version": "1", "spec": {"x": 99}})
        assert status == 201
        h.es.service.run_until_idle()  # the pool is busy with another campaign: our intent stays QUEUED
        h.open()
        h.drive(until=lambda: h.mirror(1) == "QUEUED")
        assert h.ctrl.request_cancel(h.row(1)["experiment_id"], "operator")
        h.ctrl.step()
        h.es.service.run_until_idle()
        h.ctrl.step()
        assert h.row(1)["budget_state"] == "REFUNDED" and h.ctrl.view().status == "active"
        h.drive(until=lambda: h.row(2) is not None)
        assert h.row(2)["canonical_key"] == "int:0"  # the refunded key is selected again


def test_item1_a_late_failure_with_retries_left_does_not_stop_the_campaign(tmp_path):
    from dataclasses import replace
    with Harness(tmp_path, budget=3, candidates=1, concurrency=1, sim={"failures": {"execution_error": 1.0}}) as h:
        h.config = replace(h.config, max_attempts_per_key=2)
        h.start_es()
        h.open()
        assert h.drive()
        keys = [r["canonical_key"] for r in h.ctrl.store.rows("SELECT canonical_key FROM experiments")]
        assert keys == ["int:0", "int:0"] and h.ctrl.view().stop_reason == "proposals_exhausted"


def test_item2_operator_actions_wake_the_run_loop(tmp_path):
    with Harness(tmp_path, budget=3, candidates=3, concurrency=1, sim={"duration": 100.0}) as h:
        h.start_es()
        h.open()
        h.drive(until=lambda: h.mirror(1) == "RUNNING")
        eid = h.row(1)["experiment_id"]
        actions = [lambda: h.ctrl.pause("op"), lambda: h.ctrl.resume("op"),
                   lambda: h.ctrl.request_cancel(eid, "op"), lambda: h.ctrl.close_campaign("op")]
        for action in actions:
            h.ctrl.waker.wait(0)
            action()
            assert h.ctrl.waker.wait(0), action


def test_item3_write_off_refuses_unacknowledged_intents(tmp_path):
    with Harness(tmp_path, budget=2, candidates=2, concurrency=1, sim={"duration": 5.0}) as h:
        h.start_es()
        h.transport.partitioned = True
        h.open()
        h.settle()
        row = h.row(1)
        assert row["submit_state"] == "PLANNED" and row["es_state"] is None
        with pytest.raises(ValueError, match="request_cancel"):
            h.ctrl.write_off(row["experiment_id"], "alice", "give up")
        assert h.ctrl.request_cancel(row["experiment_id"], "give up")
        h.transport.partitioned = False
        assert h.drive()
        assert all(launch[0] != row["experiment_id"] for launch in h.sim.launches)
        assert h.row(1)["budget_state"] == "REFUNDED" and h.row(1)["es_tombstone"] == 1


def test_item3_and_4_a_409_row_is_written_off_not_cancelled_and_never_revived(tmp_path):
    with Harness(tmp_path, budget=2, candidates=3, concurrency=1) as h:
        h.es_check = False
        h.start_es()
        request(h.es.base_url, "PUT", f"/v1/experiments/{CAMPAIGN}.d000001.s0",
                {"campaign_id": "other", "kind": KIND, "kind_version": "1", "spec": {"x": 7}})
        h.open()
        h.drive(until=lambda: h.row(1) is not None and h.row(1)["submit_state"] == "REJECTED")
        eid = h.row(1)["experiment_id"]
        assert h.row(1)["reject_status"] == 409
        with pytest.raises(ValueError, match="409"):
            h.ctrl.request_cancel(eid, "oops")
        assert h.ctrl.write_off(eid, "op", "id collision")
        assert not h.ctrl.retry_rejected(eid)  # a written-off row is never sent again
        h.ctrl.resume("op")
        assert h.drive()
        assert h.row(1)["submit_state"] == "REJECTED" and h.row(1)["attempts"] == 1


def test_item5_hook_retries_back_off_on_the_injected_clock(tmp_path):
    attempts = []

    def broken(event, view):
        attempts.append(h.clock.now())
        raise RuntimeError("down")

    hook = EffectHook("broken", ("experiment.planned",), broken)
    with Harness(tmp_path, budget=1, candidates=1, concurrency=1, hooks=[hook], sim={"duration": 100.0}) as h:
        h.start_es()
        h.open()
        assert h.drive()
        assert len(attempts) == 3
        assert attempts[1] - attempts[0] >= 1.0 and attempts[2] - attempts[1] >= 2.0  # backoff 1 s, 2 s
        assert h.ctrl.store.row("SELECT gave_up FROM hook_failures WHERE hook='broken'")["gave_up"] == 1


def test_item6_long_poll_thread_survives_a_closed_store(tmp_path):
    with Harness(tmp_path, budget=1, candidates=1, concurrency=1) as h:
        h.start_es()
        h.open()
        h.ctrl.store.close()
        worker = threading.Thread(target=h.ctrl._long_poll, args=(threading.Event(),), daemon=True)
        worker.start()
        worker.join(5)
        assert not worker.is_alive()


def test_item7_only_the_typed_es_problem_means_lost():
    def transport_for(status, body):
        return lambda method, url, data, headers, timeout: (status, {"content-type": "application/problem+json"}, body)

    typed = b'{"type":"%s","title":"Experiment not found","status":404}' % EXPERIMENT_NOT_FOUND.encode()
    assert ESClient("http://x", transport=transport_for(404, typed)).get_experiment("e1") is None
    for body in (b"", b"<html>not found</html>",
                 b'{"type":"urn:ldm-es:problem:route-not-found","title":"Not found","status":404}'):
        with pytest.raises(ESUnavailable):
            ESClient("http://x", transport=transport_for(404, body)).get_experiment("e1")


def test_item7_real_es_unknown_id_is_reported_lost(tmp_path):
    with Harness(tmp_path, budget=1, candidates=1, concurrency=1) as h:
        h.start_es()
        assert ESClient(h.es.base_url).get_experiment("nobody.d000001.s0") is None
        with pytest.raises(ESUnavailable):  # a wrong base path: the route itself is missing
            ESClient(h.es.base_url + "/wrong").get_experiment("nobody.d000001.s0")


def test_item9_infrastructure_failures_do_not_use_up_a_key(tmp_path):
    def failures(spec, attempt):
        return "infrastructure" if spec["x"] == 0 else None

    with Harness(tmp_path, budget=3, candidates=[Candidate("c0", 0, "int:0")], concurrency=1,
                 sim={"failures": failures}) as h:
        h.start_es()
        h.open(proposer=FixedProposer(h.candidates))
        assert h.drive()
        rows = h.ctrl.store.rows("SELECT canonical_key, es_state, es_failure_class, budget_state FROM experiments")
        assert [(r["canonical_key"], r["es_failure_class"], r["budget_state"]) for r in rows] == \
            [("int:0", "infrastructure", "CONSUMED")] * 3
        assert h.ctrl.view().stop_reason == "budget_exhausted"
        h.check(synced=True)
