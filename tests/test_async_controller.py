"""Controller behaviour against a real ES (HTTP) with a simulated executor (design 2.2, 4.2-4.4, 5.2, 5.9)."""
import threading

import pytest

from async_controller_support import CAMPAIGN, KIND, MANIFEST_SEEN, Harness, ToyCodec
from experiment_service import RealClock as ESRealClock, SimulatedExecutor
from experiment_service.testing import request, running_service
from ldm_tts.async_runtime import (ActionHook, CampaignConfig, Controller, EffectHook, ESClient, ExperimentRequest,
                                   FixedProposer, RealClock, RuntimeOptions, check_invariants, task_hooks)
from ldm_tts.async_runtime.events import hook_outputs
from ldm_tts.contracts import ObjectiveSet, ObjectiveSpec
from ldm_tts.registration import parse_async_manifest


def anomaly_types(h):
    return sorted(a["type"] for a in h.ctrl.anomalies())


def test_fixed_proposer_runs_to_budget_without_duplicates(tmp_path):
    sim = {"seed": 3, "failures": {"execution_error": 0.2, "infeasible": 0.15}}
    with Harness(tmp_path, budget=8, candidates=12, concurrency=3, sim=sim) as h:
        h.start_es()
        h.open()
        assert h.drive()
        view = h.ctrl.view()
        assert (view.status, view.stop_reason, view.outstanding) == ("stopped", "budget_exhausted", 0)
        h.check(synced=True)
        items = h.es_items()
        keys = [item["client"]["candidate_key"] for item in items]
        assert len(items) == 8 and len(set(keys)) == 8
        assert len(h.sim.launches) == len(set(h.sim.launches)) == 8
        budget = h.ctrl.budget()
        assert (budget.consumed, budget.held, budget.refunded, budget.lower, budget.upper) == (8, 0, 0, 8, 8)
        observations = h.ctrl.observations()
        assert len(observations) == sum(item["state"] in ("SUCCEEDED", "FAILED") for item in items) == 8
        assert [o.metadata["ingest_seq"] for o in observations] == list(range(1, 9))
        revisions = [int(o.metadata["es"]["revision"]) for o in observations]
        assert revisions == sorted(revisions)  # ingest order is completion order, not submission order
        by_id = {item["experiment_id"]: item for item in items}
        statuses = set()
        for o in observations:
            item = by_id[o.experiment_id]
            assert o.decision_id == item["client"]["decision_id"] and o.canonical_key == item["client"]["candidate_key"]
            statuses.add(o.evaluation.status)
            if item["state"] == "SUCCEEDED":
                assert o.evaluation.metrics == {"objective": item["result"]["value"]}
        assert "succeeded" in statuses and statuses - {"succeeded"}
        assert h.ctrl.observations(after_seq=6)[0].metadata["ingest_seq"] == 7
        assert anomaly_types(h) == []


def test_short_queue_and_outstanding_limits_hold_at_every_step(tmp_path):
    peaks = {"unstarted": 0, "outstanding": 0}

    def watch(h):
        view = h.ctrl.view()
        peaks["unstarted"] = max(peaks["unstarted"], view.unstarted)
        peaks["outstanding"] = max(peaks["outstanding"], view.outstanding)

    with Harness(tmp_path, budget=7, candidates=9, concurrency=2, buffer_depth=1, pools={"default": 1}) as h:
        h.on_step = watch
        h.start_es()
        h.open()
        assert h.drive()
        h.check(synced=True)
    assert peaks == {"unstarted": 1, "outstanding": 2}  # pool 1: one running plus the one-deep buffer


def test_cancel_of_queued_refunds_and_of_running_consumes(tmp_path):
    with Harness(tmp_path, budget=4, candidates=6, concurrency=2, buffer_depth=1, pools={"default": 1}) as h:
        h.start_es()
        h.open()
        h.drive(until=lambda: h.mirror(2) == "QUEUED" and h.mirror(1) == "RUNNING")
        assert h.ctrl.request_cancel(h.row(2)["experiment_id"], "test")
        assert not h.ctrl.request_cancel(h.row(2)["experiment_id"], "again")
        h.drive(until=lambda: h.mirror(2) == "CANCELED")
        assert h.row(2)["budget_state"] == "REFUNDED" and h.row(2)["pending"] == 0
        assert h.ctrl.request_cancel(h.row(1)["experiment_id"], "test")
        h.drive(until=lambda: h.mirror(1) == "CANCELED")
        assert h.row(1)["budget_state"] == "CONSUMED" and h.row(1)["disposition"] == "no_result"
        assert h.drive()
        h.check(synced=True)
        assert h.ctrl.budget().to_dict() == {"limit": 4, "held": 0, "consumed": 4, "refunded": 1, "started": 4,
                                                  "free": 0, "lower": 4, "upper": 4}
        keys = [r["canonical_key"] for r in h.ctrl.store.rows("SELECT canonical_key FROM experiments ORDER BY decision_id")]
        assert keys.count(h.row(2)["canonical_key"]) == 2  # a never-started cancel does not block the key
        assert len(h.ctrl.observations()) == 3 and anomaly_types(h) == []


def test_cancel_losing_to_completion_is_ingested_with_a_flag(tmp_path):
    with Harness(tmp_path, budget=2, candidates=2, concurrency=1, sim={"cancel_running": "none"}) as h:
        h.start_es()
        h.open()
        h.drive(until=lambda: h.mirror(1) == "RUNNING")
        h.ctrl.request_cancel(h.row(1)["experiment_id"], "too slow")
        h.drive(until=lambda: h.row(1)["cancel_acked_at"] is not None)
        assert h.row(1)["es_cancel_requested_at"] is not None and h.mirror(1) == "RUNNING"
        assert h.row(1)["budget_state"] == "HOLD"  # a 202 never settles anything
        assert h.drive()
        first = h.ctrl.observations()[0]
        assert first.experiment_id == h.row(1)["experiment_id"]
        assert first.metadata["flags"] == ["after_cancel_request"] and first.evaluation.status == "succeeded"
        h.check(synced=True)


def test_write_off_releases_pending_but_not_budget_or_key(tmp_path):
    # pool 2 > concurrency 1: the written-off experiment still occupies one ES slot (physical occupancy is the ES's)
    with Harness(tmp_path, budget=3, candidates=5, concurrency=1, pools={"default": 2}) as h:
        h.start_es()
        h.open()
        h.drive(until=lambda: h.mirror(1) == "RUNNING")
        first = h.row(1)["experiment_id"]
        h.sim.set_unreachable(f"sim:{first}#1")
        h.drive(until=lambda: h.row(1)["es_condition"] == "unreachable")
        with pytest.raises(ValueError):
            h.ctrl.write_off(first, "", "")
        assert h.ctrl.write_off(first, "alice", "node lost")
        assert h.ctrl.view().outstanding == 0 and h.ctrl.budget().consumed == 1
        assert h.drive()  # two more experiments; budget exhausted while #1 is still RUNNING at the ES
        assert h.ctrl.view().stop_reason == "budget_exhausted"
        assert h.row(1)["canonical_key"] not in [h.row(d)["canonical_key"] for d in (2, 3)]
        h.sim.set_unreachable(f"sim:{first}#1", False)
        h.drive(until=lambda: h.mirror(1) == "SUCCEEDED")
        late = [o for o in h.ctrl.observations() if o.experiment_id == first]
        assert late and late[0].metadata["flags"] == ["after_write_off"]
        assert h.ctrl.budget().consumed == 3 and h.row(1)["pending"] == 0
        h.check(synced=True)


def test_unexpected_cancel_and_unknown_experiment_are_anomalies(tmp_path):
    with Harness(tmp_path, budget=5, candidates=8, concurrency=2, buffer_depth=1, pools={"default": 1}) as h:
        h.start_es()
        h.open()
        h.drive(until=lambda: h.mirror(2) == "QUEUED")
        h.es.service.cancel(h.row(2)["experiment_id"], {"campaign_id": CAMPAIGN})
        h.drive(until=lambda: h.mirror(2) == "CANCELED")
        assert anomaly_types(h) == ["unexpected_cancel"] and h.ctrl.view().paused == ()
        assert h.row(2)["budget_state"] == "REFUNDED"
        h.es_check = False
        status, _, _ = request(h.es.base_url, "PUT", "/v1/experiments/intruder.1",
                               {"campaign_id": CAMPAIGN, "kind": KIND, "kind_version": "1", "spec": {"x": 99}})
        assert status == 201
        h.drive(until=lambda: "unknown_experiment" in anomaly_types(h))
        decided = h.ctrl.store.value("SELECT COUNT(*) FROM decisions")
        h.drive(until=lambda: h.ctrl.view().outstanding == 0 and h.es_states().get("intruder.1") == "SUCCEEDED")
        assert h.ctrl.store.value("SELECT COUNT(*) FROM decisions") == decided  # paused: no new decisions
        assert not h.ctrl.finished
        assert h.ctrl.resume("operator")
        assert h.drive()
        assert h.ctrl.view().paused == ()  # the open anomaly does not re-pause on the next anti-entropy pass
        assert h.ctrl.budget().consumed == 5
        assert check_invariants(h.ctrl.store, h.config) == []


def test_lost_experiment_is_detected_and_pauses(tmp_path):
    with Harness(tmp_path, budget=3, candidates=3, concurrency=1, sim={"duration": 100.0}) as h:
        h.start_es()
        h.open()
        h.drive(until=lambda: h.mirror(1) == "RUNNING")
        h.stop_es()
        h.es_db = tmp_path / "es-replaced" / "es.sqlite3"  # an ES that lost its database
        h.es_check = False
        h.start_es()
        h.drive(until=lambda: "lost_experiment" in anomaly_types(h))
        view = h.ctrl.view()
        assert view.paused and view.outstanding == 1 and view.budget.held == 1


def test_conflicting_submission_is_rejected_and_pauses_dispatch(tmp_path):
    with Harness(tmp_path, budget=4, candidates=6, concurrency=1) as h:
        h.es_check = False  # the ES legitimately holds a different experiment under our first id
        h.start_es()
        status, _, _ = request(h.es.base_url, "PUT", f"/v1/experiments/{CAMPAIGN}.d000001.s0",
                               {"campaign_id": "other", "kind": KIND, "kind_version": "1", "spec": {"x": 999}})
        assert status == 201  # e.g. a restored backup of another campaign reused our id space
        h.open()
        h.drive(until=lambda: "put_rejected" in anomaly_types(h))
        assert h.row(1)["submit_state"] == "REJECTED" and h.ctrl.view().paused
        h.drive(until=lambda: h.clock.now() > 200)
        assert h.ctrl.store.value("SELECT COUNT(*) FROM decisions") == 1  # paused, and the 409 is never retried
        assert h.ctrl.store.value("SELECT attempts FROM experiments WHERE decision_id=1") == 1
        assert h.ctrl.write_off(h.row(1)["experiment_id"], "operator", "id collision")
        h.ctrl.resume("operator")
        assert h.drive()
        assert h.ctrl.budget().consumed == 4 and len(h.ctrl.observations()) == 3
        assert check_invariants(h.ctrl.store, h.config) == []
        assert anomaly_types(h) == ["put_rejected"]


def test_rejected_request_can_be_closed_with_a_tombstone(tmp_path):
    class BadLimits(ToyCodec):
        def to_request(self, candidate):
            limits = {"max_runtime_seconds": -1 if candidate.payload == 0 else 1000}
            return ExperimentRequest(KIND, "1", {"x": candidate.payload}, limits)

    with Harness(tmp_path, budget=3, candidates=4, concurrency=1) as h:
        h.start_es()
        h.open(codec=BadLimits())
        h.drive(until=lambda: h.row(1) is not None and h.row(1)["submit_state"] == "REJECTED")
        assert "put_rejected" in anomaly_types(h)
        assert h.ctrl.request_cancel(h.row(1)["experiment_id"], "invalid limits")
        h.drive(until=lambda: h.mirror(1) == "CANCELED")
        row = h.row(1)
        assert (row["submit_state"], row["es_tombstone"], row["budget_state"]) == ("ACKED", 1, "REFUNDED")
        h.ctrl.resolve_anomaly(h.ctrl.anomalies()[0]["id"], "operator")
        h.ctrl.resume("operator")
        h.crash()
        h.candidates = h.candidates[1:]  # the operator drops the offending candidate
        h.open(codec=BadLimits())
        assert h.drive()
        assert h.ctrl.budget().consumed == 3
        h.check(synced=True)


def test_invalid_result_is_recorded_but_not_booked(tmp_path):
    class Picky(ToyCodec):
        def metrics(self, candidate, result):
            if candidate.payload == 1:
                raise ValueError("unreadable measurement")
            return super().metrics(candidate, result)

    objectives = ObjectiveSet.from_specs([ObjectiveSpec("objective", "maximize")])
    with Harness(tmp_path, budget=4, candidates=6, concurrency=2, objectives=objectives) as h:
        h.start_es()
        h.open(codec=Picky())
        assert h.drive()
        invalid = h.ctrl.store.row("SELECT * FROM experiments WHERE canonical_key='int:1'")
        assert invalid["disposition"] == "invalid" and invalid["budget_state"] == "CONSUMED"
        assert "invalid_result" in anomaly_types(h) and h.ctrl.view().paused == ()
        assert all(o.canonical_key != "int:1" for o in h.ctrl.observations())
        assert h.ctrl.store.value("SELECT COUNT(*) FROM experiments WHERE canonical_key='int:1'") == 1
        h.check(synced=True)


def test_collect_only_after_close_books_late_results_without_deciding(tmp_path):
    with Harness(tmp_path, budget=6, candidates=6, concurrency=3) as h:
        h.start_es()
        h.open()
        h.drive(until=lambda: all(h.mirror(d) == "RUNNING" for d in (1, 2, 3)))
        assert h.ctrl.close_campaign("deadline")
        h.crash()
        h.open(collect_only=True, proposer=FixedProposer([], name="other"))
        assert h.drive()
        observations = h.ctrl.observations()
        assert len(observations) == 3 and all(o.metadata["flags"] == ["after_close"] for o in observations)
        assert h.ctrl.store.value("SELECT COUNT(*) FROM decisions") == 3
        h.check(synced=True)


def test_hooks_see_every_event_after_builtins_and_action_requests_are_adjudicated(tmp_path):
    seen, calls = [], []

    def audit(event, view):
        assert h.ctrl.dispatcher.cursor("outbox") >= event.seq and h.ctrl.dispatcher.cursor("pacing") >= event.seq
        if event.seq not in seen:
            seen.append(event.seq)

    def nudge(event, view):
        calls.append(event.seq)
        if view.observation_count != 2:
            return []
        return [{"action": "request_research", "note": "plateau"}, {"action": "notify_agents", "text": "hi"},
                {"action": "request_pause", "reason": "plateau"}, {"action": "launch_experiment"}]

    hooks = [EffectHook("audit", ("*",), audit),
             ActionHook("nudge", ("observation.ingested",), nudge, frozenset({"request_research", "request_pause"}))]
    with Harness(tmp_path, budget=4, candidates=4, concurrency=1, hooks=hooks) as h:
        h.start_es()
        h.open()
        h.drive(until=lambda: h.ctrl.view().paused)
        outputs = hook_outputs(h.ctrl.store, "nudge")
        decided = [o for o in outputs if o["requests"]]
        assert len(decided) == 1
        verdicts = [(r["action"], r["accepted"], r["reason"]) for r in decided[0]["requests"]]
        assert verdicts == [("request_research", False, "lane_unavailable"),
                            ("notify_agents", False, "undeclared_action"),
                            ("request_pause", True, "paused"), ("launch_experiment", False, "unknown_action")]
        assert decided[0]["requests"][0]["request_id"] == f"nudge:{decided[0]['event_seq']}:0"
        assert h.ctrl.view().paused == ("hook.nudge:plateau",)
        assert len(calls) == len(set(calls)) == len(outputs)
        h.ctrl.resume("operator")
        assert h.drive()
        h.ctrl.step()
        assert seen == [e["seq"] for e in h.ctrl.events()]


def test_failing_hooks_observer_skipped_action_hook_pauses_or_skips(tmp_path):
    def broken(event, view):
        raise RuntimeError("boom")

    hooks = [EffectHook("flaky_observer", ("observation.ingested",), broken),
             ActionHook("strict", ("observation.ingested",), broken, frozenset({"request_pause"})),
             ActionHook("lenient", ("observation.ingested",), broken, frozenset({"request_pause"}), on_failure="skip")]
    with Harness(tmp_path, budget=3, candidates=3, concurrency=1, hooks=hooks) as h:
        h.start_es()
        h.open()
        h.drive(until=lambda: h.ctrl.view().paused)
        assert h.ctrl.view().paused[0].startswith("anomaly.hook_failed:strict:")
        failures = h.ctrl.store.rows("SELECT hook, attempts, gave_up FROM hook_failures ORDER BY hook")
        assert {(r["hook"], r["attempts"], r["gave_up"]) for r in failures} >= {("flaky_observer", 3, 1),
                                                                                ("strict", 3, 1), ("lenient", 3, 1)}
        assert [a["subject_id"].split(":")[0] for a in h.ctrl.anomalies()] == ["strict", "lenient"]


def test_task_hooks_load_from_the_async_manifest(tmp_path):
    manifest = parse_async_manifest({
        "experiment_codec": "async_controller_support:ToyCodec", "executors": {KIND: "x.y:z"},
        "hooks": [{"name": "observer", "hook": "async_controller_support:manifest_observer",
                   "subscribe": ["experiment.*"]},
                  {"name": "pauser", "hook": "async_controller_support:manifest_action",
                   "subscribe": ["observation.ingested"], "actions": ["request_pause"], "version": "3"}]})
    hooks = task_hooks(manifest)
    assert [(h.name, h.kind) for h in hooks] == [("observer", "effect"), ("pauser", "action")]
    MANIFEST_SEEN.clear()
    with Harness(tmp_path, budget=2, candidates=2, concurrency=1, hooks=hooks) as h:
        h.start_es()
        h.open()
        h.drive(until=lambda: h.ctrl.view().paused)
        assert h.ctrl.view().paused == ("hook.pauser:manifest",)
        assert MANIFEST_SEEN and all(isinstance(seq, int) for seq in MANIFEST_SEEN)
        assert h.ctrl.frozen_config()["plugins"]["action_hooks"] == ["pauser@3"]


def test_production_run_loop_with_real_clock_and_long_poll(tmp_path):
    clock = ESRealClock()
    sim = SimulatedExecutor(clock, kind=KIND, duration=("uniform", 0.02, 0.1), seed=5)
    options = RuntimeOptions(poll_interval=0.2, long_poll_seconds=1.0, anti_entropy_interval=5.0,
                             retry_base_seconds=0.05, retry_max_seconds=0.5)
    with running_service([sim], pools={"default": 2}, clock=clock, autorun=True, poll_interval=0.01) as es:
        config = CampaignConfig("live", budget=6, concurrency=2)
        candidates = [c for c in Harness(tmp_path).candidates]
        ctrl = Controller(tmp_path / "live.sqlite3", config, codec=ToyCodec(), client=ESClient(es.base_url),
                          clock=RealClock(), proposer=FixedProposer(candidates), options=options)
        try:
            runner = threading.Thread(target=ctrl.run, daemon=True)
            runner.start()
            runner.join(timeout=60)
            assert not runner.is_alive()
            assert ctrl.finished and ctrl.view().stop_reason == "budget_exhausted"
            assert len(ctrl.observations()) == 6
            items = es.service.changes("live", "", limit=1000)["items"]
            assert check_invariants(ctrl.store, config, items, synced=True) == []
        finally:
            ctrl.close()
