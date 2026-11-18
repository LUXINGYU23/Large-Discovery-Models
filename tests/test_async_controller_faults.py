"""Fault injection for the controller (design 4.1 W1-W8, 8.2): crash points, a faulty transport, restarts."""
import pytest

from async_controller_support import CAMPAIGN, KIND, Harness
from experiment_service.testing import request
from ldm_tts.async_runtime import ActionHook, CrashPlan, EffectHook, RuntimeOptions, SimulatedCrash
from ldm_tts.async_runtime.events import hook_outputs
from ldm_tts.async_runtime.faults import CRASH_POINTS, PACING_TRANSACTIONS, POLICY_TRANSACTIONS, hook_points

# Operator-only transactions that this scenario never commits (each is a single-row operator action;
# crashing before or after its commit simply means the action did or did not happen).
NOT_IN_SCENARIO = {"reject", "close", "pause", "resume", "resolve", "retry_rejected", "selection",
                   *PACING_TRANSACTIONS,  # the pacing transactions have their own scenario (test_async_pacing_faults)
                   *POLICY_TRANSACTIONS, "policy_turn"}  # so do the policy-prior ones (test_async_policy_prior)
HOOK_POINTS = (hook_points("outbox", "wake") + hook_points("pacing", "wake") + hook_points("notify", "effect")
               + hook_points("flaky", "effect") + ("hook.flaky.failure.before_commit", "hook.flaky.failure.after_commit")
               + hook_points("judge", "action"))
SCENARIO_POINTS = tuple(p for p in CRASH_POINTS if p.split(".")[0] not in NOT_IN_SCENARIO) + HOOK_POINTS


class Scenario:
    """Six experiments through a one-slot pool with a one-deep buffer, a lost PUT response, a 503 on a
    cancel, a cancel of an intent before it is sent (tombstone), a cancel of a running experiment, a
    write-off of a running experiment, an idempotent side-effect hook and an action hook."""

    def __init__(self, tmp_path):
        self.applied, self.deliveries, self.judged, self.flaked = {}, [], [], set()
        hooks = [EffectHook("notify", ("observation.ingested", "experiment.finished"), self.notify),
                 EffectHook("flaky", ("experiment.finished",), self.flaky),
                 ActionHook("judge", ("experiment.finished",), self.judge, frozenset({"request_research"}))]
        self.h = Harness(tmp_path, budget=6, candidates=9, concurrency=2, buffer_depth=1, pools={"default": 1},
                         hooks=hooks, sim={"seed": 11, "failures": {"execution_error": 0.15}},
                         options=RuntimeOptions(anti_entropy_interval=20.0, retry_base_seconds=1.0, retry_max_seconds=8.0,
                                                unreachable_alert_seconds=30.0, long_poll_seconds=0))
        self.crashes = 0

    def flaky(self, event, view):
        """Fails the first delivery of every event (review item 8: the failure transactions get crash points)."""
        if event.seq not in self.flaked:
            self.flaked.add(event.seq)
            raise RuntimeError("transient")

    def notify(self, event, view):
        self.deliveries.append(event.seq)
        self.applied.setdefault(event.seq, event.type)  # the effect is keyed by the event sequence number

    def judge(self, event, view):
        self.judged.append(event.seq)
        return [{"action": "request_research", "why": event.subject_id}]

    def script(self, h=None):
        """Operator actions conditioned only on persistent state, so a restart replays them safely."""
        h = self.h
        if h.ctrl is None:
            return
        third = h.row(3)
        if third is not None and third["pending"] and third["cancel_requested_at"] is None:
            h.ctrl.request_cancel(third["experiment_id"], "script: drop before sending")
        fifth = h.row(5)
        if fifth is not None and fifth["es_state"] == "RUNNING" and fifth["cancel_requested_at"] is None:
            h.ctrl.request_cancel(fifth["experiment_id"], "script: stop running")
        second = h.row(2)
        if second is not None and second["es_state"] == "RUNNING" and second["written_off_at"] is None:
            h.ctrl.write_off(second["experiment_id"], "script", "presumed lost")

    def run(self, plan=None):
        h = self.h
        h.start_es()
        h.transport.add("drop_response", method="PUT", times=1)
        h.transport.add("status", method="POST", status=503, times=1)
        h.on_step = self.script
        faults = plan
        while True:
            try:
                if h.ctrl is None:
                    h.open(faults=faults)
                    faults = None
                    h.check()
                self.script()
                assert h.drive()
                break
            except SimulatedCrash:
                self.crashes += 1
                h.crash()
        h.ctrl.step()  # let hooks see the final events
        h.check(synced=True)
        return self

    def outcome(self):
        h = self.h
        return {"es": h.es_states(), "budget": h.ctrl.budget().to_dict(), "stop": h.ctrl.view().stop_reason,
                "observations": sorted((o.experiment_id, o.evaluation.status, tuple(o.metadata["flags"]))
                                       for o in h.ctrl.observations())}


@pytest.fixture(scope="module")
def dry_run(tmp_path_factory):
    plan = CrashPlan()
    scenario = Scenario(tmp_path_factory.mktemp("dry"))
    with scenario.h:
        scenario.run(plan)
        outcome = scenario.outcome()
    return plan.hits, outcome


def test_scenario_reaches_every_crash_point_and_the_list_is_complete(dry_run):
    hits, outcome = dry_run
    unlisted = {p for p in hits if p not in CRASH_POINTS and not p.startswith("hook.")}
    assert not unlisted, f"transactions without a listed crash point: {sorted(unlisted)}"
    assert not [p for p in SCENARIO_POINTS if p not in hits]
    states = sorted(outcome["es"].values())
    assert states.count("CANCELED") == 2 and outcome["stop"] == "budget_exhausted"
    assert outcome["budget"]["refunded"] == 1 and outcome["budget"]["consumed"] == 6
    assert any("after_write_off" in flags for _, _, flags in outcome["observations"])


@pytest.mark.parametrize("point", SCENARIO_POINTS)
def test_crash_at_every_point_then_restart_preserves_invariants(point, dry_run, tmp_path):
    hits, expected = dry_run
    occurrences = sorted({1, min(2, hits[point]), hits[point]})
    for occurrence in occurrences:
        scenario = Scenario(tmp_path / f"{occurrence}")
        with scenario.h as h:
            plan = CrashPlan(point, occurrence)
            scenario.run(plan)
            assert plan.fired and scenario.crashes == 1
            assert scenario.outcome() == expected
            assert len(h.sim.launches) == len(set(h.sim.launches))  # at most one physical launch per experiment
            events = h.ctrl.events()
            notified = [e["seq"] for e in events if e["type"] in ("observation.ingested", "experiment.finished")]
            assert sorted(scenario.applied) == notified  # every event's effect exactly once
            duplicates = len(scenario.deliveries) - len(set(scenario.deliveries))
            # a crash between effect and cursor redelivers that one event; a crash in a cursor-only commit
            # (skipping events the hook does not subscribe to) redelivers nothing
            allowed = {"hook.notify.after_effect": {1}, "hook.notify.before_commit": {0, 1}}.get(point, {0})
            assert duplicates in allowed
            finished = [e["seq"] for e in events if e["type"] == "experiment.finished"]
            assert [o["event_seq"] for o in hook_outputs(h.ctrl.store, "judge")] == finished
            judged_twice = len(scenario.judged) - len(set(scenario.judged))
            assert judged_twice in ({0, 1} if point == "hook.judge.before_commit" else {0})  # never re-run once recorded


@pytest.mark.parametrize("point", ["reject.before_commit", "reject.after_commit"])
def test_crash_around_a_rejection(point, tmp_path):
    with Harness(tmp_path, budget=2, candidates=2, concurrency=1) as h:
        h.es_check = False
        h.start_es()
        request(h.es.base_url, "PUT", f"/v1/experiments/{CAMPAIGN}.d000001.s0",
                {"campaign_id": "other", "kind": KIND, "kind_version": "1", "spec": {"x": 7}})
        h.open(faults=CrashPlan(point))
        with pytest.raises(SimulatedCrash):
            h.drive(until=lambda: h.row(1) is not None and h.row(1)["submit_state"] == "REJECTED")
        h.crash()
        h.open()
        h.check()
        h.drive(until=lambda: h.row(1)["submit_state"] == "REJECTED")
        assert [a["type"] for a in h.ctrl.anomalies()] == ["put_rejected"] and h.ctrl.view().paused


# ---- faulty transport ---------------------------------------------------------------------------------

def single(tmp_path, **options):
    return Harness(tmp_path, budget=1, candidates=1, concurrency=1, **options)


def finish(h):
    assert h.drive()
    h.check(synced=True)
    assert len(h.sim.launches) == len(set(h.sim.launches))
    return h.es_items()


def test_lost_request_w3(tmp_path):
    with single(tmp_path) as h:
        h.start_es()
        h.transport.add("drop_request", method="PUT", times=2)
        h.open()
        items = finish(h)
        assert len(items) == 1 and h.row(1)["attempts"] == 2


def test_lost_response_w4_and_crash_before_ack_w6(tmp_path):
    with single(tmp_path) as h:
        h.start_es()
        h.transport.add("drop_response", method="PUT")
        h.open(faults=CrashPlan("ingest.before_commit"))  # the feed reports it first; the crash hits the recording
        with pytest.raises(SimulatedCrash):
            h.drive()
        assert len(h.es_items()) == 1  # the first PUT was accepted though its answer was lost
        h.crash()
        h.open()
        items = finish(h)
        assert len(items) == 1 and items[0]["state"] == "SUCCEEDED"
        puts = [entry for entry in h.transport.log if entry[1] == "PUT"]
        assert puts == [("drop_response", "PUT", puts[0][2])]


def test_original_request_arrives_after_its_retry_w5(tmp_path):
    with single(tmp_path) as h:
        h.start_es()
        h.transport.add("delay", method="PUT")
        h.open()
        h.drive(until=lambda: h.row(1) is not None and h.row(1)["submit_state"] == "ACKED")
        assert h.transport.held and h.row(1)["attempts"] == 1
        assert h.transport.release() == [200]  # the late original hits the primary key: no second experiment
        items = finish(h)
        assert len(items) == 1
        h.transport.held.append(("PUT", h.es.base_url + f"/v1/experiments/{items[0]['experiment_id']}",
                                 h.row(1)["request_body"].encode(), {"Content-Type": "application/json"}, 5.0))
        assert h.transport.release() == [200]
        assert h.es_items() == items


@pytest.mark.parametrize("cancel_first", [True, False])
def test_abandoned_intent_with_request_in_flight_w7(cancel_first, tmp_path):
    with Harness(tmp_path, budget=2, candidates=3, concurrency=2, pools={"default": 1}, sim={"duration": 50.0}) as h:
        h.start_es()
        h.transport.add("delay", method="PUT", path_contains=".d000002.")  # decision 2's PUT is stuck in the network
        h.open()
        h.drive(until=lambda: h.row(2) is not None and h.row(2)["attempts"] >= 1)
        if not cancel_first:
            assert h.transport.release() == [201]  # it arrives: QUEUED behind decision 1
        h.ctrl.request_cancel(h.row(2)["experiment_id"], "changed my mind")
        h.drive(until=lambda: h.mirror(2) == "CANCELED")
        if cancel_first:
            assert h.row(2)["es_tombstone"] == 1
            assert h.transport.release() == [200]  # the late PUT sees the tombstone and does nothing
        row = h.row(2)
        assert (row["budget_state"], row["es_started_at"], row["disposition"]) == ("REFUNDED", None, "no_result")
        items = finish(h)
        assert all(launch[0] != row["experiment_id"] for launch in h.sim.launches)
        assert h.ctrl.budget().consumed == 2 and len(items) == 3


def test_duplicated_requests(tmp_path):
    with Harness(tmp_path, budget=4, candidates=5, concurrency=2) as h:
        h.start_es()
        h.transport.add("duplicate", times=1000)
        h.open()
        items = finish(h)
        assert len(items) == 4


def test_backpressure_and_server_error_bursts(tmp_path):
    with Harness(tmp_path, budget=3, candidates=4, concurrency=1) as h:
        h.start_es()
        h.transport.add("status", method="PUT", status=429, headers={"Retry-After": "30"})
        h.transport.add("status", method="GET", status=503, times=3)
        h.transport.add("status", method="PUT", status=502, times=2)
        h.open()
        h.settle()
        assert h.ctrl.view().backoff_until == 30.0 and h.row(1)["submit_state"] == "PLANNED"
        h.drive(until=lambda: h.row(1)["submit_state"] == "ACKED")
        assert h.clock.now() >= 30.0
        finish(h)
        assert h.ctrl.anomalies() == []


def test_real_es_backlog_429(tmp_path):
    with Harness(tmp_path, budget=4, candidates=5, concurrency=3, buffer_depth=2, pools={"default": 1},
                 es_options={"max_queued_per_campaign": 1, "retry_after_seconds": 4}) as h:
        h.start_es()
        h.open()
        h.drive(until=lambda: h.ctrl.view().backoff_until is not None)
        throttled = h.ctrl.store.rows("SELECT * FROM experiments WHERE last_error LIKE '%429%'")
        assert len(throttled) == 1 and throttled[0]["submit_state"] == "PLANNED"
        finish(h)
        assert h.ctrl.budget().consumed == 4


def test_long_partition_keeps_budget_held_and_recovers(tmp_path):
    with Harness(tmp_path, budget=4, candidates=5, concurrency=2) as h:
        h.start_es()
        h.open()
        h.drive(until=lambda: h.mirror(1) == "RUNNING" and h.row(2) is not None)
        before = h.ctrl.budget()
        h.transport.partitioned = True
        h.drive(until=lambda: h.clock.now() > 500)
        assert [a["type"] for a in h.ctrl.anomalies()] == ["es_unreachable"]
        assert h.ctrl.budget().upper == before.upper and h.ctrl.view().outstanding == before.held
        assert h.es_states()[h.row(1)["experiment_id"]] == "SUCCEEDED"  # finished during the partition
        h.transport.partitioned = False
        finish(h)
        assert h.ctrl.anomalies() == [] and h.ctrl.budget().consumed == 4


# ---- restarts on both sides --------------------------------------------------------------------------

def test_controller_restarts_while_the_es_is_down(tmp_path):
    with Harness(tmp_path, budget=5, candidates=6, concurrency=2) as h:
        h.start_es()
        h.open()
        h.drive(until=lambda: h.mirror(2) == "RUNNING")
        h.stop_es()
        h.crash()
        h.open()  # startup anti-entropy and the outbox meet a dead ES and must not fail
        h.drive(until=lambda: h.clock.now() > 100)
        assert h.ctrl.view().outstanding == 2
        h.start_es()
        finish(h)
        assert h.ctrl.budget().consumed == 5


def test_experiments_finish_while_the_controller_is_down(tmp_path):
    with Harness(tmp_path, budget=4, candidates=6, concurrency=2) as h:
        h.start_es()
        h.open()
        h.drive(until=lambda: h.mirror(1) == "RUNNING" and h.mirror(2) == "RUNNING")
        h.crash()
        h.es.service.run_simulation()
        assert sorted(h.es_states().values()) == ["SUCCEEDED", "SUCCEEDED"]
        h.open()
        finish(h)
        assert [o.metadata["ingest_seq"] for o in h.ctrl.observations()] == [1, 2, 3, 4]


def test_both_sides_down_at_once(tmp_path):
    with Harness(tmp_path, budget=4, candidates=6, concurrency=2, buffer_depth=1, pools={"default": 1}) as h:
        h.start_es()
        h.open()
        h.drive(until=lambda: h.mirror(1) == "RUNNING" and h.mirror(2) == "QUEUED")
        h.crash()
        h.stop_es()
        h.clock.advance(100.0)
        h.start_es()
        h.open()
        finish(h)
        assert h.ctrl.budget().consumed == 4 and h.restarts == 1
