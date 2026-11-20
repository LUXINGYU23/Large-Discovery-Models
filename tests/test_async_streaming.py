"""Streaming decisions, overlapping research, policy preparation, and recovery."""
from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest

from async_controller_support import CAMPAIGN, Harness, ToyCodec, toy_candidates
from ldm_tts.async_runtime import (CampaignConfig, InlineExecutor, PacedController, PacingConfig, Proposal,
                                   ResearchOutcome, RuntimeOptions, campaign_extra)
from ldm_tts.async_runtime.policy import PolicyPrior
from ldm_tts.async_runtime.selectors import ReferenceSelector


class Source:
    name = "stream-test-source"
    version = "1"

    def __init__(self, pool_size=64):
        self.calls = []
        self.pool_size = pool_size

    def research(self, round_input, history):
        generation = round_input["generation"]
        self.calls.append((generation, len(history), tuple(round_input["pending_keys"])))
        return ResearchOutcome(proposals=tuple(Proposal(c, "researcher")
            for c in toy_candidates(self.pool_size, (generation - 1) * 1000)))


class TestPolicy(PolicyPrior):
    """A real PolicyPrior with a deterministic prior and controllable worker delay/failure."""

    __test__ = False

    def __init__(self, *, block_generation=None, block_trigger=None, fail_once=False, status="accepted"):
        contract = SimpleNamespace(feature_names=("x",), default_alpha=1.0, default_eta=1.0, sha256="test-contract")
        super().__init__(SimpleNamespace(contract=contract, profile_id="test"))
        self.block_generation, self.block_trigger = block_generation, block_trigger
        self.fail_once, self.status = fail_once, status
        self.entered, self.release = threading.Event(), threading.Event()
        self.turn_calls = 0

    def turn(self, round_input, history, outcome):
        self.turn_calls += 1
        return {"status": self.status, "degraded": False, "epoch_id": None, "artifact_sha256": None}

    def prepare(self, snapshot):
        if snapshot["generation"] == self.block_generation or snapshot["trigger"] == self.block_trigger:
            self.entered.set()
            if not self.release.wait(10):
                raise TimeoutError("test did not release the policy worker")
        if self.fail_once:
            self.fail_once = False
            raise ValueError("deliberate policy application failure")
        result = super().prepare(snapshot)
        # Exercise real coverage and residual-GP paths without an external policy container.
        result.update(prior_free=False, source="test", values={
            key: 0.0 for key in (*result["history_keys"], *result["query_keys"])})
        return result


class HeldSupplementExecutor(InlineExecutor):
    def __init__(self, clock):
        super().__init__(clock)
        self.hold_supplement = False

    def submit(self, generation, job, round_input, started_at, *, immediate=False):
        if generation < 0 and self.hold_supplement:
            self.jobs[generation] = (job, self.clock.now() + 1000)
            return started_at
        return super().submit(generation, job, round_input, started_at, immediate=immediate)

    def release_supplement(self):
        for generation, (job, _) in self.jobs.items():
            if generation < 0:
                self.jobs[generation] = (job, self.clock.now())


def config(*, budget=80, research_budget=None):
    pacing = PacingConfig.balanced(concurrency=4, reservoir_size=64, half_life=32,
        theta=0.75, dispatch_during_research="continue", research_budget=research_budget,
        research_timeout_seconds=3000)
    return CampaignConfig(CAMPAIGN, budget=budget, concurrency=4, buffer_depth=28,
                          extra=campaign_extra(pacing))


def open_stream(h, source, *, policy=None, executor=None):
    if h.ctrl is not None:
        h.ctrl.close()
    selector = ReferenceSelector(featurize=lambda c: (float(c.payload),), objective_name="objective",
                                 feature_version="toy-stream-v1")
    h.ctrl = PacedController(h.db, h.config, codec=ToyCodec(), client=h.client(), clock=h.clock,
        research=source, lane_executor=executor, selector=selector, policy=policy, options=h.options)
    return h.ctrl


def spin(h, predicate, timeout=10):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        h.ctrl.step()
        h.es.service.run_until_idle()
        if predicate():
            return True
        time.sleep(0.001)
    return False


@pytest.mark.parametrize(("pool_size", "quota", "last_quota", "workers"),
                         [(64, 32, 32, 4), (256, 64, 48, 8)])
def test_generation_quotas_reserve_scores_and_require_new_feedback(tmp_path, pool_size, quota, last_quota, workers):
    from tasks.antibody_rfd3.core.research import GenerationQuotaPacing

    budget = 16 + 2 * quota + last_quota
    source = Source(pool_size)
    pacing = PacingConfig.balanced(concurrency=workers, reservoir_size=pool_size, half_life=32,
                                  theta=0.75, dispatch_during_research="continue", research_budget=3)
    settings = CampaignConfig(CAMPAIGN, budget=budget, concurrency=workers, buffer_depth=32 - workers,
                              extra=campaign_extra(pacing))
    with Harness(tmp_path, budget=budget, concurrency=workers, buffer_depth=32 - workers) as h:
        h.start_es()
        h.ctrl = PacedController(h.db, settings, codec=ToyCodec(),
            client=h.client(), clock=h.clock, research=source, lane_executor=InlineExecutor(h.clock),
            pacing=GenerationQuotaPacing(16, quota), options=h.options)
        h.ctrl.add_proposals(tuple(Proposal(c, "initialization") for c in toy_candidates(16, 10000)),
                             generation=0, watermark=0)
        assert h.drive(until=lambda: h.ctrl.finished)
        decisions = h.ctrl.decisions()
        assert len(decisions) == len(h.ctrl.observations()) == budget
        assert [d["snapshot"]["pacing_state"]["current_generation"] for d in decisions] == (
            [0] * 16 + [1] * quota + [2] * quota + [3] * last_quota)
        assert source.calls[0][1] == 16
        assert source.calls[1][1] > 16 and source.calls[2][1] > 16 + quota
        assert source.calls[1][2] and source.calls[2][2]


def test_streaming_window_uses_latest_history_and_pending_after_restart(tmp_path):
    source = Source()
    options = RuntimeOptions(anti_entropy_interval=60, long_poll_seconds=0, max_intents_per_step=1)
    with Harness(tmp_path, budget=80, concurrency=4, buffer_depth=28, options=options) as h:
        h.start_es()
        h.config = config()
        executor = lambda: InlineExecutor(h.clock, lambda data: 0 if data["generation"] == 1 else 1000)
        ctrl = open_stream(h, source, executor=executor())
        assert h.drive(until=lambda: ctrl.view().outstanding == 32 and ctrl.view().lane.generation == 2)
        assert len(ctrl.decisions()) == 32 and ctrl.view().observed_running == 4
        assert ctrl.view().eligible <= 64 and not ctrl.plans()
        assert ctrl.generations()[1]["state"] == "running"

        h.crash()
        ctrl = open_stream(h, source, executor=executor())
        assert h.drive(until=lambda: ctrl.view().outstanding == 32, advance=False)
        assert ctrl.generations()[1]["state"] == "running" and len(ctrl.decisions()) == 32
        h.clock.advance_to(h.es.service.next_event_time())
        assert h.drive(until=lambda: len(ctrl.observations()) == 1 and len(ctrl.decisions()) == 33, advance=False)
        decision = ctrl.decisions()[-1]["snapshot"]
        assert decision["inputs"]["count"] == 1
        assert decision["head_ingest_seq"] == decision["inputs"]["head_ingest_seq"] == 1
        assert len(decision["inputs"]["pending"]) == 31
        assert ctrl.view().outstanding == 32 and ctrl.view().lane.generation == 2
        assert all(d["snapshot"]["outstanding"] < 32 for d in ctrl.decisions())
        assert len({h.row(i)["canonical_key"] for i in range(1, 34)}) == 33
        h.check()


def test_policy_readiness_does_not_block_ingest_or_old_policy_dispatch(tmp_path):
    source, policy = Source(), TestPolicy(block_generation=2)
    options = RuntimeOptions(anti_entropy_interval=60, long_poll_seconds=0, max_intents_per_step=1)
    with Harness(tmp_path, budget=80, concurrency=4, buffer_depth=28, options=options) as h:
        h.start_es()
        h.config = config()
        ctrl = open_stream(h, source, policy=policy)
        try:
            assert spin(h, lambda: policy.entered.is_set() and ctrl.view().outstanding == 32)
            assert ctrl.generations()[1]["state"] == "running"
            old_application = policy.current().application_id
            for head in range(1, 4):
                h.clock.advance_to(h.es.service.next_event_time())
                assert spin(h, lambda: len(ctrl.observations()) == head and len(ctrl.decisions()) == 32 + head)
                decision = ctrl.decisions()[-1]["snapshot"]
                assert decision["head_ingest_seq"] == head and len(decision["inputs"]["pending"]) == 31
                assert decision["inputs"]["prior_application_id"] == old_application
                assert ctrl.view().outstanding == 32 and not ctrl.view().paused
            policy.release.set()
            assert spin(h, lambda: ctrl.generations()[1]["state"] == "ready")
            current = policy.current()
            assert current.generation == 2 and current.application_id != old_application
            assert current.head_ingest_seq < ctrl.view().head_ingest_seq == 3
            assert all(current.covers(key) for key in ctrl.view().pending_keys)
            assert all(current.covers(item.canonical_key) for item in ctrl.observations())
            assert all(current.covers(item.canonical_key) for item in ctrl.view().reservoir.candidates)
            assert policy.check() == []
        finally:
            policy.release.set()


def test_failed_readiness_blocks_and_retry_reuses_persisted_research(tmp_path):
    source, policy = Source(), TestPolicy(fail_once=True)
    with Harness(tmp_path, budget=4, concurrency=4) as h:
        h.start_es()
        h.config = config(budget=4, research_budget=1)
        ctrl = open_stream(h, source, policy=policy, executor=InlineExecutor(h.clock))
        assert h.drive(until=lambda: ctrl.view().lane.state == "blocked")
        assert not ctrl.decisions() and policy.current() is None
        assert "deliberate policy application failure" in ctrl.generations()[0]["error"]
        assert ctrl.resolve_lane("retry", "test")
        assert h.drive(until=lambda: bool(ctrl.decisions()))
        assert source.calls == [(1, 0, ())] and policy.turn_calls == 1
        assert ctrl.generations()[0]["state"] == "ready" and policy.current() is not None


def test_degraded_policy_turn_is_never_applied(tmp_path):
    source, policy = Source(), TestPolicy(status="submission_rejected")
    with Harness(tmp_path, budget=4, concurrency=4) as h:
        h.start_es()
        h.config = config(budget=4, research_budget=1)
        ctrl = open_stream(h, source, policy=policy, executor=InlineExecutor(h.clock))
        assert h.drive(until=lambda: ctrl.view().lane.state == "blocked")
        assert "policy turn failed" in ctrl.generations()[0]["error"]
        assert not ctrl.decisions() and policy.current() is None
        with pytest.raises(ValueError, match="terminal and cannot be resumed"):
            ctrl.resolve_lane("retry", "test")
        assert ctrl.view().lane.state == "blocked" and policy.turn_calls == 1


def test_abandoned_timed_out_job_does_not_block_next_generation(tmp_path):
    source = Source()
    pacing = PacingConfig.balanced(concurrency=4, reservoir_size=64, half_life=32,
        theta=0.75, dispatch_during_research="continue", research_timeout_seconds=5)
    with Harness(tmp_path, budget=4, concurrency=4) as h:
        h.start_es()
        h.config = CampaignConfig(CAMPAIGN, budget=4, concurrency=4, extra=campaign_extra(pacing))
        executor = InlineExecutor(h.clock, lambda data: 20 if data["generation"] == 1 else 0)
        ctrl = open_stream(h, source, executor=executor)
        assert h.drive(until=lambda: ctrl.view().lane.state == "blocked")
        assert ctrl.resolve_lane("abandon", "test")
        assert h.drive(until=lambda: len(ctrl.generations()) >= 2 and bool(ctrl.decisions()))
        assert ctrl.generations()[0]["state"] == "abandoned"
        assert ctrl.generations()[1]["state"] == "ready" and source.calls == [(2, 0, ())]


def test_late_written_off_result_waits_for_history_coverage(tmp_path):
    source, policy = Source(), TestPolicy()
    options = RuntimeOptions(anti_entropy_interval=60, long_poll_seconds=0, max_intents_per_step=1)
    with Harness(tmp_path, budget=80, concurrency=4, buffer_depth=28, options=options) as h:
        h.start_es()
        h.config = config(research_budget=1)
        executor = HeldSupplementExecutor(h.clock)
        ctrl = open_stream(h, source, policy=policy, executor=executor)
        assert h.drive(until=lambda: h.mirror(1) == "RUNNING")
        first = h.row(1)
        executor_ref = f"sim:{first['experiment_id']}#1"
        h.sim.set_unreachable(executor_ref)
        assert h.drive(until=lambda: h.row(1)["es_condition"] == "unreachable")
        assert ctrl.write_off(first["experiment_id"], "test", "temporarily unreachable")

        # Replace the old application after the write-off: the orphaned key is no longer in H or Q.
        previous = policy.current().application_id
        ctrl.add_proposals([Proposal(toy_candidates(1, 10000)[0], "extra")])
        assert h.drive(until=lambda: policy.current().application_id != previous, advance=False)
        application = policy.current()
        assert not application.covers(first["canonical_key"])

        executor.hold_supplement = True
        h.sim.set_unreachable(executor_ref, False)
        assert h.drive(until=lambda: any(o.experiment_id == first["experiment_id"] for o in ctrl.observations())
                       and policy._supplement is not None)
        assert policy.missing_history() == (first["canonical_key"],)
        decisions = len(ctrl.decisions())
        ctrl.step()
        assert len(ctrl.decisions()) == decisions and policy.current().application_id == application.application_id

        executor.release_supplement()
        assert h.drive(until=lambda: policy.current().application_id != application.application_id, advance=False)
        assert policy.current().covers(first["canonical_key"]) and policy.missing_history() == ()
        assert h.drive(until=lambda: len(ctrl.decisions()) > decisions, advance=False)
        assert ctrl.decisions()[-1]["snapshot"]["head_ingest_seq"] == ctrl.view().head_ingest_seq


def test_interrupted_supplement_restarts_and_covers_added_candidate(tmp_path):
    source, policy = Source(), TestPolicy()
    with Harness(tmp_path, budget=80, concurrency=4, buffer_depth=28) as h:
        h.start_es()
        h.config = config(research_budget=1)
        ctrl = open_stream(h, source, policy=policy, executor=InlineExecutor(h.clock))
        assert h.drive(until=lambda: ctrl.view().outstanding == 32)
        extra = toy_candidates(1, 10000)[0]
        ctrl.add_proposals([Proposal(extra, "extra")])
        assert ctrl.step() and policy._supplement is not None
        before = policy.current().application_id
        h.crash()
        policy = TestPolicy()
        ctrl = open_stream(h, source, policy=policy, executor=InlineExecutor(h.clock))
        assert h.drive(until=lambda: policy.current().application_id != before, advance=False)
        assert policy.current().covers(extra.canonical_key)
        assert policy.check() == []


def test_running_supplement_keeps_ingesting_and_selecting_covered_candidates(tmp_path):
    source, policy = Source(), TestPolicy(block_trigger="supplement")
    options = RuntimeOptions(anti_entropy_interval=60, long_poll_seconds=0, max_intents_per_step=1)
    with Harness(tmp_path, budget=80, concurrency=4, buffer_depth=28, options=options) as h:
        h.start_es()
        h.config = config(research_budget=1)
        ctrl = open_stream(h, source, policy=policy)
        try:
            assert spin(h, lambda: ctrl.view().outstanding == 32)
            ctrl.add_proposals([Proposal(toy_candidates(1, 10000)[0], "extra")])
            assert spin(h, policy.entered.is_set)
            old_application = policy.current().application_id
            h.clock.advance_to(h.es.service.next_event_time())
            assert spin(h, lambda: len(ctrl.observations()) == 1 and len(ctrl.decisions()) == 33)
            decision = ctrl.decisions()[-1]["snapshot"]
            assert decision["head_ingest_seq"] == 1 and len(decision["inputs"]["pending"]) == 31
            assert decision["inputs"]["prior_application_id"] == old_application
        finally:
            policy.release.set()


def test_superseded_supplement_does_not_overwrite_newer_application(tmp_path):
    source, policy = Source(), TestPolicy()
    with Harness(tmp_path, budget=80, concurrency=4, buffer_depth=28) as h:
        h.start_es()
        h.config = config(research_budget=1)
        ctrl = open_stream(h, source, policy=policy, executor=InlineExecutor(h.clock))
        assert h.drive(until=lambda: ctrl.view().outstanding == 32)
        extra = toy_candidates(1, 10000)[0]
        ctrl.add_proposals([Proposal(extra, "extra")])
        assert ctrl.step() and policy._supplement is not None
        application = policy.current()
        snapshot = policy._snapshot("supplement", application.generation,
            {"epoch_id": application.epoch_id, "artifact_sha256": application.artifact_sha256})
        prepared = policy.prepare(snapshot)
        with ctrl.store.tx("prior_application") as db:
            newer = policy.commit(db, application.generation, prepared, h.clock.now())
        assert ctrl.step()
        assert policy.current().application_id == newer
        assert len(policy.applications()) == 2 and policy.current().covers(extra.canonical_key)
