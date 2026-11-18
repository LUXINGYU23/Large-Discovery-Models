"""Streaming decisions, overlapping research, policy preparation, and recovery."""
from __future__ import annotations

import threading
import time
from types import SimpleNamespace

from async_controller_support import CAMPAIGN, Harness, ToyCodec, toy_candidates
from ldm_tts.async_runtime import (CampaignConfig, InlineExecutor, PacedController, PacingConfig, Proposal,
                                   ResearchOutcome, RuntimeOptions, campaign_extra)
from ldm_tts.async_runtime.policy import PolicyPrior
from ldm_tts.async_runtime.selectors import ReferenceSelector


class Source:
    name = "stream-test-source"
    version = "1"

    def __init__(self):
        self.calls = []

    def research(self, round_input, history):
        generation = round_input["generation"]
        self.calls.append((generation, len(history), tuple(round_input["pending_keys"])))
        return ResearchOutcome(proposals=tuple(Proposal(c, "researcher")
            for c in toy_candidates(64, (generation - 1) * 1000)))


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
            h.clock.advance_to(h.es.service.next_event_time())
            assert spin(h, lambda: len(ctrl.observations()) == 1 and len(ctrl.decisions()) == 33)
            decision = ctrl.decisions()[-1]["snapshot"]
            assert decision["head_ingest_seq"] == 1 and len(decision["inputs"]["pending"]) == 31
            assert decision["inputs"]["prior_application_id"] == old_application
            assert ctrl.view().outstanding == 32 and not ctrl.view().paused
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
