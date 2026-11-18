"""Batch barriers exercise persisted plans and the HTTP experiment service."""
from __future__ import annotations

import json

import pytest

from async_controller_support import CAMPAIGN, Harness, ToyCodec, toy_candidates
from ldm_tts.async_runtime import (CampaignConfig, CampaignConfigMismatch, CrashPlan, InlineExecutor, PacedController,
    PacingConfig, Proposal, ResearchOutcome, SimulatedCrash, campaign_extra, check_invariants)
from ldm_tts.async_runtime.selectors import ReferenceSelector
from tasks.antibody_rfd3.core.workflow import describe_ldm_task, parse_args


class BatchSource:
    name = "batch-test-source"
    version = "1"

    def __init__(self, count=64, *, rejected=False, duplicate_lineage=False):
        self.count, self.rejected, self.duplicate_lineage = count, rejected, duplicate_lineage
        self.calls = []

    def research(self, round_input, history):
        generation = round_input["generation"]
        self.calls.append((generation, len(history), list(round_input["pending_keys"])))
        if self.rejected:
            return ResearchOutcome(status="rejected", reason="incomplete panel")
        proposals = [Proposal(c, "explorer" if index % 2 == 0 else "refiner")
                     for index, c in enumerate(toy_candidates(self.count, (generation - 1) * 1000))]
        if self.duplicate_lineage:
            proposals.append(Proposal(proposals[0].candidate, "another-lineage"))
        return ResearchOutcome(proposals=tuple(proposals))


def batch_config(rounds, *, batch=32, pool=64, concurrency=4):
    pacing = PacingConfig.barrier(reservoir_size=pool, selection_batch_size=batch, research_budget=rounds,
                                   research_timeout_seconds=100)
    return CampaignConfig(CAMPAIGN, budget=rounds * batch, concurrency=concurrency,
                          extra=campaign_extra(pacing))


def open_batch(h, source, *, selector=None, config=None, faults=None):
    if h.ctrl is not None:
        h.ctrl.close()
    h.config = config or h.config
    h.ctrl = PacedController(h.db, h.config, codec=ToyCodec(), client=h.client(), clock=h.clock,
                             research=source, lane_executor=InlineExecutor(h.clock), selector=selector,
                             options=h.options, faults=faults)
    return h.ctrl


def reference_selector():
    return ReferenceSelector(featurize=lambda c: (float(c.payload),), objective_name="objective",
                             feature_version="toy-batch-v1")


def test_batch_plan_is_complete_before_dispatch_and_survives_restart(tmp_path):
    source, selector = BatchSource(), reference_selector()
    with Harness(tmp_path, budget=64, concurrency=4, sim={"duration": 5}) as h:
        h.start_es()
        h.config = batch_config(2)
        ctrl = open_batch(h, source, selector=selector)
        assert h.drive(until=lambda: len(ctrl.decisions()) == 4)
        first = ctrl.plans()[0]
        selected = json.loads(first["keys_json"])
        record, inputs = json.loads(first["record_json"]), json.loads(first["inputs_json"])
        assert (first["count"], first["consumed"], ctrl.view().outstanding) == (32, 4, 4)
        assert len(inputs["candidates"]) == 64 and inputs["count"] == 32
        assert len({c["canonical_key"] for c in selected}) == 32
        assert len(record["selection"]["steps"]) == 32
        assert all(not step["metadata"]["gp_ready"] for step in record["selection"]["steps"])
        assert source.calls == [(1, 0, [])]
        ctrl = open_batch(h, source, selector=selector)
        assert ctrl.plans()[0]["keys_json"] == first["keys_json"]
        assert h.drive()
        plans = ctrl.plans()
        assert [(p["count"], p["consumed"], p["closed_reason"]) for p in plans] == [
            (32, 32, "complete"), (32, 32, "complete")]
        second = json.loads(plans[1]["record_json"])["selection"]["steps"]
        assert all(step["metadata"]["gp_ready"] for step in second)
        assert source.calls == [(1, 0, []), (2, 32, [])]
        assert len(ctrl.decisions()) == len(ctrl.observations()) == 64
        assert len({o.canonical_key for o in ctrl.observations()}) == 64
        assert len(h.sim.launches) == len(set(h.sim.launches)) == 64
        assert not check_invariants(ctrl.store, h.config, h.es_items(), synced=True)


@pytest.mark.parametrize("point", ["plan.before_commit", "plan.after_commit"])
def test_batch_plan_recovers_across_transaction_crashes(tmp_path, point):
    source = BatchSource()
    with Harness(tmp_path, budget=32, concurrency=4, sim={"duration": 1}) as h:
        h.start_es()
        h.config = batch_config(1)
        open_batch(h, source, faults=CrashPlan(point))
        with pytest.raises(SimulatedCrash):
            h.drive()
        ctrl = open_batch(h, source)
        assert h.drive()
        assert len(source.calls) == len(ctrl.plans()) == 1
        assert len(ctrl.observations()) == len({o.canonical_key for o in ctrl.observations()}) == 32
        assert len(h.sim.launches) == len(set(h.sim.launches)) == 32


def test_twenty_complete_batches_consume_640_evaluations(tmp_path):
    source = BatchSource()
    with Harness(tmp_path, budget=640, concurrency=4, sim={"duration": 1}) as h:
        h.start_es()
        h.config = batch_config(20)
        ctrl = open_batch(h, source)
        # This large lifecycle test checks invariants once at the end, not after every HTTP event.
        h.check = lambda synced=False: None
        assert h.drive(max_rounds=3000)
        assert ctrl.budget().consumed == len(ctrl.observations()) == 640
        assert [row["base_ingest_seq"] for row in ctrl.generations()] == list(range(0, 640, 32))
        assert source.calls == [(g, (g - 1) * 32, []) for g in range(1, 21)]
        assert all((p["count"], p["consumed"], p["closed_reason"]) == (32, 32, "complete")
                   for p in ctrl.plans())
        assert len(ctrl.plans()) == 20
        assert len(h.sim.launches) == len(set(h.sim.launches)) == 640
        assert not check_invariants(ctrl.store, h.config)


@pytest.mark.parametrize("source", [BatchSource(count=31), BatchSource(count=65), BatchSource(rejected=True)])
def test_incomplete_or_truncated_generation_pauses_without_a_plan(tmp_path, source):
    with Harness(tmp_path, budget=32, concurrency=4) as h:
        h.start_es()
        h.config = batch_config(1)
        ctrl = open_batch(h, source)
        assert h.drive(until=lambda: bool(ctrl.view().paused))
        assert not ctrl.plans() and not ctrl.decisions()
        assert {a["type"] for a in ctrl.anomalies()} & {"batch_candidates_incomplete", "batch_research_incomplete"}


def test_generic_barrier_allows_fewer_than_capacity_and_duplicate_lineage(tmp_path):
    source = BatchSource(count=40, duplicate_lineage=True)
    with Harness(tmp_path, budget=32, concurrency=4) as h:
        h.start_es()
        h.config = batch_config(1)
        ctrl = open_batch(h, source)
        assert h.drive()
        assert ctrl.plans()[0]["count"] == len(ctrl.observations()) == 32


@pytest.mark.parametrize("failure", ["canceled", "failed"])
def test_failed_or_canceled_evaluation_stops_the_batch(tmp_path, failure):
    sim = {"duration": 20}
    if failure == "failed":
        sim["failures"] = {"execution_error": 1.0}
    source = BatchSource()
    with Harness(tmp_path, budget=64, concurrency=4, sim=sim) as h:
        h.start_es()
        h.config = batch_config(2)
        ctrl = open_batch(h, source)
        assert h.drive(until=lambda: len(ctrl.decisions()) == 4)
        if failure == "canceled":
            assert ctrl.request_cancel(ctrl.decisions()[0]["experiment_id"])
        assert h.drive(until=lambda: bool(ctrl.view().paused))
        assert "batch_evaluation_incomplete" in {a["type"] for a in ctrl.anomalies()}
        assert len(source.calls) == 1 and len(ctrl.generations()) == 1
        assert len(ctrl.plans()) == 1


def test_abort_records_the_unconsumed_plan_as_abandoned(tmp_path):
    source = BatchSource()
    with Harness(tmp_path, budget=64, concurrency=4, sim={"duration": 20}) as h:
        h.start_es()
        h.config = batch_config(2)
        ctrl = open_batch(h, source)
        assert h.drive(until=lambda: len(ctrl.decisions()) == 4)
        assert ctrl.abort("operator_abort")
        assert h.drive()
        assert ctrl.plans()[0]["closed_reason"] == "abandoned:draining"
        assert ctrl.view().stop_reason == "operator_abort" and len(source.calls) == 1


def test_batch_budget_and_frozen_identity_are_checked(tmp_path):
    with Harness(tmp_path, budget=64, concurrency=4) as h:
        h.start_es()
        pacing = PacingConfig.barrier(reservoir_size=64, selection_batch_size=32)
        bad = CampaignConfig(CAMPAIGN, budget=63, concurrency=4, extra=campaign_extra(pacing))
        with pytest.raises(ValueError, match="multiple"):
            open_batch(h, BatchSource(), config=bad)
        ctrl = open_batch(h, BatchSource(), config=batch_config(2))
        ctrl.close()
        h.ctrl = None
        with pytest.raises(CampaignConfigMismatch, match="selection_batch_size"):
            open_batch(h, BatchSource(), config=batch_config(4, batch=16))


def test_antibody_scientific_batch_configuration_and_rejections():
    options = ["--method", "ldm_harness", "--iterations", "640", "--research-rounds", "20",
               "--batch-size", "32", "--research-candidates-per-role", "32", "--pool-size", "64",
               "--sequences-per-round", "512", "--backbones-per-round", "16", "--samples-per-call", "32",
               "--concurrency", "4", "--cpu-limit", "8"]
    args = parse_args(options)
    task = describe_ldm_task(args)
    assert (args.iterations, args.research_rounds, args.batch_size, args.concurrency) == (640, 20, 32, 4)
    assert task.proposal_search.parameters["research_candidates_per_role"] == 32
    for changed in (["--iterations", "639"], ["--pool-size", "63"], ["--sequences-per-round", "64"],
                    ["--initialization-run", "/tmp/shared-init"], ["--method", "fixed"]):
        with pytest.raises(SystemExit):
            parse_args(options + changed)
