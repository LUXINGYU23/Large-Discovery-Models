"""Contract and scheduling checks for the antibody dynamic policy integration."""
from __future__ import annotations

import json

import numpy as np
import pytest

from ldm_tts.async_runtime.ledger import Budget, StateView
from ldm_tts.async_runtime.pacing import CampaignView, LaneStatus, PacingConfig
from ldm_tts.async_runtime.reservoir import ReservoirView
from ldm_tts.async_runtime.selection import SelectionContext
from ldm_tts.contracts import Candidate, EvaluationResult, Observation
from ldm_tts.harness import PolicyResearchController, PolicyRoundInput
from ldm_tts.harness.protocol import HarnessSubmissionRequest, HarnessTurnResult
from ldm_tts.optimization.pending import PriorTable
from ldm_tts.optimization.records import SelectionSeed

from tasks.antibody_rfd3.core.experiments import AntibodyCodec, ENERGY_UTILITY
from tasks.antibody_rfd3.core.policy import (AMINO_ACIDS, FEATURE_NAMES, FEATURE_VERSION,
    AntibodyPolicyAdapter, featurize, make_selector)
from tasks.antibody_rfd3.core.research import PolicyRoundPacing
from tasks.antibody_rfd3.core.workflow import describe_ldm_task, parse_args


def candidate(sequence):
    return Candidate(sequence, {"sequence": sequence}, sequence)


def observation(sequence, energy, decision):
    item = candidate(sequence)
    metrics = AntibodyCodec("HER2", "absolut", utility=True).metrics(item, {"absolut_energy": energy})
    return Observation(item, EvaluationResult(item.candidate_id, "succeeded", metrics),
                       decision_id=decision)


def test_one_hot_and_energy_direction_are_shared_by_codec_and_selector():
    low = observation("AAAAAAAAAAA", -9.0, "d000001")
    high = observation("CAAAAAAAAAA", -3.0, "d000002")
    assert low.metrics["absolut_energy"] < high.metrics["absolut_energy"]
    assert low.metrics[ENERGY_UTILITY] > high.metrics[ENERGY_UTILITY]
    assert AntibodyCodec("HER2", "absolut").metrics(low.candidate, {"absolut_energy": -9.0}) == {
        "absolut_energy": -9.0}
    selector = make_selector()
    rows, _ = selector.training_rows([low, high])
    assert [row.scalar_score for row in rows] == [9.0, 3.0]
    assert FEATURE_VERSION == "cdrh3-one-hot-v1" and len(FEATURE_NAMES) == 220
    assert AMINO_ACIDS == "ACDEFGHIKLMNPQRSTVWY"
    encoded = np.asarray(featurize(low.candidate)).reshape(11, 20)
    assert np.all(encoded.sum(axis=1) == 1) and np.all(encoded[:, 0] == 1)
    assert np.sum((np.asarray(featurize(low.candidate)) - np.asarray(featurize(high.candidate))) ** 2) == 2
    other = candidate("YAAAAAAAAAA")
    assert np.sum((np.asarray(featurize(low.candidate)) - np.asarray(featurize(other))) ** 2) == 2
    with pytest.raises(ValueError):
        featurize(candidate("bad"))


def test_frozen_prior_and_weights_change_the_actual_reference_selection_inputs():
    history = (observation("AAAAAAAAAAA", -9, "d000001"), observation("CAAAAAAAAAA", -3, "d000002"))
    query = (candidate("DAAAAAAAAAA"), candidate("EAAAAAAAAAA"))
    selector = make_selector()
    table = PriorTable("a000003", {history[0].canonical_key: 9, history[1].canonical_key: 3,
                                    query[0].canonical_key: 20, query[1].canonical_key: -20})
    table.weights = (2.5, 1.75)
    table.prior_free = False
    selector.prior = lambda: table
    context = SelectionContext(None, query, {c.canonical_key: 0.5 for c in query}, (),
                               SelectionSeed(42, "d000003"), 1, history, 2)
    active = selector.choose(context)
    baseline = selector.choose(context, prior=None)
    assert active.inputs["prior_application_id"] == "a000003"
    assert active.record["selection"]["metadata"]["prior_applied"]
    assert active.record["selection"]["parameters"]["alpha"] == 2.5
    assert active.record["selection"]["parameters"]["eta"] == 1.75
    assert not np.allclose(active.record["selection"]["steps"][0]["mean"],
                           baseline.record["selection"]["steps"][0]["mean"])


class DisablePolicyClient:
    """Exercise the real policy controller and persisted feedback without a provider call."""

    def run_turn(self, turns, *, submission_validator, recovery_timeout_seconds=0):
        turn, = turns
        request = HarnessSubmissionRequest(turn.profile_id, turn.turn_id, 1, {"action": "disable"})
        assert submission_validator(request).decision == "accept"
        return (HarnessTurnResult(profile_id=turn.profile_id, session_id="test", turn_id=turn.turn_id,
            round_index=turn.round_index, history_from_seq=turn.history_from_seq, history_to_seq=turn.history_to_seq,
            history_digest=turn.history_digest, input_digest=turn.input_digest, replayed=False,
            submission_status="accepted", submission_id="test", submission_digest=request.digest,
            submission=request.submission, submitted_artifacts=(), validation_errors=(),
            usage={"providerCalls": 1}, tool_budget={}, artifacts={}),)


def test_async_feedback_reaches_policy_controller_and_is_persisted(tmp_path):
    item, query = candidate("AAAAAAAAAAA"), candidate("CAAAAAAAAAA")
    policy_input = PolicyRoundInput(round_index=2, history_features=np.asarray([featurize(item)]),
        history_utilities=np.asarray([9.0]), query_features=np.asarray([featurize(query)]),
        history_candidate_ids=(item.candidate_id,), history_rounds=(1,),
        measured_observations=({"candidate_id": item.candidate_id, "decision_id": "d000001", "utility": 9.0},),
        research_snapshot={}, execution_context={"mean_context": {}, "weight_context": {}})
    feedback = [{"decision_id": "d000001", "application_id": "a000001", "predictions": [{
        "candidate_id": item.candidate_id, "baseline_mean": 7.0, "active_mean": 8.0}]}]
    controller = PolicyResearchController(client=DisablePolicyClient(), adapter=AntibodyPolicyAdapter(),
                                          executor=None, root=tmp_path)
    result = controller.resolve(policy_input, feedback=feedback)
    assert result.metadata["status"] == "accepted" and result.metadata["action"] == "disable"
    saved = json.loads((tmp_path / "rounds/round_002/input.json").read_text())
    prediction_feedback = saved["execution_context"]["weight_context"]["prediction_feedback"]
    assert prediction_feedback["summary"]["count"] == 1
    assert prediction_feedback["measurements"][0]["decision_id"] == "d000001"
    assert prediction_feedback["measurements"][0]["measured_utility"] == 9.0


def view(generation, *, free, head, outstanding=0, selected=1, running=False, total=3):
    pacing = PacingConfig.balanced(concurrency=2, reservoir_size=6, research_budget=3,
                                   dispatch_during_research="continue")
    state = StateView(now=0, campaign_id="test", status="active", stop_reason=None,
        budget=Budget(limit=total, held=outstanding, consumed=total - free - outstanding, refunded=0, started=0),
        pending=(), unstarted=0, outstanding=outstanding, dedup_keys=frozenset(), observation_count=head,
        head_ingest_seq=head, paused=(), backoff_until=None, collect_only=False, slot_available=True)
    item = candidate("AAAAAAAAAAA")
    reservoir = ReservoirView((item,), (1.0,), (1.0,), 1.0, head, 1)
    return CampaignView(state=state, reservoir=reservoir, stale=False,
        lane=LaneStatus("running", generation + 1, 0, 100) if running else LaneStatus(),
        delta=1, selected_since_ready=selected, runtime_median=None, observed_running=outstanding,
        empty_rounds=0, research_used=generation + int(running), research_requested=None, draining=None,
        current_generation=generation, batch_generation=None, open_plan=None, pacing=pacing)


def test_each_reserved_generation_is_applied_before_its_evaluation():
    pacing = PolicyRoundPacing()
    first_pending = view(1, free=2, head=0, outstanding=1)
    assert pacing.decide(first_pending).reason == "reserved_for_research"
    assert pacing.research(first_pending).reason == "initial_measurement_pending"
    assert pacing.research(view(1, free=2, head=1)).action == "research"
    second_pending = view(2, free=1, head=1, outstanding=1)
    assert pacing.decide(second_pending).reason == "reserved_for_research"
    assert pacing.research(second_pending).reason == "round_measurement_pending"
    assert pacing.research(view(2, free=1, head=2)).action == "research"
    assert pacing.decide(view(2, free=1, head=2, running=True)).reason == "reserved_for_research"
    assert pacing.decide(view(3, free=1, head=2)).action == "dispatch"
    # Extra evaluations may dispatch while another research generation runs.
    assert pacing.decide(view(1, free=3, head=1, running=True, total=4)).action == "dispatch"


def test_research_and_evaluation_budgets_are_independent():
    args = parse_args(["--method", "ldm_harness", "--iterations", "5", "--research-rounds", "3"])
    task = describe_ldm_task(args)
    assert (args.iterations, args.research_rounds) == (5, 3)
    assert task.objectives[0].direction == "minimize"
    assert task.acquisition.objective_names == (ENERGY_UTILITY,)
    assert task.surrogate.dimension == 220
    with pytest.raises(SystemExit):
        parse_args(["--method", "ldm_harness", "--iterations", "2", "--research-rounds", "3"])
