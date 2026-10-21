from dataclasses import replace
import json
from pathlib import Path

import pytest

from ldm_tts.contracts import AcquisitionSpec, Candidate, EvaluationResult, Observation, RawProposal
from ldm_tts.engine import LDMEngine, LDMEngineConfig, LDMEngineState
from ldm_tts.engine.expansion import CallableReservoirExpander, ExpansionResult
from ldm_tts.engine.run_store import CampaignRuntime
from ldm_tts.harness.research_strategy import load_research_strategy
from ldm_tts.optimization import BOObservation, BOPrediction, BOSelectionResult
from ldm_tts.optimization.measurement import MeasurementContext, SelectionRequirements
from tasks.nucleobench.core.candidate import MutationContext, NucleoBenchCandidateDomain, make_start_candidate
from tasks.nucleobench.core.cases import get_case
from tasks.nucleobench.core.constants import NUCLEOBENCH_Q0_METADATA_KEY
from tasks.nucleobench.core.factory import build_surrogate_components
from tasks.nucleobench.core.hamming_gp import NucleotideHammingEncoder
from tasks.nucleobench.core.measurement import HammingShellCoverage
from tasks.nucleobench.core.selection import AcquisitionTiltedSelector
from tasks.nucleobench.core.task_spec import build_task_spec
from tasks.nucleobench.core.workflow import _campaign_budget, parse_args


@pytest.fixture
def reservoir():
    context = MutationContext(
        case=replace(get_case("malinois_k562"), sequence_length=96, editable_position_count=96),
        start_set_digest="a" * 64, start_index=0, start_sequence="A" * 96,
        editable_positions=tuple(range(96)),
    )
    domain, encoder = NucleoBenchCandidateDomain(context), NucleotideHammingEncoder(context)
    positions = [[bit for bit in range(7) if index & (1 << bit)] for index in range(1, 117)]
    positions += [[(offset + index) % 96 for index in range(distance)]
                  for distance in (12, 40, 80) for offset in range(4)]
    proposals = tuple(RawProposal({"mutations": [{"position": position, "base": "C"} for position in row]},
        "fixture", {NUCLEOBENCH_Q0_METADATA_KEY: {"occurrence_count": 1, "valid_occurrence_count": 128}})
        for row in positions)
    candidates = tuple(domain.admit(proposal) for proposal in proposals)
    assert all(isinstance(candidate, Candidate) for candidate in candidates)
    representations = {candidate.candidate_id: encoder.encode(candidate) for candidate in candidates}
    initial = make_start_candidate(context)
    history = (BOObservation.scalar(initial.candidate_id, 1.0, encoder.encode(initial).values,
                                    feature_version=encoder.version),)
    return context, domain, encoder, proposals, candidates, representations, initial, history


class LocalPreference:
    def describe(self):
        return AcquisitionSpec("ucb", ("utility",), "maximize", "fixed")

    def fit(self, history):
        pass

    def select(self, candidates, representations, *, count, round_idx):
        return BOSelectionResult(tuple(candidate.candidate_id for candidate in candidates[:count]), tuple(
            BOPrediction.scalar(candidate.candidate_id, mean=0.0, std=0.0,
                                acquisition_score=1000.0 if sum(representations[candidate.candidate_id].values) <= 8 else -1000.0)
            for candidate in candidates
        ))


def selector(policy=None, seed=42):
    return AcquisitionTiltedSelector(LocalPreference(), alpha=1, eta=100, z_clip=5, seed=seed,
                                    pool_size=64, proposal_sample_count=128, measurement_policy=policy)


@pytest.mark.parametrize("seed", [42, 43, 10086])
def test_exploration_survives_both_gates_without_changing_q0_or_budget(reservoir, seed):
    _, _, _, _, candidates, representations, _, history = reservoir
    original, covered = selector(seed=seed), selector(HammingShellCoverage(), seed)
    original.fit(history)
    covered.fit(history)
    baseline = original.select(candidates, representations, count=16, round_idx=1)
    result = covered.select(candidates, representations, count=16, round_idx=1)
    assert len(result.selected_candidate_ids) == len(set(result.selected_candidate_ids)) == 16
    assert result.metadata["bo_pool_size"] == 64
    assert result.metadata["proposal_base_measure"] == baseline.metadata["proposal_base_measure"]
    groups = result.metadata["measurement_coverage"]["groups"]
    assert len(groups) == 3
    for index, group in enumerate(groups):
        assert len(group["pooled_ids"]) == 4
        assert len(group["selected_ids"]) >= 1
        assert result.selected_candidate_ids[index] in group["selected_ids"]
    assert all(sum(representations[key].values) <= 8 for key in baseline.selected_candidate_ids)
    assert covered.select(tuple(reversed(candidates)), representations, count=16, round_idx=1).to_dict() == result.to_dict()


def test_empty_requirements_preserve_existing_selection_exactly(reservoir):
    class EmptyPolicy:
        def identity(self):
            return {"name": "empty", "version": 1}

        def plan(self, context):
            return SelectionRequirements()

    _, _, _, _, candidates, representations, _, history = reservoir
    first, second = selector(), selector(EmptyPolicy())
    first.fit(history)
    second.fit(history)
    original = first.select(candidates, representations, count=16, round_idx=1).to_dict()
    empty = second.select(candidates, representations, count=16, round_idx=1).to_dict()
    empty["metadata"].pop("measurement_coverage")
    assert empty == original


def test_shell_boundaries_shortages_and_measured_priority(reservoir):
    _, domain, encoder, _, _, _, _, history = reservoir
    candidates = tuple(domain.admit(RawProposal({"mutations": [
        {"position": position, "base": "C"} for position in range(distance)]}, "fixture"))
        for distance in (8, 9, 32, 33, 64, 65))
    representations = {item.candidate_id: encoder.encode(item) for item in candidates}
    context = MeasurementContext(candidates, representations, history, 1, 64, 16)
    plan = HammingShellCoverage().plan(context)
    assert [plan.metadata["candidate_shells"][item.candidate_id] for item in candidates] == [0, 1, 1, 2, 2, 3]
    assert [group.min_pool for group in plan.groups] == [2, 2, 1]
    measured = BOObservation.scalar(candidates[1].candidate_id, 0.0, representations[candidates[1].candidate_id].values,
                                   feature_version=encoder.version)
    plan = HammingShellCoverage(max_regions=1).plan(replace(context, history=(*history, measured)))
    assert plan.groups[0].group_id == "shell_2"
    bad_history = (replace(history[0], feature=None),)
    with pytest.raises(ValueError, match="encoded"):
        HammingShellCoverage().plan(replace(context, history=bad_history))


def test_coverage_configuration_preserves_fork_archive_and_campaign_budgets(reservoir):
    root = Path(__file__).resolve().parents[3] / "config"
    paths = [root / "research_strategies/joint_gain_1plus3.json",
             root / "nucleobench/research_strategies/joint_gain_1plus3_coverage.json"]
    old_config = json.loads(paths[0].read_text())
    new_config = json.loads(paths[1].read_text())
    old, new = load_research_strategy(old_config), load_research_strategy(new_config)
    assert old.fork_policy.identity() == new.fork_policy.identity()
    assert old.archive_policy.identity() == new.archive_policy.identity()
    budgets = []
    for path in paths:
        args = parse_args(["--search-method", "ldm_harness_compiled", "--iterations", "21",
                           "--evaluations-per-round", "16", "--harness-candidates-per-session", "32",
                           "--bo-pool-size", "64", "--harness-research-strategy-config", str(path)])
        budgets.append(_campaign_budget(args, None))
    assert budgets[0] == budgets[1]
    context = reservoir[0]
    _, built = build_surrogate_components("ldm_harness_compiled", context, evaluations_per_round=16,
                                          proposal_samples=128, bo_pool_size=64, measurement_policy=new.measurement_policy)
    assert built.describe().parameters["measurement_policy"] == new.measurement_policy.identity()
    with pytest.raises(ValueError, match="acquisition-tilted"):
        build_surrogate_components("bo", context, evaluations_per_round=16, measurement_policy=new.measurement_policy)


def test_engine_recovery_keeps_reserved_selection_and_calls_evaluator_only_sixteen_times(reservoir, tmp_path, monkeypatch):
    context, domain, encoder, proposals, _, _, initial, _ = reservoir
    expanded, evaluated = [], []

    def expand(request):
        expanded.append(request.round_idx)
        return ExpansionResult(proposals)

    class Evaluator:
        def evaluate(self, candidate):
            evaluated.append(candidate.candidate_id)
            return EvaluationResult(candidate.candidate_id, "succeeded", {"utility": 0.0})

    engine = LDMEngine(
        task_spec=build_task_spec(context.case, search_method="ldm_harness", evaluations_per_round=16,
                                 proposal_samples=128, bo_pool_size=64, harness_candidates_per_session=32,
                                 surrogate=encoder.describe()),
        runtime=CampaignRuntime.open(tmp_path, task="nucleobench", budget_limits={"external_evaluations": 16}),
        candidate_domain=domain, surrogate_encoder=encoder, selector=selector(HammingShellCoverage()),
        expander=CallableReservoirExpander(expand), evaluator=Evaluator(),
    )
    original = engine._record_evaluation
    interrupted = False

    def crash_once(*args):
        nonlocal interrupted
        if not interrupted:
            interrupted = True
            raise KeyboardInterrupt()
        return original(*args)

    monkeypatch.setattr(engine, "_record_evaluation", crash_once)
    config = LDMEngineConfig(iterations=2, reservoir_size=128, evaluations_per_round=16)
    state = LDMEngineState(observations=[Observation(initial, EvaluationResult(initial.candidate_id, "succeeded", {"utility": 1.0}),
                                                   surrogate=encoder.encode(initial), round_idx=0)], next_round=1)
    with pytest.raises(KeyboardInterrupt):
        engine.run(config, state=state)
    engine.runtime = CampaignRuntime.open(tmp_path, task="nucleobench", resume=True)
    result = engine.run(config)
    assert len(evaluated) == len(set(evaluated)) == 16
    assert expanded == [1]
    assert len(result.state.observations) == 17
    selection_event, = [row for row in engine.runtime.events() if row["event_type"] == "candidates_selected"]
    coverage = selection_event["payload"]["metadata"]["measurement_coverage"]
    assert all(set(group["selected_ids"]) <= set(evaluated) for group in coverage["groups"])
    assert engine.runtime.budget.counters["external_evaluations"] == 16
