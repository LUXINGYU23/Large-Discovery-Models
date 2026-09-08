"""Checks that every selected tuple uses the official task and oracle API."""

from __future__ import annotations

import pytest
from synthonbench.benchmark import make_example_task

from ldm_tts.contracts import Candidate, Observation, RawProposal
from tasks.synthonbench.core.candidate import SynthonCandidateDomain
from tasks.synthonbench.core.constants import OBJECTIVE_NAME
from tasks.synthonbench.core.evaluator import OfficialSynthonEvaluator


def test_evaluator_delegates_to_the_official_global_task() -> None:
    task = make_example_task(budget=2, seed=0)
    candidate = _candidate(task)
    evaluator = OfficialSynthonEvaluator(task)

    result = evaluator.evaluate(candidate)

    assert result.succeeded
    assert task.calls == 1
    assert result.metrics[OBJECTIVE_NAME] == pytest.approx(task.trace[0]["utility"])
    assert result.metadata["raw_score"] == pytest.approx(task.trace[0]["raw_score"])
    assert result.resource_usage == {"benchmark_jobs": 1}


def test_restore_reconstructs_the_official_unique_query_ledger() -> None:
    original = make_example_task(budget=2, seed=0)
    candidate = _candidate(original)
    evaluation = OfficialSynthonEvaluator(original).evaluate(candidate)
    observation = Observation(candidate=candidate, evaluation=evaluation)

    restored = make_example_task(budget=2, seed=0)
    evaluator = OfficialSynthonEvaluator(restored)
    evaluator.restore_observations((observation,))

    assert restored.calls == original.calls == 1
    assert restored.observed_ids == original.observed_ids
    assert restored.trace == original.trace
    assert restored.cached_utility((
        candidate.payload["reaction_id"], tuple(candidate.payload["synthon_ids"])
    )) == pytest.approx(evaluation.metrics[OBJECTIVE_NAME])


def _candidate(task) -> Candidate:
    reaction_id = task.allowed_reactions[0]
    synthon_ids = [
        task.space.synthon_ids(reaction_id, position)[0]
        for position in task.space.positions(reaction_id)
    ]
    domain = SynthonCandidateDomain(task.space, task.allowed_reactions, "mock")
    candidate = domain.admit(RawProposal({"reaction_id": reaction_id, "synthon_ids": synthon_ids}, "test"))
    assert isinstance(candidate, Candidate)
    return candidate
