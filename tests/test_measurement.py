import pytest

from ldm_tts.contracts import Candidate
from ldm_tts.optimization.measurement import (
    EvaluationGroup, MeasurementContext, SelectionRequirements,
)


def context():
    return MeasurementContext(tuple(Candidate(key, key, key) for key in "abcdef"), {}, (), 1, 4, 2)


@pytest.mark.parametrize("groups,message", [
    ((EvaluationGroup("one", ("a",), 1, 1), EvaluationGroup("two", ("a",), 1, 1)), "disjoint"),
    ((EvaluationGroup("one", ("z",), 1, 1),), "outside"),
    ((EvaluationGroup("one", ("a",), 5, 1),), "pool budget"),
    ((EvaluationGroup("one", ("a", "b", "c"), 3, 3),), "selection budget"),
])
def test_reservations_reject_ambiguous_groups_and_budget_overrides(groups, message):
    with pytest.raises(ValueError, match=message):
        SelectionRequirements(groups).validate(context())


def test_short_group_releases_capacity_and_reserved_candidates_are_dispatched_first():
    requirements = SelectionRequirements((EvaluationGroup("rare", ("f",), 3, 1),))
    requirements.validate(context())
    pool = requirements.select(tuple("abcdef"), 4, stage="pool")
    assert pool == tuple("fabc")
    assert requirements.select(tuple("abcf"), 2, stage="evaluation") == tuple("fa")
    assert SelectionRequirements().select(tuple("abcdef"), 2, stage="evaluation") == tuple("ab")


def test_unrelated_task_can_supply_a_custom_measurement_policy(monkeypatch):
    import sys
    from types import ModuleType
    from ldm_tts.harness.research_strategy import load_research_strategy

    class IntegerCoverage:
        def identity(self):
            return {"name": "integers", "version": 1}

        def plan(self, _context):
            return SelectionRequirements((EvaluationGroup("large", ("f",), 2, 1),))

    module = ModuleType("integer_coverage")
    module.policy = IntegerCoverage
    monkeypatch.setitem(sys.modules, module.__name__, module)
    strategy = load_research_strategy({"name": "independent",
        "measurement_policy": {"factory": "integer_coverage:policy"}})
    plan = strategy.measurement_policy.plan(context())
    plan.validate(context())
    assert plan.select(tuple("abcdef"), 2, stage="evaluation") == tuple("fa")
    assert strategy.identity()["measurement_policy"] == IntegerCoverage().identity()
