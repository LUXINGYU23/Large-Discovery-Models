from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from ldm_tts.harness.joint_gain import JointGainForks
from ldm_tts.harness.research_strategy import load_research_strategy
from ldm_tts.harness.research_tree import ForkAssignment, ForkContext, ResearchNode, ResearchOutcome


def node(name, round_index=1, descriptor=(1.0, 0.0, 0.0, 0.0)):
    return ResearchNode(name, f"nodes/{name}", round_index, 1, name, descriptor=descriptor)


@pytest.mark.parametrize("fixed,expected", [(1, ("old_a", "leaf_b")), (12, ("old_b", "leaf_a"))])
def test_joint_choice_accounts_for_complementarity_and_fixed_main_branch(fixed, expected):
    class KnownOutcomes(JointGainForks):
        def _sample_gaps(self, context, actions):
            gains = {("d", "main"): [fixed, 1, 0, 0],
                     ("a", "old_a"): [10, 0, 0, 0], ("a", "old_b"): [0, 6, 0, 0],
                     ("b", "leaf_a"): [8, 0, 0.1, 0], ("b", "leaf_b"): [0, 5, 0, 0],
                     ("c", "old_c"): [0, 0, 0, 20]}
            return np.asarray([gains.get((a.profile_id, a.parent_id), [0] * 4) for a in actions], dtype=float).T

    context = ForkContext(5, ("a", "b", "c", "d"),
                          (node("old_a"), node("old_b", 2), node("old_c", 3), node("main", 5)),
                          (node("leaf_a"), node("leaf_b")))
    assignments = KnownOutcomes().plan(context)
    assert [a.profile_id for a in assignments] == list(context.profile_ids)
    assert sum(a.purpose == "main" for a in assignments) == 1
    assert tuple(a.parent_id for a in assignments[:2]) == expected
    assert assignments[2].parent_id == "old_c"


@pytest.mark.parametrize("continuation_count", range(4))
def test_three_side_branches_choose_any_mix_of_restart_and_continuation(continuation_count):
    targets = {p: f"leaf_{p}" if i < continuation_count else f"old_{p}" for i, p in enumerate("abc")}

    class KnownOutcomes(JointGainForks):
        def _sample_gaps(self, context, actions):
            return np.asarray([[10.0 if a.profile_id == p and a.parent_id == targets[p] else 0.0
                                for a in actions] for p in "abc"])

    context = ForkContext(5, ("a", "b", "c", "d"),
                          (*tuple(node(f"old_{p}") for p in "abc"), node("main", 5)),
                          tuple(node(f"leaf_{p}") for p in "abc"))
    plan = KnownOutcomes().plan(context)
    assert {a.profile_id: a.parent_id for a in plan if a.purpose != "main"} == targets
    assert sum(a.purpose == "continue" for a in plan) == continuation_count
    assert sum(a.purpose == "restart" for a in plan) == 3 - continuation_count
    assert plan[-1] == ForkAssignment("d", "main", "main", "main")


def test_continuation_keeps_one_successor_per_lineage_and_restarts_can_share_parent():
    class KnownOutcomes(JointGainForks):
        def _sample_gaps(self, context, actions):
            return np.asarray([[10.0 if a.purpose == "continue" else 0.0 for a in actions]])

    context = ForkContext(2, ("a", "b", "c", "d"), (node("old"), node("main", 2)), (node("leaf"),))
    plan = KnownOutcomes().plan(context)
    continuation, = (a for a in plan if a.purpose == "continue")
    assert continuation.lineage_id == "leaf"
    restarts = [a for a in plan if a.purpose == "restart"]
    assert len(restarts) == 2
    assert {a.parent_id for a in restarts} == {"old"}
    assert len({a.lineage_id for a in restarts}) == 2


def test_measured_feedback_guides_the_policy_and_unknown_outcomes_do_not_change_it():
    a = node("leaf_a", descriptor=(1.0, 0, 0, 0))
    b = node("leaf_b", descriptor=(0, 1.0, 0, 0))
    outcomes = tuple(ResearchOutcome(r, "b", "continue", parent, 0.0, (f"{parent.node_id}_{r}",),
                                    1, score, f"{parent.node_id}_{r}")
                     for r in (6, 7, 8) for parent, score in ((a, -5.0), (b, 5.0)))
    context = ForkContext(9, ("a", "b", "c", "d"), (node("old"), node("main", 9)),
                          (a, b), outcomes, 0.0)
    policy = JointGainForks(samples=1024, noise_variance=0.1)
    plan = policy.plan(context)
    assert next(a.parent_id for a in plan if a.purpose == "continue") == "leaf_b"
    assert plan == policy.plan(context)
    unknown = ResearchOutcome(8, "b", "continue", b, 0.0, ("unmeasured",))
    assert policy.plan(replace(context, outcomes=(*outcomes, unknown))) == plan
    with pytest.raises(ValueError, match="future"):
        policy.plan(replace(context, outcomes=(replace(outcomes[0], round_index=9),)))


def test_duplicate_winning_measurement_is_not_independent_evidence():
    parent = node("main")
    outcome = ResearchOutcome(2, "a", "restart", parent, 0.0, ("winner",), 1, 3.0, "winner")
    context = ForkContext(3, ("a", "b", "c", "d"), (parent,), (), (outcome,), 0.0)
    actions = (ForkAssignment("a", "main", "restart", "explore"),)
    policy = JointGainForks(samples=2048)
    single = policy._sample_gaps(context, actions)
    duplicate = policy._sample_gaps(replace(context, outcomes=(outcome,) * 4), actions)
    np.testing.assert_allclose(single, duplicate, atol=1e-6)
    independent = policy._sample_gaps(replace(context, outcomes=tuple(
        replace(outcome, best_key=f"winner_{i}") for i in range(4))), actions)
    assert independent.var() < single.var()


def test_optional_factory_configuration_and_single_root_bootstrap():
    path = Path(__file__).resolve().parents[1] / "config/research_strategies/joint_gain_1plus3.json"
    strategy = load_research_strategy(json.loads(path.read_text()))
    assert isinstance(strategy.fork_policy, JointGainForks)
    context = ForkContext(1, ("a", "b", "c", "d"), (node("root"),), ())
    plan = strategy.fork_policy.plan(context)
    assert len(plan) == 4
    assert sum(a.purpose == "main" for a in plan) == 1
    assert sum(a.purpose == "restart" for a in plan) == 3
    assert {a.parent_id for a in plan} == {"root"}
    assert strategy.fork_policy.identity()["samples"] == 512
    main_profiles = [next(a.profile_id for a in strategy.fork_policy.plan(replace(context, round_index=r))
                          if a.purpose == "main") for r in range(1, 5)]
    assert set(main_profiles) == set(context.profile_ids)
    with pytest.raises(ValueError, match="four distinct"):
        strategy.fork_policy.plan(replace(context, profile_ids=("a", "b", "c")))
