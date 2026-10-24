from dataclasses import replace

import pytest

from ldm_tts.harness.research_tree import CoverageForks, DiverseArchive, ForkContext, LatestMainForks, ResearchNode


def node(i, **kwargs):
    return ResearchNode(f"main_{i}", f"nodes/{i}", i, i, "main", **kwargs)


def test_latest_main_and_rotating_two_plus_two():
    profiles = ("a", "b", "c", "d")
    mains = tuple(node(i) for i in range(1, 11))
    explored = []
    for r in range(10, 14):
        ctx = ForkContext(r, profiles, mains, ())
        assert {a.parent_id for a in LatestMainForks().plan(ctx)} == {"main_10"}
        assignments = CoverageForks().plan(ctx)
        assert [a.profile_id for a in assignments] == list(profiles)
        assert sum(a.purpose == "main" for a in assignments) == 2
        starts = [a for a in assignments if a.purpose == "restart"]
        assert len({a.parent_id for a in starts}) == 2
        assert all(a.parent_id in {"main_1", "main_3", "main_6"} for a in starts)
        explored.extend(a.profile_id for a in starts)
    assert all(explored.count(p) == 2 for p in profiles)


def test_continuation_uses_lineage_visits_not_highest_score():
    a = replace(node(1), node_id="branch_a", lineage_id="a", expansions=5, best_score=100)
    b = replace(node(2), node_id="branch_b", lineage_id="b", expansions=1, best_score=-10)
    ctx = ForkContext(4, ("a", "b", "c", "d"), (node(1), node(2), node(3)), (a, b))
    continuation, = (a for a in CoverageForks().plan(ctx) if a.purpose == "continue")
    assert continuation.parent_id == "branch_b"
    assert continuation.lineage_id == "b"


def test_single_root_bootstrap_and_invalid_profile_count():
    ctx = ForkContext(1, ("a", "b", "c", "d"), (node(1),), ())
    assert len(CoverageForks().plan(ctx)) == 4
    with pytest.raises(ValueError):
        CoverageForks().plan(replace(ctx, profile_ids=("a", "b")))


def test_archive_preserves_unknown_and_merges_duplicates():
    measured = replace(node(1), proposal_keys=("x", "y"), measured_count=2, best_score=10)
    duplicate = replace(node(2), proposal_keys=("x", "y", "z"), measured_count=1, best_score=5)
    pending = replace(node(3), proposal_keys=("p",), expansions=1)
    other = replace(node(4), proposal_keys=("q",), measured_count=1, best_score=-2)
    result = DiverseArchive().retain((duplicate, measured, pending, other))
    assert {n.node_id for n in result} == {measured.node_id, pending.node_id, other.node_id}
    assert pending.measured_count == 0 and pending.best_score is None
    assert len(DiverseArchive(capacity=1).retain((measured, pending))) == 1


def test_historical_restart_uses_least_recently_visited_anchor():
    mains = tuple(node(i, last_expanded=20 if i == 1 else 0) for i in range(1, 11))
    ctx = ForkContext(20, ("a", "b", "c", "d"), mains, ())
    restart = next(a for a in CoverageForks().plan(ctx) if a.profile_id == "d")
    assert restart.parent_id == "main_3"


def test_strategy_configurations_and_custom_factories(monkeypatch):
    import sys
    from types import ModuleType
    from ldm_tts.harness.research_strategy import ResearchStrategy, load_research_strategy

    assert not load_research_strategy({"name": "independent"}).requires_main
    assert isinstance(load_research_strategy({"name": "main_forks"}).fork_policy, LatestMainForks)
    module = ModuleType("custom_research_rules")
    module.choose = lambda: CoverageForks()
    module.archive = lambda capacity: DiverseArchive(capacity=capacity)
    module.strategy = lambda: ResearchStrategy(CoverageForks())
    monkeypatch.setitem(sys.modules, module.__name__, module)
    strategy = load_research_strategy({"name": "main_forks",
        "fork_policy": {"factory": "custom_research_rules:choose"},
        "archive_policy": {"factory": "custom_research_rules:archive", "options": {"capacity": 5}}})
    assert isinstance(strategy.fork_policy, CoverageForks)
    assert strategy.archive_policy.capacity == 5
    assert load_research_strategy({"factory": "custom_research_rules:strategy"}).requires_main
    with pytest.raises(ValueError):
        load_research_strategy({"name": "independent", "fork_policy": {"name": "latest"}})
    with pytest.raises(ValueError):
        load_research_strategy({"name": "main_forks", "fork_policy": {"name": "typo"}})


def test_main_history_forks_credit_only_measured_gain_and_never_children():
    from ldm_tts.harness.research_tree import MainHistoryForks, ResearchOutcome
    mains = tuple(node(i) for i in range(1, 6))
    child = replace(node(2), node_id="branch_2", best_score=9999)
    evidence = ResearchOutcome(2, "b", "restart", mains[1], 10, ("x",), 1, 20, "x")
    unknown = ResearchOutcome(3, "b", "restart", mains[2], 10, ("y",))
    context = ForkContext(5, ("a", "b", "c", "d"), mains, (child,), (evidence, unknown))
    assignments = MainHistoryForks().plan(context)
    assert [a.parent_id for a in assignments] == ["main_5", "main_2", "main_1", "main_3"]
    assert len({a.parent_id for a in assignments}) == 4
    assert [a.purpose for a in assignments] == ["main", "restart", "restart", "restart"]
    assert MainHistoryForks().plan(context) == assignments
    # Losses do not rank unknowns as failures; fallback is coverage.
    context = replace(context, outcomes=(replace(evidence, best_score=9), unknown))
    assert MainHistoryForks().plan(context)[1].parent_id == "main_1"
    bootstrap = MainHistoryForks().plan(replace(context, main_nodes=mains[:1], outcomes=()))
    assert {a.parent_id for a in bootstrap} == {"main_1"}
    with pytest.raises(ValueError):
        MainHistoryForks().plan(replace(context, profile_ids=("a", "b", "c")))
