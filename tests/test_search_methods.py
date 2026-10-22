from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass

import pytest

from ldm_tts.optimization.search import (
    SEARCH_METHOD_ALIASES,
    beam_search,
    best_of_n,
    canonical_search_method,
    estimate_generated,
    get_search_method,
    get_traversal_method,
    mcts,
    single_turn,
    tree_search,
)


@dataclass
class FakeState:
    state_id: str
    depth: int
    score: float | None = None
    status: str = "created"
    note: str = ""


class FakeEngine:
    def __init__(self, *, interval: int = 1, root_score: float | None = None) -> None:
        self._evaluation_interval = interval
        self.states: list[FakeState] = []
        self.evaluation_count = 0
        self.deferred: list[str] = []
        self.progress: list[tuple[str, int | str]] = []
        self.root_score = root_score
        self.counter = 0

    @property
    def evaluation_interval(self) -> int:
        return self._evaluation_interval

    def start_progress(self, total: int, *, label: str) -> None:
        self.progress.append((label, total))

    def finish_progress(self) -> None:
        self.progress.append(("finished", self.evaluation_count))

    def create_seed_state(self) -> FakeState:
        state = FakeState("root", 0, self.root_score, "seed")
        self.states.append(state)
        return state

    def should_evaluate_depth(self, depth: int, max_depth: int | None = None) -> bool:
        return depth % self.evaluation_interval == 0 or (
            max_depth is not None and depth >= max_depth
        )

    async def expand_state(
        self,
        parent: FakeState,
        count: int,
        *,
        search_note: str = "",
    ) -> list[FakeState]:
        children = []
        for _ in range(max(0, count)):
            self.counter += 1
            state = FakeState(f"s{self.counter}", parent.depth + 1, note=search_note)
            self.states.append(state)
            children.append(state)
        return children

    async def evaluate_many(self, states: list[FakeState]) -> None:
        for state in states:
            if state.score is None:
                state.score = 100.0 - int(state.state_id.removeprefix("s") or 0)
            state.status = "evaluated"
            self.evaluation_count += 1

    async def defer_evaluation_many(self, states: list[FakeState], *, reason: str) -> None:
        for state in states:
            state.status = "evaluation_deferred"
            state.note = reason
            self.deferred.append(state.state_id)

    def ranked_states(self, states: list[FakeState] | None = None) -> list[FakeState]:
        pool = self.states if states is None else states
        return sorted((state for state in pool if state.score is not None), key=lambda state: state.score)

    def best_state(self, states: list[FakeState] | None = None) -> FakeState | None:
        ranked = self.ranked_states(states)
        return ranked[0] if ranked else None

    def reward(self, state: FakeState) -> float:
        ranked = self.ranked_states()
        if state not in ranked:
            return 0.0
        return 1.0 - ranked.index(state) / max(1, len(ranked) - 1)


@pytest.mark.parametrize(
    ("breadth", "depth", "limit", "interval", "expected"),
    [
        (2, 2, None, 1, 6),
        (2, 3, None, 2, 12),
        (0, 0, None, 1, 1),
        (3, 4, 5, 1, 5),
    ],
)
def test_tree_budget_estimation(
    breadth: int,
    depth: int,
    limit: int | None,
    interval: int,
    expected: int,
) -> None:
    assert tree_search.estimate_budget(breadth, depth, limit, interval) == expected


@pytest.mark.parametrize(
    ("breadth", "depth", "width", "limit", "interval", "expected"),
    [
        (2, 2, 1, None, 1, 4),
        (2, 3, 2, None, 2, 8),
        (0, 0, 0, None, 1, 1),
        (3, 4, 2, 5, 1, 5),
    ],
)
def test_beam_budget_estimation(
    breadth: int,
    depth: int,
    width: int,
    limit: int | None,
    interval: int,
    expected: int,
) -> None:
    assert beam_search.estimate_budget(breadth, depth, width, limit, interval) == expected


def test_single_turn_expands_one_level() -> None:
    engine = FakeEngine()
    best = asyncio.run(
        single_turn.run(
            engine,
            breadth=3,
            depth=8,
            beam_width=4,
            max_evaluations=None,
            evaluate_root=False,
        )
    )
    assert best is not None
    assert engine.evaluation_count == 3
    assert {state.depth for state in engine.states if state.state_id != "root"} == {1}


def test_best_of_n_runs_independent_branches_and_enforces_cap() -> None:
    engine = FakeEngine(interval=2)
    best = asyncio.run(
        best_of_n.run(
            engine,
            breadth=3,
            depth=2,
            beam_width=1,
            max_evaluations=2,
            evaluate_root=False,
        )
    )
    assert best is not None
    assert engine.evaluation_count == 2
    assert engine.deferred
    assert engine.progress[-1][0] == "finished"


def test_tree_search_expands_deferred_frontier() -> None:
    engine = FakeEngine(interval=2)
    best = asyncio.run(
        tree_search.run(
            engine,
            breadth=2,
            depth=2,
            beam_width=1,
            max_evaluations=3,
            evaluate_root=False,
        )
    )
    assert best is not None
    assert engine.evaluation_count == 3
    assert len(engine.deferred) == 2


def test_beam_search_prunes_and_can_evaluate_root() -> None:
    engine = FakeEngine(root_score=200.0)
    best = asyncio.run(
        beam_search.run(
            engine,
            breadth=3,
            depth=2,
            beam_width=1,
            max_evaluations=4,
            evaluate_root=True,
        )
    )
    assert best is not None and best.state_id != "root"
    assert engine.evaluation_count == 4


def test_mcts_runs_with_deferred_depths() -> None:
    engine = FakeEngine(interval=2, root_score=100.0)
    best = asyncio.run(
        mcts.run(
            engine,
            breadth=2,
            depth=2,
            beam_width=2,
            max_evaluations=4,
            evaluate_root=True,
        )
    )
    assert best is not None
    assert engine.progress[-1][0] == "finished"


@pytest.mark.parametrize("breadth,depth,interval,budget,evaluate_root,expected", [
    (1, 2, 2, 1, False, 1),
    (2, 3, 2, 4, False, 4),
    (2, 2, 2, 20, False, 4),
    (1, 3, 2, 20, False, 2),
    (1, 2, 2, 1, True, 1),
    (1, 2, 2, 0, True, 0),
])
def test_mcts_evaluation_budget_and_tree_exhaustion(breadth, depth, interval, budget, evaluate_root, expected):
    engine = FakeEngine(interval=interval, root_score=100.0 if evaluate_root else None)
    asyncio.run(mcts.run(engine, breadth=breadth, depth=depth, beam_width=1,
                         max_evaluations=budget, evaluate_root=evaluate_root))
    assert engine.evaluation_count == expected
    if not evaluate_root:
        assert engine.best_state() is not None
        assert all(state.depth % interval == 0 or state.depth == depth
                   for state in engine.states if state.status == "evaluated")


def test_mcts_stops_when_expansion_returns_no_children():
    engine = FakeEngine()
    calls = []

    async def empty(parent, count, **kwargs):
        calls.append(parent.state_id)
        return []

    engine.expand_state = empty
    assert asyncio.run(mcts.run(engine, breadth=2, depth=3, beam_width=1,
                               max_evaluations=10, evaluate_root=False)) is None
    assert calls == ["root"]
    assert engine.evaluation_count == 0


def test_mcts_node_helpers_and_delegated_reward() -> None:
    root = mcts.MCTSNode(FakeState("root", 0, 2.0))
    child = mcts.MCTSNode(FakeState("child", 1, 1.0), parent=root)
    root.children.append(child)
    assert math.isinf(mcts._uct(child, 1.0))
    mcts._backpropagate(child, 0.5)
    assert child.visits == root.visits == 1
    assert child.value_mean == 0.5
    assert mcts._select(root, max_depth=2, max_children=1, exploration=1.0) is child

    engine = FakeEngine()
    engine.states = [root.state, child.state]
    assert mcts._reward(engine, child.state) == 1.0
    child.state.score = None
    assert mcts._reward(engine, child.state) == 0.0


def test_registry_resolves_aliases_and_generation_estimates() -> None:
    assert canonical_search_method("beam") == "beam_search"
    assert get_search_method("tree") is tree_search.run
    assert get_traversal_method("single") is not None
    assert estimate_generated(2, 3, 1, "best_of_n") == 6
    assert estimate_generated(2, 3, 1, "tree") == 14
    assert "single_turn" in SEARCH_METHOD_ALIASES
    with pytest.raises(ValueError, match="Unknown search method"):
        canonical_search_method("missing")
    with pytest.raises(ValueError, match="no standalone traversal"):
        get_traversal_method("mcts")
