"""Pure, replaceable parent-selection and archive policies for research forks.

Policies never perform IO or evaluate candidates. The task owns snapshots,
evidence attribution and durable round plans; this module only selects nodes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class ResearchNode:
    node_id: str
    snapshot: str
    round_index: int
    history_cursor: int
    lineage_id: str
    parent_id: str | None = None
    proposal_keys: tuple[str, ...] = ()
    measured_count: int = 0
    best_score: float | None = None
    expansions: int = 0
    last_expanded: int = 0


@dataclass(frozen=True)
class ForkAssignment:
    profile_id: str
    parent_id: str
    purpose: str
    lineage_id: str


@dataclass(frozen=True)
class ResearchOutcome:
    """One proposal batch, with its decision-time parent and subsequent evidence.

    Only measurements from this batch's round are credited. A missing best score
    is unknown, not zero utility; shared best keys identify correlated evidence.
    """

    round_index: int
    profile_id: str
    purpose: str
    parent: ResearchNode
    incumbent: float | None
    proposal_keys: tuple[str, ...]
    measured_count: int = 0
    best_score: float | None = None
    best_key: str | None = None


@dataclass(frozen=True)
class ForkContext:
    round_index: int
    profile_ids: tuple[str, ...]
    main_nodes: tuple[ResearchNode, ...]
    archive: tuple[ResearchNode, ...]
    outcomes: tuple[ResearchOutcome, ...] = ()
    incumbent: float | None = None


class ForkPolicy(Protocol):
    """Return exactly one parent assignment per profile, using available nodes."""

    def identity(self) -> dict: ...

    def plan(self, context: ForkContext) -> tuple[ForkAssignment, ...]: ...


class ArchivePolicy(Protocol):
    """Select a bounded set of leaves; measurements may arrive in later rounds."""

    def identity(self) -> dict: ...

    def retain(self, nodes: tuple[ResearchNode, ...]) -> tuple[ResearchNode, ...]: ...


class LatestMainForks:
    def identity(self) -> dict:
        return {"name": "latest_main", "version": 1}

    def plan(self, context: ForkContext) -> tuple[ForkAssignment, ...]:
        return tuple(ForkAssignment(p, context.main_nodes[-1].node_id, "main", "main")
                     for p in context.profile_ids)


class CoverageForks:
    """Latest-main branches plus one historical restart and one continuation.

    Roles rotate. Historical anchors are the first, one-third and two-thirds
    nodes among *older* main nodes. Selection is deterministic, not score-biased.
    """

    def identity(self) -> dict:
        return {"name": "coverage", "version": 1, "restart_slots": 1, "continuation_slots": 1}

    def plan(self, context: ForkContext) -> tuple[ForkAssignment, ...]:
        if len(context.profile_ids) < 3 or not context.main_nodes:
            raise ValueError("coverage requires at least three profiles and a main node")
        profiles = context.profile_ids
        offset = (context.round_index - 1) % len(profiles)
        roles = profiles[offset:] + profiles[:offset]
        older = context.main_nodes[:-1] or context.main_nodes
        indices = sorted({0, (len(older) - 1) // 3, 2 * (len(older) - 1) // 3})
        anchors = [older[i] for i in indices]
        restart = min(anchors, key=lambda n: (n.last_expanded, n.expansions, n.round_index))
        if context.archive:
            continuation = min(context.archive, key=lambda n: (n.expansions, n.last_expanded, n.node_id))
            continuation_lineage = continuation.lineage_id
        else:
            alternatives = [n for n in anchors if n.node_id != restart.node_id] or anchors
            continuation = min(alternatives, key=lambda n: (n.last_expanded, n.expansions, n.round_index))
            continuation_lineage = f"explore_{context.round_index:04d}_{roles[1]}"
        assignments = {
            roles[0]: ForkAssignment(roles[0], restart.node_id, "restart",
                                     f"explore_{context.round_index:04d}_{roles[0]}"),
            roles[1]: ForkAssignment(roles[1], continuation.node_id, "continue" if context.archive else "restart",
                                     continuation_lineage),
        }
        for profile in roles[2:]:
            assignments[profile] = ForkAssignment(profile, context.main_nodes[-1].node_id, "main", "main")
        return tuple(assignments[p] for p in profiles)


class MainHistoryForks:
    """One recent main fork and three evidence/coverage-selected main ancestors.

    Credit is measured improvement over the decision-time incumbent, never proposal
    frequency or selection count. Unknown outcomes stay unknown. Earlier siblings
    reserve their parents within a round; reuse only when history is too short.
    """

    def identity(self) -> dict:
        return {"name": "main_history_1plus3", "version": 1,
                "side_rules": ["measured_gain", "least_expanded", "temporal_coverage"]}

    def plan(self, context: ForkContext) -> tuple[ForkAssignment, ...]:
        if len(context.profile_ids) != 4 or not context.main_nodes:
            raise ValueError("main_history_1plus3 requires exactly four profiles and a main node")
        latest = context.main_nodes[-1]
        selected = [latest]
        older = context.main_nodes[:-1] or context.main_nodes
        gains = {}
        for outcome in context.outcomes:
            if outcome.best_score is not None and outcome.incumbent is not None:
                gains.setdefault(outcome.parent.node_id, []).append(outcome.best_score - outcome.incumbent)
        def coverage(n):
            return (n.expansions, n.last_expanded, n.round_index, n.node_id)
        for rule in ("gain", "coverage", "temporal"):
            available = [n for n in older if n.node_id not in {p.node_id for p in selected}] or list(older)
            if rule == "gain":
                positive = [n for n in available if max(gains.get(n.node_id, [0])) > 0]
                parent = min(positive, key=lambda n: (-max(gains[n.node_id]), *coverage(n))) if positive else min(available, key=coverage)
            elif rule == "coverage":
                parent = min(available, key=coverage)
            else:
                parent = min(available, key=lambda n: (
                    -min(abs(n.round_index - p.round_index) for p in selected), *coverage(n)))
            selected.append(parent)
        return tuple(ForkAssignment(profile, parent.node_id, "main" if i == 0 else "restart",
                                     "main" if i == 0 else f"explore_{context.round_index:04d}_{profile}")
                     for i, (profile, parent) in enumerate(zip(context.profile_ids, selected)))


@dataclass(frozen=True)
class DiverseArchive:
    """Bound leaves, merge overlapping proposal sets, retain an untested direction.

    Proposal overlap is an auditable first descriptor, not a claim of semantic
    independence. Tasks can inject an ArchivePolicy with richer descriptors.
    """

    capacity: int = 3
    overlap_threshold: float = 0.5

    def __post_init__(self) -> None:
        if (type(self.capacity) is not int or self.capacity < 1
                or type(self.overlap_threshold) not in (int, float) or not 0 < self.overlap_threshold <= 1):
            raise ValueError("archive requires positive capacity and overlap in (0, 1]")

    def identity(self) -> dict:
        return {"name": "diverse_archive", "version": 1, "capacity": self.capacity,
                "overlap_threshold": self.overlap_threshold}

    def retain(self, nodes: tuple[ResearchNode, ...]) -> tuple[ResearchNode, ...]:
        # Reserve representation for an untested direction, never call it a failure.
        pending = sorted((n for n in nodes if n.measured_count == 0),
                         key=lambda n: (n.expansions, n.last_expanded, -n.round_index, n.node_id))
        measured = sorted((n for n in nodes if n.measured_count > 0),
                          key=lambda n: (-(n.best_score if n.best_score is not None else float("-inf")),
                                         n.expansions, n.node_id))
        ordered = pending[:1] + measured + pending[1:]
        selected: list[ResearchNode] = []
        selected_keys: list[set[str]] = []
        for node in ordered:
            keys = set(node.proposal_keys)
            if any(keys and len(keys & other) / len(keys | other) >= self.overlap_threshold
                   for other in selected_keys):
                continue
            selected.append(node)
            selected_keys.append(keys)
            if len(selected) == self.capacity:
                break
        return tuple(selected)
