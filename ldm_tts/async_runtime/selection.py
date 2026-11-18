"""Replaceable stages of a selection step: proposal source, selector and pacing.

``Controller.step`` asks ``pacing.decide(view)``; on ``dispatch`` (and only if the core guards
allow) it takes ``proposer.propose(view)`` and lets ``selector.select(view, candidates, seed)``
pick one. controller-core ships a fixed proposer, a dedup-only selector and a greedy pacing;
controller-pacing replaces them (reservoir, Kriging Believer, rules D/R) behind the same protocols.
"""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from ldm_tts.contracts import Candidate

from .codec import plugin_identity
from .wire import canonical, digest


def normalized(value):
    """JSON round trip (tuples become lists), so records compare equal to what the store returns."""
    return json.loads(canonical(value))


@dataclass(frozen=True)
class Selection:
    candidate: Candidate
    probability: float = 1.0
    record: Mapping[str, Any] = field(default_factory=dict)  # audit: eligible set, probability table, ...


@dataclass(frozen=True)
class PacingDecision:
    action: str  # "dispatch" | "wait" | "stop"; rule R answers "research" | "wait"
    reason: str = ""

    def __post_init__(self):
        if self.action not in ("dispatch", "wait", "stop", "research"):
            raise ValueError(f"unknown pacing action {self.action!r}")


@runtime_checkable
class ProposalSource(Protocol):
    name: str
    version: str

    def propose(self, view) -> Sequence[Candidate]: ...


@runtime_checkable
class Selector(Protocol):
    name: str
    version: str

    def select(self, view, candidates: Sequence[Candidate], seed: int) -> Selection | None: ...


@runtime_checkable
class Pacing(Protocol):
    name: str
    version: str

    def decide(self, view) -> PacingDecision: ...


class FixedProposer:
    """A fixed, ordered candidate list. Stateless: what is still eligible is derived from the store."""

    version = "1"

    def __init__(self, candidates: Sequence[Candidate], name: str = "fixed"):
        self.candidates, self.name = tuple(candidates), name

    def propose(self, view):
        return self.candidates


class DedupOnlySelector:
    """Pending-unaware baseline: the first proposed candidate whose key is not in the dedup set."""

    name = "dedup_only"
    version = "1"

    def select(self, view, candidates, seed):
        eligible, seen = [], set()
        for candidate in candidates:
            key = candidate.canonical_key
            if key not in view.dedup_keys and key not in seen:
                seen.add(key)
                eligible.append(candidate)
        if not eligible:
            return None
        chosen = eligible[0]
        return Selection(chosen, 1.0, {
            "pending_handling": "dedup_only", "eligible_count": len(eligible),
            "eligible_digest": digest(sorted(seen)), "pending_digest": digest(sorted(view.pending_keys)),
            "probabilities": {chosen.canonical_key: 1.0}})


class GreedyPacing:
    """Dispatch whenever there is a slot (design 5.4 D3/D4); the core still enforces every guard."""

    name = "greedy"
    version = "1"

    def decide(self, view):
        return PacingDecision("dispatch") if view.slot_available else PacingDecision("wait", "no_slot")


# ---- reservoir selection (controller-pacing) -------------------------------------------------------

@dataclass(frozen=True)
class SelectionContext:
    view: Any  # CampaignView
    candidates: tuple[Candidate, ...]  # eligible reservoir candidates, first-occurrence order
    q0: Mapping[str, float]
    pending: tuple  # PendingItem, decision order
    seed: SelectionSeed
    count: int
    history: tuple  # Observation, ingest order, up to head_ingest_seq
    head_ingest_seq: int
    legacy_seed: int = 0  # the core's integer decision seed, for controller-core selectors


@dataclass(frozen=True)
class Choice:
    keys: tuple[str, ...]
    probabilities: tuple[float, ...]  # per-step conditional probability of each chosen key
    record: Mapping[str, Any]
    inputs: Mapping[str, Any] = field(default_factory=dict)


class LegacySelector:
    """A controller-core ``Selector`` (``select(view, candidates, seed)``): pending experiments act only through the
    framework's dedup; the record says so. ``count`` > 1 repeats the call on the remaining candidates."""

    pending_aware = False

    def __init__(self, selector):
        self.selector = selector
        self.name, self.version = getattr(selector, "name", "legacy"), getattr(selector, "version", "1")

    def describe(self) -> dict:
        return {"name": self.name, "version": self.version, "pending_handling": "dedup_only", "pending_aware": False}

    def choose(self, context: SelectionContext) -> Choice | None:
        remaining, keys, probabilities, steps = list(context.candidates), [], [], []
        for j in range(context.count):
            selection = self.selector.select(context.view, remaining, context.legacy_seed + j)
            if selection is None:
                break
            key = selection.candidate.canonical_key
            keys.append(key)
            probabilities.append(float(selection.probability))
            steps.append({**dict(selection.record), "selected": key})
            remaining = [c for c in remaining if c.canonical_key != key]
        if not keys:
            return None
        return Choice(tuple(keys), tuple(probabilities),
                      normalized({"selector": plugin_identity(self.selector, "selector"),
                                   "pending_handling": "dedup_only", "pending_aware": False,
                                   "probability_kind": "selector_reported", "steps": steps}),
                      {"head_ingest_seq": context.head_ingest_seq, "count": context.count})


def as_reservoir_selector(selector):
    """Paced controllers accept a ``ReferenceSelector``-like object (``choose``) or a core ``Selector``."""
    return selector if hasattr(selector, "choose") else LegacySelector(selector)


def as_reservoir_selector(selector):
    """Paced controllers accept a ``ReferenceSelector``-like object (``choose``) or a core ``Selector``."""
    return selector if hasattr(selector, "choose") else LegacySelector(selector)
