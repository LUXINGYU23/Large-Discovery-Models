"""Reservoir maintenance: proposal occurrences, information-age decay, q0 and fresh stock.

State is an immutable tuple of occurrences; every function is pure, so weights, q0
and the fresh stock S can be recomputed from occurrences and the number of booked
results (invariant I12). An occurrence's age is the number of results booked after
its visibility watermark; its weight is 2^(-age/h).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from ldm_tts.contracts import Candidate, ReservoirBuildResult


@dataclass(frozen=True)
class Occurrence:
    """One proposal of a candidate by one lineage at a visibility watermark."""

    candidate: Candidate
    lineage: str
    generation: int
    watermark: int

    def __post_init__(self) -> None:
        for value in (self.generation, self.watermark):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("occurrence generation and watermark must be non-negative integers")

    @property
    def key(self) -> str:
        return self.candidate.canonical_key


@dataclass(frozen=True)
class ReservoirView:
    """Eligible candidates (first-occurrence order), their total weights and q0."""

    candidates: tuple[Candidate, ...]
    weights: tuple[float, ...]
    q0: tuple[float, ...]
    fresh_stock: float
    results_booked: int
    occurrence_count: int

    def q0_by_key(self) -> dict[str, float]:
        return {item.canonical_key: mass for item, mass in zip(self.candidates, self.q0)}

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def occurrence_age(occurrence: Occurrence, results_booked: int) -> int:
    age = results_booked - occurrence.watermark
    if age < 0:
        raise ValueError("occurrence watermark is ahead of the booked results")
    return age


def occurrence_weight(occurrence: Occurrence, results_booked: int, half_life: float | None) -> float:
    age = occurrence_age(occurrence, results_booked)
    return 1.0 if half_life is None else 2.0 ** (-age / half_life)


def reservoir_view(
    occurrences: Iterable[Occurrence],
    *,
    results_booked: int,
    half_life: float | None,
    min_weight: float = 0.0,
    excluded_keys: Iterable[str] = (),
) -> ReservoirView:
    """q0(x) = sum of x's occurrence weights / sum over all eligible occurrences."""
    excluded = set(excluded_keys)
    totals: dict[str, list] = {}
    count = 0
    for item in occurrences:
        weight = occurrence_weight(item, results_booked, half_life)
        if item.key in excluded or weight < min_weight:
            continue
        totals.setdefault(item.key, [item.candidate, 0.0])[1] += weight
        count += 1
    stock = sum(weight for _, weight in totals.values())
    return ReservoirView(
        candidates=tuple(candidate for candidate, _ in totals.values()),
        weights=tuple(weight for _, weight in totals.values()),
        q0=tuple(weight / stock for _, weight in totals.values()),
        fresh_stock=stock,
        results_booked=results_booked,
        occurrence_count=count,
    )


def synchronous_q0(result: ReservoirBuildResult) -> dict[str, float]:
    """Empirical proposal mass of one synchronous build: admitted plus in-build duplicates."""
    counts = {item.canonical_key: 1 for item in result.candidates}
    for rejection in result.rejections:
        key = rejection.metadata.get("canonical_key")
        if rejection.reason == "duplicate" and key in counts:
            counts[key] += 1
    total = sum(counts.values())
    return {key: value / total for key, value in counts.items()}


class _Maintenance:
    name = ""
    version = "1"
    half_life: float | None = None
    min_weight = 0.0

    def withdraw(self, occurrences, *, lineage: str, keys: Iterable[str]) -> tuple[Occurrence, ...]:
        keys = set(keys)
        return tuple(item for item in occurrences if item.lineage != lineage or item.key not in keys)

    def invalidate(self, occurrences, keys: Iterable[str]) -> tuple[Occurrence, ...]:
        """Measured or in-flight candidates lose every occurrence."""
        keys = set(keys)
        return tuple(item for item in occurrences if item.key not in keys)

    def view(self, occurrences, *, results_booked: int, excluded_keys: Iterable[str] = ()) -> ReservoirView:
        return reservoir_view(occurrences, results_booked=results_booked, half_life=self.half_life,
                              min_weight=self.min_weight, excluded_keys=excluded_keys)


class ResidentDecayReservoir(_Maintenance):
    """Resident reservoir with information-age decay, renewal, withdrawal and eviction."""

    name = "resident_decay"

    def __init__(self, *, half_life: float, capacity: int, min_weight: float = 1 / 16,
                 stale_fraction: float = 0.5) -> None:
        if not math.isfinite(half_life) or half_life <= 0:
            raise ValueError("half_life must be finite and positive")
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise ValueError("capacity must be a positive integer")
        if not 0 <= min_weight < 1 or not 0 < stale_fraction <= 1:
            raise ValueError("min_weight must be in [0, 1) and stale_fraction in (0, 1]")
        self.half_life, self.capacity = float(half_life), capacity
        self.min_weight, self.stale_fraction = float(min_weight), float(stale_fraction)

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "version": self.version, "half_life": self.half_life,
                "capacity": self.capacity, "min_weight": self.min_weight,
                "stale_fraction": self.stale_fraction}

    def add(self, occurrences, new: Sequence[Occurrence], *, results_booked: int,
            blocked_keys: Iterable[str] = ()) -> tuple[Occurrence, ...]:
        """Same lineage re-proposing a candidate replaces its old occurrence (weight back to 1)."""
        blocked = set(blocked_keys)
        state = list(self.evict(occurrences, results_booked=results_booked))
        for item in new:
            occurrence_age(item, results_booked)
            if item.key in blocked:
                continue
            state = [old for old in state if (old.lineage, old.key) != (item.lineage, item.key)]
            state.append(item)
        return self._enforce_capacity(state, results_booked)

    def evict(self, occurrences, *, results_booked: int) -> tuple[Occurrence, ...]:
        return tuple(item for item in occurrences
                     if occurrence_weight(item, results_booked, self.half_life) >= self.min_weight)

    def is_stale(self, view: ReservoirView) -> bool:
        return view.fresh_stock < self.stale_fraction * self.capacity

    def _enforce_capacity(self, state: list[Occurrence], results_booked: int) -> tuple[Occurrence, ...]:
        # Candidate weight ranking is invariant in results_booked (common 2^(-n/h) factor).
        rank: dict[str, list] = {}
        for position, item in enumerate(state):
            entry = rank.setdefault(item.key, [0.0, item.watermark, position])
            entry[0] += occurrence_weight(item, results_booked, self.half_life)
            entry[1] = max(entry[1], item.watermark)
        drop = set(sorted(rank, key=lambda key: tuple(rank[key]))[:max(0, len(rank) - self.capacity)])
        return tuple(item for item in state if item.key not in drop)


class GenerationReservoir(_Maintenance):
    """Generation replacement: equivalent to the synchronous per-round ReservoirBuilder.

    A new generation clears earlier occurrences; every occurrence counts (no decay,
    no same-lineage merge); new keys beyond ``capacity`` are dropped first-come.
    """

    name = "generation_replacement"

    def __init__(self, *, capacity: int | None = None) -> None:
        if capacity is not None and (isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1):
            raise ValueError("capacity must be a positive integer")
        self.capacity = capacity

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "version": self.version, "capacity": self.capacity}

    def add(self, occurrences, new: Sequence[Occurrence], *, results_booked: int = 0,
            blocked_keys: Iterable[str] = ()) -> tuple[Occurrence, ...]:
        generations = {item.generation for item in new}
        if len(generations) > 1:
            raise ValueError("one generation per reservoir replacement")
        state = list(occurrences)
        if generations and state:
            current, incoming = max(item.generation for item in state), generations.pop()
            if incoming < current:
                raise ValueError("generation replacement cannot go back to an older generation")
            if incoming > current:
                state = []
        blocked = set(blocked_keys)
        present = {item.key for item in state}
        for item in new:
            occurrence_age(item, results_booked)
            if item.key in blocked:
                continue
            if item.key not in present:
                if self.capacity is not None and len(present) >= self.capacity:
                    continue
                present.add(item.key)
            state.append(item)
        return tuple(state)

    def evict(self, occurrences, *, results_booked: int = 0) -> tuple[Occurrence, ...]:
        return tuple(occurrences)

    def is_stale(self, view: ReservoirView) -> bool:
        return not view.candidates


__all__ = [
    "GenerationReservoir",
    "Occurrence",
    "ReservoirView",
    "ResidentDecayReservoir",
    "occurrence_age",
    "occurrence_weight",
    "reservoir_view",
    "synchronous_q0",
]
