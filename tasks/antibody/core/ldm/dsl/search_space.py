"""Search space atoms — define WHERE the BO samples and HOW.

Hierarchy::

    SearchSpaceAtom (ABC)
        LatinHyperCubeSampling(num)                            — full-space LHS
        NeighborSampling(center, fixed, radius, mut_pr, budget) — geometric sampling
        LocalSearch(center, fixed, radius, restart, steps)     — hill-climbing
        Or(*children)                                          — union (no weights)

``sample(n, rng, timeout_s, strict)`` is implemented ONCE in the base
class: loops ``sample_one``, deduplicates, detects exhaustion, enforces
timeout.  ``LocalSearch`` does not support ``sample_one`` (it is
executed via ``parallel_local_search`` with a trained GP).
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from itertools import combinations, product
from math import comb
from typing import Iterator

import numpy as np

from tasks.antibody.core.ldm.dsl.alphabet import (
    SEQ_LEN,
    aa_to_idx,
    idx_to_aa,
)
from tasks.antibody.core.ldm.dsl.exceptions import SamplingTimeout

_UNIVERSE = 20 ** SEQ_LEN
_DUP_EXHAUST_THRESHOLD = 2000


# ------------------------------------------------------------------ #
#  ABC
# ------------------------------------------------------------------ #

class SearchSpaceAtom(ABC):
    """Public ABC for all search-space atoms."""

    @property
    @abstractmethod
    def size_ub(self) -> int:
        """Tight upper bound on the set size (informational)."""

    @property
    def budget(self) -> int:
        """Number of GP evaluations this atom consumes when executed."""
        return 0

    def sample_one(self, rng: np.random.Generator) -> list[int]:
        """Sample a single sequence.  Override in sampling-capable atoms."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support sample_one"
        )

    @abstractmethod
    def __contains__(self, seq) -> bool:
        """Membership test (used for GP training-data filtering)."""

    @abstractmethod
    def __iter__(self) -> Iterator[list[int]]:
        """Enumerate all sequences (may be very large)."""

    # ------------------------------------------------------------------ #
    #  Concrete sample() — dedup, timeout, exhaustion, strict
    # ------------------------------------------------------------------ #
    def sample(
        self,
        n: int,
        rng: np.random.Generator | None = None,
        timeout_s: float = 5.0,
        strict: bool = False,
    ) -> list[list[int]]:
        """Draw up to ``n`` unique sequences."""
        rng = rng if rng is not None else np.random.default_rng()
        deadline = time.time() + timeout_s
        seen: set[tuple[int, ...]] = set()
        results: list[list[int]] = []
        dup_streak = 0

        while len(results) < n:
            if time.time() > deadline:
                break
            seq = self.sample_one(rng)
            key = tuple(seq)
            if key in seen:
                dup_streak += 1
                if dup_streak >= _DUP_EXHAUST_THRESHOLD:
                    break
            else:
                seen.add(key)
                results.append(seq)
                dup_streak = 0

        if strict and len(results) < n:
            raise SamplingTimeout(
                f"{type(self).__name__}: only {len(results)}/{n} unique "
                f"sequences in {timeout_s}s"
            )
        return results

    def __repr__(self) -> str:
        return f"{type(self).__name__}(...)"


# ------------------------------------------------------------------ #
#  Shared helper for atoms with center + fixed + radius
# ------------------------------------------------------------------ #

class _RegionMixin:
    """Shared logic for atoms parameterised by center / fixed / radius."""

    def _init_region(self, center: str, fixed: str | None, radius: int | None):
        if not isinstance(center, str) or len(center) != SEQ_LEN:
            raise ValueError(f"center must be {SEQ_LEN}-char str, got {center!r}")
        if fixed is not None:
            if not isinstance(fixed, str) or len(fixed) != SEQ_LEN:
                raise ValueError(f"fixed must be {SEQ_LEN}-char str or None, got {fixed!r}")
            if any(ch not in "*." for ch in fixed):
                raise ValueError(f"fixed chars must be '*' or '.', got {fixed!r}")
        if radius is not None:
            if not isinstance(radius, int) or radius < 0:
                raise ValueError(f"radius must be non-negative int or None, got {radius!r}")

        self._center = [aa_to_idx(c) for c in center.upper()]
        self._fixed = fixed if fixed is not None else "*" * SEQ_LEN
        self._radius = radius
        self._fixed_positions = [i for i in range(SEQ_LEN) if self._fixed[i] == "."]
        self._mutable_positions = [i for i in range(SEQ_LEN) if self._fixed[i] == "*"]

    @property
    def center(self) -> str:
        return "".join(idx_to_aa(i) for i in self._center)

    @property
    def center_idx(self) -> list[int]:
        return list(self._center)

    @property
    def fixed(self) -> str:
        return self._fixed

    @property
    def radius(self) -> int | None:
        return self._radius

    @property
    def fixed_positions(self) -> list[int]:
        return list(self._fixed_positions)

    @property
    def mutable_positions(self) -> list[int]:
        return list(self._mutable_positions)

    def _contains(self, seq_list: list[int]) -> bool:
        if len(seq_list) != SEQ_LEN:
            return False
        diffs = 0
        for i in range(SEQ_LEN):
            if seq_list[i] != self._center[i]:
                if self._fixed[i] != "*":
                    return False
                diffs += 1
        if self._radius is not None and diffs > self._radius:
            return False
        return True

    def _size_ub(self) -> int:
        n_mut = len(self._mutable_positions)
        cap = self._radius if self._radius is not None else n_mut
        cap = min(cap, n_mut)
        return sum(comb(n_mut, d) * (19 ** d) for d in range(cap + 1))

    def _iter_region(self) -> Iterator[list[int]]:
        max_d = (
            min(self._radius, len(self._mutable_positions))
            if self._radius is not None
            else len(self._mutable_positions)
        )
        for d in range(max_d + 1):
            for positions in combinations(self._mutable_positions, d):
                for substitutions in product(range(19), repeat=d):
                    seq = list(self._center)
                    for pos, sub in zip(positions, substitutions):
                        original = seq[pos]
                        new = sub if sub < original else sub + 1
                        seq[pos] = new
                    yield seq

    def _sample_one_region(self, rng: np.random.Generator) -> list[int]:
        """Stepwise geometric mutation (shared by NeighborSampling)."""
        seq = list(self._center)
        mutated: set[int] = set()
        available = self._mutable_positions

        if self._radius is not None and self._radius > 0:
            max_steps = int(self._radius / 0.5) + 1
        else:
            max_steps = len(available) if available else 1

        for _ in range(max_steps):
            if len(mutated) >= len(available):
                break
            if self._radius is not None and len(mutated) >= self._radius:
                break
            if rng.random() >= 0.5:
                break
            candidates = [p for p in available if p not in mutated]
            if not candidates:
                break
            pos = candidates[int(rng.integers(0, len(candidates)))]
            original = seq[pos]
            new = int(rng.integers(0, 19))
            if new >= original:
                new += 1
            seq[pos] = new
            mutated.add(pos)

        return seq


# ------------------------------------------------------------------ #
#  LatinHyperCubeSampling
# ------------------------------------------------------------------ #

class LatinHyperCubeSampling(SearchSpaceAtom):
    """Full-space uniform sampling via Latin Hypercube."""

    def __init__(self, num: int) -> None:
        if not isinstance(num, int) or num <= 0:
            raise ValueError(f"num must be positive int, got {num!r}")
        self._num = num

    @property
    def num(self) -> int:
        return self._num

    @property
    def budget(self) -> int:
        return self._num

    @property
    def size_ub(self) -> int:
        return _UNIVERSE

    def __contains__(self, seq) -> bool:
        return True

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng()
        while True:
            yield [int(rng.integers(0, 20)) for _ in range(SEQ_LEN)]

    def sample_one(self, rng: np.random.Generator) -> list[int]:
        return [int(rng.integers(0, 20)) for _ in range(SEQ_LEN)]

    def __repr__(self) -> str:
        return f"LatinHyperCubeSampling(num={self._num})"


# ------------------------------------------------------------------ #
#  NeighborSampling
# ------------------------------------------------------------------ #

class NeighborSampling(SearchSpaceAtom, _RegionMixin):
    """Geometric mutation sampling around a center."""

    def __init__(
        self,
        center: str,
        fixed: str | None = None,
        radius: int | None = None,
        mut_pr: float = 0.5,
        budget: int = 1000,
    ) -> None:
        self._init_region(center, fixed, radius)
        if not 0.0 < mut_pr <= 1.0:
            raise ValueError(f"mut_pr must be in (0, 1], got {mut_pr!r}")
        if not isinstance(budget, int) or budget <= 0:
            raise ValueError(f"budget must be positive int, got {budget!r}")
        self._mut_pr = mut_pr
        self._budget = budget

    @property
    def mut_pr(self) -> float:
        return self._mut_pr

    @property
    def budget(self) -> int:
        return self._budget

    @property
    def size_ub(self) -> int:
        return self._size_ub()

    def __contains__(self, seq) -> bool:
        seq_list = list(seq) if not isinstance(seq, list) else seq
        return self._contains(seq_list)

    def __iter__(self) -> Iterator[list[int]]:
        return self._iter_region()

    def sample_one(self, rng: np.random.Generator) -> list[int]:
        return self._sample_one_region(rng)

    def __repr__(self) -> str:
        parts: list[str] = []
        if self._fixed != "*" * SEQ_LEN:
            parts.append(f"fixed={self._fixed!r}")
        if self._radius is not None:
            parts.append(f"radius={self._radius}")
        parts.append(f"mut_pr={self._mut_pr}")
        parts.append(f"budget={self._budget}")
        return f"NeighborSampling('{self.center}', {', '.join(parts)})"


# ------------------------------------------------------------------ #
#  LocalSearch
# ------------------------------------------------------------------ #

class LocalSearch(SearchSpaceAtom, _RegionMixin):
    """Hill-climbing acquisition search around a center.

    Executed via ``parallel_local_search`` with a trained GP — does NOT
    support ``sample()``.
    """

    def __init__(
        self,
        center: str,
        fixed: str | None = None,
        radius: int | None = None,
        restart: int = 3,
        steps: int = 200,
    ) -> None:
        self._init_region(center, fixed, radius)
        if not isinstance(restart, int) or restart <= 0:
            raise ValueError(f"restart must be positive int, got {restart!r}")
        if not isinstance(steps, int) or steps <= 0:
            raise ValueError(f"steps must be positive int, got {steps!r}")
        self._restart = restart
        self._steps = steps

    @property
    def restart(self) -> int:
        return self._restart

    @property
    def steps(self) -> int:
        return self._steps

    @property
    def budget(self) -> int:
        return self._restart * (self._steps + 1)  # +1 for center evaluation

    @property
    def size_ub(self) -> int:
        return self._size_ub()

    def __contains__(self, seq) -> bool:
        seq_list = list(seq) if not isinstance(seq, list) else seq
        return self._contains(seq_list)

    def __iter__(self) -> Iterator[list[int]]:
        return self._iter_region()

    def __repr__(self) -> str:
        parts: list[str] = []
        if self._fixed != "*" * SEQ_LEN:
            parts.append(f"fixed={self._fixed!r}")
        if self._radius is not None:
            parts.append(f"radius={self._radius}")
        parts.append(f"restart={self._restart}")
        parts.append(f"steps={self._steps}")
        return f"LocalSearch('{self.center}', {', '.join(parts)})"


# ------------------------------------------------------------------ #
#  Or — union (no weights)
# ------------------------------------------------------------------ #

class Or(SearchSpaceAtom):
    """Union of children. No weights — each child carries its own budget."""

    def __init__(self, *children: SearchSpaceAtom) -> None:
        if len(children) < 2:
            raise ValueError("Or requires at least 2 children")
        for c in children:
            if not isinstance(c, SearchSpaceAtom):
                raise TypeError(f"Or child must be SearchSpaceAtom, got {type(c).__name__}")
        self._children = list(children)

    @property
    def children(self) -> list[SearchSpaceAtom]:
        return list(self._children)

    @property
    def budget(self) -> int:
        return sum(c.budget for c in self._children)

    @property
    def size_ub(self) -> int:
        return sum(c.size_ub for c in self._children)

    def __repr__(self) -> str:
        return " | ".join(repr(c) for c in self._children)

    def __contains__(self, seq) -> bool:
        return any(c.__contains__(seq) for c in self._children)

    def __iter__(self) -> Iterator[list[int]]:
        seen: set[tuple[int, ...]] = set()
        for child in self._children:
            for seq in child:
                key = tuple(seq)
                if key not in seen:
                    seen.add(key)
                    yield seq

    def sample_one(self, rng: np.random.Generator) -> list[int]:
        idx = int(rng.integers(0, len(self._children)))
        return self._children[idx].sample_one(rng)


# ------------------------------------------------------------------ #
#  Operator binding — only | (Or)
# ------------------------------------------------------------------ #

def _bind_operators() -> None:
    def __or__(self, other):
        if isinstance(other, SearchSpaceAtom):
            return Or(self, other)
        return NotImplemented

    SearchSpaceAtom.__or__ = __or__


_bind_operators()
