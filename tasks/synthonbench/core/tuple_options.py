"""Deterministic complete-tuple options for the direct LLM baseline."""

from __future__ import annotations

from collections.abc import Sequence
from math import prod
from typing import Protocol

import numpy as np

COMPLETE_TUPLE_OPTION_COUNT = 8


class _SynthonOption(Protocol):
    synthon_id: int


def complete_tuple_options(
    slots: Sequence[Sequence[_SynthonOption]],
    rng: np.random.Generator,
) -> tuple[tuple[int, ...], ...]:
    """Sample complete tuples uniformly without enumerating the Cartesian space."""

    total = prod(len(slot) for slot in slots)
    count = min(COMPLETE_TUPLE_OPTION_COUNT, total)
    flat_indices = rng.choice(total, size=count, replace=False)
    return tuple(_tuple_at_index(slots, int(index)) for index in flat_indices)


def _tuple_at_index(
    slots: Sequence[Sequence[_SynthonOption]],
    index: int,
) -> tuple[int, ...]:
    selected = []
    for slot in slots:
        selected.append(slot[index % len(slot)].synthon_id)
        index //= len(slot)
    return tuple(selected)


__all__ = ["complete_tuple_options"]
