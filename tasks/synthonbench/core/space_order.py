"""Canonical ordering for identifiers read from the official SynthonBench space."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def ordered_reactions(reactions: Iterable[object]) -> tuple[str, ...]:
    """Return reaction IDs in a process-independent order."""

    return tuple(sorted(str(item) for item in reactions))


def ordered_positions(space: Any, reaction_id: str) -> tuple[int, ...]:
    """Return reaction slots in their canonical numerical order."""

    return tuple(sorted(int(item) for item in space.positions(reaction_id)))


def ordered_synthon_ids(space: Any, reaction_id: str, position: int) -> tuple[int, ...]:
    """Return public synthon IDs in a process-independent order."""

    return tuple(sorted(int(item) for item in space.synthon_ids(reaction_id, position)))


__all__ = ["ordered_positions", "ordered_reactions", "ordered_synthon_ids"]
