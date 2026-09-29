"""Shared alphabet utilities for DSL atoms.

The 20 canonical amino acids used across :class:`NeighborSampling`,
:class:`Or`, and :class:`BiasAtom` implementations.
"""
from __future__ import annotations

AA: str = "ACDEFGHIKLMNPQRSTVWY"
"""The 20 standard amino acids, indexed by position 0..19."""

SEQ_LEN: int = 11
"""Fixed CDRH3 sequence length for AntBO."""

AA_TO_IDX: dict[str, int] = {aa: i for i, aa in enumerate(AA)}
"""Letter -> column index map."""


def idx_to_aa(i: int) -> str:
    """Index (0..19) -> amino acid letter."""
    return AA[int(i)]


def aa_to_idx(aa: str) -> int:
    """Amino acid letter -> index (0..19)."""
    return AA_TO_IDX[aa.upper()]


def hamming(a: "list[int]", b: "list[int]") -> int:
    """Hamming distance between two integer-encoded sequences."""
    if len(a) != len(b):
        raise ValueError(f"length mismatch: {len(a)} vs {len(b)}")
    return sum(1 for x, y in zip(a, b) if x != y)
