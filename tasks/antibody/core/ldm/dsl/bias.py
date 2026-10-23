"""Bias atoms — define HOW BO ranks candidates (additive soft score).

Hierarchy::

    BiasAtom (ABC, public via tasks.antibody.core.ldm.__init__)
        MaxCysteine(int)
        MaxHydrophobicRun(int)
        MaxAromatic(int)
        NetChargeRange(min, max)
        NoNGlycosylation()
    BiasSum(*atoms)        # additive composite, supports `+`

Each atom implements ``__call__(seq) -> float``. Higher = better (less
penalty). Score is added to the BO acquisition with weight ``bias_weight``.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod

from tasks.antibody.core.ldm.dsl.alphabet import idx_to_aa

HYDROPHOBIC = set("AILMFWVY")
AROMATIC = set("FWY")
POSITIVE = set("RKH")
NEGATIVE = set("DE")
_N_GLYCO = re.compile("N[^P][ST][^P]")


class BiasAtom(ABC):
    """Public ABC. ``core/`` external code only sees this type."""

    @abstractmethod
    def __call__(self, seq) -> float:
        """Return additive bias score; higher = more preferred."""

    def __add__(self, other) -> "BiasSum":
        if isinstance(other, BiasSum):
            return BiasSum(*self._to_list(), *other._to_list())
        if isinstance(other, BiasAtom):
            return BiasSum(*self._to_list(), other)
        return NotImplemented

    def _to_list(self) -> list["BiasAtom"]:
        if isinstance(self, BiasSum):
            return list(self._atoms)
        return [self]


class _PenaltyAtom(BiasAtom):
    """Helper: atom whose score is a sum of negative penalties."""

    def __init__(self, threshold: float, penalty_per_unit: float = 1.0) -> None:
        self.threshold = float(threshold)
        self.penalty_per_unit = float(penalty_per_unit)

    def penalty(self, amount: float) -> float:
        excess = max(0.0, amount - self.threshold)
        return -self.penalty_per_unit * excess


class MaxCysteine(_PenaltyAtom):
    """Penalty if Cys count > ``value``."""

    def __init__(self, value: int) -> None:
        super().__init__(threshold=float(value), penalty_per_unit=1.0)

    def __call__(self, seq) -> float:
        s = "".join(idx_to_aa(int(aa)) for aa in seq)
        return self.penalty(s.count("C"))

    def __repr__(self) -> str:
        return f"MaxCysteine({int(self.threshold)})"


class MaxHydrophobicRun(_PenaltyAtom):
    """Penalty if longest hydrophobic run > ``value``."""

    def __init__(self, value: int) -> None:
        super().__init__(threshold=float(value), penalty_per_unit=0.5)

    def __call__(self, seq) -> float:
        s = "".join(idx_to_aa(int(aa)) for aa in seq)
        best = cur = 0
        for ch in s:
            cur = cur + 1 if ch in HYDROPHOBIC else 0
            best = max(best, cur)
        return self.penalty(best)

    def __repr__(self) -> str:
        return f"MaxHydrophobicRun({int(self.threshold)})"


class MaxAromatic(BiasAtom):
    """Penalty if aromatic count > ``value``; bonus if >= 1 (capped at 2)."""

    def __init__(self, value: int) -> None:
        self.value = int(value)

    def __call__(self, seq) -> float:
        s = "".join(idx_to_aa(int(aa)) for aa in seq)
        cnt = sum(1 for ch in s if ch in AROMATIC)
        score = 0.0
        if cnt > self.value:
            score -= 0.25 * (cnt - self.value)
        score += 0.15 * min(cnt, 2)
        return score

    def __repr__(self) -> str:
        return f"MaxAromatic({self.value})"


class NetChargeRange(BiasAtom):
    """Penalty if |net charge| > ``max_v`` (lower bound treated softly)."""

    def __init__(self, min_v: float, max_v: float) -> None:
        if max_v < min_v:
            raise ValueError(f"max_v {max_v} < min_v {min_v}")
        self.min_v = float(min_v)
        self.max_v = float(max_v)

    def __call__(self, seq) -> float:
        s = "".join(idx_to_aa(int(aa)) for aa in seq)
        charge = sum(0.1 if ch == "H" else (1 if ch in POSITIVE else (-1 if ch in NEGATIVE else 0)) for ch in s)
        score = 0.0
        if charge < self.min_v:
            score -= 0.5 * (self.min_v - charge)
        elif charge > self.max_v:
            score -= 0.5 * (charge - self.max_v)
        return score

    def __repr__(self) -> str:
        return f"NetChargeRange({self.min_v}, {self.max_v})"


class NoNGlycosylation(BiasAtom):
    """Penalty if the sequence contains an N-X-S/T N-linked glycosylation motif."""

    def __call__(self, seq) -> float:
        s = "".join(idx_to_aa(int(aa)) for aa in seq)
        return -1.0 if _N_GLYCO.search(s) else 0.0

    def __repr__(self) -> str:
        return "NoNGlycosylation()"


class BiasSum(BiasAtom):
    """Composite of multiple bias atoms, evaluated as ``sum(a(seq) for a in atoms)``."""

    def __init__(self, *atoms: BiasAtom) -> None:
        for a in atoms:
            if not isinstance(a, BiasAtom):
                raise TypeError(f"BiasSum child must be BiasAtom, got {type(a).__name__}")
        self._atoms: list[BiasAtom] = list(atoms)

    @property
    def atoms(self) -> list[BiasAtom]:
        return list(self._atoms)

    def __call__(self, seq) -> float:
        return float(sum(a(seq) for a in self._atoms))

    def __repr__(self) -> str:
        return " + ".join(repr(a) for a in self._atoms)