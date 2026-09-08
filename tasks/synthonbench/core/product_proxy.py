"""Task-local product proxy built only from public synthon structures."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from ldm_tts.contracts import Candidate
from tasks.synthonbench.core.space_order import (
    ordered_positions,
    ordered_reactions,
    ordered_synthon_ids,
)


DEFAULT_MORGAN_RADIUS = 2


class SynthonProductProxy:
    """Represent a tuple by the sum of raw-connector Count-Morgan vectors.

    The proxy deliberately does not assemble a product molecule.  It uses only
    the public SMILES attached to each released synthon and keeps the ``[U]``
    and ``[Np]`` connector atoms intact, so it cannot inspect an official score
    table or depend on reaction execution at campaign time.
    """

    def __init__(
        self,
        space: Any,
        allowed_reactions: Sequence[str],
        *,
        fingerprint_bits: int,
        radius: int = DEFAULT_MORGAN_RADIUS,
        include_chirality: bool = False,
    ) -> None:
        if fingerprint_bits < 1 or radius < 1:
            raise ValueError("fingerprint_bits and radius must be positive")
        self.space = space
        self.reactions = ordered_reactions(allowed_reactions)
        if not self.reactions:
            raise ValueError("product proxy requires at least one allowed reaction")
        self._reaction_set = frozenset(self.reactions)
        self.fingerprint_bits = int(fingerprint_bits)
        self.radius = int(radius)
        self.include_chirality = bool(include_chirality)
        self._counts: dict[tuple[str, int, int], np.ndarray] = {}
        self._generator: Any | None = None

    @property
    def version(self) -> str:
        chirality = "chiral" if self.include_chirality else "achiral"
        return (
            f"raw_connector_count_morgan_r{self.radius}_"
            f"{self.fingerprint_bits}bit_{chirality}_v1"
        )

    def candidate_counts(self, candidate: Candidate) -> np.ndarray:
        """Return the public product proxy for a validated candidate payload."""

        reaction_id, synthon_ids = _candidate_tuple(candidate)
        return self.tuple_counts(reaction_id, synthon_ids)

    def tuple_counts(self, reaction_id: str, synthon_ids: Sequence[int]) -> np.ndarray:
        """Sum Count-Morgan vectors in official reaction-slot order."""

        positions = self._positions(reaction_id)
        normalized_ids = _normalize_synthon_ids(synthon_ids, len(positions))
        result = np.zeros(self.fingerprint_bits, dtype=float)
        for position, synthon_id in zip(positions, normalized_ids, strict=True):
            result += self._cached_counts(reaction_id, position, synthon_id)
        return result

    def synthon_counts(self, reaction_id: str, position: int, synthon_id: int) -> np.ndarray:
        """Return one raw-connector Count-Morgan vector as an independent array."""

        return self._cached_counts(reaction_id, int(position), int(synthon_id)).copy()

    def _cached_counts(self, reaction_id: str, position: int, synthon_id: int) -> np.ndarray:
        self._validate_synthon_key(reaction_id, position, synthon_id)
        key = (reaction_id, position, synthon_id)
        cached = self._counts.get(key)
        if cached is None:
            cached = self._count_morgan(self.space.synthon_smiles(*key))
            cached.setflags(write=False)
            self._counts[key] = cached
        return cached

    def _positions(self, reaction_id: str) -> tuple[int, ...]:
        if reaction_id not in self._reaction_set:
            raise ValueError(f"reaction {reaction_id!r} is outside the configured SynthonBench space")
        positions = ordered_positions(self.space, reaction_id)
        if not positions:
            raise ValueError(f"reaction {reaction_id!r} has no synthon positions")
        return positions

    def _validate_synthon_key(self, reaction_id: str, position: int, synthon_id: int) -> None:
        if position not in self._positions(reaction_id):
            raise ValueError(f"position {position} is invalid for reaction {reaction_id!r}")
        valid_ids = set(ordered_synthon_ids(self.space, reaction_id, position))
        if synthon_id not in valid_ids:
            raise ValueError(
                f"synthon ID {synthon_id} is invalid for reaction {reaction_id!r} position {position}"
            )

    def _count_morgan(self, smiles: str | None) -> np.ndarray:
        if not isinstance(smiles, str) or not smiles:
            raise ValueError("synthon product proxy requires a non-empty public SMILES string")
        from rdkit import Chem
        from rdkit.Chem import rdFingerprintGenerator

        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise ValueError(f"RDKit cannot parse source synthon SMILES: {smiles!r}")
        if self._generator is None:
            self._generator = rdFingerprintGenerator.GetMorganGenerator(
                radius=self.radius,
                fpSize=self.fingerprint_bits,
                includeChirality=self.include_chirality,
            )
        fingerprint = self._generator.GetCountFingerprint(molecule)
        output = np.zeros(self.fingerprint_bits, dtype=float)
        for index, count in fingerprint.GetNonzeroElements().items():
            output[int(index)] = float(count)
        return output


def _candidate_tuple(candidate: Candidate) -> tuple[str, tuple[int, ...]]:
    payload = candidate.payload
    if not isinstance(payload, Mapping):
        raise TypeError("SynthonBench candidate payload must be a mapping")
    reaction_id = payload.get("reaction_id")
    synthon_ids = payload.get("synthon_ids")
    if not isinstance(reaction_id, str) or not reaction_id:
        raise ValueError("SynthonBench candidate requires a non-empty reaction_id")
    if not isinstance(synthon_ids, list):
        raise TypeError("SynthonBench candidate synthon_ids must be a list")
    return reaction_id, _normalize_synthon_ids(synthon_ids, len(synthon_ids))


def _normalize_synthon_ids(raw_ids: Sequence[int], expected_count: int) -> tuple[int, ...]:
    if len(raw_ids) != expected_count:
        raise ValueError("candidate synthon tuple does not match reaction slot count")
    result: list[int] = []
    for raw_id in raw_ids:
        if isinstance(raw_id, bool) or not isinstance(raw_id, (int, np.integer)):
            raise TypeError("SynthonBench synthon IDs must be integers")
        result.append(int(raw_id))
    return tuple(result)


__all__ = ["DEFAULT_MORGAN_RADIUS", "SynthonProductProxy"]
