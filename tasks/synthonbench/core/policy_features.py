"""Public, task-local features for compiled SynthonBench policy means."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from ldm_tts.contracts import Candidate
from tasks.synthonbench.core.space_order import (
    ordered_positions,
    ordered_reactions,
    ordered_synthon_ids,
)

DESCRIPTOR_NAMES = (
    "molecular_weight",
    "crippen_logp",
    "tpsa",
    "hbond_donor_count",
    "hbond_acceptor_count",
    "rotatable_bond_count",
    "ring_count",
    "heavy_atom_count",
    "formal_charge",
    "fraction_csp3",
)


class SynthonPolicyFeatureEncoder:
    """Encode official tuples without assembling products or reading oracle data."""

    def __init__(self, space: Any, allowed_reactions: Sequence[str]) -> None:
        self.space = space
        self.reactions = ordered_reactions(allowed_reactions)
        if not self.reactions:
            raise ValueError("Synthon policy features require allowed reactions")
        self._reaction_index = {
            reaction_id: index for index, reaction_id in enumerate(self.reactions)
        }
        self._positions = {
            reaction_id: ordered_positions(space, reaction_id)
            for reaction_id in self.reactions
        }
        if any(not positions for positions in self._positions.values()):
            raise ValueError("Every Synthon policy reaction must have public slots")
        self.max_slots = max(len(positions) for positions in self._positions.values())
        self._capacities = {
            reaction_id: int(space.product_count_estimate(reaction_id))
            for reaction_id in self.reactions
        }
        if any(capacity < 1 for capacity in self._capacities.values()):
            raise ValueError("Synthon policy reaction capacities must be positive")
        self._descriptors, self.space_digest, descriptor_version = (
            _load_public_descriptors(space, self.reactions, self._positions)
        )
        descriptor_matrix = np.asarray(list(self._descriptors.values()), dtype=float)
        self.descriptor_location = descriptor_matrix.mean(axis=0)
        scale = descriptor_matrix.std(axis=0)
        self.descriptor_scale = np.where(scale > 0.0, scale, 1.0)
        self.feature_names, self.feature_groups = _feature_contract(
            self.reactions,
            self.max_slots,
        )
        self.dimension = len(self.feature_names)
        self.version = _feature_version(
            space_digest=self.space_digest,
            reactions=self.reactions,
            descriptor_version=descriptor_version,
            max_slots=self.max_slots,
        )

    def encode_candidate(self, candidate: Candidate) -> np.ndarray:
        payload = candidate.payload
        if not isinstance(payload, Mapping):
            raise TypeError("Synthon policy candidate payload must be a mapping")
        reaction_id = payload.get("reaction_id")
        synthon_ids = payload.get("synthon_ids")
        if not isinstance(reaction_id, str) or not isinstance(synthon_ids, list):
            raise ValueError("Synthon policy candidate requires reaction_id and synthon_ids")
        return self.encode_tuple(reaction_id, synthon_ids)

    def encode_candidate_id(self, candidate_id: str) -> np.ndarray:
        reaction_id, synthon_ids = candidate_tuple_from_id(candidate_id)
        return self.encode_tuple(reaction_id, synthon_ids)

    def encode_tuple(
        self,
        reaction_id: str,
        synthon_ids: Sequence[int],
    ) -> np.ndarray:
        positions = self._positions.get(reaction_id)
        if positions is None:
            raise ValueError("Synthon policy reaction is outside the allowed space")
        normalized_ids = _integer_ids(synthon_ids)
        if len(normalized_ids) != len(positions):
            raise ValueError("Synthon policy tuple does not match the reaction slot count")
        values = np.zeros(self.dimension, dtype=float)
        values[self._reaction_index[reaction_id]] = 1.0
        offset = len(self.reactions)
        values[offset] = len(positions)
        values[offset + 1] = math.log1p(self._capacities[reaction_id])
        offset += 2
        values[offset : offset + len(positions)] = 1.0
        offset += self.max_slots
        width = len(DESCRIPTOR_NAMES)
        for slot_index, (position, synthon_id) in enumerate(
            zip(positions, normalized_ids, strict=True)
        ):
            descriptor = self._descriptors.get((reaction_id, position, synthon_id))
            if descriptor is None:
                raise ValueError("Synthon policy tuple contains an invalid public synthon")
            start = offset + slot_index * width
            values[start : start + width] = (
                descriptor - self.descriptor_location
            ) / self.descriptor_scale
        if not np.isfinite(values).all():
            raise ValueError("Synthon policy features must be finite")
        return values

    def describe_tuple(
        self,
        reaction_id: str,
        synthon_ids: Sequence[int],
    ) -> dict[str, Any]:
        positions = self._positions.get(reaction_id)
        normalized_ids = _integer_ids(synthon_ids)
        if positions is None or len(positions) != len(normalized_ids):
            raise ValueError("Cannot describe an invalid Synthon policy tuple")
        synthons = []
        for position, synthon_id in zip(positions, normalized_ids, strict=True):
            key = (reaction_id, position, synthon_id)
            if key not in self._descriptors:
                raise ValueError("Cannot describe an invalid public synthon")
            smiles = self.space.synthon_smiles(*key)
            synthons.append(
                {
                    "position": position,
                    "synthon_id": synthon_id,
                    "smiles": smiles,
                }
            )
        return {
            "reaction_id": reaction_id,
            "synthon_ids": list(normalized_ids),
            "synthons": synthons,
        }

    def descriptor_statistics(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "space_sha256": self.space_digest,
            "names": list(DESCRIPTOR_NAMES),
            "location": self.descriptor_location.tolist(),
            "scale": self.descriptor_scale.tolist(),
        }


def candidate_tuple_from_id(candidate_id: str) -> tuple[str, tuple[int, ...]]:
    prefix = "synthonbench:"
    if not candidate_id.startswith(prefix):
        raise ValueError("Synthon policy history has an invalid candidate ID")
    product_id = candidate_id[len(prefix) :]
    reaction_id, separator, raw_ids = product_id.partition("|")
    if not separator or not reaction_id or not raw_ids:
        raise ValueError("Synthon policy history has an invalid product ID")
    try:
        synthon_ids = tuple(int(value) for value in raw_ids.split("_"))
    except ValueError as exc:
        raise ValueError("Synthon policy product ID contains invalid synthon IDs") from exc
    return reaction_id, synthon_ids


def _load_public_descriptors(
    space: Any,
    reactions: Sequence[str],
    positions_by_reaction: Mapping[str, Sequence[int]],
) -> tuple[dict[tuple[str, int, int], np.ndarray], str, str]:
    from rdkit import rdBase

    digest = hashlib.sha256()
    descriptors: dict[tuple[str, int, int], np.ndarray] = {}
    for reaction_id in reactions:
        for position in positions_by_reaction[reaction_id]:
            for synthon_id in ordered_synthon_ids(space, reaction_id, position):
                smiles = space.synthon_smiles(reaction_id, position, synthon_id)
                if not isinstance(smiles, str) or not smiles:
                    raise ValueError("Synthon policy features require public SMILES")
                key = (reaction_id, position, synthon_id)
                descriptors[key] = _descriptor_vector(smiles)
                digest.update(
                    json.dumps(
                        [reaction_id, position, synthon_id, smiles],
                        ensure_ascii=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                )
                digest.update(b"\n")
    if not descriptors:
        raise ValueError("Synthon policy feature space has no public synthons")
    return (
        descriptors,
        digest.hexdigest(),
        f"rdkit_{rdBase.rdkitVersion}_public_synthon_descriptors_v1",
    )


def _descriptor_vector(smiles: str) -> np.ndarray:
    from rdkit import Chem
    from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"RDKit cannot parse source synthon SMILES: {smiles!r}")
    values = np.asarray(
        (
            Descriptors.MolWt(molecule),
            Crippen.MolLogP(molecule),
            rdMolDescriptors.CalcTPSA(molecule),
            Lipinski.NumHDonors(molecule),
            Lipinski.NumHAcceptors(molecule),
            Lipinski.NumRotatableBonds(molecule),
            rdMolDescriptors.CalcNumRings(molecule),
            molecule.GetNumHeavyAtoms(),
            sum(atom.GetFormalCharge() for atom in molecule.GetAtoms()),
            rdMolDescriptors.CalcFractionCSP3(molecule),
        ),
        dtype=float,
    )
    if not np.isfinite(values).all():
        raise ValueError("RDKit produced non-finite public synthon descriptors")
    values.setflags(write=False)
    return values


def _feature_contract(
    reactions: Sequence[str],
    max_slots: int,
) -> tuple[tuple[str, ...], dict[str, tuple[int, int]]]:
    names = [f"reaction={json.dumps(value)}" for value in reactions]
    groups = {"reaction": (0, len(names))}
    start = len(names)
    names.extend(("reaction_slot_count", "reaction_log1p_product_capacity"))
    groups["reaction_context"] = (start, len(names))
    start = len(names)
    names.extend(f"slot_{index}_present" for index in range(max_slots))
    groups["slot_presence"] = (start, len(names))
    for slot_index in range(max_slots):
        start = len(names)
        names.extend(
            f"slot_{slot_index}_{descriptor}" for descriptor in DESCRIPTOR_NAMES
        )
        groups[f"slot_{slot_index}_descriptors"] = (start, len(names))
    return tuple(names), groups


def _feature_version(
    *,
    space_digest: str,
    reactions: Sequence[str],
    descriptor_version: str,
    max_slots: int,
) -> str:
    payload = json.dumps(
        {
            "space_sha256": space_digest,
            "reactions": list(reactions),
            "descriptor_version": descriptor_version,
            "descriptor_names": list(DESCRIPTOR_NAMES),
            "max_slots": max_slots,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return f"synthon_policy_public_features_{digest[:16]}_v1"


def _integer_ids(values: Sequence[int]) -> tuple[int, ...]:
    normalized = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError("Synthon policy synthon IDs must be integers")
        normalized.append(int(value))
    return tuple(normalized)


__all__ = [
    "DESCRIPTOR_NAMES",
    "SynthonPolicyFeatureEncoder",
    "candidate_tuple_from_id",
]
