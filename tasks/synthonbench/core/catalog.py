"""Public, deterministic synthon slates for independent LLM proposals."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

ROLE_CYCLE = ("explore", "exploit", "diversify", "scaffold_shift")
REACTION_ALLOCATIONS = ("product_weighted", "uniform")


@dataclass(frozen=True)
class SynthonOption:
    """One exact synthon option shown in a prompt."""

    position: int
    synthon_id: int
    smiles: str

    def to_dict(self) -> dict[str, object]:
        return {"position": self.position, "synthon_id": self.synthon_id, "smiles": self.smiles}


@dataclass(frozen=True)
class ProposalSlotPlan:
    """A public finite action set for one independent proposal request."""

    round_idx: int
    proposal_index: int
    reaction_id: str
    role: str
    reaction_probability: float
    slot_options: tuple[tuple[SynthonOption, ...], ...]
    uniqueness_anchor_position: int | None = None

    def allowed_ids(self) -> tuple[tuple[int, ...], ...]:
        return tuple(tuple(option.synthon_id for option in slot) for slot in self.slot_options)

    def metadata(self) -> dict[str, object]:
        return {
            "round_idx": self.round_idx,
            "proposal_index": self.proposal_index,
            "reaction_id": self.reaction_id,
            "proposal_role": self.role,
            "reaction_probability": self.reaction_probability,
            "slot_synthon_ids": [list(ids) for ids in self.allowed_ids()],
            "uniqueness_anchor_position": self.uniqueness_anchor_position,
        }


class SynthonProposalCatalog:
    """Draw reproducible reaction-conditioned public candidate slates."""

    def __init__(
        self,
        space: Any,
        *,
        allowed_reactions: Sequence[str],
        slate_size: int,
        seed: int,
        reaction_allocation: str = "product_weighted",
        direct_unique: bool = False,
        direct_proposal_count: int | None = None,
    ) -> None:
        if slate_size < 1:
            raise ValueError("slate_size must be positive")
        if seed < 0:
            raise ValueError("catalog seed must be non-negative")
        if reaction_allocation not in REACTION_ALLOCATIONS:
            raise ValueError(f"unknown reaction allocation: {reaction_allocation!r}")
        if direct_unique and (direct_proposal_count is None or direct_proposal_count < 1):
            raise ValueError("direct_unique catalog requires a positive direct_proposal_count")
        self.space = space
        self.reactions = tuple(str(item) for item in allowed_reactions)
        if not self.reactions:
            raise ValueError("proposal catalog requires at least one reaction")
        self.slate_size = int(slate_size)
        self.seed = int(seed)
        self.reaction_allocation = reaction_allocation
        self.direct_unique = bool(direct_unique)
        self.direct_proposal_count = direct_proposal_count
        self._reaction_probabilities = _reaction_probabilities(space, self.reactions, reaction_allocation)

    def build_plan(self, *, round_idx: int, proposal_index: int) -> ProposalSlotPlan:
        if round_idx < 0 or proposal_index < 0:
            raise ValueError("round_idx and proposal_index must be non-negative")
        rng = np.random.default_rng(_slot_seed(self.seed, round_idx, proposal_index))
        reaction_index = int(rng.choice(len(self.reactions), p=self._reaction_probabilities))
        reaction_id = self.reactions[reaction_index]
        if self.direct_unique:
            options, anchor_position = self._direct_options(round_idx, proposal_index, reaction_id, rng)
        else:
            options = tuple(
                _sample_slot(self.space, reaction_id, position, self.slate_size, rng)
                for position in self.space.positions(reaction_id)
            )
            anchor_position = None
        return ProposalSlotPlan(
            round_idx=round_idx,
            proposal_index=proposal_index,
            reaction_id=reaction_id,
            role=ROLE_CYCLE[(round_idx + proposal_index) % len(ROLE_CYCLE)],
            reaction_probability=float(self._reaction_probabilities[reaction_index]),
            slot_options=options,
            uniqueness_anchor_position=anchor_position,
        )

    def _direct_options(self, round_idx: int, proposal_index: int, reaction_id: str, rng) -> tuple[tuple[tuple[SynthonOption, ...], ...], int]:
        positions = tuple(self.space.positions(reaction_id))
        anchor = max(positions, key=lambda position: len(tuple(self.space.synthon_ids(reaction_id, position))))
        ordinal = self._direct_reaction_ordinal(round_idx, proposal_index, reaction_id)
        anchor_ids = tuple(int(item) for item in self.space.synthon_ids(reaction_id, anchor))
        if ordinal >= len(anchor_ids):
            raise ValueError(f"direct LLM slate cannot assign a unique anchor for reaction {reaction_id!r}")
        anchor_rng = np.random.default_rng(_anchor_seed(self.seed, reaction_id, anchor))
        anchor_id = int(anchor_rng.permutation(anchor_ids)[ordinal])
        options = tuple(
            (_option(self.space, reaction_id, position, anchor_id),)
            if position == anchor
            else _sample_slot(self.space, reaction_id, position, self.slate_size, rng)
            for position in positions
        )
        return options, anchor

    def _direct_reaction_ordinal(self, round_idx: int, proposal_index: int, reaction_id: str) -> int:
        assert self.direct_proposal_count is not None
        previous = (
            self._reaction_id(previous_round, previous_index)
            for previous_round in range(round_idx + 1)
            for previous_index in range(self.direct_proposal_count)
            if previous_round < round_idx or previous_index < proposal_index
        )
        return sum(item == reaction_id for item in previous)

    def _reaction_id(self, round_idx: int, proposal_index: int) -> str:
        rng = np.random.default_rng(_slot_seed(self.seed, round_idx, proposal_index))
        index = int(rng.choice(len(self.reactions), p=self._reaction_probabilities))
        return self.reactions[index]


def validate_payload_against_plan(payload: dict[str, object], plan: ProposalSlotPlan) -> None:
    """Require every LLM-selected ID to come from its supplied public slate."""

    if payload.get("reaction_id") != plan.reaction_id:
        raise ValueError("proposal reaction_id differs from the assigned reaction slate")
    raw_ids = payload.get("synthon_ids")
    if not isinstance(raw_ids, list) or len(raw_ids) != len(plan.slot_options):
        raise ValueError("proposal synthon_ids do not match the assigned reaction arity")
    for index, (raw_id, allowed_ids) in enumerate(zip(raw_ids, plan.allowed_ids(), strict=True)):
        if isinstance(raw_id, bool) or not isinstance(raw_id, int) or raw_id not in allowed_ids:
            raise ValueError(f"synthon_ids[{index}] is not present in the supplied slate")


def _reaction_probabilities(space: Any, reactions: Sequence[str], allocation: str) -> np.ndarray:
    if allocation == "uniform":
        return np.full(len(reactions), 1.0 / len(reactions), dtype=float)
    weights = np.asarray([space.product_count_estimate(reaction) for reaction in reactions], dtype=float)
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("reaction product counts must be finite and positive")
    return weights / float(weights.sum())


def _sample_slot(space: Any, reaction_id: str, position: int, slate_size: int,
                 rng: np.random.Generator) -> tuple[SynthonOption, ...]:
    ids = tuple(int(item) for item in space.synthon_ids(reaction_id, position))
    if not ids:
        raise ValueError(f"reaction {reaction_id!r} position {position} has no synthons")
    selected = rng.choice(ids, size=min(slate_size, len(ids)), replace=False)
    return tuple(_option(space, reaction_id, position, int(item)) for item in selected)


def _option(space: Any, reaction_id: str, position: int, synthon_id: int) -> SynthonOption:
    smiles = space.synthon_smiles(reaction_id, position, synthon_id)
    if not isinstance(smiles, str) or not smiles:
        raise ValueError(f"synthon {synthon_id} lacks a public SMILES value")
    return SynthonOption(position=position, synthon_id=synthon_id, smiles=smiles)


def _slot_seed(seed: int, round_idx: int, proposal_index: int) -> int:
    payload = json.dumps([seed, round_idx, proposal_index], separators=(",", ":")).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _anchor_seed(seed: int, reaction_id: str, position: int) -> int:
    payload = json.dumps([seed, reaction_id, position], separators=(",", ":")).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


__all__ = [
    "REACTION_ALLOCATIONS",
    "ProposalSlotPlan",
    "SynthonOption",
    "SynthonProposalCatalog",
    "validate_payload_against_plan",
]
