"""Public, deterministic synthon slates for independent LLM proposals."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

ROLE_CYCLE = ("explore", "exploit", "diversify", "scaffold_shift")
REACTION_ALLOCATIONS = ("product_weighted", "uniform")
DIRECT_TUPLE_OPTION_COUNT = 8


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
    uniqueness_anchor_id: int | None = None
    direct_tuple_options: tuple[tuple[int, ...], ...] = ()

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
            "uniqueness_anchor_id": self.uniqueness_anchor_id,
            "direct_tuple_options": [list(item) for item in self.direct_tuple_options],
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
        direct_start_round: int = 0,
    ) -> None:
        if slate_size < 1:
            raise ValueError("slate_size must be positive")
        if seed < 0:
            raise ValueError("catalog seed must be non-negative")
        if reaction_allocation not in REACTION_ALLOCATIONS:
            raise ValueError(f"unknown reaction allocation: {reaction_allocation!r}")
        if direct_unique and (direct_proposal_count is None or direct_proposal_count < 1):
            raise ValueError("direct_unique catalog requires a positive direct_proposal_count")
        if direct_start_round < 0:
            raise ValueError("direct_start_round must be non-negative")
        self.space = space
        self.reactions = tuple(str(item) for item in allowed_reactions)
        if not self.reactions:
            raise ValueError("proposal catalog requires at least one reaction")
        self.slate_size = int(slate_size)
        self.seed = int(seed)
        self.reaction_allocation = reaction_allocation
        self.direct_unique = bool(direct_unique)
        self.direct_proposal_count = direct_proposal_count
        self.direct_start_round = int(direct_start_round)
        self._reaction_probabilities = _reaction_probabilities(space, self.reactions, reaction_allocation)

    def build_plan(
        self,
        *,
        round_idx: int,
        proposal_index: int,
        excluded_anchor_ids: Mapping[str, set[int]] | None = None,
    ) -> ProposalSlotPlan:
        if round_idx < 0 or proposal_index < 0:
            raise ValueError("round_idx and proposal_index must be non-negative")
        rng = np.random.default_rng(_slot_seed(self.seed, round_idx, proposal_index))
        if self.direct_unique:
            reaction_id, ordinal = self._direct_assignment(
                round_idx,
                proposal_index,
                excluded_anchor_ids or {},
            )
            reaction_index = self.reactions.index(reaction_id)
            options, anchor_position, anchor_id, direct_tuple_options = self._direct_options(
                round_idx,
                proposal_index,
                reaction_id,
                rng,
                set((excluded_anchor_ids or {}).get(reaction_id, set())),
                ordinal,
            )
        else:
            reaction_index = int(rng.choice(len(self.reactions), p=self._reaction_probabilities))
            reaction_id = self.reactions[reaction_index]
            options = tuple(
                _sample_slot(self.space, reaction_id, position, self.slate_size, rng)
                for position in self.space.positions(reaction_id)
            )
            anchor_position = None
            anchor_id = None
            direct_tuple_options = ()
        return ProposalSlotPlan(
            round_idx=round_idx,
            proposal_index=proposal_index,
            reaction_id=reaction_id,
            role=ROLE_CYCLE[(round_idx + proposal_index) % len(ROLE_CYCLE)],
            reaction_probability=float(self._reaction_probabilities[reaction_index]),
            slot_options=options,
            uniqueness_anchor_position=anchor_position,
            uniqueness_anchor_id=anchor_id,
            direct_tuple_options=direct_tuple_options,
        )

    def direct_anchor_position(self, reaction_id: str) -> int:
        positions = tuple(self.space.positions(reaction_id))
        return max(positions, key=lambda position: len(tuple(self.space.synthon_ids(reaction_id, position))))

    def _direct_options(self, round_idx: int, proposal_index: int, reaction_id: str, rng,
                        excluded_anchor_ids: set[int], ordinal: int) -> tuple[tuple[tuple[SynthonOption, ...], ...], int, int, tuple[tuple[int, ...], ...]]:
        positions = tuple(self.space.positions(reaction_id))
        anchor = self.direct_anchor_position(reaction_id)
        anchor_ids = tuple(int(item) for item in self.space.synthon_ids(reaction_id, anchor))
        available_anchor_ids = tuple(item for item in anchor_ids if item not in excluded_anchor_ids)
        if ordinal >= len(available_anchor_ids):
            raise ValueError(f"direct LLM slate cannot assign a unique anchor for reaction {reaction_id!r}")
        anchor_rng = np.random.default_rng(_anchor_seed(self.seed, reaction_id, anchor))
        ordered_anchor_ids = tuple(item for item in anchor_rng.permutation(anchor_ids) if item in available_anchor_ids)
        anchor_id = int(ordered_anchor_ids[ordinal])
        options = tuple(
            (_option(self.space, reaction_id, position, anchor_id),)
            if position == anchor
            else _sample_slot(self.space, reaction_id, position, self.slate_size, rng)
            for position in positions
        )
        return options, anchor, anchor_id, _direct_tuple_options(options)

    def _direct_assignment(self, round_idx: int, proposal_index: int,
                           excluded_anchor_ids: Mapping[str, set[int]]) -> tuple[str, int]:
        assert self.direct_proposal_count is not None
        remaining = {
            reaction: self._available_anchor_count(reaction, excluded_anchor_ids)
            for reaction in self.reactions
        }
        assigned = {reaction: 0 for reaction in self.reactions}
        for current_round in range(self.direct_start_round, round_idx + 1):
            for current_index in range(self.direct_proposal_count):
                weights = np.asarray(
                    [probability if remaining[reaction] else 0.0
                     for reaction, probability in zip(self.reactions, self._reaction_probabilities, strict=True)],
                    dtype=float,
                )
                if not float(weights.sum()):
                    raise ValueError("direct LLM slate has no remaining unique public anchors")
                rng = np.random.default_rng(_slot_seed(self.seed, current_round, current_index))
                reaction = self.reactions[int(rng.choice(len(self.reactions), p=weights / weights.sum()))]
                if current_round == round_idx and current_index == proposal_index:
                    return reaction, assigned[reaction]
                remaining[reaction] -= 1
                assigned[reaction] += 1
        raise RuntimeError("direct LLM reaction assignment did not reach the requested proposal")

    def _available_anchor_count(self, reaction_id: str,
                                excluded_anchor_ids: Mapping[str, set[int]]) -> int:
        anchor = self.direct_anchor_position(reaction_id)
        ids = {int(item) for item in self.space.synthon_ids(reaction_id, anchor)}
        return len(ids - set(excluded_anchor_ids.get(reaction_id, set())))


def validate_payload_against_plan(payload: dict[str, object], plan: ProposalSlotPlan) -> None:
    """Require every LLM-selected ID to come from its supplied public slate."""

    if payload.get("reaction_id") != plan.reaction_id:
        raise ValueError("proposal reaction_id differs from the assigned reaction slate")
    raw_ids = payload.get("synthon_ids")
    if not isinstance(raw_ids, list) or len(raw_ids) != len(plan.slot_options):
        raise ValueError("proposal synthon_ids do not match the assigned reaction arity")
    if plan.direct_tuple_options and tuple(raw_ids) not in plan.direct_tuple_options:
        raise ValueError("synthon_ids must exactly match one supplied direct candidate option")
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


def _direct_tuple_options(
    slots: tuple[tuple[SynthonOption, ...], ...],
) -> tuple[tuple[int, ...], ...]:
    count = min(DIRECT_TUPLE_OPTION_COUNT, int(np.prod([len(slot) for slot in slots])))
    return tuple(_tuple_at_index(slots, index) for index in range(count))


def _tuple_at_index(slots: tuple[tuple[SynthonOption, ...], ...], index: int) -> tuple[int, ...]:
    selected = []
    for slot in slots:
        selected.append(slot[index % len(slot)].synthon_id)
        index //= len(slot)
    return tuple(selected)


__all__ = [
    "REACTION_ALLOCATIONS",
    "ProposalSlotPlan",
    "SynthonOption",
    "SynthonProposalCatalog",
    "validate_payload_against_plan",
]
