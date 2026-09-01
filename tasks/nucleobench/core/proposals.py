"""Task-local proposal utilities shared by BO and LDM methods."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from collections.abc import Mapping
from dataclasses import replace

from ldm_tts.contracts import RawProposal
from ldm_tts.engine.expansion import ExpansionRequest, ExpansionResult
from tasks.nucleobench.core.candidate import (
    CandidatePayloadError,
    MutationContext,
    NucleoBenchCandidateDomain,
    prepare_candidate_payload,
)
from tasks.nucleobench.core.constants import NUCLEOBENCH_Q0_METADATA_KEY


def attach_empirical_base_measure(
    proposals: tuple[RawProposal, ...],
    request: ExpansionRequest,
    domain: NucleoBenchCandidateDomain,
) -> tuple[RawProposal, ...]:
    """Attach q0 before shared reservoir deduplication."""

    evaluated = {item.canonical_key for item in request.observations}
    keys: dict[int, str] = {}
    counts: Counter[str] = Counter()
    for index, proposal in enumerate(proposals):
        try:
            prepared = prepare_candidate_payload(proposal.payload, domain.context)
        except CandidatePayloadError:
            continue
        if prepared.canonical_key in evaluated:
            continue
        keys[index] = prepared.canonical_key
        counts[prepared.canonical_key] += 1
    total = sum(counts.values())
    return tuple(
        _annotate_q0(proposal, keys.get(index), counts, total)
        for index, proposal in enumerate(proposals)
    )


class ScoreBlindMutationPoolExpander:
    """Generate a deterministic finite BO pool around start and incumbent."""

    def __init__(
        self,
        context: MutationContext,
        *,
        seed: int,
        distance_steps: tuple[int, ...] = (1, 2, 4, 8),
    ) -> None:
        if seed < 0:
            raise ValueError("seed must be non-negative")
        self.context = context
        self.seed = seed
        self.distance_steps = tuple(
            dict.fromkeys(
                min(step, len(context.editable_positions))
                for step in distance_steps
                if step > 0
            )
        )
        if not self.distance_steps:
            raise ValueError("distance_steps must contain a positive distance")

    def expand(self, request: ExpansionRequest) -> ExpansionResult:
        parents = [_editable_bases(self.context, {"mutations": []})]
        if request.parent is not None:
            incumbent = _editable_bases(self.context, request.parent.payload)
            if incumbent != parents[0]:
                parents.append(incumbent)
        evaluated = {item.canonical_key for item in request.observations}
        rng = random.Random(_pool_seed(self.seed, request, evaluated))
        proposals: list[RawProposal] = []
        seen: set[str] = set()
        max_attempts = max(256, request.reservoir_size * 128)
        for attempt in range(max_attempts):
            parent = parents[attempt % len(parents)]
            distance = self.distance_steps[
                (attempt // len(parents)) % len(self.distance_steps)
            ]
            child = list(parent)
            for offset in rng.sample(range(len(child)), distance):
                child[offset] = rng.choice(
                    tuple(base for base in "ACGT" if base != child[offset])
                )
            payload = _patch_from_editable_bases(self.context, child)
            if not payload["mutations"]:
                continue
            prepared = prepare_candidate_payload(payload, self.context)
            if prepared.canonical_key in evaluated or prepared.canonical_key in seen:
                continue
            seen.add(prepared.canonical_key)
            proposals.append(
                RawProposal(
                    prepared.payload,
                    "nucleobench_score_blind_bo",
                    {"round_idx": request.round_idx},
                )
            )
            if len(proposals) == request.reservoir_size:
                break
        if len(proposals) != request.reservoir_size:
            raise ValueError(
                "score-blind mutation search could not fill the requested unseen pool"
            )
        return ExpansionResult(
            tuple(proposals),
            metadata={
                "mode": "score_blind_mutation_pool",
                "distance_steps": list(self.distance_steps),
                "parent_count": len(parents),
            },
        )


def _annotate_q0(
    proposal: RawProposal,
    canonical_key: str | None,
    counts: Mapping[str, int],
    total: int,
) -> RawProposal:
    if canonical_key is None:
        return proposal
    occurrence_count = counts[canonical_key]
    return replace(
        proposal,
        metadata={
            **proposal.metadata,
            NUCLEOBENCH_Q0_METADATA_KEY: {
                "occurrence_count": occurrence_count,
                "valid_occurrence_count": total,
                "probability": occurrence_count / total,
            },
        },
    )


def _editable_bases(context: MutationContext, payload: object) -> tuple[str, ...]:
    prepared = prepare_candidate_payload(payload, context, allow_empty=True)
    mutations = {
        int(item["position"]): str(item["base"])
        for item in prepared.payload["mutations"]
    }
    return tuple(
        mutations.get(position, context.start_sequence[position])
        for position in context.editable_positions
    )


def _patch_from_editable_bases(
    context: MutationContext,
    bases: list[str],
) -> dict[str, list[dict[str, object]]]:
    return {
        "mutations": [
            {"position": position, "base": base}
            for position, base in zip(context.editable_positions, bases, strict=True)
            if base != context.start_sequence[position]
        ]
    }


def _pool_seed(seed: int, request: ExpansionRequest, evaluated: set[str]) -> int:
    payload = json.dumps(
        {
            "seed": seed,
            "round_idx": request.round_idx,
            "evaluated": sorted(evaluated),
            "parent": None if request.parent is None else request.parent.canonical_key,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


__all__ = ["ScoreBlindMutationPoolExpander", "attach_empirical_base_measure"]
