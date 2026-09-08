"""Empirical q0 construction before shared reservoir deduplication."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import replace

from ldm_tts.contracts import RawProposal
from ldm_tts.engine.expansion import ExpansionRequest
from tasks.synthonbench.core.candidate import (
    CandidatePayloadError,
    SynthonCandidateDomain,
    prepare_candidate_payload,
)
from tasks.synthonbench.core.constants import Q0_METADATA_KEY


def attach_empirical_base_measure(
    proposals: tuple[RawProposal, ...], request: ExpansionRequest, domain: SynthonCandidateDomain
) -> tuple[RawProposal, ...]:
    """Attach q0=c(x)/valid_occurrences to valid unseen raw proposals."""

    evaluated = {item.canonical_key for item in request.observations}
    accepted_keys, counts = _valid_unseen_occurrences(proposals, domain, evaluated)
    valid_occurrences = sum(counts.values())
    return tuple(
        _annotate(proposal, accepted_keys.get(index), counts, valid_occurrences)
        for index, proposal in enumerate(proposals)
    )


def _valid_unseen_occurrences(
    proposals: tuple[RawProposal, ...], domain: SynthonCandidateDomain, evaluated: set[str]
) -> tuple[dict[int, str], Counter[str]]:
    keys: dict[int, str] = {}
    counts: Counter[str] = Counter()
    for index, proposal in enumerate(proposals):
        try:
            prepared = prepare_candidate_payload(proposal.payload, domain.space, domain.allowed_reactions)
        except CandidatePayloadError:
            continue
        if prepared.product_id in evaluated:
            continue
        keys[index] = prepared.product_id
        counts[prepared.product_id] += 1
    return keys, counts


def _annotate(
    proposal: RawProposal,
    canonical_key: str | None,
    counts: Mapping[str, int],
    valid_occurrences: int,
) -> RawProposal:
    if canonical_key is None:
        return proposal
    occurrence_count = counts[canonical_key]
    return replace(
        proposal,
        metadata={
            **proposal.metadata,
            Q0_METADATA_KEY: {
                "occurrence_count": occurrence_count,
                "valid_occurrence_count": valid_occurrences,
                "probability": occurrence_count / valid_occurrences,
            },
        },
    )


__all__ = ["attach_empirical_base_measure"]
