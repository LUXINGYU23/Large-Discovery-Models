"""Task-local empirical proposal distribution for Iron Mind."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import replace

from ldm_tts.contracts import RawProposal
from ldm_tts.engine.expansion import ExpansionRequest

from tasks.iron_mind.core.candidate import (
    IRON_MIND_Q0_METADATA_KEY,
    CandidatePayloadError,
    IronMindCandidateDomain,
    prepare_candidate_payload,
)


def attach_empirical_base_measure(
    proposals: tuple[RawProposal, ...],
    request: ExpansionRequest,
    domain: IronMindCandidateDomain,
) -> tuple[RawProposal, ...]:
    """Annotate valid unseen proposal occurrences before canonical deduplication."""

    evaluated = {item.canonical_key for item in request.observations}
    keys: dict[int, str] = {}
    counts: Counter[str] = Counter()
    for index, proposal in enumerate(proposals):
        try:
            prepared = prepare_candidate_payload(proposal.payload, domain.schema, domain.table)
        except CandidatePayloadError:
            continue
        if prepared.canonical_key in evaluated:
            continue
        keys[index] = prepared.canonical_key
        counts[prepared.canonical_key] += 1
    valid_occurrences = sum(counts.values())
    return tuple(
        _annotate_proposal(
            proposal,
            keys.get(index),
            counts=counts,
            valid_occurrences=valid_occurrences,
        )
        for index, proposal in enumerate(proposals)
    )


def _annotate_proposal(
    proposal: RawProposal,
    canonical_key: str | None,
    *,
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
            IRON_MIND_Q0_METADATA_KEY: {
                "occurrence_count": occurrence_count,
                "valid_occurrence_count": valid_occurrences,
                "probability": occurrence_count / valid_occurrences,
            },
        },
    )


__all__ = ["attach_empirical_base_measure"]
