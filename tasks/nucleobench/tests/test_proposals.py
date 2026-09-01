from __future__ import annotations

from dataclasses import replace

import pytest

from ldm_tts.contracts import (
    Candidate,
    EvaluationResult,
    Observation,
    RawProposal,
    ReservoirBuilder,
)
from ldm_tts.engine.expansion import ExpansionRequest
from tasks.nucleobench.core.candidate import MutationContext, NucleoBenchCandidateDomain
from tasks.nucleobench.core.cases import get_case
from tasks.nucleobench.core.constants import NUCLEOBENCH_Q0_METADATA_KEY
from tasks.nucleobench.core.proposals import (
    ScoreBlindMutationPoolExpander,
    attach_empirical_base_measure,
)


def _context() -> MutationContext:
    case = replace(
        get_case("malinois_k562"),
        sequence_length=8,
        editable_position_count=4,
    )
    return MutationContext(
        case=case,
        start_set_digest="b" * 64,
        start_index=3,
        start_sequence="AAAAAAAA",
        editable_positions=(0, 2, 4, 6),
    )


def _admit(domain: NucleoBenchCandidateDomain, payload: dict) -> Candidate:
    candidate = domain.admit(RawProposal(payload, "test"))
    assert isinstance(candidate, Candidate)
    return candidate


def _observation(candidate: Candidate, score: float = 1.0) -> Observation:
    return Observation(
        candidate,
        EvaluationResult(
            candidate.candidate_id,
            "succeeded",
            metrics={"utility": score},
        ),
    )


def test_q0_counts_unseen_occurrences_before_shared_deduplication() -> None:
    domain = NucleoBenchCandidateDomain(_context())
    payload_a = {"mutations": [{"position": 0, "base": "C"}]}
    payload_b = {"mutations": [{"position": 2, "base": "G"}]}
    historical_payload = {"mutations": [{"position": 4, "base": "T"}]}
    historical = _admit(domain, historical_payload)
    request = ExpansionRequest(
        round_idx=2,
        reservoir_size=65,
        observations=(_observation(historical),),
    )
    raw = (
        *(RawProposal(payload_a, f"agent-a-{index}") for index in range(48)),
        *(RawProposal(payload_b, f"agent-b-{index}") for index in range(16)),
        RawProposal(historical_payload, "agent-d"),
    )

    annotated = attach_empirical_base_measure(raw, request, domain)
    reservoir = ReservoirBuilder(domain).build(
        annotated,
        evaluated_keys=(historical.canonical_key,),
    )
    records = {
        item.payload["mutations"][0]["position"]: item.metadata[NUCLEOBENCH_Q0_METADATA_KEY]
        for item in reservoir.candidates
    }

    assert records[0]["probability"] == pytest.approx(0.75)
    assert records[2]["probability"] == pytest.approx(0.25)
    assert records[0]["valid_occurrence_count"] == 64
    assert NUCLEOBENCH_Q0_METADATA_KEY not in annotated[-1].metadata
    assert reservoir.drop_counts == {"duplicate": 62, "already_evaluated": 1}


def test_score_blind_bo_pool_is_deterministic_unique_and_unseen() -> None:
    context = _context()
    domain = NucleoBenchCandidateDomain(context)
    incumbent = _admit(
        domain,
        {"mutations": [{"position": 0, "base": "C"}, {"position": 2, "base": "G"}]},
    )
    evaluated = _admit(domain, {"mutations": [{"position": 4, "base": "T"}]})
    request = ExpansionRequest(
        round_idx=3,
        reservoir_size=12,
        observations=(_observation(incumbent, 2.0), _observation(evaluated, 1.0)),
        parent=incumbent,
    )
    expander = ScoreBlindMutationPoolExpander(context, seed=19)

    first = expander.expand(request)
    second = expander.expand(request)
    first_candidates = tuple(domain.admit(item) for item in first.proposals)

    assert [item.payload for item in first.proposals] == [item.payload for item in second.proposals]
    assert len(first.proposals) == 12
    assert all(isinstance(item, Candidate) for item in first_candidates)
    assert len({item.canonical_key for item in first_candidates if isinstance(item, Candidate)}) == 12
    assert evaluated.canonical_key not in {
        item.canonical_key for item in first_candidates if isinstance(item, Candidate)
    }
    assert first.metadata == {
        "mode": "score_blind_mutation_pool",
        "distance_steps": [1, 2, 4],
        "parent_count": 2,
    }
