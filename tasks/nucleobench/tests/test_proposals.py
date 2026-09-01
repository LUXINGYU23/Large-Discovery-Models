from __future__ import annotations

import json
import threading
from dataclasses import replace
from pathlib import Path

import pytest

from ldm_tts.contracts import (
    Candidate,
    EvaluationResult,
    Observation,
    RawProposal,
    ReservoirBuilder,
)
from ldm_tts.data import DataCollectionSink
from ldm_tts.engine.expansion import ExpansionRequest
from ldm_tts.transport import ProposalRequest, ProposalResponse
from tasks.nucleobench.core.candidate import MutationContext, NucleoBenchCandidateDomain
from tasks.nucleobench.core.cases import get_case
from tasks.nucleobench.core.constants import NUCLEOBENCH_Q0_METADATA_KEY
from tasks.nucleobench.core.prompting import build_mutation_prompt_messages
from tasks.nucleobench.core.proposals import (
    DirectMutationProposalExpander,
    ProposalGenerationError,
    ProposalResponseError,
    ScoreBlindMutationPoolExpander,
    attach_empirical_base_measure,
    build_openai_mutation_client,
    parse_mutation_response,
)


class ScriptedProposalClient:
    def __init__(self, operation) -> None:
        self.operation = operation
        self.requests: list[ProposalRequest] = []
        self._lock = threading.Lock()

    def propose(self, request: ProposalRequest) -> ProposalResponse:
        with self._lock:
            self.requests.append(request)
        return ProposalResponse(
            text=json.dumps(self.operation(request), separators=(",", ":"))
        )


class ConcurrentProposalClient(ScriptedProposalClient):
    def __init__(self, operation) -> None:
        super().__init__(operation)
        self.active = 0
        self.max_active = 0
        self.barrier = threading.Barrier(2)

    def propose(self, request: ProposalRequest) -> ProposalResponse:
        with self._lock:
            self.requests.append(request)
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        self.barrier.wait(timeout=2)
        try:
            return ProposalResponse(
                text=json.dumps(self.operation(request), separators=(",", ":"))
            )
        finally:
            with self._lock:
                self.active -= 1


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


def _payload(index: int) -> dict[str, list[dict[str, object]]]:
    choices = (
        (0, "C"),
        (0, "G"),
        (0, "T"),
        (2, "C"),
        (2, "G"),
        (2, "T"),
        (4, "C"),
        (4, "G"),
    )
    position, base = choices[index % len(choices)]
    return {"mutations": [{"position": position, "base": base}]}


def _response_payload(payloads: list[dict]) -> dict:
    if len(payloads) == 1:
        return payloads[0]
    return {
        "candidates": [
            {"proposal_index": index, **payload}
            for index, payload in enumerate(payloads)
        ]
    }


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
        item.payload["mutations"][0]["position"]: item.metadata[
            NUCLEOBENCH_Q0_METADATA_KEY
        ]
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

    assert [item.payload for item in first.proposals] == [
        item.payload for item in second.proposals
    ]
    assert len(first.proposals) == 12
    assert all(isinstance(item, Candidate) for item in first_candidates)
    assert (
        len(
            {
                item.canonical_key
                for item in first_candidates
                if isinstance(item, Candidate)
            }
        )
        == 12
    )
    assert evaluated.canonical_key not in {
        item.canonical_key for item in first_candidates if isinstance(item, Candidate)
    }
    assert first.metadata == {
        "mode": "score_blind_mutation_pool",
        "distance_steps": [1, 2, 4],
        "parent_count": 2,
    }


def test_prompts_use_full_short_context_and_bounded_long_windows() -> None:
    short = _context()
    short_messages = build_mutation_prompt_messages(
        ExpansionRequest(round_idx=1, reservoir_size=2),
        short,
        candidate_count=2,
        request_id="request-1",
        lineage_index=0,
        wave_index=0,
        same_round_agreement_allowed=True,
    )
    short_text = short_messages[1]["content"]
    assert '"kind":"full_start_sequence"' in short_text
    assert short.start_sequence in short_text
    assert '"candidate_count":2' in short_text

    long_case = replace(
        get_case("bpnet_ctcf"),
        sequence_length=3_000,
        editable_position_count=3_000,
    )
    long_context = MutationContext(
        case=long_case,
        start_set_digest="c" * 64,
        start_index=1,
        start_sequence="A" * 3_000,
        editable_positions=tuple(range(3_000)),
    )
    long_messages = build_mutation_prompt_messages(
        ExpansionRequest(round_idx=2, reservoir_size=1),
        long_context,
        candidate_count=1,
        request_id="request-2",
        lineage_index=3,
        wave_index=0,
        same_round_agreement_allowed=False,
    )
    long_text = long_messages[1]["content"]
    assert '"kind":"editable_windows"' in long_text
    assert "A" * 3_000 not in long_text
    assert len(long_text) < 12_000
    assert "api_key" not in short_text + long_text


def test_strict_response_parser_keeps_valid_batch_peers() -> None:
    with pytest.raises(ProposalResponseError, match="complete JSON"):
        parse_mutation_response("prefix {}", expected_count=1)
    with pytest.raises(ProposalResponseError, match="unexpected field"):
        parse_mutation_response(
            json.dumps({"mutations": [], "sequence": "ACGT"}),
            expected_count=1,
        )

    parsed = parse_mutation_response(
        json.dumps(
            {
                "candidates": [
                    {"proposal_index": 0, **_payload(0)},
                    {"proposal_index": 1, **_payload(1), "reason": "extra"},
                ]
            }
        ),
        expected_count=2,
    )
    assert parsed.proposals == ((0, _payload(0)),)
    assert parsed.errors[0]["reason"] == "unexpected_response_fields"


def test_ldm_uses_four_stable_concurrent_lineages_and_preserves_q0_occurrences() -> (
    None
):
    def operation(request: ProposalRequest) -> dict:
        lineage = int(request.metadata["lineage_index"])
        payloads = [_payload(lineage), _payload(lineage + 1)]
        return _response_payload(payloads)

    client = ConcurrentProposalClient(operation)
    result = DirectMutationProposalExpander(
        client,
        NucleoBenchCandidateDomain(_context()),
        search_method="ldm",
        evaluations_per_round=2,
        seed=7,
        max_workers=2,
    ).expand(ExpansionRequest(round_idx=3, reservoir_size=8))

    assert len(result.proposals) == 8
    assert len(result.attempts) == 4
    assert client.max_active == 2
    assert [request.metadata["request_id"] for request in client.requests] == [
        f"nucleobench-r0003-s0007-l{index:03d}-w00" for index in range(4)
    ]
    assert [request.metadata["seed_lineage"] for request in client.requests] == [
        f"7:{index}" for index in range(4)
    ]
    q0 = [item.metadata[NUCLEOBENCH_Q0_METADATA_KEY] for item in result.proposals]
    assert [item["valid_occurrence_count"] for item in q0] == [8] * 8
    assert max(item["occurrence_count"] for item in q0) > 1


def test_direct_llm_refills_rejections_to_a_unique_real_evaluation_batch(
    tmp_path: Path,
) -> None:
    context = _context()
    historical_payload = _payload(6)
    base_domain = NucleoBenchCandidateDomain(context)
    historical = _admit(base_domain, historical_payload)

    def operation(request: ProposalRequest) -> dict:
        lineage = int(request.metadata["lineage_index"])
        wave = int(request.metadata["wave_index"])
        if wave == 0:
            payload = (
                historical_payload
                if lineage == 0
                else {"mutations": [{"position": 1, "base": "C"}]}
            )
        elif wave == 1:
            payload = _payload(0)
        else:
            payload = _payload(lineage + 1)
        return _response_payload([payload])

    sink = DataCollectionSink(tmp_path / "collection")
    domain = NucleoBenchCandidateDomain(context, sink=sink)
    client = ScriptedProposalClient(operation)
    request = ExpansionRequest(
        round_idx=4,
        reservoir_size=2,
        observations=(_observation(historical),),
    )
    result = DirectMutationProposalExpander(
        client,
        domain,
        search_method="llm",
        evaluations_per_round=2,
        max_request_waves=3,
    ).expand(request)
    reservoir = ReservoirBuilder(domain).build(
        result.proposals,
        evaluated_keys=(historical.canonical_key,),
    )

    assert len(client.requests) == 5
    assert len(result.proposals) == len(reservoir.candidates) == 2
    assert result.selection_mode == "reservoir_order"
    assert result.metadata["rejection_counts"] == {
        "historical_duplicate": 1,
        "non_mutable_position": 1,
        "same_round_duplicate": 1,
    }
    assert NUCLEOBENCH_Q0_METADATA_KEY not in result.proposals[0].metadata
    assert sink.paths is not None
    assert len(sink.paths.ir_path.read_text(encoding="utf-8").splitlines()) == 2


def test_generation_exhaustion_reports_precise_reasons_and_attempts() -> None:
    client = ScriptedProposalClient(
        lambda request: {"mutations": [{"position": 0, "base": "A"}]}
    )
    expander = DirectMutationProposalExpander(
        client,
        NucleoBenchCandidateDomain(_context()),
        search_method="llm",
        evaluations_per_round=1,
        max_request_waves=2,
    )

    with pytest.raises(ProposalGenerationError, match="still missing") as error:
        expander.expand(ExpansionRequest(round_idx=1, reservoir_size=1))

    assert len(error.value.attempts) == 2
    assert error.value.metadata["rejection_counts"] == {"unchanged_base": 2}
    assert error.value.metadata["missing_occurrence_count"] == 1


def test_openai_client_configuration_is_generic_and_secret_free() -> None:
    client = build_openai_mutation_client(
        base_url="https://provider.example",
        model="model-name",
        api_key="test-secret",
        wire_api="responses",
        timeout_seconds=30.0,
        max_tokens=512,
        temperature=0.8,
        reasoning="high",
    )

    assert client.wire_api == "responses"
    assert client.extra_body == {
        "reasoning": {"effort": "high"},
        "text": {"format": {"type": "json_object"}},
    }
    assert "test-secret" not in repr(client.extra_body)
