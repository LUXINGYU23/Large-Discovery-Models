"""Strict proposal, tuple, and empirical-base-measure checks."""

from __future__ import annotations

import json
import threading
from typing import Any

import pytest
from synthonbench.space import Synthon, SynthonSpace

from ldm_tts.contracts import (
    Candidate,
    EvaluationResult,
    Observation,
    RawProposal,
    ReservoirBuilder,
)
from ldm_tts.engine.expansion import ExpansionRequest
from ldm_tts.transport import ProposalRequest, ProposalResponse
from tasks.synthonbench.core import proposals
from tasks.synthonbench.core.candidate import SynthonCandidateDomain
from tasks.synthonbench.core.catalog import SynthonProposalCatalog
from tasks.synthonbench.core.constants import Q0_METADATA_KEY
from tasks.synthonbench.core.proposal_base_measure import attach_empirical_base_measure
from tasks.synthonbench.core.proposal_parsing import (
    parse_synthon_batch_response,
    parse_synthon_response,
    parse_synthon_responses,
)
from tasks.synthonbench.core.proposal_transport import build_openai_synthon_client
from tasks.synthonbench.core.proposals import SynthonBenchProposalExpander
from tasks.synthonbench.core.proposals import PROPOSAL_SOURCE
from tasks.synthonbench.core.search import SynthonInitializationExpander

REQUEST_SIZE = 64


def test_direct_transport_retries_transient_provider_failures() -> None:
    client = build_openai_synthon_client(
        base_url="https://example.invalid/v1",
        model="test-model",
        api_key="",
        timeout_seconds=10.0,
        max_tokens=256,
        temperature=0.7,
        json_mode=False,
    )

    assert client.max_retries == 3
    assert client.retry_backoff_seconds == 10.0
    assert client.breaker.failure_threshold == 32


class IndexedProposalClient:
    def __init__(self) -> None:
        self.requests: list[ProposalRequest] = []

    def propose(self, request: ProposalRequest) -> ProposalResponse:
        self.requests.append(request)
        return _response_from_request(request)


class BarrierProposalClient(IndexedProposalClient):
    def __init__(self) -> None:
        super().__init__()
        self.barrier = threading.Barrier(2)

    def propose(self, request: ProposalRequest) -> ProposalResponse:
        self.barrier.wait(timeout=2)
        return super().propose(request)


def test_catalog_is_deterministic_and_reaction_slot_specific() -> None:
    catalog = _catalog(seed=13)
    first = catalog.build_plan(round_idx=2, proposal_index=1)
    repeat = catalog.build_plan(round_idx=2, proposal_index=1)

    assert first == repeat
    assert first.reaction_id in {"r1", "r2"}
    assert len(first.slot_options) == len(_space().positions(first.reaction_id))
    assert all(option.smiles for slot in first.slot_options for option in slot)


def test_unique_anchor_catalog_assigns_distinct_fixed_anchors_within_a_round() -> None:
    catalog = SynthonProposalCatalog(
        _space(),
        allowed_reactions=("r1",),
        slate_size=2,
        seed=7,
        unique_anchors=True,
        proposals_per_round=2,
        restrict_to_complete_tuples=True,
    )
    first = catalog.build_plan(round_idx=0, proposal_index=0)
    second = catalog.build_plan(round_idx=0, proposal_index=1)

    assert first.uniqueness_anchor_position == second.uniqueness_anchor_position
    anchor = first.uniqueness_anchor_position
    assert anchor is not None
    first_anchor = next(slot for slot in first.slot_options if slot[0].position == anchor)
    second_anchor = next(slot for slot in second.slot_options if slot[0].position == anchor)
    assert len(first_anchor) == len(second_anchor) == 1
    assert first_anchor[0].synthon_id != second_anchor[0].synthon_id
    assert first.complete_tuple_options
    assert first.complete_tuple_options[0][0] == first_anchor[0].synthon_id


def test_unobserved_anchors_remain_available_in_later_rounds() -> None:
    catalog = SynthonProposalCatalog(
        _space(),
        allowed_reactions=("r1",),
        slate_size=2,
        seed=7,
        unique_anchors=True,
        proposals_per_round=2,
    )

    first_round = [catalog.build_plan(round_idx=0, proposal_index=index) for index in range(2)]
    second_round = [catalog.build_plan(round_idx=1, proposal_index=index) for index in range(2)]

    assert {plan.uniqueness_anchor_id for plan in first_round} == {1, 2}
    assert {plan.uniqueness_anchor_id for plan in second_round} == {1, 2}


def test_unique_anchor_catalog_excludes_history_anchor_ids() -> None:
    catalog = SynthonProposalCatalog(
        _space(),
        allowed_reactions=("r2",),
        slate_size=2,
        seed=7,
        unique_anchors=True,
        proposals_per_round=1,
    )
    baseline = catalog.build_plan(round_idx=0, proposal_index=0)
    anchor_id = baseline.uniqueness_anchor_id
    assert anchor_id is not None

    plan = catalog.build_plan(
        round_idx=0,
        proposal_index=0,
        excluded_anchor_ids={"r2": {anchor_id}},
    )

    assert plan.uniqueness_anchor_id != anchor_id


def test_anchor_exclusion_includes_evaluated_llm_proposals() -> None:
    catalog = SynthonProposalCatalog(
        _space(),
        allowed_reactions=("r1",),
        slate_size=2,
        seed=7,
        unique_anchors=True,
        proposals_per_round=2,
    )
    plan = catalog.build_plan(round_idx=0, proposal_index=0)
    payload = {
        "reaction_id": plan.reaction_id,
        "synthon_ids": [slot[0].synthon_id for slot in plan.slot_options],
    }
    candidate = Candidate("observed", payload, "r1|observed", source=PROPOSAL_SOURCE)
    observation = Observation(
        candidate,
        EvaluationResult("observed", "succeeded", metrics={"synthon_utility": -4.0}),
    )
    request = ExpansionRequest(round_idx=1, reservoir_size=2, observations=(observation,))

    excluded = proposals.excluded_anchor_ids(request, catalog)

    assert excluded == {"r1": {plan.uniqueness_anchor_id}}


def test_direct_catalog_requires_one_complete_tuple_option() -> None:
    catalog = SynthonProposalCatalog(
        _space(), allowed_reactions=("r1",), slate_size=2, seed=7,
        unique_anchors=True, proposals_per_round=1, restrict_to_complete_tuples=True,
    )
    plan = catalog.build_plan(round_idx=0, proposal_index=0)
    expected = {
        "reaction_id": "r1",
        "synthon_ids": list(plan.complete_tuple_options[0]),
    }
    mixed = {
        "reaction_id": "r1",
        "synthon_ids": [plan.complete_tuple_options[0][0], 999],
    }

    assert parse_synthon_response(json.dumps(expected), plan) == expected
    with pytest.raises(ValueError, match="complete candidate option"):
        parse_synthon_response(json.dumps(mixed), plan)


def test_parser_rejects_any_tuple_outside_the_assigned_slate() -> None:
    plan = _catalog(seed=0).build_plan(round_idx=0, proposal_index=0)
    valid = {
        "reaction_id": plan.reaction_id,
        "synthon_ids": [slot[0].synthon_id for slot in plan.slot_options],
    }

    assert parse_synthon_response(json.dumps(valid), plan) == valid
    invalid = {**valid, "reaction_id": "r2" if plan.reaction_id == "r1" else "r1"}
    with pytest.raises(ValueError, match="assigned reaction slate"):
        parse_synthon_response(json.dumps(invalid), plan)


def test_parser_isolates_type_errors_without_refills() -> None:
    plan = _catalog(seed=0).build_plan(round_idx=0, proposal_index=0)
    response = ProposalResponse(text=json.dumps({"reaction_id": 1, "synthon_ids": []}))

    parsed = parse_synthon_responses((response,), (plan,))

    assert not parsed.proposals
    assert parsed.errors[0].message == "reaction_id must be a string"


def test_batch_parser_preserves_valid_peers_and_reports_the_failed_slot() -> None:
    plans = tuple(_catalog(seed=0).build_plan(round_idx=0, proposal_index=index)
                  for index in range(2))
    candidates = [
        {
            "proposal_index": plan.proposal_index,
            "reaction_id": plan.reaction_id,
            "synthon_ids": [slot[0].synthon_id for slot in plan.slot_options],
        }
        for plan in plans
    ]
    candidates[1]["reaction_id"] = "missing"

    parsed = parse_synthon_batch_response(
        json.dumps({"candidates": candidates}),
        plans,
    )

    assert [item.proposal_index for item in parsed.proposals] == [0]
    assert parsed.errors[0].proposal_index == 1
    assert "assigned reaction slate" in parsed.errors[0].message


def test_batch_parser_requires_the_exact_candidate_count() -> None:
    plans = tuple(_catalog(seed=0).build_plan(round_idx=0, proposal_index=index)
                  for index in range(2))

    with pytest.raises(ValueError, match="exactly 2 candidates"):
        parse_synthon_batch_response(json.dumps({"candidates": []}), plans)


def test_batch_response_failure_counts_every_missing_candidate() -> None:
    class InvalidBatchClient:
        def propose(self, request: ProposalRequest) -> ProposalResponse:
            return ProposalResponse(text='{"candidates":[]}')

    result = _expander(InvalidBatchClient()).expand(_request())

    assert result.metadata["invalid_response_count"] == 4
    assert result.metadata["invalid_candidate_count"] == 64
    assert result.metadata["candidate_count_parsed"] == 0


def test_expander_issues_four_independent_requests_of_sixteen_candidates() -> None:
    client = IndexedProposalClient()
    result = _expander(client).expand(_request())

    assert len(result.attempts) == 4
    assert len(result.proposals) == REQUEST_SIZE
    assert [item.metadata["proposal_indices"] for item in client.requests] == [
        list(range(start, start + 16)) for start in range(0, REQUEST_SIZE, 16)
    ]
    assert result.metadata["request_count"] == 4
    assert result.metadata["candidates_per_request"] == 16
    assert result.metadata["candidate_count_requested"] == 64
    assert result.metadata["candidate_count_parsed"] == 64
    assert result.metadata["sampling_mode"] == (
        "local_concurrent_independent_minibatch_requests"
    )
    assert result.metadata["reaction_allocation"] == "product_weighted"


def test_endpoint_requests_use_local_workers_without_changing_candidate_count(monkeypatch) -> None:
    client = BarrierProposalClient()
    monkeypatch.setattr(proposals, "supports_local_concurrency", lambda _: True)

    result = _expander(client).expand(_request())

    assert len(result.attempts) == 4
    assert len(result.proposals) == REQUEST_SIZE


def test_q0_counts_valid_occurrences_before_shared_reservoir_deduplication() -> None:
    domain = _domain()
    payload_a = {"reaction_id": "r1", "synthon_ids": [1, 11]}
    payload_b = {"reaction_id": "r1", "synthon_ids": [2, 11]}
    raw = (RawProposal(payload_a, "test"), RawProposal(payload_a, "test"), RawProposal(payload_b, "test"))

    annotated = attach_empirical_base_measure(raw, _request(reservoir_size=3), domain)
    reservoir = ReservoirBuilder(domain).build(annotated)
    by_key = {item.canonical_key: item for item in reservoir.candidates}

    assert reservoir.drop_counts == {"duplicate": 1}
    assert by_key["r1|1_11"].metadata[Q0_METADATA_KEY]["probability"] == pytest.approx(2 / 3)
    assert by_key["r1|2_11"].metadata[Q0_METADATA_KEY]["probability"] == pytest.approx(1 / 3)


def test_shared_initialization_is_invariant_to_official_collection_order() -> None:
    space = _UnorderedSpace()
    request = _request(reservoir_size=3)
    first = SynthonInitializationExpander(
        space, ("r2", "r1"), seed=17, attach_q0=False,
    ).expand(request)
    second = SynthonInitializationExpander(
        space, ("r1", "r2"), seed=17, attach_q0=False,
    ).expand(request)

    assert [item.payload for item in first.proposals] == [item.payload for item in second.proposals]


def _space() -> SynthonSpace:
    return SynthonSpace([
        Synthon(1, 1, "r1", "CC"),
        Synthon(2, 1, "r1", "CO"),
        Synthon(11, 2, "r1", "N"),
        Synthon(12, 2, "r1", "O"),
        Synthon(21, 1, "r2", "c1ccccc1"),
        Synthon(22, 1, "r2", "CCN"),
    ])


class _UnorderedSpace:
    def __init__(self) -> None:
        self._space = _space()

    def positions(self, reaction_id: str):
        return set(self._space.positions(reaction_id))

    def synthon_ids(self, reaction_id: str, position: int):
        return set(self._space.synthon_ids(reaction_id, position))

    def product_count_estimate(self, reaction_id: str) -> int:
        return self._space.product_count_estimate(reaction_id)


def _domain() -> SynthonCandidateDomain:
    return SynthonCandidateDomain(_space(), ("r1", "r2"), "kif11")


def _catalog(seed: int) -> SynthonProposalCatalog:
    return SynthonProposalCatalog(
        _space(),
        allowed_reactions=("r1", "r2"),
        slate_size=2,
        seed=seed,
        reaction_allocation="product_weighted",
    )


def _expander(client: Any) -> SynthonBenchProposalExpander:
    return SynthonBenchProposalExpander(
        client,
        _domain(),
        _catalog(seed=3),
        target="kif11",
        candidates_per_request=16,
        max_workers=2,
    )


def _request(**overrides: Any) -> ExpansionRequest:
    values = {"round_idx": 0, "reservoir_size": REQUEST_SIZE}
    values.update(overrides)
    return ExpansionRequest(**values)


def _response_from_request(request: ProposalRequest) -> ProposalResponse:
    plans = request.metadata.get("proposal_plans")
    assert isinstance(plans, list)
    return ProposalResponse(text=json.dumps({
        "candidates": [
            {
                "proposal_index": plan["proposal_index"],
                "reaction_id": plan["reaction_id"],
                "synthon_ids": [ids[0] for ids in plan["slot_synthon_ids"]],
            }
            for plan in plans
        ]
    }))
