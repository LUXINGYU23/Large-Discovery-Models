"""Strict proposal, tuple, and empirical-base-measure checks."""

from __future__ import annotations

import json
import threading
from typing import Any

import pytest
from synthonbench.space import Synthon, SynthonSpace

from ldm_tts.contracts import RawProposal, ReservoirBuilder
from ldm_tts.engine.expansion import ExpansionRequest
from ldm_tts.transport import ProposalRequest, ProposalResponse
from tasks.synthonbench.core import proposals
from tasks.synthonbench.core.candidate import SynthonCandidateDomain
from tasks.synthonbench.core.catalog import SynthonProposalCatalog
from tasks.synthonbench.core.constants import Q0_METADATA_KEY
from tasks.synthonbench.core.proposal_base_measure import attach_empirical_base_measure
from tasks.synthonbench.core.proposal_parsing import (
    parse_synthon_response,
    parse_synthon_responses,
)
from tasks.synthonbench.core.proposals import SynthonBenchProposalExpander

REQUEST_SIZE = 4


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


def test_direct_catalog_assigns_distinct_fixed_anchors() -> None:
    catalog = SynthonProposalCatalog(
        _space(),
        allowed_reactions=("r1",),
        slate_size=2,
        seed=7,
        direct_unique=True,
        direct_proposal_count=2,
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


def test_expander_keeps_independent_requests_and_records_the_actual_allocation() -> None:
    client = IndexedProposalClient()
    result = _expander(client).expand(_request())

    assert len(result.attempts) == REQUEST_SIZE
    assert len(result.proposals) == REQUEST_SIZE
    assert [item.metadata["proposal_index"] for item in client.requests] == list(range(REQUEST_SIZE))
    assert result.metadata["sampling_mode"] == "local_concurrent_independent_requests"
    assert result.metadata["reaction_allocation"] == "product_weighted"


def test_endpoint_requests_use_local_workers_without_changing_candidate_count(monkeypatch) -> None:
    client = BarrierProposalClient()
    monkeypatch.setattr(proposals, "supports_local_concurrency", lambda _: True)

    result = _expander(client).expand(_request())

    assert len(result.attempts) == REQUEST_SIZE
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


def _space() -> SynthonSpace:
    return SynthonSpace([
        Synthon(1, 1, "r1", "CC"),
        Synthon(2, 1, "r1", "CO"),
        Synthon(11, 2, "r1", "N"),
        Synthon(12, 2, "r1", "O"),
        Synthon(21, 1, "r2", "c1ccccc1"),
        Synthon(22, 1, "r2", "CCN"),
    ])


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
    return SynthonBenchProposalExpander(client, _domain(), _catalog(seed=3), target="kif11", max_workers=2)


def _request(**overrides: Any) -> ExpansionRequest:
    values = {"round_idx": 0, "reservoir_size": REQUEST_SIZE}
    values.update(overrides)
    return ExpansionRequest(**values)


def _response_from_request(request: ProposalRequest) -> ProposalResponse:
    return ProposalResponse(text=json.dumps({
        "reaction_id": request.metadata["reaction_id"],
        "synthon_ids": [ids[0] for ids in request.metadata["slot_synthon_ids"]],
    }))
