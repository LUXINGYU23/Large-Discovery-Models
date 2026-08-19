"""Tests for strict concurrent Iron Mind reaction proposal expansion."""

from __future__ import annotations

import csv
import hashlib
import json
import threading
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pytest

from ldm_tts.contracts import Candidate, RawProposal
from ldm_tts.contracts.evaluation import EvaluationResult, Observation
from ldm_tts.engine.expansion import ExpansionRequest
from ldm_tts.transport import CallableProposalClient, ProposalRequest, ProposalResponse
from tasks.iron_mind.core import proposals
from tasks.iron_mind.core.candidate import IronMindCandidateDomain
from tasks.iron_mind.core.data import FrozenReactionTable, ReactionRow
from tasks.iron_mind.core.prompting import BASELINE_PROMPT_POLICY
from tasks.iron_mind.core.proposals import (
    IronMindProposalExpander,
    build_openai_reaction_client,
    build_reaction_proposal_request,
    parse_reaction_response,
    parse_reaction_responses,
)
from tasks.iron_mind.core.schema import ReactionDatasetSchema, load_reaction_schemas

TASK_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = TASK_ROOT / "resources" / "reaction_schemas.json"
MOCK_ORACLE_PATH = TASK_ROOT / "resources" / "mock_oracle.csv"
TEST_CANDIDATE_COUNT = 4

class IndexedProposalClient:
    def __init__(self, payloads: list[dict[str, Any]]) -> None:
        self.payloads = payloads
        self.requests: list[ProposalRequest] = []

    def propose(self, request: ProposalRequest) -> ProposalResponse:
        self.requests.append(request)
        index = int(request.metadata["proposal_index"])
        return _response(self.payloads[index])

class FixedProposalClient:
    def __init__(self, response: ProposalResponse) -> None:
        self.response = response

    def propose(self, _request: ProposalRequest) -> ProposalResponse:
        return self.response

class BarrierProposalClient(IndexedProposalClient):
    def __init__(self, payloads: list[dict[str, Any]]) -> None:
        super().__init__(payloads)
        self.barrier = threading.Barrier(2)

    def propose(self, request: ProposalRequest) -> ProposalResponse:
        self.barrier.wait(timeout=1)
        return super().propose(request)

def _schema() -> ReactionDatasetSchema:
    return load_reaction_schemas(SCHEMA_PATH)["buchwald_hartwig"]

def _mock_rows(schema: ReactionDatasetSchema) -> tuple[ReactionRow, ...]:
    rows = []
    for row_id, source in enumerate(_oracle_sources(), start=1):
        conditions = {name: source[name] for name in schema.factor_names}
        rows.append(
            ReactionRow(
                row_id=row_id,
                conditions=MappingProxyType(conditions),
                measurements=MappingProxyType({"yield": float(source["reaction_score"])}),
                raw_row_sha256=hashlib.sha256(
                    json.dumps(source, sort_keys=True).encode("utf-8")
                ).hexdigest(),
            )
        )
    return tuple(rows)

def _domain() -> IronMindCandidateDomain:
    schema = _schema()
    rows = _mock_rows(schema)
    indexed = {
        tuple(row.conditions[name] for name in schema.factor_names): (row,) for row in rows
    }
    table = FrozenReactionTable(schema, rows, MappingProxyType(indexed))
    return IronMindCandidateDomain(schema, table)
def _candidate_payloads() -> list[dict[str, Any]]:
    schema = _schema()
    return [
        {
            "dataset_id": row["dataset_id"],
            "conditions": {name: row[name] for name in schema.factor_names},
        }
        for row in _oracle_sources()
    ]
def _oracle_sources() -> list[dict[str, str]]:
    with MOCK_ORACLE_PATH.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))
def _response(payload: dict[str, Any]) -> ProposalResponse:
    return ProposalResponse(text=json.dumps(payload, separators=(",", ":")))
def _request(**overrides: Any) -> ExpansionRequest:
    values = {"round_idx": 0, "reservoir_size": TEST_CANDIDATE_COUNT}
    values.update(overrides)
    return ExpansionRequest(**values)


def test_parser_accepts_only_one_complete_json_object() -> None:
    valid = _response(_candidate_payloads()[0]).text
    invalid_responses = (
        f"prefix {valid}",
        f"```json\n{valid}\n```",
        valid + valid,
        "[]",
        json.dumps({"candidates": [_candidate_payloads()[0]]}),
    )

    for text in invalid_responses:
        with pytest.raises(ValueError):
            parse_reaction_response(text, index=1)


def test_parser_enforces_candidate_envelope_before_domain_admission() -> None:
    candidate = _candidate_payloads()[0]
    invalid_payloads = (
        {"other": candidate},
        {**candidate, "extra": True},
        {"dataset_id": candidate["dataset_id"], "conditions": []},
    )

    for payload in invalid_payloads:
        with pytest.raises(ValueError):
            parse_reaction_response(_response(payload).text, index=1)

    reversed_conditions = dict(reversed(tuple(candidate["conditions"].items())))
    candidate["conditions"] = reversed_conditions
    parsed = parse_reaction_response(_response(candidate).text, index=1)
    assert parsed == candidate
    assert list(parsed["conditions"]) == list(reversed_conditions)

    candidate["conditions"]["base"] = "btmg"
    assert parse_reaction_response(_response(candidate).text, index=1) == candidate


def test_parser_preserves_semantic_duplicates_for_the_reservoir_builder() -> None:
    candidates = _candidate_payloads()
    candidates[1] = {
        "dataset_id": candidates[0]["dataset_id"],
        "conditions": dict(reversed(tuple(candidates[0]["conditions"].items()))),
    }
    parsed = parse_reaction_responses(
        tuple(_response(candidate) for candidate in candidates),
        candidate_count=TEST_CANDIDATE_COUNT,
    )

    assert len(parsed.proposals) == TEST_CANDIDATE_COUNT
    assert parsed.errors == ()


def test_parser_requires_exact_response_count() -> None:
    responses = tuple(_response(candidate) for candidate in _candidate_payloads()[:3])

    with pytest.raises(ValueError, match="exactly 4 responses"):
        parse_reaction_responses(responses, candidate_count=TEST_CANDIDATE_COUNT)


def test_callable_and_endpoint_clients_share_the_same_strict_parser() -> None:
    payloads = _candidate_payloads()
    callable_client = CallableProposalClient(
        lambda request: _response(payloads[int(request.metadata["proposal_index"])])
    )
    endpoint_client = IndexedProposalClient(payloads)
    callable_result = IronMindProposalExpander(
        callable_client,
        _domain(),
        prompt_policy=BASELINE_PROMPT_POLICY,
    ).expand(_request())
    endpoint_result = IronMindProposalExpander(
        endpoint_client,
        _domain(),
        prompt_policy=BASELINE_PROMPT_POLICY,
    ).expand(_request())

    assert [proposal.payload for proposal in callable_result.proposals] == [
        proposal.payload for proposal in endpoint_result.proposals
    ]
    assert len(callable_result.attempts) == len(endpoint_result.attempts) == 4
    assert [item.metadata["proposal_index"] for item in endpoint_client.requests] == [0, 1, 2, 3]


def test_openai_path_executes_independent_requests_with_local_workers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = BarrierProposalClient(_candidate_payloads())
    monkeypatch.setattr(proposals, "_can_parallelize", lambda _client: True)

    result = IronMindProposalExpander(
        client,
        _domain(),
        max_workers=2,
        prompt_policy=BASELINE_PROMPT_POLICY,
    ).expand(_request())

    assert len(result.proposals) == TEST_CANDIDATE_COUNT
    assert len(result.attempts) == TEST_CANDIDATE_COUNT


def test_expander_records_all_invalid_responses_without_hiding_attempts() -> None:
    response = ProposalResponse(text="not JSON", metadata={"provider": "test"})
    result = IronMindProposalExpander(FixedProposalClient(response), _domain()).expand(_request())

    assert result.proposals == ()
    assert result.attempts == (response,) * TEST_CANDIDATE_COUNT
    assert result.metadata["invalid_response_count"] == TEST_CANDIDATE_COUNT
    assert [item["proposal_index"] for item in result.metadata["response_errors"]] == [0, 1, 2, 3]
    assert result.metadata["response_count"] == TEST_CANDIDATE_COUNT


def test_parser_keeps_valid_peers_when_one_response_is_invalid() -> None:
    parsed = parse_reaction_responses(
        (
            _response(_candidate_payloads()[0]),
            ProposalResponse(text="not JSON"),
            _response(_candidate_payloads()[2]),
            ProposalResponse(text="[]"),
        ),
        candidate_count=TEST_CANDIDATE_COUNT,
    )

    assert [item.proposal_index for item in parsed.proposals] == [0, 2]
    assert [item.proposal_index for item in parsed.errors] == [1, 3]


def test_before_requests_is_consumed_once_with_the_full_request_count() -> None:
    calls: list[int] = []
    endpoint = IronMindProposalExpander(
        IndexedProposalClient(_candidate_payloads()),
        _domain(),
        before_requests=calls.append,
        prompt_policy=BASELINE_PROMPT_POLICY,
    )
    mock = IronMindProposalExpander(
        CallableProposalClient(
            lambda request: _response(
                _candidate_payloads()[int(request.metadata["proposal_index"])]
            )
        ),
        _domain(),
        prompt_policy=BASELINE_PROMPT_POLICY,
    )

    endpoint.expand(_request())
    mock.expand(_request())

    assert calls == [TEST_CANDIDATE_COUNT]


def test_request_contains_schema_observations_and_independent_slot() -> None:
    domain = _domain()
    candidate = domain.admit(RawProposal(_candidate_payloads()[0], "seed"))
    assert isinstance(candidate, Candidate)
    observation = Observation(
        candidate=candidate,
        evaluation=EvaluationResult(
            candidate_id=candidate.candidate_id,
            status="succeeded",
            metrics={"reaction_score": 45.92980056},
        ),
    )
    request = build_reaction_proposal_request(
        _request(observations=(observation,)),
        _schema(),
        proposal_index=2,
    )
    content = "\n".join(message["content"] for message in request.messages)

    assert _schema().schema_sha256 in content
    assert "reaction_score" in content
    assert candidate.canonical_key in content
    assert "Independent proposal slot: 3 of 4" in content
    assert '"direction":"maximize"' in content
    assert "Proposal policy: portfolio_v1." in content
    assert "Required slot focus (hard allocation):" in content
    assert "Return exactly one" in content
    assert request.metadata["sampling_mode"] == "local_concurrent_independent_requests"
    assert request.metadata["proposal_index"] == 2
    assert request.metadata["prompt_policy"] == "portfolio_v1"
    assert request.metadata["proposal_role"] == "underexplored_coverage"
    assert len(request.metadata["prompt_sha256"]) == 64
    assert "api_key" not in request.metadata


def test_request_uses_the_runtime_reservoir_size() -> None:
    request = build_reaction_proposal_request(
        _request(reservoir_size=64),
        _schema(),
        proposal_index=63,
    )
    content = "\n".join(message["content"] for message in request.messages)

    assert request.metadata["proposal_count"] == 64
    assert request.metadata["slot_focus_capacity"] >= 64
    assert "Independent proposal slot: 64 of 64" in content


def test_openai_client_uses_the_standard_chat_completion_contract() -> None:
    client = build_openai_reaction_client(
        base_url="https://example.invalid/v1",
        model="test-model",
        api_key="",
        timeout_seconds=10.0,
        max_tokens=256,
    )

    assert client.max_retries == 0
    assert client.require_models_preflight is False
    assert client.extra_body == {}

    json_client = build_openai_reaction_client(
        base_url="https://example.invalid/v1",
        model="test-model",
        api_key="",
        timeout_seconds=10.0,
        max_tokens=256,
        json_mode=True,
        extra_body={"thinking": {"type": "disabled"}},
    )

    assert json_client.extra_body == {
        "thinking": {"type": "disabled"},
        "response_format": {"type": "json_object"},
    }
    with pytest.raises(ValueError, match="cannot be combined"):
        build_openai_reaction_client(
            base_url="https://example.invalid/v1",
            model="test-model",
            api_key="",
            timeout_seconds=10.0,
            max_tokens=256,
            json_mode=True,
            extra_body={"response_format": {"type": "text"}},
        )
