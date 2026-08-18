"""Tests for strict Iron Mind reaction proposal expansion."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pytest

from ldm_tts.contracts import Candidate, RawProposal
from ldm_tts.contracts.evaluation import EvaluationResult, Observation
from ldm_tts.data import DataCollectionSink
from ldm_tts.engine.expansion import ExpansionRequest
from ldm_tts.transport import CallableProposalClient, ProposalRequest, ProposalResponse
from tasks.iron_mind.core.candidate import IronMindCandidateDomain
from tasks.iron_mind.core.data import FrozenReactionTable, ReactionRow
from tasks.iron_mind.core.proposals import (
    DEEPSEEK_REACTION_EXTRA_BODY,
    IronMindProposalExpander,
    build_deepseek_reaction_client,
    build_reaction_proposal_request,
    parse_reaction_candidates,
)
from tasks.iron_mind.core.schema import ReactionDatasetSchema, load_reaction_schemas


TASK_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = TASK_ROOT / "resources" / "reaction_schemas.json"
MOCK_ORACLE_PATH = TASK_ROOT / "resources" / "mock_oracle.csv"
REQUIRED_CANDIDATE_COUNT = 4


class StaticProposalClient:
    """A test-only proposal client that returns one fixed response."""

    def __init__(self, response: ProposalResponse) -> None:
        self.response = response
        self.requests: list[ProposalRequest] = []

    def propose(self, request: ProposalRequest) -> ProposalResponse:
        self.requests.append(request)
        return self.response


def _schema() -> ReactionDatasetSchema:
    return load_reaction_schemas(SCHEMA_PATH)["buchwald_hartwig"]


def _mock_rows(schema: ReactionDatasetSchema) -> tuple[ReactionRow, ...]:
    source_rows = _oracle_sources()
    rows = []
    for row_id, source in enumerate(source_rows, start=1):
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


def _domain(sink: DataCollectionSink | None = None) -> IronMindCandidateDomain:
    schema = _schema()
    rows = _mock_rows(schema)
    indexed = {
        tuple(row.conditions[name] for name in schema.factor_names): (row,) for row in rows
    }
    table = FrozenReactionTable(
        schema=schema,
        rows=rows,
        rows_by_conditions=MappingProxyType(indexed),
    )
    return IronMindCandidateDomain(schema, table, sink or DataCollectionSink.disabled())


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


def _response_text(candidates: list[dict[str, Any]] | None = None) -> str:
    return json.dumps({"candidates": candidates or _candidate_payloads()})


def _request(**overrides: Any) -> ExpansionRequest:
    values = {"round_idx": 0, "reservoir_size": REQUIRED_CANDIDATE_COUNT}
    values.update(overrides)
    return ExpansionRequest(**values)


def test_parser_accepts_only_one_complete_json_object() -> None:
    valid = _response_text()
    invalid_responses = (f"prefix {valid}", f"```json\n{valid}\n```", valid + valid, "[]")

    for text in invalid_responses:
        with pytest.raises(ValueError, match="complete JSON object"):
            parse_reaction_candidates(text, domain=_domain())


def test_parser_enforces_envelope_and_preserves_untrusted_payloads() -> None:
    candidates = _candidate_payloads()
    invalid_envelopes = (
        {"other": candidates},
        {"candidates": candidates, "extra": True},
        {"candidates": candidates[:3]},
        {"candidates": ["not-an-object"] * REQUIRED_CANDIDATE_COUNT},
    )

    for payload in invalid_envelopes:
        with pytest.raises(ValueError):
            parse_reaction_candidates(json.dumps(payload), domain=_domain())

    reversed_conditions = dict(reversed(tuple(candidates[0]["conditions"].items())))
    candidates[0] = {"dataset_id": "buchwald_hartwig", "conditions": reversed_conditions}
    parsed = parse_reaction_candidates(_response_text(candidates), domain=_domain())
    assert parsed[0] == candidates[0]
    assert list(parsed[0]["conditions"]) == list(reversed_conditions)

    candidates[0]["conditions"]["base"] = "btmg"
    with pytest.raises(ValueError, match="candidate 1 rejected: unknown_category"):
        parse_reaction_candidates(_response_text(candidates), domain=_domain())


def test_parser_rejects_semantic_duplicates_before_collection(tmp_path: Path) -> None:
    candidates = _candidate_payloads()
    candidates[1] = {
        "dataset_id": candidates[0]["dataset_id"],
        "conditions": dict(reversed(tuple(candidates[0]["conditions"].items()))),
    }
    collection_dir = tmp_path / "collection"
    domain = _domain(DataCollectionSink(collection_dir))

    with pytest.raises(ValueError, match="candidates must be distinct"):
        parse_reaction_candidates(_response_text(candidates), domain=domain)

    assert not (collection_dir / "ldm_ir.jsonl").exists()


def test_callable_and_endpoint_clients_share_the_same_strict_parser() -> None:
    text = _response_text()
    callable_client = CallableProposalClient(lambda _: text)
    endpoint_client = StaticProposalClient(ProposalResponse(text=text))
    callable_result = IronMindProposalExpander(callable_client, _domain()).expand(_request())
    endpoint_result = IronMindProposalExpander(endpoint_client, _domain()).expand(_request())

    assert [proposal.payload for proposal in callable_result.proposals] == [
        proposal.payload for proposal in endpoint_result.proposals
    ]
    assert len(callable_result.attempts) == len(endpoint_result.attempts) == 1
    assert len(endpoint_client.requests) == 1


def test_expander_returns_attempt_only_result_on_strict_parse_failure() -> None:
    response = ProposalResponse(text="not JSON", metadata={"provider": "test"})
    result = IronMindProposalExpander(StaticProposalClient(response), _domain()).expand(_request())

    assert result.proposals == ()
    assert result.attempts == (response,)
    assert result.metadata["parse_error"].isascii()


def test_before_request_is_consumed_once_only_by_the_injected_endpoint_path() -> None:
    calls: list[str] = []
    callback = lambda: calls.append("llm_request")
    response = ProposalResponse(text=_response_text())
    endpoint = IronMindProposalExpander(
        StaticProposalClient(response), _domain(), before_request=callback
    )
    mock = IronMindProposalExpander(CallableProposalClient(lambda _: _response_text()), _domain())

    endpoint.expand(_request())
    mock.expand(_request())

    assert calls == ["llm_request"]


def test_request_contains_the_schema_observations_and_do_not_repeat_keys() -> None:
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
        _request(observations=(observation,), context={"do_not_repeat_keys": ["seed-key"]}),
        _schema(),
    )
    content = "\n".join(message["content"] for message in request.messages)

    assert _schema().schema_sha256 in content
    assert "reaction_score" in content
    assert "seed-key" in content
    assert "JSON" in content
    assert "exactly four" in content
    envelope_example = {
        "candidates": [
            {
                "dataset_id": _schema().dataset_id,
                "conditions": {
                    factor.name: f"<allowed {factor.name} category>"
                    for factor in _schema().factors
                },
            }
        ]
    }
    assert json.dumps(envelope_example, ensure_ascii=False, separators=(",", ":")) in content
    assert "placeholders" in content
    for factor in _schema().factors:
        for category in factor.categories:
            assert category in content
    assert "api_key" not in request.metadata


def test_deepseek_client_contract_enforces_json_output_without_request_secrets() -> None:
    client = build_deepseek_reaction_client(
        base_url="https://api.deepseek.com",
        model="deepseek-v4-flash",
        api_key="",
        timeout_seconds=10.0,
        max_tokens=256,
    )

    assert client.max_retries == 0
    assert client.require_models_preflight is True
    assert client.extra_body == DEEPSEEK_REACTION_EXTRA_BODY == {
        "response_format": {"type": "json_object"},
        "thinking": {"type": "disabled"},
    }
