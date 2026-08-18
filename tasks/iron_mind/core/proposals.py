"""Strict shared-path proposal expansion for Iron Mind reaction conditions."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from typing import Any

from ldm_tts.contracts import Candidate, CandidateDomainAdapter, CandidateRejection, RawProposal
from ldm_tts.engine.expansion import ExpansionRequest, ExpansionResult
from ldm_tts.transport import ProposalClient, ProposalRequest
from ldm_tts.transport.openai import OpenAICompatibleProposalClient

from tasks.iron_mind.core.candidate import IronMindCandidateDomain
from tasks.iron_mind.core.schema import ReactionDatasetSchema


REQUIRED_CANDIDATE_COUNT = 4
PROPOSAL_SOURCE = "iron_mind_reaction_proposal"
DEEPSEEK_REACTION_EXTRA_BODY = {
    "response_format": {"type": "json_object"},
    "thinking": {"type": "disabled"},
}


class IronMindProposalExpander:
    """Expand one proposal client response through the strict reaction parser."""

    def __init__(
        self,
        client: ProposalClient,
        domain: IronMindCandidateDomain,
        *,
        before_request: Callable[[], None] | None = None,
    ) -> None:
        self.client = client
        self.domain = domain
        self.before_request = before_request

    def expand(self, request: ExpansionRequest) -> ExpansionResult:
        """Return four raw proposals or one attempt-only parse failure result."""

        proposal_request = build_reaction_proposal_request(request, self.domain.schema)
        if self.before_request is not None:
            self.before_request()
        response = self.client.propose(proposal_request)
        try:
            payloads = parse_reaction_candidates(response.text, domain=self.domain)
        except ValueError as exc:
            return ExpansionResult(
                attempts=(response,),
                metadata={
                    "mode": "proposal_client",
                    "round_idx": request.round_idx,
                    "parse_error": str(exc),
                },
            )
        return ExpansionResult(
            proposals=tuple(
                RawProposal(
                    payload,
                    PROPOSAL_SOURCE,
                    {"collectable": True, "round_idx": request.round_idx},
                )
                for payload in payloads
            ),
            attempts=(response,),
            metadata={"mode": "proposal_client", "round_idx": request.round_idx},
        )


def build_deepseek_reaction_client(
    *,
    base_url: str,
    model: str,
    api_key: str,
    timeout_seconds: float,
    max_tokens: int,
) -> OpenAICompatibleProposalClient:
    """Build the real proposal transport without placing credentials in requests."""

    return OpenAICompatibleProposalClient(
        url=base_url,
        model=model,
        api_key=api_key,
        timeout_seconds=timeout_seconds,
        max_tokens=max_tokens,
        max_retries=0,
        extra_body=DEEPSEEK_REACTION_EXTRA_BODY,
        require_models_preflight=True,
    )


def build_reaction_proposal_request(
    request: ExpansionRequest, schema: ReactionDatasetSchema
) -> ProposalRequest:
    """Build the exact English prompt used by both mock and endpoint clients."""

    _require_fixed_candidate_count(request)
    factors = [
        {"name": factor.name, "categories": list(factor.categories)}
        for factor in schema.factors
    ]
    observations = _proposal_observations(request.observations)
    do_not_repeat = _do_not_repeat_keys(request)
    envelope_example = _candidate_envelope_example(schema)
    content = "\n".join(
        (
            "Task: propose source-valid categorical reaction conditions.",
            f"Dataset ID: {schema.dataset_id}",
            f"Schema SHA-256: {schema.schema_sha256}",
            "Allowed categorical factors: " + _json_text(factors),
            "Observed evaluations: " + _json_text(observations),
            "Do-not-repeat canonical keys: " + _json_text(do_not_repeat),
            "Return exactly four distinct candidates as one complete JSON object.",
            "The JSON root must contain only the candidates array.",
            "Required JSON envelope example (one candidate shown; return exactly four): "
            + _json_text(envelope_example),
            "Replace the placeholders with allowed categories; each candidate must contain only dataset_id and conditions.",
            "Every factor must appear only inside conditions.",
            "Do not return markdown, prose, scores, ids, or any extra fields.",
        )
    )
    return ProposalRequest(
        messages=(
            {
                "role": "system",
                "content": "You propose reaction conditions under an exact source-pinned schema.",
            },
            {"role": "user", "content": content},
        ),
        metadata={
            "round_idx": request.round_idx,
            "dataset_id": schema.dataset_id,
            "schema_sha256": schema.schema_sha256,
            "candidate_count": REQUIRED_CANDIDATE_COUNT,
        },
    )


def parse_reaction_candidates(
    text: str, *, domain: CandidateDomainAdapter
) -> tuple[dict[str, Any], ...]:
    """Parse exactly four unique, task-admitted payloads without rewriting them."""

    response = _load_complete_json_object(text)
    if set(response) != {"candidates"}:
        raise ValueError("proposal response must contain only the candidates field.")
    candidates = response["candidates"]
    if not isinstance(candidates, list) or len(candidates) != REQUIRED_CANDIDATE_COUNT:
        raise ValueError("proposal response must contain exactly four candidates.")

    payloads = []
    canonical_keys = set()
    for index, payload in enumerate(candidates, start=1):
        admitted = _admit_payload(payload, domain, index)
        if admitted.canonical_key in canonical_keys:
            raise ValueError("proposal response candidates must be distinct.")
        canonical_keys.add(admitted.canonical_key)
        payloads.append(dict(payload))
    return tuple(payloads)


def _load_complete_json_object(text: str) -> dict[str, Any]:
    if not isinstance(text, str):
        raise ValueError("proposal response must be one complete JSON object.")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("proposal response must be one complete JSON object.") from exc
    if not isinstance(payload, dict):
        raise ValueError("proposal response must be one complete JSON object.")
    return payload


def _admit_payload(
    payload: Any, domain: CandidateDomainAdapter, index: int
) -> Candidate:
    if not isinstance(payload, dict):
        raise ValueError(f"proposal response candidate {index} must be a JSON object.")
    admitted = domain.admit(RawProposal(payload, PROPOSAL_SOURCE))
    if isinstance(admitted, CandidateRejection):
        raise ValueError(f"proposal candidate {index} rejected: {admitted.reason}")
    if not isinstance(admitted, Candidate):
        raise TypeError("candidate domain must return Candidate or CandidateRejection.")
    return admitted


def _require_fixed_candidate_count(request: ExpansionRequest) -> None:
    if request.reservoir_size != REQUIRED_CANDIDATE_COUNT:
        raise ValueError("Iron Mind proposal expansion requires exactly four candidates.")


def _candidate_envelope_example(schema: ReactionDatasetSchema) -> dict[str, Any]:
    return {
        "candidates": [
            {
                "dataset_id": schema.dataset_id,
                "conditions": {
                    factor.name: f"<allowed {factor.name} category>" for factor in schema.factors
                },
            }
        ]
    }


def _proposal_observations(observations: Sequence[Any]) -> list[dict[str, Any]]:
    return [
        {
            "candidate": observation.candidate.payload,
            "metrics": dict(observation.metrics),
        }
        for observation in observations
    ]


def _do_not_repeat_keys(request: ExpansionRequest) -> list[str]:
    raw_keys = request.context.get("do_not_repeat_keys", ())
    if isinstance(raw_keys, str) or not isinstance(raw_keys, Sequence):
        raise ValueError("do_not_repeat_keys must be a sequence of strings.")
    if any(not isinstance(key, str) or not key for key in raw_keys):
        raise ValueError("do_not_repeat_keys must contain non-empty strings.")
    return sorted({*raw_keys, *(item.canonical_key for item in request.observations)})


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
