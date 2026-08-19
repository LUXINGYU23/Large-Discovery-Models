"""Strict concurrent proposal expansion for Iron Mind reaction conditions."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from ldm_tts.contracts import RawProposal
from ldm_tts.engine.expansion import ExpansionRequest, ExpansionResult
from ldm_tts.transport import ProposalClient, ProposalRequest, ProposalResponse
from ldm_tts.transport.openai import OpenAICompatibleProposalClient

from tasks.iron_mind.core.candidate import IronMindCandidateDomain
from tasks.iron_mind.core.schema import ReactionDatasetSchema


PROPOSAL_SOURCE = "iron_mind_reaction_proposal"
SAMPLING_MODE = "local_concurrent_independent_requests"
DEFAULT_PROPOSAL_MAX_WORKERS = 64


@dataclass(frozen=True)
class ParsedReactionProposal:
    proposal_index: int
    payload: dict[str, Any]


@dataclass(frozen=True)
class ResponseParseError:
    proposal_index: int
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {"proposal_index": self.proposal_index, "message": self.message}


@dataclass(frozen=True)
class ParsedReactionResponses:
    proposals: tuple[ParsedReactionProposal, ...]
    errors: tuple[ResponseParseError, ...]


class IronMindProposalExpander:
    """Expand independently generated one-candidate proposal responses."""

    def __init__(
        self,
        client: ProposalClient,
        domain: IronMindCandidateDomain,
        *,
        before_requests: Callable[[int], None] | None = None,
        max_workers: int = DEFAULT_PROPOSAL_MAX_WORKERS,
    ) -> None:
        if max_workers < 1:
            raise ValueError("proposal max_workers must be positive")
        self.client = client
        self.domain = domain
        self.before_requests = before_requests
        self.max_workers = max_workers

    def expand(self, request: ExpansionRequest) -> ExpansionResult:
        """Launch one request per candidate and return a strict reservoir."""

        proposal_requests = tuple(
            build_reaction_proposal_request(
                request,
                self.domain.schema,
                proposal_index=index,
            )
            for index in range(request.reservoir_size)
        )
        if self.before_requests is not None:
            self.before_requests(len(proposal_requests))
        responses = self._propose_all(proposal_requests)
        parsed = parse_reaction_responses(
            responses,
            candidate_count=request.reservoir_size,
        )
        return ExpansionResult(
            proposals=tuple(
                RawProposal(
                    item.payload,
                    PROPOSAL_SOURCE,
                    {
                        "collectable": True,
                        "round_idx": request.round_idx,
                        "proposal_index": item.proposal_index,
                        "sampling_mode": SAMPLING_MODE,
                    },
                )
                for item in parsed.proposals
            ),
            attempts=responses,
            metadata=_expansion_metadata(
                request,
                responses,
                self.max_workers,
                parsed.errors,
            ),
        )

    def _propose_all(
        self, proposal_requests: Sequence[ProposalRequest]
    ) -> tuple[ProposalResponse, ...]:
        if len(proposal_requests) == 1 or not _can_parallelize(self.client):
            return tuple(self.client.propose(item) for item in proposal_requests)
        worker_count = min(self.max_workers, len(proposal_requests))
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [executor.submit(self.client.propose, item) for item in proposal_requests]
            return tuple(future.result() for future in futures)


def build_openai_reaction_client(
    *,
    base_url: str,
    model: str,
    api_key: str,
    timeout_seconds: float,
    max_tokens: int,
    temperature: float = 0.7,
) -> OpenAICompatibleProposalClient:
    """Build the shared OpenAI-compatible transport for a real campaign."""

    return OpenAICompatibleProposalClient(
        url=base_url,
        model=model,
        api_key=api_key,
        timeout_seconds=timeout_seconds,
        max_tokens=max_tokens,
        temperature=temperature,
        max_retries=0,
        require_models_preflight=False,
    )


def build_reaction_proposal_request(
    request: ExpansionRequest,
    schema: ReactionDatasetSchema,
    *,
    proposal_index: int,
) -> ProposalRequest:
    """Build one independent one-candidate proposal request."""

    proposal_count = _candidate_count(request.reservoir_size)
    if proposal_index < 0 or proposal_index >= proposal_count:
        raise ValueError("proposal index must be inside the requested reservoir")
    factors = [
        {
            "name": factor.name,
            "type": factor.parameter_type,
            "options": list(factor.options),
        }
        for factor in schema.factors
    ]
    content = "\n".join(
        (
            "Task: propose one source-valid finite reaction condition.",
            f"Dataset ID: {schema.dataset_id}",
            f"Schema SHA-256: {schema.schema_sha256}",
            f"Independent proposal slot: {proposal_index + 1} of {proposal_count}.",
            "Allowed factors and exact options: " + _json_text(factors),
            "Observed evaluations: " + _json_text(_proposal_observations(request.observations)),
            "Do-not-repeat canonical keys: " + _json_text(_do_not_repeat_keys(request)),
            "Return exactly one source-valid candidate as one complete JSON object.",
            "The JSON root must contain only dataset_id and conditions.",
            "Required JSON object: " + _json_text(_candidate_example(schema)),
            "Use only the exact typed options shown above; every factor must appear only inside conditions.",
            "Do not return markdown, prose, scores, ids, or extra fields.",
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
            "proposal_index": proposal_index,
            "proposal_count": proposal_count,
            "sampling_mode": SAMPLING_MODE,
        },
    )


def parse_reaction_responses(
    responses: Sequence[ProposalResponse],
    *,
    candidate_count: int,
) -> ParsedReactionResponses:
    """Parse each independent response without discarding valid peers."""

    candidate_count = _candidate_count(candidate_count)
    if len(responses) != candidate_count:
        raise ValueError(
            f"proposal expansion must return exactly {candidate_count} responses."
        )
    proposals = []
    errors = []
    for index, response in enumerate(responses, start=1):
        try:
            payload = parse_reaction_response(response.text, index=index)
        except ValueError as exc:
            errors.append(ResponseParseError(index - 1, str(exc)))
            continue
        proposals.append(ParsedReactionProposal(index - 1, payload))
    return ParsedReactionResponses(tuple(proposals), tuple(errors))


def parse_reaction_response(
    text: str,
    *,
    index: int,
) -> dict[str, Any]:
    """Parse one complete candidate JSON object at the response boundary."""

    payload = _load_complete_json_object(text)
    if set(payload) != {"dataset_id", "conditions"}:
        raise ValueError(f"proposal response {index} must contain only dataset_id and conditions.")
    if not isinstance(payload["dataset_id"], str):
        raise ValueError(f"proposal response {index} dataset_id must be a string.")
    if not isinstance(payload["conditions"], dict):
        raise ValueError(f"proposal response {index} conditions must be a JSON object.")
    return dict(payload)


def _expansion_metadata(
    request: ExpansionRequest,
    responses: Sequence[ProposalResponse],
    max_workers: int,
    parse_errors: Sequence[ResponseParseError] = (),
) -> dict[str, Any]:
    metadata = {
        "mode": "proposal_client",
        "sampling_mode": SAMPLING_MODE,
        "round_idx": request.round_idx,
        "request_count": request.reservoir_size,
        "response_count": len(responses),
        "max_workers": min(max_workers, request.reservoir_size),
    }
    if parse_errors:
        metadata["invalid_response_count"] = len(parse_errors)
        metadata["response_errors"] = [item.to_dict() for item in parse_errors]
    return metadata


def _can_parallelize(client: ProposalClient) -> bool:
    return isinstance(client, OpenAICompatibleProposalClient)


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


def _candidate_count(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("candidate count must be a positive integer.")
    return value


def _candidate_example(schema: ReactionDatasetSchema) -> dict[str, Any]:
    return {
        "dataset_id": schema.dataset_id,
        "conditions": {factor.name: factor.options[0] for factor in schema.factors},
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
    return sorted(item.canonical_key for item in request.observations)


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
