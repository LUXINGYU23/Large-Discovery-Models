"""Strict concurrent proposal expansion for Iron Mind reaction conditions."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from ldm_tts.contracts import RawProposal
from ldm_tts.engine.expansion import ExpansionRequest, ExpansionResult
from ldm_tts.transport import ProposalClient, ProposalRequest, ProposalResponse
from ldm_tts.transport.openai import OpenAICompatibleProposalClient

from tasks.iron_mind.core.candidate import IronMindCandidateDomain
from tasks.iron_mind.core.prompting import (
    DEFAULT_PROMPT_POLICY,
    ProposalSlotPlan,
    build_reaction_prompt_messages,
    build_slot_plan,
    prompt_sha256,
    validate_prompt_policy,
    validate_slot_focus,
)
from tasks.iron_mind.core.schema import ReactionDatasetSchema


PROPOSAL_SOURCE = "iron_mind_reaction_proposal"
SAMPLING_MODE = "local_concurrent_independent_requests"
DEFAULT_PROPOSAL_MAX_WORKERS = 64


@dataclass(frozen=True)
class ParsedReactionProposal:
    proposal_index: int
    payload: dict[str, Any]
    slot_plan: ProposalSlotPlan | None = None


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
        prompt_policy: str = DEFAULT_PROMPT_POLICY,
    ) -> None:
        if max_workers < 1:
            raise ValueError("proposal max_workers must be positive")
        self.client = client
        self.domain = domain
        self.before_requests = before_requests
        self.max_workers = max_workers
        self.prompt_policy = validate_prompt_policy(prompt_policy)

    def expand(self, request: ExpansionRequest) -> ExpansionResult:
        """Launch one request per candidate and return a strict reservoir."""

        slot_plans = tuple(
            build_slot_plan(
                request,
                self.domain.schema,
                proposal_index=index,
                policy=self.prompt_policy,
            )
            for index in range(request.reservoir_size)
        )
        proposal_requests = tuple(
            build_reaction_proposal_request(
                request,
                self.domain.schema,
                proposal_index=index,
                slot_plan=slot_plans[index],
            )
            for index in range(request.reservoir_size)
        )
        if self.before_requests is not None:
            self.before_requests(len(proposal_requests))
        responses = self._propose_all(proposal_requests)
        parsed = parse_reaction_responses(
            responses,
            candidate_count=request.reservoir_size,
            slot_plans=slot_plans,
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
                        **_slot_plan_metadata(item.slot_plan),
                        "prompt_sha256": proposal_requests[item.proposal_index].metadata[
                            "prompt_sha256"
                        ],
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
                self.prompt_policy,
                slot_plans,
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
    json_mode: bool = False,
    extra_body: Mapping[str, Any] | None = None,
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
        extra_body=_request_extra_body(json_mode, extra_body),
        require_models_preflight=False,
    )


def build_reaction_proposal_request(
    request: ExpansionRequest,
    schema: ReactionDatasetSchema,
    *,
    proposal_index: int,
    prompt_policy: str = DEFAULT_PROMPT_POLICY,
    slot_plan: ProposalSlotPlan | None = None,
) -> ProposalRequest:
    """Build one independent one-candidate proposal request."""

    proposal_count = _candidate_count(request.reservoir_size)
    if proposal_index < 0 or proposal_index >= proposal_count:
        raise ValueError("proposal index must be inside the requested reservoir")
    plan = slot_plan or build_slot_plan(
        request,
        schema,
        proposal_index=proposal_index,
        policy=prompt_policy,
    )
    messages = build_reaction_prompt_messages(
        request,
        schema,
        plan,
        proposal_index=proposal_index,
    )
    return ProposalRequest(
        messages=messages,
        metadata={
            "round_idx": request.round_idx,
            "dataset_id": schema.dataset_id,
            "schema_sha256": schema.schema_sha256,
            "proposal_index": proposal_index,
            "proposal_count": proposal_count,
            "sampling_mode": SAMPLING_MODE,
            "prompt_sha256": prompt_sha256(messages),
            **plan.metadata(),
        },
    )


def parse_reaction_responses(
    responses: Sequence[ProposalResponse],
    *,
    candidate_count: int,
    slot_plans: Sequence[ProposalSlotPlan] | None = None,
) -> ParsedReactionResponses:
    """Parse each independent response without discarding valid peers."""

    candidate_count = _candidate_count(candidate_count)
    if len(responses) != candidate_count:
        raise ValueError(
            f"proposal expansion must return exactly {candidate_count} responses."
        )
    if slot_plans is not None and len(slot_plans) != candidate_count:
        raise ValueError("proposal slot plans must match the requested candidate count.")
    proposals = []
    errors = []
    for index, response in enumerate(responses, start=1):
        try:
            payload = parse_reaction_response(response.text, index=index)
            slot_plan = None if slot_plans is None else slot_plans[index - 1]
            if slot_plan is not None:
                validate_slot_focus(payload, slot_plan)
        except ValueError as exc:
            errors.append(ResponseParseError(index - 1, str(exc)))
            continue
        proposals.append(ParsedReactionProposal(index - 1, payload, slot_plan))
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
    prompt_policy: str = DEFAULT_PROMPT_POLICY,
    slot_plans: Sequence[ProposalSlotPlan] = (),
) -> dict[str, Any]:
    metadata = {
        "mode": "proposal_client",
        "sampling_mode": SAMPLING_MODE,
        "round_idx": request.round_idx,
        "request_count": request.reservoir_size,
        "response_count": len(responses),
        "max_workers": min(max_workers, request.reservoir_size),
        "prompt_policy": prompt_policy,
        "proposal_role_counts": _slot_role_counts(slot_plans),
    }
    if parse_errors:
        metadata["invalid_response_count"] = len(parse_errors)
        metadata["response_errors"] = [item.to_dict() for item in parse_errors]
    return metadata


def _can_parallelize(client: ProposalClient) -> bool:
    return isinstance(client, OpenAICompatibleProposalClient)


def _request_extra_body(
    json_mode: bool,
    extra_body: Mapping[str, Any] | None,
) -> dict[str, Any]:
    body = dict(extra_body or {})
    if not json_mode:
        return body
    if "response_format" in body:
        raise ValueError(
            "--llm-json-mode cannot be combined with response_format in --llm-extra-body-json."
        )
    body["response_format"] = {"type": "json_object"}
    return body


def _slot_plan_metadata(plan: ProposalSlotPlan | None) -> dict[str, Any]:
    return {} if plan is None else plan.metadata()


def _slot_role_counts(slot_plans: Sequence[ProposalSlotPlan]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for plan in slot_plans:
        counts[plan.role] = counts.get(plan.role, 0) + 1
    return counts


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
