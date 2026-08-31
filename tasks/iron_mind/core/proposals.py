"""Strict concurrent proposal expansion for Iron Mind reaction conditions."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
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
)
from tasks.iron_mind.core.proposal_base_measure import attach_empirical_base_measure
from tasks.iron_mind.core.proposal_parsing import (
    ParsedReactionResponses,
    ResponseParseError,
    parse_reaction_responses,
    validate_candidate_count,
)
from tasks.iron_mind.core.schema import ReactionDatasetSchema

PROPOSAL_SOURCE = "iron_mind_reaction_proposal"
SAMPLING_MODE = "local_concurrent_independent_requests"
DEFAULT_PROPOSAL_MAX_WORKERS = 64


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
        slot_seed: int = 0,
    ) -> None:
        if max_workers < 1:
            raise ValueError("proposal max_workers must be positive")
        if slot_seed < 0:
            raise ValueError("slot seed must be non-negative")
        self.client = client
        self.domain = domain
        self.before_requests = before_requests
        self.max_workers = max_workers
        self.prompt_policy = validate_prompt_policy(prompt_policy)
        self.slot_seed = slot_seed

    def expand(self, request: ExpansionRequest) -> ExpansionResult:
        """Launch one request per candidate and return a strict reservoir."""

        slot_plans = _build_slot_plans(
            request,
            self.domain.schema,
            self.prompt_policy,
            self.slot_seed,
        )
        proposal_requests = _build_proposal_requests(
            request,
            self.domain.schema,
            slot_plans,
        )
        if self.before_requests is not None:
            self.before_requests(len(proposal_requests))
        responses = self._propose_all(proposal_requests)
        parsed = parse_reaction_responses(
            responses,
            candidate_count=request.reservoir_size,
            slot_plans=slot_plans,
        )
        proposals = attach_empirical_base_measure(
            _raw_proposals(parsed, request, proposal_requests),
            request,
            self.domain,
        )
        return ExpansionResult(
            proposals=proposals,
            attempts=responses,
            metadata=_expansion_metadata(
                request,
                responses,
                self.max_workers,
                parsed.errors,
                self.prompt_policy,
                slot_plans,
                self.slot_seed,
            ),
        )

    def _propose_all(
        self, proposal_requests: Sequence[ProposalRequest]
    ) -> tuple[ProposalResponse, ...]:
        if len(proposal_requests) == 1 or not isinstance(
            self.client, OpenAICompatibleProposalClient
        ):
            return tuple(self.client.propose(item) for item in proposal_requests)
        worker_count = min(self.max_workers, len(proposal_requests))
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [executor.submit(self.client.propose, item) for item in proposal_requests]
            return tuple(future.result() for future in futures)


def _build_slot_plans(
    request: ExpansionRequest,
    schema: ReactionDatasetSchema,
    prompt_policy: str,
    slot_seed: int,
) -> tuple[ProposalSlotPlan, ...]:
    return tuple(
        build_slot_plan(
            request,
            schema,
            proposal_index=index,
            policy=prompt_policy,
            slot_seed=slot_seed,
        )
        for index in range(request.reservoir_size)
    )


def _build_proposal_requests(
    request: ExpansionRequest,
    schema: ReactionDatasetSchema,
    slot_plans: Sequence[ProposalSlotPlan],
) -> tuple[ProposalRequest, ...]:
    return tuple(
        build_reaction_proposal_request(
            request,
            schema,
            proposal_index=index,
            slot_plan=slot_plans[index],
        )
        for index in range(request.reservoir_size)
    )


def _raw_proposals(
    parsed: ParsedReactionResponses,
    request: ExpansionRequest,
    proposal_requests: Sequence[ProposalRequest],
) -> tuple[RawProposal, ...]:
    return tuple(
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

    proposal_count = validate_candidate_count(request.reservoir_size)
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


def _expansion_metadata(
    request: ExpansionRequest,
    responses: Sequence[ProposalResponse],
    max_workers: int,
    parse_errors: Sequence[ResponseParseError] = (),
    prompt_policy: str = DEFAULT_PROMPT_POLICY,
    slot_plans: Sequence[ProposalSlotPlan] = (),
    slot_seed: int = 0,
) -> dict[str, Any]:
    metadata = {
        "mode": "proposal_client",
        "sampling_mode": SAMPLING_MODE,
        "round_idx": request.round_idx,
        "request_count": request.reservoir_size,
        "response_count": len(responses),
        "max_workers": min(max_workers, request.reservoir_size),
        "prompt_policy": prompt_policy,
        "prompt_slot_seed": slot_seed,
        "proposal_role_counts": _slot_role_counts(slot_plans),
    }
    if parse_errors:
        metadata["invalid_response_count"] = len(parse_errors)
        metadata["response_errors"] = [item.to_dict() for item in parse_errors]
    return metadata


def _slot_plan_metadata(plan: ProposalSlotPlan | None) -> dict[str, Any]:
    return {} if plan is None else plan.metadata()


def _slot_role_counts(slot_plans: Sequence[ProposalSlotPlan]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for plan in slot_plans:
        counts[plan.role] = counts.get(plan.role, 0) + 1
    return counts
