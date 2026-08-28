"""Concurrent, one-candidate-per-request SynthonBench proposal expansion."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from ldm_tts.contracts import RawProposal
from ldm_tts.engine.expansion import ExpansionRequest, ExpansionResult
from ldm_tts.transport import ProposalClient, ProposalRequest, ProposalResponse
from tasks.synthonbench.core.candidate import SynthonCandidateDomain
from tasks.synthonbench.core.catalog import SynthonProposalCatalog
from tasks.synthonbench.core.space_order import ordered_positions
from tasks.synthonbench.core.prompting import (
    DEFAULT_PROMPT_POLICY,
    build_synthon_prompt_messages,
    prompt_sha256,
    validate_prompt_policy,
)
from tasks.synthonbench.core.proposal_base_measure import attach_empirical_base_measure
from tasks.synthonbench.core.proposal_parsing import parse_synthon_responses
from tasks.synthonbench.core.proposal_transport import supports_local_concurrency

PROPOSAL_SOURCE = "synthonbench_independent_llm"
SAMPLING_MODE = "local_concurrent_independent_requests"


class SynthonBenchProposalExpander:
    """Issue M independent single-choice requests without replacement/refill logic."""

    def __init__(
        self,
        client: ProposalClient,
        domain: SynthonCandidateDomain,
        catalog: SynthonProposalCatalog,
        *,
        target: str,
        before_requests: Callable[[int], None] | None = None,
        max_workers: int,
        prompt_policy: str = DEFAULT_PROMPT_POLICY,
    ) -> None:
        if max_workers < 1:
            raise ValueError("proposal max_workers must be positive")
        self.client = client
        self.domain = domain
        self.catalog = catalog
        self.target = str(target)
        self.before_requests = before_requests
        self.max_workers = int(max_workers)
        self.prompt_policy = validate_prompt_policy(prompt_policy)

    def expand(self, request: ExpansionRequest) -> ExpansionResult:
        """Construct one slate and one endpoint call for every requested proposal."""

        excluded = excluded_anchor_ids(request, self.catalog)
        plans = tuple(
            self.catalog.build_plan(
                round_idx=request.round_idx,
                proposal_index=index,
                excluded_anchor_ids=excluded,
            )
            for index in range(request.reservoir_size)
        )
        proposal_requests = tuple(
            _proposal_request(
                request,
                plan,
                space=self.catalog.space,
                target=self.target,
                prompt_policy=self.prompt_policy,
            )
            for plan in plans
        )
        if self.before_requests is not None:
            self.before_requests(len(proposal_requests))
        responses = self._propose_all(proposal_requests)
        parsed = parse_synthon_responses(responses, plans)
        proposals = attach_empirical_base_measure(
            _raw_proposals(parsed.proposals, request, proposal_requests), request, self.domain
        )
        return ExpansionResult(
            proposals=proposals,
            attempts=responses,
            metadata=_metadata(
                request,
                plans,
                responses,
                parsed.errors,
                self.max_workers,
                self.prompt_policy,
                self.catalog.reaction_allocation,
            ),
        )

    def _propose_all(self, requests: Sequence[ProposalRequest]) -> tuple[ProposalResponse, ...]:
        if len(requests) == 1 or not supports_local_concurrency(self.client):
            return tuple(self.client.propose(item) for item in requests)
        worker_count = min(self.max_workers, len(requests))
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [executor.submit(self.client.propose, item) for item in requests]
            return tuple(future.result() for future in futures)


def _proposal_request(
    request: ExpansionRequest,
    plan,
    *,
    space: object,
    target: str,
    prompt_policy: str,
) -> ProposalRequest:
    messages = build_synthon_prompt_messages(
        request,
        plan,
        target=target,
        space=space,
        prompt_policy=prompt_policy,
    )
    return ProposalRequest(
        messages=messages,
        metadata={
            "round_idx": request.round_idx,
            "proposal_index": plan.proposal_index,
            "proposal_count": request.reservoir_size,
            "sampling_mode": SAMPLING_MODE,
            "prompt_policy": prompt_policy,
            "prompt_sha256": prompt_sha256(messages),
            **plan.metadata(),
        },
    )


def _raw_proposals(parsed, request: ExpansionRequest,
                   requests: Sequence[ProposalRequest]) -> tuple[RawProposal, ...]:
    return tuple(
        RawProposal(
            item.payload,
            PROPOSAL_SOURCE,
            {
                "collectable": True,
                "round_idx": request.round_idx,
                "sampling_mode": SAMPLING_MODE,
                "prompt_sha256": requests[item.proposal_index].metadata["prompt_sha256"],
                **item.slot_plan.metadata(),
            },
        )
        for item in parsed
    )


def _metadata(request: ExpansionRequest, plans, responses, errors, max_workers: int,
              prompt_policy: str, reaction_allocation: str) -> dict[str, Any]:
    return {
        "sampling_mode": SAMPLING_MODE,
        "round_idx": request.round_idx,
        "request_count": request.reservoir_size,
        "response_count": len(responses),
        "max_workers": min(max_workers, request.reservoir_size),
        "prompt_policy": prompt_policy,
        "reaction_allocation": reaction_allocation,
        "proposal_role_counts": _role_counts(plans),
        "invalid_response_count": len(errors),
        "response_errors": [item.to_dict() for item in errors],
    }


def _role_counts(plans) -> dict[str, int]:
    counts: dict[str, int] = {}
    for plan in plans:
        counts[plan.role] = counts.get(plan.role, 0) + 1
    return counts


def excluded_anchor_ids(
    request: ExpansionRequest,
    catalog: SynthonProposalCatalog,
) -> dict[str, set[int]]:
    if not catalog.unique_anchors:
        return {}
    excluded: dict[str, set[int]] = {}
    for observation in request.observations:
        payload = observation.candidate.payload
        reaction_id = payload.get("reaction_id") if isinstance(payload, dict) else None
        synthon_ids = payload.get("synthon_ids") if isinstance(payload, dict) else None
        if not isinstance(reaction_id, str) or not isinstance(synthon_ids, list):
            raise TypeError("SynthonBench observations must retain reaction_id and synthon_ids")
        positions = ordered_positions(catalog.space, reaction_id)
        anchor = catalog.anchor_position(reaction_id)
        anchor_index = positions.index(anchor)
        anchor_id = synthon_ids[anchor_index]
        if isinstance(anchor_id, bool) or not isinstance(anchor_id, int):
            raise TypeError("SynthonBench observation anchor must be an integer")
        excluded.setdefault(reaction_id, set()).add(anchor_id)
    return excluded


__all__ = ["SynthonBenchProposalExpander", "excluded_anchor_ids"]
