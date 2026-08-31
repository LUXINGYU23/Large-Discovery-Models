"""Concurrent single and minibatch SynthonBench proposal expansion."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from ldm_tts.contracts import RawProposal
from ldm_tts.engine.expansion import ExpansionRequest, ExpansionResult
from ldm_tts.transport import ProposalClient, ProposalRequest, ProposalResponse
from ldm_tts.transport.openai import OpenAICompatibleProposalClient
from tasks.synthonbench.core.candidate import SynthonCandidateDomain
from tasks.synthonbench.core.catalog import SynthonProposalCatalog
from tasks.synthonbench.core.prompting import (
    DEFAULT_PROMPT_POLICY,
    build_synthon_batch_prompt_messages,
    build_synthon_prompt_messages,
    prompt_sha256,
    validate_prompt_policy,
)
from tasks.synthonbench.core.proposal_base_measure import attach_empirical_base_measure
from tasks.synthonbench.core.proposal_parsing import (
    parse_synthon_batch_responses,
    parse_synthon_responses,
)
from tasks.synthonbench.core.space_order import ordered_positions

PROPOSAL_SOURCE = "synthonbench_independent_llm"
SINGLE_SAMPLING_MODE = "local_concurrent_independent_requests"
BATCH_SAMPLING_MODE = "local_concurrent_independent_minibatch_requests"


class SynthonBenchProposalExpander:
    """Issue independent requests that emit a fixed number of proposal occurrences."""

    def __init__(
        self,
        client: ProposalClient,
        domain: SynthonCandidateDomain,
        catalog: SynthonProposalCatalog,
        *,
        target: str,
        before_requests: Callable[[int], None] | None = None,
        candidates_per_request: int,
        max_workers: int,
        prompt_policy: str = DEFAULT_PROMPT_POLICY,
    ) -> None:
        if max_workers < 1:
            raise ValueError("proposal max_workers must be positive")
        if candidates_per_request < 1:
            raise ValueError("proposal candidates_per_request must be positive")
        self.client = client
        self.domain = domain
        self.catalog = catalog
        self.target = str(target)
        self.before_requests = before_requests
        self.candidates_per_request = int(candidates_per_request)
        self.max_workers = int(max_workers)
        self.prompt_policy = validate_prompt_policy(prompt_policy)

    def expand(self, request: ExpansionRequest) -> ExpansionResult:
        """Construct independent slots and group them into fixed-size endpoint requests."""

        if request.reservoir_size % self.candidates_per_request:
            raise ValueError("reservoir size must divide evenly by candidates_per_request")

        excluded = excluded_anchor_ids(request, self.catalog)
        plans = tuple(
            self.catalog.build_plan(
                round_idx=request.round_idx,
                proposal_index=index,
                excluded_anchor_ids=excluded,
            )
            for index in range(request.reservoir_size)
        )
        plan_batches = _batches(plans, self.candidates_per_request)
        proposal_requests = tuple(
            _proposal_request(
                request,
                batch,
                request_index=request_index,
                space=self.catalog.space,
                target=self.target,
                prompt_policy=self.prompt_policy,
            )
            for request_index, batch in enumerate(plan_batches)
        )
        if self.before_requests is not None:
            self.before_requests(len(proposal_requests))
        responses = self._propose_all(proposal_requests)
        parsed = (
            parse_synthon_responses(responses, plans)
            if self.candidates_per_request == 1
            else parse_synthon_batch_responses(responses, plan_batches)
        )
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
                len(parsed.proposals),
                parsed.errors,
                self.candidates_per_request,
                self.max_workers,
                self.prompt_policy,
                self.catalog.reaction_allocation,
            ),
        )

    def _propose_all(self, requests: Sequence[ProposalRequest]) -> tuple[ProposalResponse, ...]:
        if len(requests) == 1 or not isinstance(
            self.client, OpenAICompatibleProposalClient
        ):
            return tuple(self.client.propose(item) for item in requests)
        worker_count = min(self.max_workers, len(requests))
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [executor.submit(self.client.propose, item) for item in requests]
            return tuple(future.result() for future in futures)


def _proposal_request(
    request: ExpansionRequest,
    plans,
    *,
    request_index: int,
    space: object,
    target: str,
    prompt_policy: str,
) -> ProposalRequest:
    messages = (
        build_synthon_prompt_messages(
            request,
            plans[0],
            target=target,
            space=space,
            prompt_policy=prompt_policy,
        )
        if len(plans) == 1
        else build_synthon_batch_prompt_messages(
            request,
            plans,
            target=target,
            space=space,
            prompt_policy=prompt_policy,
        )
    )
    sampling_mode = _sampling_mode(len(plans))
    plan_metadata = [plan.metadata() for plan in plans]
    return ProposalRequest(
        messages=messages,
        metadata={
            "round_idx": request.round_idx,
            "request_index": request_index,
            "proposal_indices": [plan.proposal_index for plan in plans],
            "proposal_count": request.reservoir_size,
            "candidates_per_request": len(plans),
            "sampling_mode": sampling_mode,
            "prompt_policy": prompt_policy,
            "prompt_sha256": prompt_sha256(messages),
            **(plan_metadata[0] if len(plans) == 1 else {"proposal_plans": plan_metadata}),
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
                "request_index": item.request_index,
                "sampling_mode": requests[item.request_index].metadata["sampling_mode"],
                "prompt_sha256": requests[item.request_index].metadata["prompt_sha256"],
                **item.slot_plan.metadata(),
            },
        )
        for item in parsed
    )


def _metadata(request: ExpansionRequest, plans, responses, parsed_count: int, errors,
              candidates_per_request: int, max_workers: int, prompt_policy: str,
              reaction_allocation: str) -> dict[str, Any]:
    return {
        "sampling_mode": _sampling_mode(candidates_per_request),
        "round_idx": request.round_idx,
        "request_count": len(responses),
        "response_count": len(responses),
        "candidates_per_request": candidates_per_request,
        "candidate_count_requested": request.reservoir_size,
        "candidate_count_parsed": parsed_count,
        "max_workers": min(max_workers, len(responses)),
        "prompt_policy": prompt_policy,
        "reaction_allocation": reaction_allocation,
        "proposal_role_counts": _role_counts(plans),
        "invalid_response_count": len({item.request_index for item in errors}),
        "invalid_candidate_count": request.reservoir_size - parsed_count,
        "response_errors": [item.to_dict() for item in errors],
    }


def _batches(plans, size: int):
    return tuple(tuple(plans[start:start + size]) for start in range(0, len(plans), size))


def _sampling_mode(candidates_per_request: int) -> str:
    return BATCH_SAMPLING_MODE if candidates_per_request > 1 else SINGLE_SAMPLING_MODE


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
