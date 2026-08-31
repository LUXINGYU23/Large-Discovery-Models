"""Strict parsing for single and minibatch SynthonBench proposal responses."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ldm_tts.transport import ProposalResponse
from tasks.synthonbench.core.catalog import (
    ProposalSlotPlan,
    validate_payload_against_plan,
)


@dataclass(frozen=True)
class ParsedSynthonProposal:
    """One parsed valid occurrence with its request and public slate."""

    request_index: int
    proposal_index: int
    payload: dict[str, object]
    slot_plan: ProposalSlotPlan


@dataclass(frozen=True)
class ResponseParseError:
    """One isolated request- or candidate-level parsing failure."""

    request_index: int
    proposal_index: int | None
    message: str

    def to_dict(self) -> dict[str, object]:
        return {
            "request_index": self.request_index,
            "proposal_index": self.proposal_index,
            "message": self.message,
        }


@dataclass(frozen=True)
class ParsedSynthonResponses:
    """The independently accepted responses and rejected peers."""

    proposals: tuple[ParsedSynthonProposal, ...]
    errors: tuple[ResponseParseError, ...]


def parse_synthon_responses(
    responses: Sequence[ProposalResponse], slot_plans: Sequence[ProposalSlotPlan]
) -> ParsedSynthonResponses:
    """Parse every response without issuing refills for malformed peers."""

    if len(responses) != len(slot_plans):
        raise ValueError("proposal responses and slot plans must have equal lengths")
    proposals: list[ParsedSynthonProposal] = []
    errors: list[ResponseParseError] = []
    for request_index, (response, plan) in enumerate(zip(responses, slot_plans, strict=True)):
        try:
            proposals.append(ParsedSynthonProposal(
                request_index=request_index,
                proposal_index=plan.proposal_index,
                payload=parse_synthon_response(response.text, plan),
                slot_plan=plan,
            ))
        except (TypeError, ValueError) as exc:
            errors.append(ResponseParseError(request_index, plan.proposal_index, str(exc)))
    return ParsedSynthonResponses(tuple(proposals), tuple(errors))


def parse_synthon_batch_responses(
    responses: Sequence[ProposalResponse],
    plan_batches: Sequence[Sequence[ProposalSlotPlan]],
) -> ParsedSynthonResponses:
    """Parse each minibatch response while preserving valid peer occurrences."""

    if len(responses) != len(plan_batches):
        raise ValueError("proposal responses and plan batches must have equal lengths")
    proposals: list[ParsedSynthonProposal] = []
    errors: list[ResponseParseError] = []
    for request_index, (response, plans) in enumerate(
        zip(responses, plan_batches, strict=True)
    ):
        try:
            parsed = parse_synthon_batch_response(
                response.text,
                plans,
                request_index=request_index,
            )
        except (TypeError, ValueError) as exc:
            errors.append(ResponseParseError(request_index, None, str(exc)))
            continue
        proposals.extend(parsed.proposals)
        errors.extend(parsed.errors)
    return ParsedSynthonResponses(tuple(proposals), tuple(errors))


def parse_synthon_batch_response(
    text: str,
    plans: Sequence[ProposalSlotPlan],
    *,
    request_index: int = 0,
) -> ParsedSynthonResponses:
    """Parse one exact-size candidate array keyed by proposal_index."""

    if not plans:
        raise ValueError("proposal batch must contain at least one slot")
    envelope = _load_json_object(text)
    if set(envelope) != {"candidates"}:
        raise ValueError("batch response must contain exactly candidates")
    items = envelope["candidates"]
    if not isinstance(items, list):
        raise TypeError("batch response candidates must be an array")
    if len(items) != len(plans):
        raise ValueError(f"batch response must contain exactly {len(plans)} candidates")

    plans_by_index = {plan.proposal_index: plan for plan in plans}
    if len(plans_by_index) != len(plans):
        raise ValueError("proposal batch contains duplicate proposal indices")
    proposals: list[ParsedSynthonProposal] = []
    errors: list[ResponseParseError] = []
    seen: set[int] = set()
    for item in items:
        proposal_index: int | None = None
        try:
            proposal_index, payload = _batch_candidate(item)
            if proposal_index not in plans_by_index:
                raise ValueError(f"proposal_index {proposal_index} is not assigned to this request")
            if proposal_index in seen:
                raise ValueError(f"proposal_index {proposal_index} appears more than once")
            seen.add(proposal_index)
            plan = plans_by_index[proposal_index]
            _validate_types(payload)
            validate_payload_against_plan(payload, plan)
            proposals.append(ParsedSynthonProposal(
                request_index=request_index,
                proposal_index=proposal_index,
                payload=payload,
                slot_plan=plan,
            ))
        except (TypeError, ValueError) as exc:
            errors.append(ResponseParseError(request_index, proposal_index, str(exc)))

    for missing in sorted(set(plans_by_index) - seen):
        errors.append(ResponseParseError(
            request_index,
            missing,
            f"proposal_index {missing} is missing from the batch response",
        ))
    proposals.sort(key=lambda item: item.proposal_index)
    return ParsedSynthonResponses(tuple(proposals), tuple(errors))


def parse_synthon_response(text: str, plan: ProposalSlotPlan) -> dict[str, object]:
    """Accept exactly one JSON object in the response and nothing else."""

    payload = _load_json_object(text)
    if set(payload) != {"reaction_id", "synthon_ids"}:
        raise ValueError("response must contain exactly reaction_id and synthon_ids")
    _validate_types(payload)
    validate_payload_against_plan(payload, plan)
    return payload


def _load_json_object(text: Any) -> dict[str, object]:
    if not isinstance(text, str):
        raise TypeError("proposal response must be one JSON object")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("proposal response must be one complete JSON object") from exc
    if not isinstance(payload, dict):
        raise TypeError("proposal response must be one JSON object")
    return dict(payload)


def _batch_candidate(value: Any) -> tuple[int, dict[str, object]]:
    if not isinstance(value, dict):
        raise TypeError("each batch candidate must be one JSON object")
    if set(value) != {"proposal_index", "reaction_id", "synthon_ids"}:
        raise ValueError(
            "each batch candidate must contain exactly proposal_index, reaction_id, and synthon_ids"
        )
    proposal_index = value["proposal_index"]
    if isinstance(proposal_index, bool) or not isinstance(proposal_index, int):
        raise TypeError("proposal_index must be an integer")
    return proposal_index, {
        "reaction_id": value["reaction_id"],
        "synthon_ids": value["synthon_ids"],
    }


def _validate_types(payload: dict[str, object]) -> None:
    if not isinstance(payload["reaction_id"], str):
        raise TypeError("reaction_id must be a string")
    ids = payload["synthon_ids"]
    if not isinstance(ids, list):
        raise TypeError("synthon_ids must be an array")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in ids):
        raise ValueError("synthon_ids must contain only integers")


__all__ = [
    "ParsedSynthonProposal",
    "ParsedSynthonResponses",
    "ResponseParseError",
    "parse_synthon_batch_response",
    "parse_synthon_batch_responses",
    "parse_synthon_response",
    "parse_synthon_responses",
]
