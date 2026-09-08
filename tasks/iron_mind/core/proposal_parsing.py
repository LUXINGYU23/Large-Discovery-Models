"""Strict response parsing for Iron Mind proposal requests."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ldm_tts.transport import ProposalResponse

from tasks.iron_mind.core.prompting import ProposalSlotPlan, validate_slot_focus


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


def parse_reaction_responses(
    responses: Sequence[ProposalResponse],
    *,
    candidate_count: int,
    slot_plans: Sequence[ProposalSlotPlan] | None = None,
) -> ParsedReactionResponses:
    """Parse each independent response without discarding valid peers."""

    candidate_count = validate_candidate_count(candidate_count)
    if len(responses) != candidate_count:
        raise ValueError(f"proposal expansion must return exactly {candidate_count} responses.")
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


def parse_reaction_response(text: str, *, index: int) -> dict[str, Any]:
    """Parse one complete candidate JSON object at the response boundary."""

    payload = _load_complete_json_object(text)
    if set(payload) != {"dataset_id", "conditions"}:
        raise ValueError(f"proposal response {index} must contain only dataset_id and conditions.")
    if not isinstance(payload["dataset_id"], str):
        raise ValueError(f"proposal response {index} dataset_id must be a string.")
    if not isinstance(payload["conditions"], dict):
        raise ValueError(f"proposal response {index} conditions must be a JSON object.")
    return dict(payload)


def validate_candidate_count(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("candidate count must be a positive integer.")
    return value


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


__all__ = [
    "ParsedReactionProposal",
    "ParsedReactionResponses",
    "ResponseParseError",
    "parse_reaction_response",
    "parse_reaction_responses",
    "validate_candidate_count",
]
