"""Strict one-response parsing for independent SynthonBench proposals."""

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
    """One parsed valid response with its originating public slate."""

    proposal_index: int
    payload: dict[str, object]
    slot_plan: ProposalSlotPlan


@dataclass(frozen=True)
class ResponseParseError:
    """One isolated response-format or slate-membership failure."""

    proposal_index: int
    message: str

    def to_dict(self) -> dict[str, object]:
        return {"proposal_index": self.proposal_index, "message": self.message}


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
    for index, (response, plan) in enumerate(zip(responses, slot_plans, strict=True)):
        try:
            proposals.append(ParsedSynthonProposal(index, parse_synthon_response(response.text, plan), plan))
        except (TypeError, ValueError) as exc:
            errors.append(ResponseParseError(index, str(exc)))
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
    "parse_synthon_response",
    "parse_synthon_responses",
]
