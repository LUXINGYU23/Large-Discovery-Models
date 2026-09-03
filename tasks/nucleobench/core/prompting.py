"""Prompt construction for direct NucleoBench mutation proposals."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from ldm_tts.contracts import Observation
from ldm_tts.engine.expansion import ExpansionRequest
from tasks.nucleobench.core.candidate import MutationContext

FULL_SEQUENCE_MAX_LENGTH = 1_000
LONG_CONTEXT_WINDOW_COUNT = 8
LONG_CONTEXT_WINDOW_RADIUS = 24


def build_mutation_prompt_messages(
    request: ExpansionRequest,
    context: MutationContext,
    *,
    candidate_count: int,
    request_id: str,
    lineage_index: int,
    wave_index: int,
    same_round_agreement_allowed: bool,
    additional_exclusions: Sequence[Mapping[str, Any]] = (),
) -> tuple[dict[str, str], dict[str, str]]:
    """Build one credential-free prompt with an exact mutation-patch contract."""

    if candidate_count < 1:
        raise ValueError("candidate_count must be positive")
    payload = {
        "message_type": "nucleobench_proposal_request",
        "request_id": request_id,
        "round_index": request.round_idx,
        "wave_index": wave_index,
        "case": {
            "case_id": context.case.case_id,
            "model_family": context.case.model_family,
            "model_name": context.case.model_name,
            "target": context.case.target,
            "sequence_length": context.case.sequence_length,
            "editable_position_count": len(context.editable_positions),
            "objective": "maximize measured utility; higher numeric values are better",
        },
        "sequence_context": _sequence_context(
            context,
            focus_index=request.round_idx + lineage_index,
        ),
        "new_measured_observations": _latest_measurements(request.observations),
        "evaluated_candidates": _evaluated_candidates(request.observations),
        "additional_forbidden_candidates": [
            dict(item) for item in additional_exclusions
        ],
        "novelty_contract": {
            "evaluated_candidates_are_forbidden": True,
            "previously_proposed_but_unmeasured_candidates_are_allowed": True,
            "agreement_with_other_requests_in_this_round_is_allowed": (
                same_round_agreement_allowed
            ),
        },
        "submission_contract": _submission_contract(candidate_count),
    }
    user = (
        "Propose nucleotide mutation patches for this black-box sequence-design task. "
        "Use measured results as evidence, but never invent scores. Every mutation must use "
        "a zero-based editable position shown in sequence_context and replace its start base "
        "with a different DNA base. Never submit a patch in evaluated_candidates. "
        "Also avoid any patch in additional_forbidden_candidates. "
        "Candidates proposed in earlier rounds but absent from evaluated_candidates remain "
        "eligible. Return only the exact JSON structure in submission_contract; do not return "
        "a full sequence, markdown, prose, reasoning, or extra fields.\n\n"
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    )
    return (
        {
            "role": "system",
            "content": (
                "You are a proposal component in a budgeted black-box DNA sequence search. "
                "Submit valid start-relative mutation patches as strict JSON only."
            ),
        },
        {"role": "user", "content": user},
    )


def prompt_sha256(messages: Sequence[Mapping[str, Any]]) -> str:
    encoded = json.dumps(
        list(messages),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sequence_context(context: MutationContext, *, focus_index: int) -> dict[str, Any]:
    if context.case.sequence_length <= FULL_SEQUENCE_MAX_LENGTH:
        return {
            "kind": "full_start_sequence",
            "start_sequence": context.start_sequence,
            "editable_positions": list(context.editable_positions),
        }

    editable = context.editable_positions
    window_count = min(LONG_CONTEXT_WINDOW_COUNT, len(editable))
    offset = focus_index % len(editable)
    anchor_indices = sorted(
        {
            (offset + index * len(editable) // window_count) % len(editable)
            for index in range(window_count)
        }
    )
    windows = []
    for anchor_index in anchor_indices:
        anchor = editable[anchor_index]
        start = max(0, anchor - LONG_CONTEXT_WINDOW_RADIUS)
        end = min(context.case.sequence_length, anchor + LONG_CONTEXT_WINDOW_RADIUS + 1)
        shown = [position for position in editable if start <= position < end]
        windows.append(
            {
                "start": start,
                "end_exclusive": end,
                "start_bases": context.start_sequence[start:end],
                "editable_positions": shown,
            }
        )
    return {
        "kind": "editable_windows",
        "position_indexing": "zero_based",
        "windows": windows,
    }


def _latest_measurements(observations: Sequence[Observation]) -> list[dict[str, Any]]:
    successful = [item for item in observations if item.evaluation.succeeded]
    if not successful:
        return []
    latest_round = max(
        (-1 if item.round_idx is None else item.round_idx) for item in successful
    )
    return [
        {
            "mutations": _mutations(item),
            "utility": item.evaluation.metrics.get("utility"),
        }
        for item in successful
        if (-1 if item.round_idx is None else item.round_idx) == latest_round
    ]


def _evaluated_candidates(observations: Sequence[Observation]) -> list[dict[str, Any]]:
    return [
        {
            "canonical_key": item.canonical_key,
            "mutations": _mutations(item),
        }
        for item in observations
    ]


def _mutations(observation: Observation) -> list[dict[str, Any]]:
    payload = observation.candidate.payload
    if not isinstance(payload, Mapping) or not isinstance(
        payload.get("mutations"), list
    ):
        raise TypeError("NucleoBench observations must retain mutation-patch payloads")
    return [dict(item) for item in payload["mutations"]]


def _submission_contract(candidate_count: int) -> dict[str, Any]:
    mutation = {"position": "integer", "base": "one of A, C, G, T"}
    if candidate_count == 1:
        return {
            "candidate_count": 1,
            "json_shape": {"mutations": [mutation]},
        }
    return {
        "candidate_count": candidate_count,
        "json_shape": {
            "candidates": [
                {
                    "proposal_index": f"integer from 0 through {candidate_count - 1}",
                    "mutations": [mutation],
                }
            ]
        },
    }


__all__ = ["build_mutation_prompt_messages", "prompt_sha256"]
