"""Prompt-policy orchestration for finite Iron Mind reaction search."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

from ldm_tts.engine.expansion import ExpansionRequest

from tasks.iron_mind.core.prompt_baseline import build_baseline_messages
from tasks.iron_mind.core.prompt_policy import (
    BASELINE_PROMPT_POLICY,
    DEFAULT_PROMPT_POLICY,
    DIRECT_PROMPT_POLICY,
    EVIDENCE_ROLE_INSTRUCTIONS,
    INITIAL_ROLE_INSTRUCTIONS,
    PORTFOLIO_PROMPT_POLICY,
    PROMPT_POLICIES,
    ProposalSlotPlan,
)
from tasks.iron_mind.core.prompt_portfolio import build_portfolio_messages
from tasks.iron_mind.core.schema import ReactionDatasetSchema, ReactionFactor, ReactionValue


def validate_prompt_policy(value: str) -> str:
    """Return one supported prompt policy name."""

    policy = str(value).strip()
    if policy not in PROMPT_POLICIES:
        choices = ", ".join(sorted(PROMPT_POLICIES))
        raise ValueError(f"Unknown Iron Mind prompt policy {policy!r}; choose one of {choices}.")
    return policy


def build_slot_plan(
    request: ExpansionRequest,
    schema: ReactionDatasetSchema,
    *,
    proposal_index: int,
    policy: str = DEFAULT_PROMPT_POLICY,
    slot_seed: int = 0,
) -> ProposalSlotPlan:
    """Allocate one semantically distinct prompt slot without touching scores."""

    _validate_slot_index(request.reservoir_size, proposal_index)
    if slot_seed < 0:
        raise ValueError("slot seed must be non-negative")
    policy = validate_prompt_policy(policy)
    if policy == BASELINE_PROMPT_POLICY:
        return ProposalSlotPlan(policy, "baseline", "Propose one novel condition from the schema.")
    if policy == DIRECT_PROMPT_POLICY:
        focus, capacity, position = _slot_focus(
            schema,
            request.reservoir_size,
            proposal_index,
            round_idx=request.round_idx,
            slot_seed=slot_seed,
        )
        return ProposalSlotPlan(
            policy,
            "direct_search",
            "Choose an unevaluated condition in the assigned focus without using a GP ranking.",
            focus,
            capacity,
            position,
        )
    role, instruction = _role_instruction(request.observations, proposal_index)
    focus, capacity, position = _slot_focus(
        schema,
        request.reservoir_size,
        proposal_index,
        round_idx=request.round_idx,
        slot_seed=slot_seed,
    )
    return ProposalSlotPlan(policy, role, instruction, focus, capacity, position)


def build_reaction_prompt_messages(
    request: ExpansionRequest,
    schema: ReactionDatasetSchema,
    plan: ProposalSlotPlan,
    *,
    proposal_index: int,
) -> tuple[dict[str, str], dict[str, str]]:
    """Render the prompt for one independently issued proposal request."""

    _validate_slot_index(request.reservoir_size, proposal_index)
    if plan.policy == BASELINE_PROMPT_POLICY:
        return build_baseline_messages(request, schema, proposal_index)
    return build_portfolio_messages(request, schema, plan, proposal_index)


def prompt_sha256(messages: Sequence[Mapping[str, Any]]) -> str:
    """Return a stable digest of the exact rendered prompt, not its contents."""

    encoded = json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def validate_slot_focus(payload: Mapping[str, Any], plan: ProposalSlotPlan) -> None:
    """Enforce the portfolio allocation before a proposal reaches the reservoir."""

    if not plan.focus:
        return
    conditions = payload.get("conditions")
    if not isinstance(conditions, Mapping):
        raise ValueError("proposal conditions must satisfy the assigned slot focus.")
    missing = [name for name, value in plan.focus if conditions.get(name) != value]
    if missing:
        raise ValueError(
            "proposal does not satisfy assigned slot focus factor(s): " + ", ".join(missing)
        )


def _role_instruction(observations: Sequence[Any], proposal_index: int) -> tuple[str, str]:
    choices = EVIDENCE_ROLE_INSTRUCTIONS if observations else INITIAL_ROLE_INSTRUCTIONS
    return choices[proposal_index % len(choices)]


def _slot_focus(
    schema: ReactionDatasetSchema,
    reservoir_size: int,
    proposal_index: int,
    *,
    round_idx: int,
    slot_seed: int,
) -> tuple[tuple[tuple[str, ReactionValue], ...], int, int]:
    factors, capacity = _focus_factors(schema, reservoir_size)
    stride = _balanced_stride(capacity, reservoir_size)
    batch_offset = (round_idx + slot_seed) * reservoir_size
    focus_position = ((batch_offset + proposal_index) * stride) % capacity
    position = focus_position
    focus = []
    for factor in factors:
        option_index = position % len(factor.options)
        focus.append((factor.name, factor.options[option_index]))
        position //= len(factor.options)
    return tuple(focus), capacity, focus_position


def _balanced_stride(capacity: int, sample_count: int) -> int:
    stride = max(1, capacity // sample_count + 1)
    while math.gcd(stride, capacity) != 1:
        stride += 1
    return stride


def _focus_factors(
    schema: ReactionDatasetSchema,
    reservoir_size: int,
) -> tuple[tuple[ReactionFactor, ...], int]:
    capacity = 1
    selected = []
    ordered = sorted(schema.factors, key=lambda item: (-len(item.options), item.name))
    for factor in ordered:
        selected.append(factor)
        capacity *= len(factor.options)
        if capacity >= reservoir_size:
            break
    return tuple(selected), capacity


def _validate_slot_index(reservoir_size: int, proposal_index: int) -> None:
    if proposal_index < 0 or proposal_index >= reservoir_size:
        raise ValueError("proposal index must be inside the requested reservoir")


__all__ = [
    "BASELINE_PROMPT_POLICY",
    "DEFAULT_PROMPT_POLICY",
    "DIRECT_PROMPT_POLICY",
    "PORTFOLIO_PROMPT_POLICY",
    "PROMPT_POLICIES",
    "ProposalSlotPlan",
    "build_reaction_prompt_messages",
    "build_slot_plan",
    "prompt_sha256",
    "validate_prompt_policy",
    "validate_slot_focus",
]
