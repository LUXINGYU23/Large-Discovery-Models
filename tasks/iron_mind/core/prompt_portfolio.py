"""Rendering helpers for the optimized Iron Mind portfolio prompt."""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from ldm_tts.engine.expansion import ExpansionRequest

from tasks.iron_mind.core.prompt_policy import ProposalSlotPlan, SYSTEM_PROMPT
from tasks.iron_mind.core.schema import ReactionDatasetSchema, ReactionValue


def build_portfolio_messages(
    request: ExpansionRequest,
    schema: ReactionDatasetSchema,
    plan: ProposalSlotPlan,
    proposal_index: int,
) -> tuple[dict[str, str], dict[str, str]]:
    """Render task evidence and a hard focus allocation for one slot."""

    lines = (
        _campaign_lines(request, schema, proposal_index),
        _schema_lines(schema),
        _history_lines(request.observations, schema),
        _portfolio_lines(plan),
        _output_lines(schema, plan),
    )
    return (
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(line for group in lines for line in group)},
    )


def _campaign_lines(
    request: ExpansionRequest,
    schema: ReactionDatasetSchema,
    proposal_index: int,
) -> tuple[str, ...]:
    return (
        "Task: propose one source-valid finite reaction condition.",
        f"Dataset ID: {schema.dataset_id}",
        f"Schema SHA-256: {schema.schema_sha256}",
        "Objective: "
        + _json_text({"name": schema.objective, "direction": schema.direction, "meaning": _direction_text(schema)}),
        f"Campaign round: {request.round_idx}",
        f"Completed external evaluations: {len(request.observations)}",
        f"Independent proposal slot: {proposal_index + 1} of {request.reservoir_size}.",
    )


def _schema_lines(schema: ReactionDatasetSchema) -> tuple[str, ...]:
    factors = [
        {"name": factor.name, "type": factor.parameter_type, "options": list(factor.options)}
        for factor in schema.factors
    ]
    return ("Allowed factors and exact typed options: " + _json_text(factors),)


def _history_lines(observations: Sequence[Any], schema: ReactionDatasetSchema) -> tuple[str, ...]:
    return (
        "Observed evaluations, ordered by objective: "
        + _json_text(_ordered_observations(observations, schema)),
        "Observed option coverage: " + _json_text(_option_coverage(observations, schema)),
        "Do-not-repeat canonical keys: " + _json_text(_do_not_repeat_keys(observations)),
    )


def _portfolio_lines(plan: ProposalSlotPlan) -> tuple[str, ...]:
    return (
        "Proposal policy: portfolio_v1.",
        f"Assigned portfolio role: {plan.role}.",
        "Role instruction: " + plan.role_instruction,
        "Required slot focus (hard allocation): " + _json_text(plan.focus_payload()),
        "Your conditions MUST contain every exact value in the required slot focus.",
    )


def _output_lines(schema: ReactionDatasetSchema, plan: ProposalSlotPlan) -> tuple[str, ...]:
    example = {"dataset_id": schema.dataset_id, "conditions": _example_conditions(schema, plan)}
    return (
        "Use the supplied evidence and chemical knowledge, but do not estimate outcomes.",
        "Return exactly one source-valid candidate as one complete JSON object.",
        "The JSON root must contain only dataset_id and conditions.",
        "Required JSON object: " + _json_text(example),
        "Use only the exact typed options shown above; every factor must appear only inside conditions.",
        "Do not return markdown, prose, scores, ids, rationale, or extra fields.",
    )


def _ordered_observations(
    observations: Sequence[Any], schema: ReactionDatasetSchema
) -> list[dict[str, Any]]:
    ranked = sorted(
        observations,
        key=lambda item: _objective_value(item, schema),
        reverse=schema.direction == "maximize",
    )
    return [
        {"rank": index, "candidate": item.candidate.payload, "metrics": dict(item.metrics)}
        for index, item in enumerate(ranked, start=1)
    ]


def _option_coverage(observations: Sequence[Any], schema: ReactionDatasetSchema) -> list[dict[str, Any]]:
    counts = {factor.name: Counter() for factor in schema.factors}
    for item in observations:
        conditions = item.candidate.payload.get("conditions", {})
        if isinstance(conditions, Mapping):
            for factor in schema.factors:
                counts[factor.name][conditions.get(factor.name)] += 1
    return [
        {"factor": factor.name, "option_counts": [[option, counts[factor.name][option]] for option in factor.options]}
        for factor in schema.factors
    ]


def _objective_value(observation: Any, schema: ReactionDatasetSchema) -> float:
    value = observation.metrics.get(schema.objective)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return -math.inf if schema.direction == "maximize" else math.inf
    return float(value)


def _do_not_repeat_keys(observations: Sequence[Any]) -> list[str]:
    return sorted(item.candidate.canonical_key for item in observations)


def _example_conditions(
    schema: ReactionDatasetSchema,
    plan: ProposalSlotPlan,
) -> dict[str, ReactionValue]:
    conditions = {factor.name: factor.options[0] for factor in schema.factors}
    conditions.update(plan.focus_payload())
    return conditions


def _direction_text(schema: ReactionDatasetSchema) -> str:
    return "higher is better" if schema.direction == "maximize" else "lower is better"


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
