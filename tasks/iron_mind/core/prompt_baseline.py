"""Frozen legacy prompt used as the Iron Mind prompt ablation baseline."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from ldm_tts.engine.expansion import ExpansionRequest

from tasks.iron_mind.core.schema import ReactionDatasetSchema, ReactionValue


BASELINE_SYSTEM_PROMPT = "You propose reaction conditions under an exact source-pinned schema."


def build_baseline_messages(
    request: ExpansionRequest,
    schema: ReactionDatasetSchema,
    proposal_index: int,
) -> tuple[dict[str, str], dict[str, str]]:
    """Render the prompt used before portfolio prompting was introduced."""

    factors = [
        {"name": factor.name, "type": factor.parameter_type, "options": list(factor.options)}
        for factor in schema.factors
    ]
    example = {"dataset_id": schema.dataset_id, "conditions": _example_conditions(schema)}
    lines = (
        "Task: propose one source-valid finite reaction condition.",
        f"Dataset ID: {schema.dataset_id}",
        f"Schema SHA-256: {schema.schema_sha256}",
        f"Independent proposal slot: {proposal_index + 1} of {request.reservoir_size}.",
        "Allowed factors and exact options: " + _json_text(factors),
        "Observed evaluations: " + _json_text(_observations(request.observations)),
        "Do-not-repeat canonical keys: " + _json_text(_do_not_repeat_keys(request)),
        "Return exactly one source-valid candidate as one complete JSON object.",
        "The JSON root must contain only dataset_id and conditions.",
        "Required JSON object: " + _json_text(example),
        "Use only the exact typed options shown above; every factor must appear only inside conditions.",
        "Do not return markdown, prose, scores, ids, or extra fields.",
    )
    return (
        {"role": "system", "content": BASELINE_SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(lines)},
    )


def _example_conditions(schema: ReactionDatasetSchema) -> dict[str, ReactionValue]:
    return {factor.name: factor.options[0] for factor in schema.factors}


def _observations(observations: Sequence[Any]) -> list[dict[str, Any]]:
    return [
        {"candidate": item.candidate.payload, "metrics": dict(item.metrics)}
        for item in observations
    ]


def _do_not_repeat_keys(request: ExpansionRequest) -> list[str]:
    return sorted(item.canonical_key for item in request.observations)


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
