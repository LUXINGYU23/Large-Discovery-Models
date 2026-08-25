"""Versioned, leakage-free prompts for SynthonBench tuple proposals."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence

from ldm_tts.contracts import Observation
from ldm_tts.engine.expansion import ExpansionRequest
from tasks.synthonbench.core.catalog import ProposalSlotPlan
from tasks.synthonbench.core.constants import OBJECTIVE_NAME
from tasks.synthonbench.core.space_order import ordered_positions

DEFAULT_PROMPT_POLICY = "policy_v1"
PROMPT_POLICIES = ("baseline_v1", "direct_v1", "policy_v1")
HISTORY_TOP_COUNT = 8
HISTORY_RECENT_COUNT = 4

ROLE_INSTRUCTIONS = {
    "explore": "Choose a chemically plausible tuple that explores motifs absent from the best observations.",
    "exploit": "Prefer components chemically analogous to fragments in the highest-utility observations.",
    "diversify": "Combine promising chemistry with fragments unlike the repeatedly observed motifs.",
    "scaffold_shift": "Test a coherent scaffold shift while preserving target-relevant functionality.",
}


def validate_prompt_policy(policy: str) -> str:
    normalized = str(policy).strip()
    if normalized not in PROMPT_POLICIES:
        raise ValueError(f"prompt policy must be one of {PROMPT_POLICIES}")
    return normalized


def build_synthon_prompt_messages(
    request: ExpansionRequest,
    plan: ProposalSlotPlan,
    *,
    target: str,
    space: object,
    prompt_policy: str = DEFAULT_PROMPT_POLICY,
) -> list[dict[str, str]]:
    """Build one self-contained prompt for exactly one tuple response."""

    policy = validate_prompt_policy(prompt_policy)
    history = _history_summary(request.observations, space)
    user = _user_prompt(plan, target=target, policy=policy, history=history)
    return [
        {"role": "system", "content": _system_prompt()},
        {"role": "user", "content": user},
    ]


def prompt_sha256(messages: Sequence[dict[str, str]]) -> str:
    """Return a stable digest without persisting endpoint credentials."""

    payload = json.dumps(list(messages), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _system_prompt() -> str:
    return (
        "You are a molecular-design proposal component in a budgeted black-box search. "
        "Choose one valid tuple only from the supplied reaction-specific synthon slate. "
        "Do not invent identifiers, do not predict scores, and return JSON only."
    )


def _user_prompt(plan: ProposalSlotPlan, *, target: str, policy: str,
                 history: list[dict[str, object]]) -> str:
    payload = {
        "target_label": target,
        "objective": "maximize the measured SynthonBench utility; a higher numeric value is better",
        "observed_history": history,
        "reaction_id": plan.reaction_id,
        "proposal_role": plan.role,
        **_candidate_options(plan, policy),
        "output_contract": {
            "reaction_id": plan.reaction_id,
            "synthon_ids": "copy one complete integer array from the supplied options",
        },
    }
    instruction = _policy_instruction(policy, plan.role)
    return (
        "Select one ordered synthon tuple for the fixed reaction below. "
        "The history contains only previous charged measurements; it contains no hidden scores. "
        f"{instruction}\n\n"
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        + "\n\nReturn exactly one JSON object containing only reaction_id and synthon_ids. "
        "Copy the complete tuple unchanged; do not return an option number or explanatory text."
    )


def _policy_instruction(policy: str, role: str) -> str:
    if policy == "baseline_v1":
        return "Choose a chemically sensible valid combination from the shown options."
    if policy == "direct_v1":
        return (
            "Choose exactly one complete object from candidate_options. Copy its reaction_id "
            "and entire synthon_ids array without mixing components from different objects."
        )
    return ROLE_INSTRUCTIONS[role]


def _slot_payload(plan: ProposalSlotPlan) -> list[dict[str, object]]:
    return [
        {
            "position": options[0].position,
            "options": [option.to_dict() for option in options],
        }
        for options in plan.slot_options
    ]


def _candidate_options(plan: ProposalSlotPlan, policy: str) -> dict[str, object]:
    if policy == "direct_v1" and plan.complete_tuple_options:
        return {
            "candidate_options": [
                _direct_candidate_option(plan, item) for item in plan.complete_tuple_options
            ]
        }
    return {"slot_options": _slot_payload(plan)}


def _direct_candidate_option(
    plan: ProposalSlotPlan,
    synthon_ids: tuple[int, ...],
) -> dict[str, object]:
    components = []
    for slot, synthon_id in zip(plan.slot_options, synthon_ids, strict=True):
        option = next(item for item in slot if item.synthon_id == synthon_id)
        components.append(option.to_dict())
    return {
        "reaction_id": plan.reaction_id,
        "synthon_ids": list(synthon_ids),
        "components": components,
    }


def _history_summary(
    observations: Sequence[Observation],
    space: object,
) -> list[dict[str, object]]:
    successful = [item for item in observations if item.evaluation.succeeded]
    ranked = sorted(successful, key=_objective_value, reverse=True)
    selected = _unique_observations(
        [*ranked[:HISTORY_TOP_COUNT], *successful[-HISTORY_RECENT_COUNT:]]
    )
    return [_history_item(item, space) for item in selected]


def _objective_value(observation: Observation) -> float:
    value = observation.evaluation.metrics.get(OBJECTIVE_NAME)
    return float("-inf") if value is None else float(value)


def _unique_observations(observations: Sequence[Observation]) -> list[Observation]:
    seen: set[str] = set()
    result: list[Observation] = []
    for observation in observations:
        if observation.candidate_id in seen:
            continue
        seen.add(observation.candidate_id)
        result.append(observation)
    return result


def _history_item(observation: Observation, space: object) -> dict[str, object]:
    payload = observation.candidate.payload
    if not isinstance(payload, dict):
        raise TypeError("SynthonBench observations must retain mapping candidate payloads")
    reaction_id = str(payload["reaction_id"])
    synthon_ids = [int(item) for item in payload["synthon_ids"]]
    return {
        "reaction_id": reaction_id,
        "synthon_ids": synthon_ids,
        "components": _history_components(space, reaction_id, synthon_ids),
        OBJECTIVE_NAME: observation.evaluation.metrics[OBJECTIVE_NAME],
    }


def _history_components(
    space: object,
    reaction_id: str,
    synthon_ids: Sequence[int],
) -> list[dict[str, object]]:
    positions = ordered_positions(space, reaction_id)
    if len(positions) != len(synthon_ids):
        raise ValueError("observed synthon tuple does not match the official reaction arity")
    return [
        {
            "position": position,
            "synthon_id": synthon_id,
            "smiles": space.synthon_smiles(reaction_id, position, synthon_id),
        }
        for position, synthon_id in zip(positions, synthon_ids, strict=True)
    ]


__all__ = [
    "DEFAULT_PROMPT_POLICY",
    "PROMPT_POLICIES",
    "build_synthon_prompt_messages",
    "prompt_sha256",
    "validate_prompt_policy",
]
