"""Versioned, leakage-free prompts for SynthonBench tuple proposals."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence

from ldm_tts.contracts import Observation
from ldm_tts.engine.expansion import ExpansionRequest
from tasks.synthonbench.core.catalog import ProposalSlotPlan
from tasks.synthonbench.core.constants import OBJECTIVE_NAME

DEFAULT_PROMPT_POLICY = "policy_v1"
PROMPT_POLICIES = ("baseline_v1", "direct_v1", "policy_v1")
HISTORY_LIMIT = 8

ROLE_INSTRUCTIONS = {
    "explore": "Favor a chemically distinct combination from the observed tuples.",
    "exploit": "Use observed outcomes to select a plausible local analogue from this slate.",
    "diversify": "Avoid repeating the most common observed fragments when this slate permits it.",
    "scaffold_shift": "Test a different structural motif while keeping every chosen ID exact.",
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
    prompt_policy: str = DEFAULT_PROMPT_POLICY,
) -> list[dict[str, str]]:
    """Build one self-contained prompt for exactly one tuple response."""

    policy = validate_prompt_policy(prompt_policy)
    history = _history_summary(request.observations)
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
        "objective": "maximize the measured SynthonBench utility",
        "observed_history": history,
        "reaction_id": plan.reaction_id,
        "proposal_role": plan.role,
        **_candidate_options(plan, policy),
        "output": {"reaction_id": plan.reaction_id, "synthon_ids": ["one ID per slot"]},
    }
    instruction = _policy_instruction(policy, plan.role)
    return (
        "Select one ordered synthon tuple for the fixed reaction below. "
        "The history contains only previous charged measurements; it contains no hidden scores. "
        f"{instruction}\n\n"
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        + "\n\nReturn exactly {\"reaction_id\":\"...\",\"synthon_ids\":[...]} with integer IDs."
    )


def _policy_instruction(policy: str, role: str) -> str:
    if policy == "baseline_v1":
        return "Choose a chemically sensible valid combination from the shown options."
    if policy == "direct_v1":
        return "Choose exactly one complete tuple from candidate_options; copy its IDs without mixing options."
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
    if policy == "direct_v1" and plan.direct_tuple_options:
        return {"candidate_options": [{"synthon_ids": list(item)} for item in plan.direct_tuple_options]}
    return {"slot_options": _slot_payload(plan)}


def _history_summary(observations: Sequence[Observation]) -> list[dict[str, object]]:
    successful = [item for item in observations if item.evaluation.succeeded]
    ranked = sorted(successful, key=_objective_value, reverse=True)
    selected = _unique_observations([*ranked[: HISTORY_LIMIT // 2], *successful[-HISTORY_LIMIT // 2 :]])
    return [_history_item(item) for item in selected]


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


def _history_item(observation: Observation) -> dict[str, object]:
    payload = observation.candidate.payload
    if not isinstance(payload, dict):
        raise TypeError("SynthonBench observations must retain mapping candidate payloads")
    return {
        "reaction_id": payload["reaction_id"],
        "synthon_ids": payload["synthon_ids"],
        OBJECTIVE_NAME: observation.evaluation.metrics[OBJECTIVE_NAME],
    }


__all__ = [
    "DEFAULT_PROMPT_POLICY",
    "PROMPT_POLICIES",
    "build_synthon_prompt_messages",
    "prompt_sha256",
    "validate_prompt_policy",
]
