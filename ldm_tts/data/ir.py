"""Construction and validation of the ldm-2.0 intermediate representation."""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

SCHEMA_VERSION = "ldm-2.0"

TASK_ID_ALIASES = {
    "small_molecule": "smallmol",
    "small-molecule": "smallmol",
    "molecule": "smallmol",
    "antibody": "protein",
    "antibody_sequence": "protein",
}


class LDMDataCollectionError(ValueError):
    """Raised when an ldm-2.0 record is structurally invalid."""


def jdump(value: Any, indent: int | None = None) -> str:
    """Dump JSON with stable unicode handling used by the data pipeline."""

    return json.dumps(value, ensure_ascii=False, indent=indent)


def normalize_task_id(task_id: str) -> str:
    """Map execution task names onto the ldm-2.0 task ids."""

    text = str(task_id).strip()
    return TASK_ID_ALIASES.get(text, text)


def validate_ir_record(ir: Mapping[str, Any]) -> None:
    """Validate the minimum contract every ldm-2.0 training record must satisfy."""

    required = {"schema_version", "task", "search_state", "request", "action"}
    missing = sorted(required - set(ir))
    if missing:
        raise LDMDataCollectionError(f"IR record missing top-level field(s): {missing}")
    if ir.get("schema_version") != SCHEMA_VERSION:
        raise LDMDataCollectionError(
            f"IR schema_version must be {SCHEMA_VERSION!r}, got {ir.get('schema_version')!r}"
        )
    for field in ("task", "search_state", "request", "action"):
        if not isinstance(ir.get(field), Mapping):
            raise LDMDataCollectionError(f"IR field {field!r} must be an object")

    task = ir["task"]
    if not task.get("id") or not task.get("domain"):
        raise LDMDataCollectionError("IR task must include id and domain")
    objectives = task.get("objectives")
    if not isinstance(objectives, list) or not objectives:
        raise LDMDataCollectionError("IR task.objectives must be a non-empty list")
    for objective in objectives:
        if objective.get("direction") not in {"minimize", "maximize"}:
            raise LDMDataCollectionError(
                f"objective {objective.get('name')!r} has invalid direction "
                f"{objective.get('direction')!r}"
            )

    design_space = ir["search_state"].get("design_space")
    if not isinstance(design_space, Mapping):
        raise LDMDataCollectionError("IR search_state.design_space must be an object")
    if design_space.get("representation") not in {"parameter_edits", "complete_design"}:
        raise LDMDataCollectionError(
            "IR design_space.representation must be parameter_edits or complete_design"
        )

    allowed = ir["request"].get("allowed_actions")
    if not isinstance(allowed, list) or not allowed:
        raise LDMDataCollectionError("IR request.allowed_actions must be a non-empty list")
    action_type = ir["action"].get("type")
    if action_type not in {"propose", "expand_design_space", "add_new_parameter"}:
        raise LDMDataCollectionError(f"invalid action.type {action_type!r}")
    if action_type not in allowed:
        raise LDMDataCollectionError(
            f"action.type {action_type!r} is not in request.allowed_actions {allowed!r}"
        )
    if not isinstance(ir["action"].get("payload"), Mapping):
        raise LDMDataCollectionError("IR action.payload must be an object")


def make_complete_design_ir(
    *,
    task_id: str,
    domain: str,
    task_description: str,
    objectives: Sequence[Mapping[str, Any]],
    design_space_description: str,
    observations: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    request_description: str,
    num_candidates: int | None = None,
    round_idx: int | None = None,
    num_evaluated: int | None = None,
    best_so_far: Mapping[str, Any] | None = None,
    progress: Mapping[str, Any] | None = None,
    do_not_repeat: Sequence[Any] | None = None,
    allows_new_parameters: bool = True,
    reasoning_available: bool = True,
    reasoning: str | None = None,
    summary: str | None = None,
    raw_context: Mapping[str, Any] | None = None,
    active_parameters: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a standard ldm-2.0 row for complete-design proposal tasks."""

    normalized_task = normalize_task_id(task_id)
    ir = {
        "schema_version": SCHEMA_VERSION,
        "task": {
            "id": normalized_task,
            "domain": domain,
            "description": task_description,
            "objectives": [dict(objective) for objective in objectives],
            "reasoning_available": bool(reasoning_available),
        },
        "search_state": {
            "round": round_idx,
            "num_evaluated": num_evaluated,
            "design_space": {
                "representation": "complete_design",
                "active_parameters": [dict(param) for param in (active_parameters or [])],
                "inactive_parameters": [],
                "expansion_history": [],
                "allows_new_parameters": bool(allows_new_parameters),
                "description": design_space_description,
            },
            "observations": [dict(observation) for observation in observations],
            "best_so_far": dict(best_so_far) if best_so_far else None,
            "surrogate_feedback": None,
            "progress": dict(progress) if progress else None,
            "do_not_repeat": list(do_not_repeat or []),
        },
        "request": {
            "allowed_actions": ["propose"],
            "num_candidates": int(num_candidates if num_candidates is not None else len(candidates)),
            "max_edits_per_candidate": None,
            "description": request_description,
        },
        "action": {
            "type": "propose",
            "reasoning": reasoning,
            "payload": {"candidates": [dict(candidate) for candidate in candidates]},
            "summary": summary,
        },
    }
    if raw_context:
        ir["raw_context"] = dict(raw_context)
    validate_ir_record(ir)
    return ir


def make_parameter_edit_ir(
    *,
    task_id: str,
    domain: str,
    task_description: str,
    objectives: Sequence[Mapping[str, Any]],
    active_parameters: Sequence[Mapping[str, Any]],
    inactive_parameters: Sequence[Mapping[str, Any]],
    action: Mapping[str, Any],
    request_description: str,
    design_space_description: str = "",
    allowed_actions: Sequence[str] | None = None,
    num_candidates: int = 1,
    max_edits_per_candidate: int | None = None,
    round_idx: int | None = None,
    num_evaluated: int | None = None,
    observations: Sequence[Mapping[str, Any]] | None = None,
    best_so_far: Mapping[str, Any] | None = None,
    surrogate_feedback: Mapping[str, Any] | None = None,
    progress: Mapping[str, Any] | None = None,
    do_not_repeat: Sequence[Any] | None = None,
    expansion_history: Sequence[Mapping[str, Any]] | None = None,
    applied_this_transition: Sequence[Mapping[str, Any]] | None = None,
    allows_new_parameters: bool = True,
    reasoning_available: bool = True,
    raw_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a standard ldm-2.0 row for parameter-edit proposal tasks."""

    normalized_task = normalize_task_id(task_id)
    allowed = list(allowed_actions or ["propose"])
    if inactive_parameters and "expand_design_space" not in allowed:
        allowed.append("expand_design_space")
    design_space = {
        "representation": "parameter_edits",
        "active_parameters": [dict(param) for param in active_parameters],
        "inactive_parameters": [dict(param) for param in inactive_parameters],
        "expansion_history": [dict(item) for item in (expansion_history or [])],
        "allows_new_parameters": bool(allows_new_parameters),
        "description": design_space_description,
    }
    if applied_this_transition is not None:
        design_space["applied_this_transition"] = [
            dict(item) for item in applied_this_transition
        ]
    ir = {
        "schema_version": SCHEMA_VERSION,
        "task": {
            "id": normalized_task,
            "domain": domain,
            "description": task_description,
            "objectives": [dict(objective) for objective in objectives],
            "reasoning_available": bool(reasoning_available),
        },
        "search_state": {
            "round": round_idx,
            "num_evaluated": num_evaluated,
            "design_space": design_space,
            "observations": [dict(observation) for observation in (observations or [])],
            "best_so_far": dict(best_so_far) if best_so_far else None,
            "surrogate_feedback": dict(surrogate_feedback) if surrogate_feedback else None,
            "progress": dict(progress) if progress else None,
            "do_not_repeat": list(do_not_repeat or []),
        },
        "request": {
            "allowed_actions": allowed,
            "num_candidates": int(num_candidates),
            "max_edits_per_candidate": max_edits_per_candidate,
            "description": request_description,
        },
        "action": dict(action),
    }
    if raw_context:
        ir["raw_context"] = dict(raw_context)
    validate_ir_record(ir)
    return ir
