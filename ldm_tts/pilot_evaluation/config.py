"""Strict configuration loading for the reusable pilot-evaluation matrix."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ldm_tts.cli.runner import load_config


SUPPORTED_METHODS = (
    "ldm",
    "ldm_harness",
    "ldm_harness_compiled",
    "bo",
    "llm",
    "harness",
    "blind_harness_compiled",
)
BASELINE_METHODS = frozenset(("ldm", "bo", "llm"))
STEP_KINDS = ("round", "evaluation_index")


@dataclass(frozen=True)
class EvaluationCase:
    """One benchmark case expressed as standard runner overrides."""

    case_id: str
    overrides: tuple[str, ...]


@dataclass(frozen=True)
class TrajectorySpec:
    """Task-neutral mapping from a child trajectory CSV to comparable calls."""

    step_column: str
    step_kind: str
    objective_column: str
    direction: str


@dataclass(frozen=True)
class PilotEvaluationSpec:
    """Validated matrix specification with no task-specific control flow."""

    task: str
    name: str
    base_config: Path
    cases: tuple[EvaluationCase, ...]
    methods: tuple[str, ...]
    method_overrides: dict[str, tuple[str, ...]]
    seeds: tuple[int, ...]
    optimization_rounds: int
    initialization_mode: str
    output_root: Path
    trajectory: TrajectorySpec
    result_fields: dict[str, str]
    policy_fields: dict[str, str] = field(default_factory=dict)
    policy_mean_fields: tuple[str, ...] = ()
    selection_protocol: str = "best_so_far"

    def __post_init__(self) -> None:
        if self.selection_protocol not in ("best_so_far", "final_submission"):
            raise ValueError("selection_protocol must be best_so_far or final_submission")
        if any(
            not isinstance(name, str) or not name
            or not isinstance(path, str) or not all(path.split("."))
            for name, path in self.policy_fields.items()
        ):
            raise ValueError("policy_fields must map non-empty names to dotted paths")
        if (
            any(name not in self.policy_fields for name in self.policy_mean_fields)
            or len(set(self.policy_mean_fields)) != len(self.policy_mean_fields)
        ):
            raise ValueError("policy_mean_fields must be unique names from policy_fields")

    @property
    def iterations(self) -> int:
        return self.optimization_rounds + 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "name": self.name,
            "base_config": str(self.base_config),
            "cases": [{"id": item.case_id, "overrides": list(item.overrides)} for item in self.cases],
            "methods": list(self.methods),
            "method_overrides": {name: list(values) for name, values in self.method_overrides.items()},
            "seeds": list(self.seeds),
            "optimization_rounds": self.optimization_rounds,
            "initialization_mode": self.initialization_mode,
            "output_root": str(self.output_root),
            "trajectory": self.trajectory.__dict__,
            "result_fields": dict(self.result_fields),
            "policy_fields": dict(self.policy_fields),
            "policy_mean_fields": list(self.policy_mean_fields),
            "selection_protocol": self.selection_protocol,
        }


def load_pilot_evaluation_spec(path: Path) -> PilotEvaluationSpec:
    """Load one versioned matrix specification and reject ambiguous fields."""

    resolved = Path(path).resolve()
    raw = load_config(resolved)
    _require_exact_keys(raw)
    protocol = raw.get("selection_protocol", "best_so_far")
    methods = _methods(raw.get("methods"), require_baselines=protocol != "final_submission")
    policy_fields = raw.get("policy_fields", {})
    policy_mean_fields = raw.get("policy_mean_fields", [])
    if not isinstance(policy_fields, dict) or not isinstance(policy_mean_fields, list):
        raise ValueError("policy_fields must be an object and policy_mean_fields an array")
    if any(not isinstance(name, str) for name in policy_mean_fields):
        raise ValueError("policy_mean_fields must contain field names")
    return PilotEvaluationSpec(
        task=_required_string(raw.get("task"), "task"),
        name=_required_string(raw.get("name"), "name"),
        base_config=_resolve_base_config(resolved, raw),
        cases=_cases(raw.get("cases")),
        methods=methods,
        method_overrides=_method_overrides(raw.get("method_overrides"), methods),
        seeds=_seeds(raw.get("seeds")),
        optimization_rounds=_positive_int(raw.get("optimization_rounds"), "optimization_rounds"),
        initialization_mode=_required_string(
            raw.get("initialization_mode"), "initialization_mode"
        ),
        output_root=_output_root(raw.get("output_root")),
        trajectory=_trajectory(raw.get("trajectory")),
        result_fields=_result_fields(raw.get("result_fields")),
        policy_fields=policy_fields,
        policy_mean_fields=tuple(policy_mean_fields),
        selection_protocol=protocol,
    )


def _require_exact_keys(raw: dict[str, Any]) -> None:
    expected = {
        "schema_version", "name", "task", "base_config", "cases", "methods", "method_overrides", "seeds",
        "optimization_rounds", "initialization_mode", "output_root", "trajectory", "result_fields",
    }
    optional = {"policy_fields", "policy_mean_fields", "selection_protocol"}
    if not expected <= set(raw) or set(raw) - expected - optional or raw.get("schema_version") != 1:
        raise ValueError("pilot evaluation config must use schema_version=1 and the documented fields")


def _methods(value: Any, *, require_baselines: bool = True) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("pilot evaluation methods must be a non-empty list")
    methods = tuple(value)
    if any(not isinstance(item, str) or item not in SUPPORTED_METHODS for item in methods):
        raise ValueError(f"pilot evaluation methods must come from {list(SUPPORTED_METHODS)}")
    if len(set(methods)) != len(methods):
        raise ValueError("pilot evaluation methods must be unique")
    if require_baselines and not BASELINE_METHODS <= set(methods):
        raise ValueError("pilot evaluation methods must include ldm, bo, and llm")
    return methods


def _resolve_base_config(path: Path, raw: dict[str, Any]) -> Path:
    candidate = Path(_required_string(raw.get("base_config"), "base_config"))
    resolved = (path.parent / candidate).resolve() if not candidate.is_absolute() else candidate
    if not resolved.is_file():
        raise ValueError(f"pilot evaluation base config does not exist: {resolved}")
    return resolved


def _cases(value: Any) -> tuple[EvaluationCase, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("pilot evaluation cases must be a non-empty list")
    cases = []
    for item in value:
        if not isinstance(item, dict) or set(item) != {"id", "overrides"}:
            raise ValueError("each pilot evaluation case needs only id and overrides")
        overrides = item["overrides"]
        if not isinstance(overrides, list) or not all(isinstance(entry, str) and "=" in entry for entry in overrides):
            raise ValueError("case overrides must be PATH=VALUE strings")
        cases.append(EvaluationCase(_required_string(item.get("id"), "id"), tuple(overrides)))
    if len({item.case_id for item in cases}) != len(cases):
        raise ValueError("pilot evaluation case IDs must be unique")
    return tuple(cases)


def _seeds(value: Any) -> tuple[int, ...]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError("pilot evaluation requires exactly three seeds")
    if any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in value):
        raise ValueError("pilot evaluation seeds must be non-negative integers")
    if len(set(value)) != len(value):
        raise ValueError("pilot evaluation seeds must be unique")
    return tuple(value)


def _method_overrides(value: Any, methods: tuple[str, ...]) -> dict[str, tuple[str, ...]]:
    if not isinstance(value, dict) or set(value) != set(methods):
        raise ValueError("method_overrides must define exactly the configured methods")
    result: dict[str, tuple[str, ...]] = {}
    for method in methods:
        overrides = value[method]
        if not isinstance(overrides, list) or not all(
            isinstance(item, str) and "=" in item for item in overrides
        ):
            raise ValueError("method overrides must be PATH=VALUE strings")
        result[method] = tuple(overrides)
    return result


def _trajectory(value: Any) -> TrajectorySpec:
    if not isinstance(value, dict) or set(value) != {"step_column", "step_kind", "objective_column", "direction"}:
        raise ValueError("trajectory requires step_column, step_kind, objective_column, and direction")
    result = TrajectorySpec(
        **{key: _required_string(value.get(key), key) for key in value}
    )
    if result.step_kind not in STEP_KINDS or result.direction not in {"maximize", "minimize"}:
        raise ValueError("trajectory step_kind or direction is invalid")
    return result


def _result_fields(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError("result_fields must be a name-to-dotted-path mapping")
    result = {str(name): str(path) for name, path in value.items()}
    if any(not name or not path for name, path in result.items()):
        raise ValueError("result_fields names and paths must be non-empty")
    return result


def _output_root(value: Any) -> Path:
    raw = os.path.expandvars(_required_string(value, "output_root"))
    if "$" in raw:
        raise ValueError("pilot evaluation output_root contains an unresolved environment variable")
    return Path(raw).resolve()


def _required_string(value: Any, key: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"pilot evaluation {key} must be a non-empty string")
    return value.strip()


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"pilot evaluation {name} must be a positive integer")
    return value


__all__ = ["PilotEvaluationSpec", "load_pilot_evaluation_spec"]
