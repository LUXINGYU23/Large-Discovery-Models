"""Strict configuration loading for the reusable quick-comparison matrix."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ldm_tts.cli.runner import load_config


METHODS = ("ldm", "bo", "llm")
STEP_KINDS = ("round", "evaluation_index")


@dataclass(frozen=True)
class ComparisonCase:
    """One fixed benchmark case expressed as standard runner overrides."""

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
class QuickCompareSpec:
    """Validated matrix specification with no task-specific control flow."""

    path: Path
    task: str
    name: str
    base_config: Path
    cases: tuple[ComparisonCase, ...]
    seeds: tuple[int, ...]
    optimization_rounds: int
    initialization_mode: str
    output_root: Path
    trajectory: TrajectorySpec
    result_fields: dict[str, str]

    @property
    def iterations(self) -> int:
        return self.optimization_rounds + 1

    @property
    def digest(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "name": self.name,
            "base_config": str(self.base_config),
            "cases": [{"id": item.case_id, "overrides": list(item.overrides)} for item in self.cases],
            "methods": list(METHODS),
            "seeds": list(self.seeds),
            "optimization_rounds": self.optimization_rounds,
            "initialization_mode": self.initialization_mode,
            "output_root": str(self.output_root),
            "trajectory": self.trajectory.__dict__,
            "result_fields": dict(self.result_fields),
        }


def load_quick_compare_spec(path: Path) -> QuickCompareSpec:
    """Load one versioned matrix specification and reject ambiguous fields."""

    resolved = Path(path).resolve()
    raw = load_config(resolved)
    _require_exact_keys(raw)
    return QuickCompareSpec(
        path=resolved,
        task=_required_string(raw, "task"),
        name=_required_string(raw, "name"),
        base_config=_resolve_base_config(resolved, raw),
        cases=_cases(raw.get("cases")),
        seeds=_seeds(raw.get("seeds")),
        optimization_rounds=_positive_int(raw.get("optimization_rounds"), "optimization_rounds"),
        initialization_mode=_required_string(raw, "initialization_mode"),
        output_root=_output_root(raw.get("output_root")),
        trajectory=_trajectory(raw.get("trajectory")),
        result_fields=_result_fields(raw.get("result_fields")),
    )


def _require_exact_keys(raw: dict[str, Any]) -> None:
    expected = {
        "schema_version", "name", "task", "base_config", "cases", "methods", "seeds",
        "optimization_rounds", "initialization_mode", "output_root", "trajectory", "result_fields",
    }
    if set(raw) != expected or raw.get("schema_version") != 1:
        raise ValueError("quick comparison config must use schema_version=1 and the documented fields")
    if tuple(raw.get("methods", ())) != METHODS:
        raise ValueError("quick comparison methods must be exactly [ldm, bo, llm]")


def _resolve_base_config(path: Path, raw: dict[str, Any]) -> Path:
    candidate = Path(_required_string(raw, "base_config"))
    resolved = (path.parent / candidate).resolve() if not candidate.is_absolute() else candidate
    if not resolved.is_file():
        raise ValueError(f"quick comparison base config does not exist: {resolved}")
    return resolved


def _cases(value: Any) -> tuple[ComparisonCase, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("quick comparison cases must be a non-empty list")
    cases = []
    for item in value:
        if not isinstance(item, dict) or set(item) != {"id", "overrides"}:
            raise ValueError("each quick comparison case needs only id and overrides")
        overrides = item["overrides"]
        if not isinstance(overrides, list) or not all(isinstance(entry, str) and "=" in entry for entry in overrides):
            raise ValueError("case overrides must be PATH=VALUE strings")
        cases.append(ComparisonCase(_required_string(item, "id"), tuple(overrides)))
    if len({item.case_id for item in cases}) != len(cases):
        raise ValueError("quick comparison case IDs must be unique")
    return tuple(cases)


def _seeds(value: Any) -> tuple[int, ...]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError("quick comparison requires exactly three seeds")
    if any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in value):
        raise ValueError("quick comparison seeds must be non-negative integers")
    if len(set(value)) != len(value):
        raise ValueError("quick comparison seeds must be unique")
    return tuple(value)


def _trajectory(value: Any) -> TrajectorySpec:
    if not isinstance(value, dict) or set(value) != {"step_column", "step_kind", "objective_column", "direction"}:
        raise ValueError("trajectory requires step_column, step_kind, objective_column, and direction")
    result = TrajectorySpec(**{key: _required_string(value, key) for key in value})
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
    raw = os.path.expandvars(_required_string({"output_root": value}, "output_root"))
    if "$" in raw:
        raise ValueError("quick comparison output_root contains an unresolved environment variable")
    return Path(raw).resolve()


def _required_string(raw: dict[str, Any], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"quick comparison {key} must be a non-empty string")
    return value.strip()


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"quick comparison {name} must be a positive integer")
    return value


__all__ = ["ComparisonCase", "METHODS", "QuickCompareSpec", "TrajectorySpec", "load_quick_compare_spec"]
