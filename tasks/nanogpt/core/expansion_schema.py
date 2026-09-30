"""Structured parameter-space primitives for LDM task adapters."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ldm_tts.contracts.evaluation import as_float


@dataclass(frozen=True)
class OperationParameter:
    """One editable parameter in a structured operation space."""

    name: str
    kind: str
    min_value: float | None = None
    max_value: float | None = None
    choices: tuple[Any, ...] = ()
    scale: str = "linear"


@dataclass(frozen=True)
class OperationSchema:
    """Ordered structured operation space."""

    version: str
    description: str
    parameters: dict[str, OperationParameter]
    path: Path | None = None


@dataclass
class ValidatedOperation:
    """A schema-valid operation emitted by an LLM response."""

    name: str
    op: str
    value: Any
    rationale: str = ""


def canonical_name(name: str) -> str:
    """Normalize parameter names used in schemas and parsed code."""

    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").upper()


def load_operation_schema(path: Path, project_root: Path) -> OperationSchema:
    """Load an operation schema JSON file."""

    schema_path = path if path.is_absolute() else project_root / path
    data = json.loads(schema_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Operation schema {schema_path} must be a JSON object.")
    version = str(data.get("version") or "").strip()
    if not version:
        raise ValueError(f"Operation schema {schema_path} is missing a non-empty version.")
    raw_parameters = data.get("parameters")
    if not isinstance(raw_parameters, dict) or not raw_parameters:
        raise ValueError(f"Operation schema {schema_path} must define parameters.")
    parameters: dict[str, OperationParameter] = {}
    raw_order = data.get("parameter_order")
    if isinstance(raw_order, list) and raw_order:
        ordered_names = [str(name) for name in raw_order]
    else:
        ordered_names = [str(name) for name in raw_parameters]
    raw_parameter_by_canonical = {
        canonical_name(str(name)): (name, spec)
        for name, spec in raw_parameters.items()
    }
    for raw_name in ordered_names:
        canonical_raw_name = canonical_name(str(raw_name))
        if canonical_raw_name not in raw_parameter_by_canonical:
            raise ValueError(f"Parameter order references unknown parameter {raw_name!r}.")
        original_name, raw_spec = raw_parameter_by_canonical[canonical_raw_name]
        if not isinstance(raw_spec, dict):
            raise ValueError(f"Parameter {original_name!r} spec must be an object.")
        parameters[canonical_name(str(original_name))] = operation_parameter_from_schema(
            original_name,
            raw_spec,
        )
    return OperationSchema(
        version=version,
        description=str(data.get("description") or ""),
        parameters=parameters,
        path=schema_path,
    )


def operation_parameter_from_schema(raw_name: Any, raw_spec: dict[str, Any]) -> OperationParameter:
    name = canonical_name(str(raw_name))
    kind = str(raw_spec.get("type") or "").strip().lower()
    if kind not in {"int", "float", "choice"}:
        raise ValueError(f"Parameter {name} has unsupported type {kind!r}.")
    if kind == "choice":
        raw_choices = raw_spec.get("choices")
        if not isinstance(raw_choices, list) or not raw_choices:
            raise ValueError(f"Choice parameter {name} must define a non-empty choices list.")
        parameter = OperationParameter(name=name, kind=kind, choices=tuple(raw_choices))
    else:
        parameter = OperationParameter(
            name=name,
            kind=kind,
            min_value=as_float(raw_spec.get("min")),
            max_value=as_float(raw_spec.get("max")),
            scale=str(raw_spec.get("scale") or "linear"),
        )
    return normalize_operation_parameter(parameter)


def operation_schema_to_json(schema: OperationSchema) -> dict[str, Any]:
    """Serialize an operation schema to the repository JSON shape."""

    parameters: dict[str, Any] = {}
    for parameter in schema.parameters.values():
        if parameter.kind == "choice":
            parameters[parameter.name] = {
                "type": "choice",
                "choices": list(parameter.choices),
            }
        else:
            parameters[parameter.name] = {
                "type": parameter.kind,
                "min": parameter.min_value,
                "max": parameter.max_value,
                "scale": parameter.scale,
            }
    return {
        "version": schema.version,
        "description": schema.description,
        "source_path": None if schema.path is None else str(schema.path),
        "parameter_order": list(schema.parameters),
        "parameters": parameters,
    }


def operation_representation_version(schema: OperationSchema) -> str:
    return f"operation_schema:{schema.version}"


def operation_schema_signature(schema: OperationSchema) -> str:
    return hashlib.sha1(
        json.dumps(operation_schema_to_json(schema), sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]


def replace_operation_schema(
    base: OperationSchema,
    parameters: dict[str, OperationParameter],
    *,
    version_suffix: str,
    description_prefix: str,
) -> OperationSchema:
    ordered = {
        canonical_name(name): normalize_operation_parameter(parameter)
        for name, parameter in parameters.items()
    }
    payload = {
        "base_version": base.version,
        "parameter_order": list(ordered),
        "parameters": {
            name: operation_parameter_to_json(parameter)
            for name, parameter in ordered.items()
        },
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    description = f"{description_prefix} {base.description}".strip()
    return OperationSchema(
        version=f"{base.version}:{version_suffix}:{digest}",
        description=description,
        parameters=ordered,
        path=base.path,
    )


def initial_active_operation_schema(full_schema: OperationSchema, args: Any) -> OperationSchema:
    spec = str(getattr(args, "initial_operation_features", "5") or "5")
    names = initial_operation_parameter_names(full_schema, spec)
    parameters = {name: full_schema.parameters[name] for name in names}
    return replace_operation_schema(
        full_schema,
        parameters,
        version_suffix="active",
        description_prefix="Active reservoir expansion schema.",
    )


def initial_operation_parameter_names(full_schema: OperationSchema, spec: str) -> list[str]:
    names = list(full_schema.parameters)
    text = str(spec or "").strip()
    if not text or text.lower() == "all":
        return names
    if re.fullmatch(r"\d+", text):
        count = max(1, min(len(names), int(text)))
        return names[:count]
    selected: list[str] = []
    for raw_name in text.split(","):
        name = canonical_name(raw_name)
        if not name:
            continue
        if name not in full_schema.parameters:
            raise ValueError(f"Unknown expansion-schema parameter {raw_name!r}.")
        if name not in selected:
            selected.append(name)
    if not selected:
        raise ValueError("--initial-expansion-parameters did not select any schema parameters.")
    return selected


def normalize_operation_parameter(parameter: OperationParameter) -> OperationParameter:
    name = canonical_name(parameter.name)
    kind = str(parameter.kind).strip().lower()
    if kind not in {"int", "float", "choice"}:
        raise ValueError(f"Parameter {name} has unsupported type {kind!r}.")
    choices = tuple(parameter.choices)
    min_value = parameter.min_value
    max_value = parameter.max_value
    if kind == "choice":
        if not choices:
            raise ValueError(f"Choice parameter {name} must define choices.")
        min_value = None
        max_value = None
    else:
        min_value = as_float(min_value)
        max_value = as_float(max_value)
        if min_value is None or max_value is None or min_value > max_value:
            raise ValueError(f"Numeric parameter {name} must define valid min/max values.")
    scale = str(parameter.scale or "linear").strip().lower()
    if scale not in {"linear", "log"}:
        raise ValueError(f"Parameter {name} has unsupported scale {scale!r}.")
    if scale == "log" and kind != "choice" and (
        min_value is None or min_value <= 0 or max_value is None or max_value <= 0
    ):
        raise ValueError(f"Log-scaled parameter {name} must have positive min/max.")
    return OperationParameter(
        name=name,
        kind=kind,
        min_value=min_value,
        max_value=max_value,
        choices=choices,
        scale=scale,
    )


def operation_parameter_to_json(parameter: OperationParameter) -> dict[str, Any]:
    parameter = normalize_operation_parameter(parameter)
    if parameter.kind == "choice":
        return {
            "name": parameter.name,
            "type": "choice",
            "choices": list(parameter.choices),
        }
    return {
        "name": parameter.name,
        "type": parameter.kind,
        "min": parameter.min_value,
        "max": parameter.max_value,
        "scale": parameter.scale,
    }


def operation_parameter_from_payload(payload: Any) -> OperationParameter:
    if not isinstance(payload, dict):
        raise ValueError("expansion parameter payload must be an object.")
    name = canonical_name(str(payload.get("name") or payload.get("parameter") or ""))
    if not name:
        raise ValueError("expansion parameter payload must include a non-empty name.")
    kind = str(payload.get("type") or payload.get("kind") or "").strip().lower()
    if kind == "numeric":
        kind = "float"
    if kind == "choice":
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError(f"Choice parameter {name} must include non-empty choices.")
        parameter = OperationParameter(name=name, kind="choice", choices=tuple(choices))
    else:
        if kind not in {"int", "float"}:
            raise ValueError(f"Expansion parameter {name} has unsupported type {kind!r}.")
        parameter = OperationParameter(
            name=name,
            kind=kind,
            min_value=as_float(payload.get("min")),
            max_value=as_float(payload.get("max")),
            scale=str(payload.get("scale") or "linear"),
        )
    return normalize_operation_parameter(parameter)


def operation_representation_dimension(schema: OperationSchema) -> int:
    total = 0
    for parameter in schema.parameters.values():
        total += len(parameter.choices) if parameter.kind == "choice" else 1
        total += 1
    return total


def normalize_operation_numeric(value: float, parameter: OperationParameter) -> float:
    lo = float(parameter.min_value)
    hi = float(parameter.max_value)
    value = min(max(float(value), lo), hi)
    if hi == lo:
        return 0.0
    if parameter.scale == "log":
        return (math.log(value) - math.log(lo)) / (math.log(hi) - math.log(lo))
    return (value - lo) / (hi - lo)


def choice_values_equal(left: Any, right: Any) -> bool:
    if isinstance(right, str):
        return str(left) == right
    if isinstance(right, bool):
        return isinstance(left, bool) and left is right
    if isinstance(right, int) and not isinstance(right, bool):
        number = as_float(left)
        return number is not None and abs(number - int(right)) <= 1e-9
    if isinstance(right, float):
        number = as_float(left)
        return number is not None and abs(number - float(right)) <= 1e-9
    return left == right


def validate_operation_value(value: Any, parameter: OperationParameter, *, index: int) -> Any:
    if parameter.kind == "choice":
        for choice in parameter.choices:
            if choice_values_equal(value, choice):
                return choice
        raise ValueError(f"operation {index} value {value!r} is not in choices for {parameter.name}.")
    if isinstance(value, bool):
        raise ValueError(f"operation {index} value for {parameter.name} must not be boolean.")
    number = as_float(value)
    if number is None:
        raise ValueError(f"operation {index} value for {parameter.name} must be numeric.")
    if number < float(parameter.min_value) or number > float(parameter.max_value):
        raise ValueError(
            f"operation {index} value for {parameter.name}={number} outside "
            f"[{parameter.min_value}, {parameter.max_value}]."
        )
    if parameter.kind == "int":
        if abs(number - round(number)) > 1e-9:
            raise ValueError(f"operation {index} value for {parameter.name} must be an integer.")
        return int(round(number))
    return float(number)


def validate_operation_payload(
    payload: Any,
    schema: OperationSchema,
    *,
    max_operations: int,
) -> list[ValidatedOperation]:
    if not isinstance(payload, dict):
        raise ValueError("payload must be an object containing operations.")
    raw_operations = payload.get("operations")
    if not isinstance(raw_operations, list) or not raw_operations:
        raise ValueError("payload.operations must be a non-empty list.")
    if len(raw_operations) > max(1, int(max_operations)):
        raise ValueError(f"too many operations: {len(raw_operations)} > {max_operations}.")
    seen: set[str] = set()
    validated: list[ValidatedOperation] = []
    for index, raw in enumerate(raw_operations, start=1):
        if not isinstance(raw, dict):
            raise ValueError(f"operation {index} must be an object.")
        name = canonical_name(str(raw.get("name") or ""))
        if name not in schema.parameters:
            raise ValueError(f"operation {index} uses unknown parameter {name!r}.")
        if name in seen:
            raise ValueError(f"operation {index} repeats parameter {name}.")
        seen.add(name)
        parameter = schema.parameters[name]
        op = str(raw.get("op") or "").strip()
        expected_op = "set_choice" if parameter.kind == "choice" else "set_numeric"
        if op != expected_op:
            raise ValueError(f"operation {index} for {name} must use op={expected_op!r}, got {op!r}.")
        value = validate_operation_value(raw.get("value"), parameter, index=index)
        rationale = str(raw.get("rationale") or "").strip()
        validated.append(ValidatedOperation(name=name, op=op, value=value, rationale=rationale))
    return validated
