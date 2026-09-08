"""Strict loader for tracked mock reaction schema resources."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from tasks.iron_mind.core.schema import (
    OBSERVATION_POLICIES,
    SCHEMA_VERSION,
    ReactionDatasetSchema,
    ReactionFactor,
    canonical_schema_payload,
    schema_sha256,
)


def load_tracked_reaction_schemas(path: Path) -> dict[str, ReactionDatasetSchema]:
    document = _read_json_object(path, "reaction schema resource")
    _require_exact_keys(document, {"schema_version", "datasets"}, "schema resource")
    if document["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"Unsupported reaction schema version: {document['schema_version']!r}.")
    raw_datasets = document["datasets"]
    if not isinstance(raw_datasets, dict) or not raw_datasets:
        raise ValueError("Reaction schema resource must contain a non-empty dataset object.")
    return {
        dataset_id: _parse_dataset_schema(dataset_id, raw_schema)
        for dataset_id, raw_schema in raw_datasets.items()
    }


def _parse_dataset_schema(dataset_id: str, raw_schema: Any) -> ReactionDatasetSchema:
    schema = _require_object(raw_schema, f"schema {dataset_id!r}")
    expected = {
        "schema_version", "dataset_id", "factors", "measurements", "objective",
        "observation_policy", "schema_sha256",
    }
    _require_exact_keys(schema, expected, f"schema {dataset_id!r}")
    if schema["schema_version"] != SCHEMA_VERSION or schema["dataset_id"] != dataset_id:
        raise ValueError(f"Schema identity does not match dataset {dataset_id!r}.")
    factors = _parse_factors(schema["factors"], dataset_id)
    measurements = _parse_string_list(schema["measurements"], "measurements")
    objective = _require_object(schema["objective"], "objective")
    _require_exact_keys(objective, {"name", "direction"}, "objective")
    policy = schema["observation_policy"]
    if policy not in OBSERVATION_POLICIES:
        raise ValueError(f"Unknown observation policy for {dataset_id!r}: {policy!r}.")
    payload = canonical_schema_payload(
        dataset_id=dataset_id,
        factors=factors,
        measurements=measurements,
        objective=_require_string(objective["name"], "objective name"),
        direction=_require_direction(objective["direction"]),
        observation_policy=policy,
    )
    digest = schema_sha256(payload)
    if digest != _require_digest(schema["schema_sha256"], "schema SHA-256"):
        raise ValueError(f"Schema SHA-256 mismatch for dataset {dataset_id!r}.")
    return ReactionDatasetSchema(
        dataset_id, factors, measurements, payload["objective"]["name"],
        payload["objective"]["direction"], policy, digest,
    )


def _parse_factors(raw_factors: Any, dataset_id: str) -> tuple[ReactionFactor, ...]:
    if not isinstance(raw_factors, list) or not raw_factors:
        raise ValueError(f"Schema {dataset_id!r} must declare at least one factor.")
    factors = [_parse_factor(value) for value in raw_factors]
    names = tuple(factor.name for factor in factors)
    if len(names) != len(set(names)):
        raise ValueError(f"Schema {dataset_id!r} repeats a factor name.")
    return tuple(factors)


def _parse_factor(raw_factor: Any) -> ReactionFactor:
    factor = _require_object(raw_factor, "factor")
    if set(factor) == {"name", "categories"}:
        return ReactionFactor(
            _require_string(factor["name"], "factor name"),
            _parse_string_list(factor["categories"], "factor categories"),
        )
    _require_exact_keys(factor, {"name", "type", "options"}, "factor")
    options = factor["options"]
    if not isinstance(options, list):
        raise ValueError("factor options must be a list.")
    return ReactionFactor(
        _require_string(factor["name"], "factor name"),
        tuple(options),
        _require_string(factor["type"], "factor type"),
    )


def _parse_string_list(raw_values: Any, label: str) -> tuple[str, ...]:
    if not isinstance(raw_values, list) or not raw_values:
        raise ValueError(f"{label} must be a non-empty list of strings.")
    values = tuple(_require_string(value, label) for value in raw_values)
    if len(values) != len(set(values)):
        raise ValueError(f"{label} must not contain duplicates.")
    return values


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read {label}: {path}.") from exc
    return _require_object(payload, label)


def _require_object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object.")
    return value


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{label} fields do not match the tracked contract.")


def _require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string.")
    return value


def _require_direction(value: Any) -> str:
    direction = _require_string(value, "objective direction")
    if direction not in {"maximize", "minimize"}:
        raise ValueError(f"Unsupported objective direction: {direction!r}.")
    return direction


def _require_digest(value: Any, label: str) -> str:
    digest = _require_string(value, label)
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError(f"{label} must be a lowercase SHA-256 hex digest.")
    return digest


__all__ = ["load_tracked_reaction_schemas"]
