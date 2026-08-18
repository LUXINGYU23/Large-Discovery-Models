"""Versioned categorical schemas for Iron Mind reaction tables."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = 1
CANONICAL_JSON_SEPARATORS = (",", ":")
OBSERVATION_POLICIES = frozenset({"single_row", "replicated_rows"})


@dataclass(frozen=True)
class ReactionFactor:
    """One ordered categorical reaction factor."""

    name: str
    categories: tuple[str, ...]


@dataclass(frozen=True)
class ReactionDatasetSchema:
    """Immutable task-facing view of one pinned reaction dataset."""

    dataset_id: str
    factors: tuple[ReactionFactor, ...]
    measurements: tuple[str, ...]
    objective: str
    direction: str
    observation_policy: str
    schema_sha256: str

    @property
    def factor_names(self) -> tuple[str, ...]:
        return tuple(factor.name for factor in self.factors)

    @property
    def category_counts(self) -> tuple[int, ...]:
        return tuple(len(factor.categories) for factor in self.factors)

    @property
    def one_hot_dimension(self) -> int:
        return sum(self.category_counts)

    @property
    def allows_replicates(self) -> bool:
        return self.observation_policy == "replicated_rows"

    def categories_for(self, factor_name: str) -> tuple[str, ...]:
        for factor in self.factors:
            if factor.name == factor_name:
                return factor.categories
        raise KeyError(f"Unknown reaction factor {factor_name!r}.")

    def canonical_payload(self) -> dict[str, Any]:
        return canonical_schema_payload(
            dataset_id=self.dataset_id,
            factors=self.factors,
            measurements=self.measurements,
            objective=self.objective,
            direction=self.direction,
            observation_policy=self.observation_policy,
        )


def load_reaction_schemas(path: Path) -> dict[str, ReactionDatasetSchema]:
    """Load and validate all tracked reaction schemas keyed by dataset ID."""

    document = _read_json_object(path, "reaction schema resource")
    _require_exact_keys(document, {"schema_version", "datasets"}, "schema resource")
    if document["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"Unsupported reaction schema version: {document['schema_version']!r}.")
    raw_datasets = document["datasets"]
    if not isinstance(raw_datasets, dict) or not raw_datasets:
        raise ValueError("Reaction schema resource must contain a non-empty dataset object.")

    schemas = {
        dataset_id: _parse_dataset_schema(dataset_id, raw_schema)
        for dataset_id, raw_schema in raw_datasets.items()
    }
    if len(schemas) != len(raw_datasets) or any(not dataset_id for dataset_id in schemas):
        raise ValueError("Reaction schema resource has invalid dataset identifiers.")
    return schemas


def canonical_schema_payload(
    *,
    dataset_id: str,
    factors: Sequence[ReactionFactor],
    measurements: Sequence[str],
    objective: str,
    direction: str,
    observation_policy: str,
) -> dict[str, Any]:
    """Build the canonical payload whose digest pins a schema definition."""

    return {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": dataset_id,
        "factors": [
            {"name": factor.name, "categories": list(factor.categories)} for factor in factors
        ],
        "measurements": list(measurements),
        "objective": {"name": objective, "direction": direction},
        "observation_policy": observation_policy,
    }


def schema_sha256(payload: Mapping[str, Any]) -> str:
    """Return the SHA-256 of canonical UTF-8 schema JSON."""

    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=CANONICAL_JSON_SEPARATORS,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _parse_dataset_schema(dataset_id: str, raw_schema: Any) -> ReactionDatasetSchema:
    if not isinstance(dataset_id, str) or not dataset_id:
        raise ValueError("Reaction schema dataset IDs must be non-empty strings.")
    schema = _require_object(raw_schema, f"schema {dataset_id!r}")
    expected = {
        "schema_version",
        "dataset_id",
        "factors",
        "measurements",
        "objective",
        "observation_policy",
        "schema_sha256",
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
    expected_digest = _require_digest(schema["schema_sha256"], "schema SHA-256")
    actual_digest = schema_sha256(payload)
    if actual_digest != expected_digest:
        raise ValueError(f"Schema SHA-256 mismatch for dataset {dataset_id!r}.")
    return ReactionDatasetSchema(
        dataset_id=dataset_id,
        factors=factors,
        measurements=measurements,
        objective=payload["objective"]["name"],
        direction=payload["objective"]["direction"],
        observation_policy=policy,
        schema_sha256=actual_digest,
    )


def _parse_factors(raw_factors: Any, dataset_id: str) -> tuple[ReactionFactor, ...]:
    if not isinstance(raw_factors, list) or not raw_factors:
        raise ValueError(f"Schema {dataset_id!r} must declare at least one factor.")
    factors = []
    for raw_factor in raw_factors:
        factor = _require_object(raw_factor, "factor")
        _require_exact_keys(factor, {"name", "categories"}, "factor")
        factors.append(
            ReactionFactor(
                name=_require_string(factor["name"], "factor name"),
                categories=_parse_string_list(factor["categories"], "factor categories"),
            )
        )
    names = tuple(factor.name for factor in factors)
    if len(names) != len(set(names)):
        raise ValueError(f"Schema {dataset_id!r} repeats a factor name.")
    return tuple(factors)


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
