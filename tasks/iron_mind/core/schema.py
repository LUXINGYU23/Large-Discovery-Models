"""Versioned finite schemas for Iron Mind reaction tables."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence, TypeAlias


SCHEMA_VERSION = 1
CANONICAL_JSON_SEPARATORS = (",", ":")
OBSERVATION_POLICIES = frozenset({"single_row", "replicated_rows"})
PARAMETER_TYPES = frozenset({"categorical", "discrete"})
ReactionValue: TypeAlias = str | int | float


@dataclass(frozen=True)
class ReactionFactor:
    """One ordered finite reaction factor from an Olympus config."""

    name: str
    options: tuple[ReactionValue, ...]
    parameter_type: str = "categorical"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Reaction factor name must not be empty.")
        if self.parameter_type not in PARAMETER_TYPES:
            raise ValueError(f"Unsupported reaction parameter type: {self.parameter_type!r}.")
        normalized = tuple(self.normalize(value) for value in self.options)
        if not normalized or len(normalized) != len(set(normalized)):
            raise ValueError("Reaction factor options must be non-empty and unique.")
        object.__setattr__(self, "options", normalized)

    def normalize(self, value: Any) -> ReactionValue:
        """Validate and normalize one JSON value against this factor type."""

        if self.parameter_type == "categorical":
            if not isinstance(value, str) or not value:
                raise ValueError(f"Categorical factor {self.name!r} requires a string.")
            return value
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Discrete factor {self.name!r} requires a number.")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"Discrete factor {self.name!r} requires a finite number.")
        return numeric

    def admit(self, value: Any) -> ReactionValue:
        """Return the contract-owned option equal to an untrusted value."""

        normalized = self.normalize(value)
        for option in self.options:
            if normalized == option:
                return option
        raise ValueError(f"Unknown option {value!r} for factor {self.name!r}.")

    def parse_csv(self, raw_value: str) -> ReactionValue:
        """Parse one headerless CSV field using the declared parameter type."""

        value: Any = raw_value if self.parameter_type == "categorical" else float(raw_value)
        return self.admit(value)


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
        return tuple(len(factor.options) for factor in self.factors)

    @property
    def one_hot_dimension(self) -> int:
        return sum(self.category_counts)

    @property
    def allows_replicates(self) -> bool:
        return self.observation_policy == "replicated_rows"

    def categories_for(self, factor_name: str) -> tuple[ReactionValue, ...]:
        for factor in self.factors:
            if factor.name == factor_name:
                return factor.options
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
    """Load tracked schemas used by the deterministic mock boundary."""

    from tasks.iron_mind.core.schema_resource import load_tracked_reaction_schemas

    return load_tracked_reaction_schemas(path)


def load_reaction_schema_from_config(
    path: Path,
    *,
    dataset_id: str,
    observation_policy: str,
    expected_sha256: str,
) -> ReactionDatasetSchema:
    """Build one real schema from its exact source-pinned Olympus config."""

    config = _read_json_object(path, "reaction config")
    factors = parse_config_factors(config.get("parameters"))
    measurements = parse_config_measurements(config.get("measurements"))
    payload = canonical_schema_payload(
        dataset_id=dataset_id,
        factors=factors,
        measurements=measurements,
        objective="reaction_score",
        direction="maximize",
        observation_policy=observation_policy,
    )
    actual_sha256 = schema_sha256(payload)
    if actual_sha256 != expected_sha256:
        raise ValueError(f"Config schema digest mismatch for dataset {dataset_id!r}.")
    return ReactionDatasetSchema(
        dataset_id,
        factors,
        measurements,
        "reaction_score",
        "maximize",
        observation_policy,
        actual_sha256,
    )


def parse_config_factors(raw_parameters: Any) -> tuple[ReactionFactor, ...]:
    """Parse ordered categorical and discrete Olympus parameters."""

    if not isinstance(raw_parameters, list) or not raw_parameters:
        raise ValueError("Reaction config parameters must be a non-empty list.")
    factors = []
    for raw_parameter in raw_parameters:
        parameter = _require_object(raw_parameter, "reaction config parameter")
        parameter_type = parameter.get("type")
        if parameter_type not in PARAMETER_TYPES:
            raise ValueError(f"Unsupported reaction parameter type: {parameter_type!r}.")
        name = _require_string(parameter.get("name"), "reaction parameter name")
        options = parameter.get("options")
        if not isinstance(options, list):
            raise ValueError("Reaction config parameter options must be a list.")
        factors.append(ReactionFactor(name, tuple(options), parameter_type))
    _require_unique((factor.name for factor in factors), "Reaction parameter names")
    return tuple(factors)


def parse_config_measurements(raw_measurements: Any) -> tuple[str, ...]:
    """Parse ordered continuous Olympus measurement names."""

    if not isinstance(raw_measurements, list) or not raw_measurements:
        raise ValueError("Reaction config measurements must be a non-empty list.")
    names = []
    for raw_measurement in raw_measurements:
        measurement = _require_object(raw_measurement, "reaction config measurement")
        if measurement.get("type") != "continuous":
            raise ValueError("Reaction config measurement type must be continuous.")
        names.append(_require_string(measurement.get("name"), "measurement name"))
    _require_unique(names, "Reaction measurement names")
    return tuple(names)


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
        "factors": [_canonical_factor(factor) for factor in factors],
        "measurements": list(measurements),
        "objective": {"name": objective, "direction": direction},
        "observation_policy": observation_policy,
    }


def schema_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=CANONICAL_JSON_SEPARATORS
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_factor(factor: ReactionFactor) -> dict[str, Any]:
    if factor.parameter_type == "categorical":
        return {"name": factor.name, "categories": list(factor.options)}
    return {
        "name": factor.name,
        "type": factor.parameter_type,
        "options": list(factor.options),
    }


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


def _require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string.")
    return value


def _require_unique(values: Sequence[Any] | Any, label: str) -> None:
    items = tuple(values)
    if len(items) != len(set(items)):
        raise ValueError(f"{label} must not contain duplicates.")


__all__ = [
    "ReactionDatasetSchema",
    "ReactionFactor",
    "ReactionValue",
    "canonical_schema_payload",
    "load_reaction_schema_from_config",
    "load_reaction_schemas",
    "parse_config_factors",
    "parse_config_measurements",
    "schema_sha256",
]
