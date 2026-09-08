"""Digest-verified loading for pinned Iron Mind reaction tables."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from tasks.iron_mind.core.schema import (
    ReactionDatasetSchema,
    ReactionValue,
    canonical_schema_payload,
    parse_config_factors,
    parse_config_measurements,
    schema_sha256,
)


HASH_CHUNK_BYTES = 1024 * 1024


@dataclass(frozen=True)
class ReactionRow:
    """One source row with stable one-based identity and raw-row digest."""

    row_id: int
    conditions: Mapping[str, ReactionValue]
    measurements: Mapping[str, float]
    raw_row_sha256: str


@dataclass(frozen=True)
class FrozenReactionTable:
    """An immutable, digest-verified table indexed by exact condition tuples."""

    schema: ReactionDatasetSchema
    rows: tuple[ReactionRow, ...]
    rows_by_conditions: Mapping[tuple[ReactionValue, ...], tuple[ReactionRow, ...]]

    def rows_for_conditions(
        self, conditions: Mapping[str, ReactionValue]
    ) -> tuple[ReactionRow, ...]:
        if set(conditions) != set(self.schema.factor_names):
            raise ValueError("Conditions do not match the tracked reaction schema.")
        key = tuple(conditions[name] for name in self.schema.factor_names)
        for factor, value in zip(self.schema.factors, key, strict=True):
            factor.admit(value)
        return self.rows_by_conditions.get(key, ())


def load_frozen_reaction_table(
    *,
    schema: ReactionDatasetSchema,
    config_path: Path,
    data_path: Path,
    artifact_contract: Mapping[str, Any],
) -> FrozenReactionTable:
    """Load a source-pinned headerless table after all contract checks pass."""

    config_contract, data_contract, row_count = _artifact_contract_fields(artifact_contract)
    _verify_artifact(config_path, config_contract, "config")
    _verify_artifact(data_path, data_contract, "data")
    config = _read_json_object(config_path)
    _validate_config_schema(config, schema)
    rows = _read_rows(data_path, schema)
    if len(rows) != row_count:
        raise ValueError(f"Frozen table row count mismatch: expected {row_count}, got {len(rows)}.")
    return FrozenReactionTable(
        schema=schema,
        rows=rows,
        rows_by_conditions=_index_rows(rows, schema),
    )


def _artifact_contract_fields(
    artifact_contract: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any], int]:
    if not isinstance(artifact_contract, Mapping):
        raise ValueError("Artifact contract must be a mapping.")
    artifacts = artifact_contract.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("Artifact contract must declare an artifacts mapping.")
    config_contract = artifacts.get("config")
    data_contract = artifacts.get("data")
    row_count = artifact_contract.get("row_count")
    if not isinstance(config_contract, Mapping) or not isinstance(data_contract, Mapping):
        raise ValueError("Artifact contract must declare config and data artifacts.")
    if isinstance(row_count, bool) or not isinstance(row_count, int) or row_count < 1:
        raise ValueError("Artifact contract row_count must be a positive integer.")
    return config_contract, data_contract, row_count


def _verify_artifact(path: Path, contract: Mapping[str, Any], label: str) -> None:
    expected_bytes = contract.get("bytes")
    expected_digest = contract.get("sha256")
    if isinstance(expected_bytes, bool) or not isinstance(expected_bytes, int) or expected_bytes < 0:
        raise ValueError(f"{label} artifact contract has invalid bytes.")
    if not _is_sha256(expected_digest):
        raise ValueError(f"{label} artifact contract has invalid SHA-256.")
    if not path.is_file():
        raise ValueError(f"{label} artifact does not exist: {path}.")
    if path.stat().st_size != expected_bytes:
        raise ValueError(f"{label} artifact byte size mismatch.")
    if _sha256_file(path) != expected_digest:
        raise ValueError(f"{label} artifact SHA-256 mismatch.")


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read reaction config: {path}.") from exc
    if not isinstance(payload, dict):
        raise ValueError("Reaction config must be a JSON object.")
    return payload


def _validate_config_schema(config: Mapping[str, Any], schema: ReactionDatasetSchema) -> None:
    factors = parse_config_factors(config.get("parameters"))
    measurements = parse_config_measurements(config.get("measurements"))
    payload = canonical_schema_payload(
        dataset_id=schema.dataset_id,
        factors=factors,
        measurements=measurements,
        objective=schema.objective,
        direction=schema.direction,
        observation_policy=schema.observation_policy,
    )
    if schema_sha256(payload) != schema.schema_sha256:
        raise ValueError(f"Config schema digest mismatch for dataset {schema.dataset_id!r}.")


def _read_rows(data_path: Path, schema: ReactionDatasetSchema) -> tuple[ReactionRow, ...]:
    raw_lines = data_path.read_bytes().splitlines()
    if not raw_lines:
        raise ValueError("Frozen reaction table has no data rows.")
    rows = []
    seen_keys: set[tuple[str, ...]] = set()
    for row_id, raw_line in enumerate(raw_lines, start=1):
        row = _build_row(raw_line, row_id, schema)
        key = tuple(row.conditions[name] for name in schema.factor_names)
        if not schema.allows_replicates and key in seen_keys:
            raise ValueError(f"Duplicate observation key at row {row_id}.")
        seen_keys.add(key)
        rows.append(row)
    return tuple(rows)


def _build_row(raw_line: bytes, row_id: int, schema: ReactionDatasetSchema) -> ReactionRow:
    fields = _parse_csv_line(raw_line, row_id)
    expected_count = len(schema.factors) + len(schema.measurements)
    if len(fields) != expected_count:
        raise ValueError(
            f"Row {row_id} has invalid column count: expected {expected_count}, got {len(fields)}."
        )
    conditions = _row_conditions(fields[: len(schema.factors)], schema, row_id)
    measurements = _row_measurements(fields[len(schema.factors) :], schema, row_id)
    return ReactionRow(
        row_id=row_id,
        conditions=MappingProxyType(conditions),
        measurements=MappingProxyType(measurements),
        raw_row_sha256=hashlib.sha256(raw_line).hexdigest(),
    )


def _parse_csv_line(raw_line: bytes, row_id: int) -> list[str]:
    try:
        text = raw_line.decode("utf-8")
        parsed_rows = list(csv.reader([text]))
    except (UnicodeDecodeError, csv.Error) as exc:
        raise ValueError(f"Could not parse CSV row {row_id}.") from exc
    if len(parsed_rows) != 1:
        raise ValueError(f"Could not parse CSV row {row_id}.")
    return parsed_rows[0]


def _row_conditions(
    fields: list[str], schema: ReactionDatasetSchema, row_id: int
) -> dict[str, ReactionValue]:
    conditions = {}
    for factor, raw_value in zip(schema.factors, fields, strict=True):
        try:
            conditions[factor.name] = factor.parse_csv(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Unknown option {raw_value!r} for factor {factor.name!r} at row {row_id}."
            ) from exc
    return conditions


def _row_measurements(
    raw_values: list[str], schema: ReactionDatasetSchema, row_id: int
) -> dict[str, float]:
    measurements = {}
    for name, raw_value in zip(schema.measurements, raw_values, strict=True):
        try:
            value = float(raw_value)
        except ValueError as exc:
            raise ValueError(f"Invalid measurement {name!r} at row {row_id}.") from exc
        if not math.isfinite(value):
            raise ValueError(f"Non-finite measurement {name!r} at row {row_id}.")
        measurements[name] = value
    return measurements


def _index_rows(
    rows: tuple[ReactionRow, ...], schema: ReactionDatasetSchema
) -> Mapping[tuple[ReactionValue, ...], tuple[ReactionRow, ...]]:
    grouped: dict[tuple[ReactionValue, ...], list[ReactionRow]] = defaultdict(list)
    for row in rows:
        key = tuple(row.conditions[name] for name in schema.factor_names)
        grouped[key].append(row)
    return MappingProxyType({key: tuple(value) for key, value in grouped.items()})


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )
