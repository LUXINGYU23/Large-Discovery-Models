"""Strict loading of source-pinned NucleoBench case declarations."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from tasks.nucleobench.core.constants import (
    CASE_STATES,
    CATALOG_PATH,
    FAMILY_PROTOCOLS,
    OFFICIAL_CASE_COUNT,
    OFFICIAL_START_COUNT,
    UPSTREAM_COMMIT,
)


CASE_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
CATALOG_FIELDS = {
    "schema_version",
    "benchmark_commit",
    "case_count",
    "official_start_count",
    "cases",
}
CASE_FIELDS = {
    "case_id",
    "model_family",
    "model_name",
    "target",
    "model_selector",
    "sequence_length",
    "editable_position_count",
    "max_seconds",
    "state",
}


class CaseCatalogError(ValueError):
    """Raised when the versioned case catalog violates its protocol."""


@dataclass(frozen=True)
class NucleoBenchCase:
    case_id: str
    model_family: str
    model_name: str
    target: str
    model_selector: dict[str, Any]
    sequence_length: int
    editable_position_count: int
    max_seconds: int
    state: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_case_catalog(path: Path = CATALOG_PATH) -> tuple[NucleoBenchCase, ...]:
    """Load and validate all cases in the pinned catalog."""

    path = Path(path).resolve()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise CaseCatalogError(f"Case catalog does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise CaseCatalogError(
            f"Invalid JSON in {path}: line {exc.lineno}, column {exc.colno}"
        ) from exc
    data = _object(payload, "catalog", path)
    _reject_unknown(data, CATALOG_FIELDS, "catalog", path)
    if data.get("schema_version") != 1:
        raise CaseCatalogError(f"Unsupported catalog schema_version in {path}")
    if data.get("benchmark_commit") != UPSTREAM_COMMIT:
        raise CaseCatalogError(
            f"Catalog benchmark_commit does not match the task pin in {path}"
        )
    _expect_int(data.get("case_count"), "case_count", OFFICIAL_CASE_COUNT, path)
    _expect_int(
        data.get("official_start_count"),
        "official_start_count",
        OFFICIAL_START_COUNT,
        path,
    )
    raw_cases = data.get("cases")
    if not isinstance(raw_cases, list):
        raise CaseCatalogError(f"cases must be an array in {path}")
    if len(raw_cases) != OFFICIAL_CASE_COUNT:
        raise CaseCatalogError(
            f"cases must contain {OFFICIAL_CASE_COUNT} entries in {path}"
        )

    cases = tuple(_load_case(raw, path) for raw in raw_cases)
    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise CaseCatalogError(f"case_id values must be unique in {path}")
    return cases


def get_case(case_id: str, path: Path = CATALOG_PATH) -> NucleoBenchCase:
    """Return one declared case by stable ID."""

    cases = load_case_catalog(path)
    try:
        return next(case for case in cases if case.case_id == case_id)
    except StopIteration as exc:
        raise CaseCatalogError(
            f"Unknown NucleoBench case {case_id!r}; expected one of "
            f"{[case.case_id for case in cases]}"
        ) from exc


def _load_case(raw: Any, path: Path) -> NucleoBenchCase:
    data = _object(raw, "cases[]", path)
    _reject_unknown(data, CASE_FIELDS, "case", path)
    missing = sorted(CASE_FIELDS.difference(data))
    if missing:
        raise CaseCatalogError(f"Missing case field(s) in {path}: {', '.join(missing)}")

    case_id = _string(data["case_id"], "case_id", path)
    if not CASE_ID_PATTERN.fullmatch(case_id):
        raise CaseCatalogError(f"Invalid case_id {case_id!r} in {path}")
    family = _string(data["model_family"], f"{case_id}.model_family", path)
    if family not in FAMILY_PROTOCOLS:
        raise CaseCatalogError(
            f"Unknown model_family {family!r} for {case_id!r} in {path}"
        )
    protocol = FAMILY_PROTOCOLS[family]
    model_name = _string(data["model_name"], f"{case_id}.model_name", path)
    if model_name != protocol["model_name"]:
        raise CaseCatalogError(f"Incorrect model_name for {case_id!r} in {path}")
    target = _string(data["target"], f"{case_id}.target", path)
    selector = _object(data["model_selector"], f"{case_id}.model_selector", path)
    _validate_selector(family, target, selector, case_id, path)

    sequence_length = _positive_int(
        data["sequence_length"], f"{case_id}.sequence_length", path
    )
    editable_count = _positive_int(
        data["editable_position_count"],
        f"{case_id}.editable_position_count",
        path,
    )
    max_seconds = _positive_int(data["max_seconds"], f"{case_id}.max_seconds", path)
    for field, value in (
        ("sequence_length", sequence_length),
        ("editable_position_count", editable_count),
        ("max_seconds", max_seconds),
    ):
        if value != protocol[field]:
            raise CaseCatalogError(f"Incorrect {field} for {case_id!r} in {path}")
    if editable_count > sequence_length:
        raise CaseCatalogError(
            f"editable_position_count exceeds sequence_length for {case_id!r} in {path}"
        )

    state = _string(data["state"], f"{case_id}.state", path)
    if state not in CASE_STATES:
        raise CaseCatalogError(
            f"Unknown state {state!r} for {case_id!r}; expected {sorted(CASE_STATES)}"
        )
    return NucleoBenchCase(
        case_id=case_id,
        model_family=family,
        model_name=model_name,
        target=target,
        model_selector=dict(selector),
        sequence_length=sequence_length,
        editable_position_count=editable_count,
        max_seconds=max_seconds,
        state=state,
    )


def _validate_selector(
    family: str,
    target: str,
    selector: dict[str, Any],
    case_id: str,
    path: Path,
) -> None:
    if family == "malinois":
        _reject_unknown(selector, {"target_feature"}, "model_selector", path)
        if set(selector) != {"target_feature"}:
            raise CaseCatalogError(f"Missing target_feature for {case_id!r} in {path}")
        feature = selector["target_feature"]
        if (
            isinstance(feature, bool)
            or not isinstance(feature, int)
            or feature not in range(3)
        ):
            raise CaseCatalogError(f"Invalid target_feature for {case_id!r} in {path}")
        return
    if family == "bpnet":
        _reject_unknown(selector, {"protein"}, "model_selector", path)
        if set(selector) != {"protein"} or selector["protein"] != target:
            raise CaseCatalogError(
                f"BPNet protein must equal target for {case_id!r} in {path}"
            )
        return
    if family == "rinalmo":
        if selector:
            raise CaseCatalogError(
                f"RiNALMo selector must be empty for {case_id!r} in {path}"
            )
        return
    _reject_unknown(selector, {"aggregation_type"}, "model_selector", path)
    if selector != {"aggregation_type": "muscle_not_liver"}:
        raise CaseCatalogError(f"Invalid Enformer selector for {case_id!r} in {path}")


def _object(value: Any, field: str, path: Path) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CaseCatalogError(f"{field} must be an object in {path}")
    return value


def _string(value: Any, field: str, path: Path) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CaseCatalogError(f"{field} must be a non-empty string in {path}")
    return value.strip()


def _positive_int(value: Any, field: str, path: Path) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise CaseCatalogError(f"{field} must be a positive integer in {path}")
    return value


def _expect_int(value: Any, field: str, expected: int, path: Path) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value != expected:
        raise CaseCatalogError(f"{field} must equal {expected} in {path}")


def _reject_unknown(
    data: Mapping[str, Any], allowed: set[str], field: str, path: Path
) -> None:
    unknown = sorted(str(key) for key in data if key not in allowed)
    if unknown:
        raise CaseCatalogError(
            f"Unknown {field} field(s) in {path}: {', '.join(unknown)}"
        )


__all__ = [
    "CaseCatalogError",
    "NucleoBenchCase",
    "get_case",
    "load_case_catalog",
]
