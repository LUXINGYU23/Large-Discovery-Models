"""Low-level validation primitives for Iron Mind tiny-campaign evidence."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tasks.iron_mind.core.qualification_support import QualificationRecordError


PROVIDER = {
    "kind": "openai_compatible",
    "base_url": "https://api.deepseek.com",
    "model": "deepseek-v4-flash",
}


class TinyCampaignRecordError(QualificationRecordError):
    """Raised when one run cannot support real tiny-campaign evidence."""


def read_json_lines(path: Path) -> list[Mapping[str, Any]]:
    try:
        rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]
    except (OSError, json.JSONDecodeError) as exc:
        raise TinyCampaignRecordError(f"Could not read campaign events: {path}") from exc
    if not rows or any(not isinstance(row, Mapping) for row in rows):
        raise TinyCampaignRecordError("Campaign events must be non-empty JSON objects.")
    return rows


def one_event(events: Sequence[Mapping[str, Any]], event_type: str) -> Mapping[str, Any]:
    matches = [event for event in events if event.get("event_type") == event_type]
    if len(matches) != 1:
        raise TinyCampaignRecordError(f"Tiny campaign requires exactly one {event_type} event.")
    return matches[0]


def event_payload(event: Mapping[str, Any], label: str) -> Mapping[str, Any]:
    return require_mapping(event.get("payload"), f"{label} payload")


def require_endpoint_preflight(events: Sequence[Mapping[str, Any]]) -> None:
    payload = event_payload(
        one_event(events, "endpoint_preflight_succeeded"),
        "endpoint_preflight_succeeded",
    )
    expected = {
        "status": "ok",
        "request_model": PROVIDER["model"],
        "response_model": PROVIDER["model"],
        "model_visible": True,
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise TinyCampaignRecordError("Endpoint preflight did not verify the configured model identity.")
    if not isinstance(payload.get("model_count"), int) or payload["model_count"] < 1:
        raise TinyCampaignRecordError("Endpoint preflight did not report a visible model list.")


def candidate_ids(value: object, *, count: int, label: str) -> list[tuple[str, str]]:
    parsed = [candidate_key(item) for item in require_sequence(value, label)]
    if len(parsed) != count or len({item[0] for item in parsed}) != count:
        raise TinyCampaignRecordError(f"{label.capitalize()} must contain {count} distinct values.")
    return parsed


def candidate_key(value: object) -> tuple[str, str]:
    if not isinstance(value, str) or not value.startswith("iron-mind-"):
        raise TinyCampaignRecordError("Candidate id must use the Iron Mind canonical prefix.")
    key = value.removeprefix("iron-mind-")
    if len(key) != 64 or any(char not in "0123456789abcdef" for char in key):
        raise TinyCampaignRecordError("Candidate id must contain a lowercase SHA-256 canonical key.")
    return value, key


def reaction_score(evaluation: Mapping[str, Any], candidate_id: str) -> float:
    metrics = require_mapping(evaluation.get("metrics"), "evaluation metrics")
    if evaluation.get("candidate_id") != candidate_id or evaluation.get("status") != "succeeded":
        raise TinyCampaignRecordError("Evaluation did not succeed for the selected candidate.")
    if set(metrics) != {"reaction_score"} or evaluation.get("resource_usage") != {"benchmark_jobs": 1.0}:
        raise TinyCampaignRecordError("Evaluation must report one reaction_score and one benchmark job.")
    return finite_number(metrics["reaction_score"], "evaluation reaction_score")


def integer_mapping(value: object, label: str) -> dict[str, int]:
    parsed: dict[str, int] = {}
    for key, item in require_mapping(value, label).items():
        parsed[str(key)] = int(finite_number(item, label))
        if parsed[str(key)] != item:
            raise TinyCampaignRecordError(f"{label.capitalize()} contains a non-integral value.")
    return parsed


def finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise TinyCampaignRecordError(f"{label.capitalize()} must be a finite numeric value.")
    return float(value)


def sha256_digest(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise TinyCampaignRecordError(f"{label.capitalize()} must be a lowercase SHA-256 value.")
    return value


def require_mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TinyCampaignRecordError(f"{label.capitalize()} must be an object.")
    return value


def require_sequence(value: object, label: str) -> Sequence[Any]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise TinyCampaignRecordError(f"{label.capitalize()} must be an array.")
    return value


__all__ = [
    "PROVIDER",
    "TinyCampaignRecordError",
    "candidate_ids",
    "candidate_key",
    "event_payload",
    "finite_number",
    "integer_mapping",
    "one_event",
    "reaction_score",
    "read_json_lines",
    "require_endpoint_preflight",
    "require_mapping",
    "require_sequence",
    "sha256_digest",
]
