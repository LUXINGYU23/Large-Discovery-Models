"""Wire-level conventions shared with the experiment service (by contract, not by import).

``canonical``/``digest`` reproduce the ES algorithm byte for byte: the ES recomputes
``spec_digest`` on every PUT and answers 422 when the client's value differs, so any drift
here surfaces immediately as a rejected submission.
"""
from __future__ import annotations

import hashlib
import json
import re

ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._~-]{0,199}")
CAMPAIGN_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._~-]{0,150}")
STATES = frozenset({"QUEUED", "RUNNING", "SUCCEEDED", "FAILED", "CANCELED"})
TERMINAL = frozenset({"SUCCEEDED", "FAILED", "CANCELED"})
TERMINAL_SQL = "('SUCCEEDED','FAILED','CANCELED')"


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def digest(value) -> str:
    return "sha256:" + hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def revision_number(value) -> int:
    """ES revisions are zero-padded decimal strings; compare them as integers."""
    if isinstance(value, bool):
        raise ValueError("revision must be a decimal string")
    if isinstance(value, int) and value >= 0:
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    raise ValueError(f"invalid revision {value!r}")


def decision_label(decision_id: int) -> str:
    return f"d{decision_id:06d}"


def experiment_id_for(campaign_id: str, decision_id: int, step: int = 0) -> str:
    """Deterministic identity (design 3.3): replaying a decision yields the same id."""
    experiment_id = f"{campaign_id}.{decision_label(decision_id)}.s{step}"
    if not ID_PATTERN.fullmatch(experiment_id):
        raise ValueError(f"derived experiment id {experiment_id!r} is not URL-safe")
    return experiment_id
