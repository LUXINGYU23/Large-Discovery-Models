"""Append-only, strictly allowlisted scientific feedback.

The journal is owned by the controller, not by an agent. A partial final write,
an unexpected field, or a changed entry is an error rather than a new result.
"""

import fcntl
import json
import math
import os
import re
from pathlib import Path

from .exposure import valid_case_id

FIELDS = frozenset({
    "seq", "program_id", "epoch", "round", "branch", "status", "risk",
    "risk_by_case", "legal_rate", "failed_solves", "seconds_per_solve",
    "message", "selection",
})
MESSAGES = frozenset({"", "interface rejected", "all planned solves failed"})


def _number(value, lower, upper=None):
    return (type(value) in (int, float) and math.isfinite(value)
            and value >= lower and (upper is None or value <= upper))


def validate(record):
    if not isinstance(record, dict) or set(record) != FIELDS:
        raise ValueError("ledger record has unexpected or missing fields")
    if (type(record["seq"]) is not int or record["seq"] < 0
            or not isinstance(record["program_id"], str)
            or re.fullmatch(r"[a-f0-9]{64}", record["program_id"]) is None):
        raise ValueError("invalid record identity")
    if (any(type(record[k]) is not int or record[k] < 0 for k in ("epoch", "round"))
            or type(record["branch"]) is not int or record["branch"] < -1):
        raise ValueError("invalid record position")
    if record["selection"] not in {"initial", "reserved", "acquisition"}:
        raise ValueError("invalid selection reason")
    if (record["round"] == 0) != (record["branch"] == -1 and record["selection"] == "initial"):
        raise ValueError("initial record identity differs")
    if record["round"] > 0 and (record["branch"] < 0 or record["selection"] == "initial"):
        raise ValueError("non-initial records require a branch and selection reason")
    if record["status"] not in {"valid", "rejected", "failed"} or record["message"] not in MESSAGES:
        raise ValueError("invalid status or public reason")
    if (not _number(record["risk"], 0, 1) or not _number(record["legal_rate"], 0, 1)
            or not _number(record["seconds_per_solve"], 0)
            or type(record["failed_solves"]) is not int or record["failed_solves"] < 0):
        raise ValueError("invalid numerical feedback")
    cases = record["risk_by_case"]
    if not isinstance(cases, dict):
        raise ValueError("invalid public case risks")
    if cases:
        if any(not valid_case_id(k) or not _number(v, 0, 1) for k, v in cases.items()):
            raise ValueError("invalid public case risks")
        if not math.isclose(sum(cases.values()) / len(cases), record["risk"], abs_tol=1e-10):
            raise ValueError("case risks do not reproduce the parent-macro risk")
    if record["status"] == "rejected" and (record["risk"] != 1 or record["legal_rate"] != 0
                                           or record["message"] != "interface rejected"):
        raise ValueError("invalid interface rejection feedback")
    if record["status"] == "failed" and (record["risk"] != 1 or record["legal_rate"] != 0
                                         or record["message"] != "all planned solves failed"):
        raise ValueError("invalid all-failed feedback")
    if record["status"] == "valid" and record["message"]:
        raise ValueError("valid record has a failure reason")
    return record


def _read(stream):
    stream.seek(0)
    rows, ids = [], set()
    for line in stream:
        if not line.endswith("\n"):
            raise ValueError("incomplete ledger line; reconcile before resuming")
        row = validate(json.loads(line))
        if row["seq"] != len(rows) or row["program_id"] in ids:
            raise ValueError("ledger order or source identity differs")
        rows.append(row)
        ids.add(row["program_id"])
    return rows


def read(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as stream:
        fcntl.flock(stream, fcntl.LOCK_SH)
        return _read(stream)


def append(path, record):
    """Append once; replaying an identical completed record is idempotent."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    validate(record)
    with path.open("a+", encoding="utf-8") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        rows = _read(stream)
        previous = next((r for r in rows if r["program_id"] == record["program_id"]), None)
        if previous is not None:
            if previous != record:
                raise ValueError("replayed record changed")
            return previous
        if record["seq"] != len(rows):
            raise ValueError("record sequence is not the next ledger slot")
        stream.seek(0, os.SEEK_END)
        stream.write(json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return record
