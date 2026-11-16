"""A predeclared, one-turn planted note for development continuations."""

import hashlib
import re


def validate(value, rounds):
    if not isinstance(value, dict) or set(value) != {"round", "target_tag", "note"}:
        raise ValueError("intervention requires round, target_tag and note")
    if type(value["round"]) is not int or not 1 <= value["round"] <= rounds:
        raise ValueError("intervention round is outside the run")
    if (not isinstance(value["target_tag"], str)
            or re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", value["target_tag"]) is None):
        raise ValueError("invalid intervention target tag")
    if not isinstance(value["note"], str) or not 1 <= len(value["note"]) <= 4096:
        raise ValueError("invalid planted note")
    return dict(value)


def identity(value):
    return {"kind": "planted_note", "round": value["round"], "target_tag": value["target_tag"],
            "note_sha256": hashlib.sha256(value["note"].encode()).hexdigest()}
