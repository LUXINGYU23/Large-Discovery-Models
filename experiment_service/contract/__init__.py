"""Machine-readable wire contract of the ES (design 3.9): ``es-contract.schema.json``.

``load()`` returns the JSON Schema document (also served at ``GET /v1/contract``);
``validate(instance, definition)`` checks a body against one of its ``$defs`` with the
minimal standard-library validator in ``validator.py``.
"""
from __future__ import annotations

import json
from pathlib import Path

from .validator import SchemaError, check

SCHEMA_PATH = Path(__file__).with_name("es-contract.schema.json")
_CACHE: list = []


def load() -> dict:
    """The contract document (parsed once; callers must not mutate it)."""
    if not _CACHE:
        _CACHE.append(json.loads(SCHEMA_PATH.read_text(encoding="utf-8")))
    return _CACHE[0]


def validate(instance, definition: str) -> list[str]:
    """Problems (empty when valid) of ``instance`` against ``#/$defs/<definition>``."""
    return check(instance, {"$ref": f"#/$defs/{definition}"}, load())


__all__ = ["SCHEMA_PATH", "SchemaError", "check", "load", "validate"]
