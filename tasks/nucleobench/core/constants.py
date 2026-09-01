"""Pinned public protocol constants for NucleoBench."""

from __future__ import annotations

from pathlib import Path


TASK_ID = "nucleobench"
UPSTREAM_URL = "https://github.com/move37-labs/nucleobench"
UPSTREAM_COMMIT = "a6d8b040a4fa80b18266ee5416c904d01f842428"
UPSTREAM_PACKAGE_VERSION = "2.0.10"
OFFICIAL_CASE_COUNT = 17
OFFICIAL_START_COUNT = 100
CASE_STATES = frozenset({"planned", "prepared", "qualified"})
CATALOG_PATH = (
    Path(__file__).resolve().parents[1] / "resources" / "cases" / "catalog.json"
)

FAMILY_PROTOCOLS = {
    "malinois": {
        "model_name": "malinois",
        "sequence_length": 200,
        "editable_position_count": 200,
        "max_seconds": 28_800,
    },
    "bpnet": {
        "model_name": "bpnet",
        "sequence_length": 3_000,
        "editable_position_count": 3_000,
        "max_seconds": 28_800,
    },
    "rinalmo": {
        "model_name": "rinalmo_mrl",
        "sequence_length": 100,
        "editable_position_count": 100,
        "max_seconds": 28_800,
    },
    "enformer": {
        "model_name": "enformer",
        "sequence_length": 196_608,
        "editable_position_count": 256,
        "max_seconds": 43_200,
    },
}


__all__ = [
    "CASE_STATES",
    "CATALOG_PATH",
    "FAMILY_PROTOCOLS",
    "OFFICIAL_CASE_COUNT",
    "OFFICIAL_START_COUNT",
    "TASK_ID",
    "UPSTREAM_COMMIT",
    "UPSTREAM_PACKAGE_VERSION",
    "UPSTREAM_URL",
]
