"""Draft workflow boundary for the Iron Mind task."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


TASK_ID = "iron_mind"
DRAFT_MESSAGE = "Iron Mind task implementation is incomplete."


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Iron Mind LDM task.")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--out-dir", type=Path, default=Path("runs/mock"))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def describe_ldm_task(*args: Any, **kwargs: Any) -> Any:
    del args, kwargs
    raise RuntimeError(DRAFT_MESSAGE)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.dry_run:
        print(json.dumps({"status": "draft", "task": TASK_ID}, sort_keys=True))
        return 0
    raise SystemExit(DRAFT_MESSAGE)
