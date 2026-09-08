"""Stable shared-runner adapter for Iron Mind."""

from __future__ import annotations

from typing import Any

from tasks.iron_mind.core import workflow as _workflow


def parse_args(argv: list[str] | None = None) -> Any:
    return _workflow.parse_args(argv)


def describe_ldm_task(*args: Any, **kwargs: Any) -> Any:
    return _workflow.describe_ldm_task(*args, **kwargs)


def main(argv: list[str] | None = None) -> int:
    return _workflow.main(argv)
