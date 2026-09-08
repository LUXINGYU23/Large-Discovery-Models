"""Stable shared-runner adapter for the SynthonBench task."""

from __future__ import annotations

from typing import Any

from tasks.synthonbench.core import workflow as _workflow


def parse_args(argv: list[str] | None = None) -> Any:
    """Expose the task CLI parser at the registered runner boundary."""

    return _workflow.parse_args(argv)


def describe_ldm_task(*args: Any, **kwargs: Any) -> Any:
    """Expose the declared task semantics at the registered runner boundary."""

    return _workflow.describe_ldm_task(*args, **kwargs)


def main(argv: list[str] | None = None) -> int:
    """Run the task-local workflow through the shared LDM engine."""

    return _workflow.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
