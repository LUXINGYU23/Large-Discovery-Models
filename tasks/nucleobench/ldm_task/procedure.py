"""Stable shared-runner adapter for the NucleoBench task."""

from __future__ import annotations

import argparse

from ldm_tts.contracts import LDMTaskSpec
from tasks.nucleobench.core import workflow


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return workflow.parse_args(argv)


def describe_ldm_task(args: argparse.Namespace) -> LDMTaskSpec:
    return workflow.describe_ldm_task(args)


def main(argv: list[str] | None = None) -> int:
    return workflow.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
