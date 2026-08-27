"""Command-line entrypoint for the task-neutral quick comparison matrix."""

from __future__ import annotations

import argparse
from pathlib import Path

from ldm_tts.quick_compare.config import load_quick_compare_spec
from ldm_tts.quick_compare.execution import run_comparison


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a fixed-budget LDM, BO, and LLM comparison.")
    parser.add_argument("config", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--case", action="append")
    parser.add_argument("--method", action="append")
    parser.add_argument("--seed", action="append", type=int)
    args = parser.parse_args(argv)
    return run_comparison(
        load_quick_compare_spec(args.config), resume=args.resume, dry_run=args.dry_run,
        cases=args.case, methods=args.method, seeds=args.seed,
    )


__all__ = ["main"]
