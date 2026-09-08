"""Command-line entrypoint for the task-neutral pilot evaluation matrix."""

from __future__ import annotations

import argparse
from pathlib import Path

from ldm_tts.pilot_evaluation.config import load_pilot_evaluation_spec
from ldm_tts.pilot_evaluation.execution import run_evaluation


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a fixed-round task-method pilot evaluation."
    )
    parser.add_argument("config", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--case", action="append")
    parser.add_argument("--method", action="append")
    parser.add_argument("--seed", action="append", type=int)
    args = parser.parse_args(argv)
    return run_evaluation(
        load_pilot_evaluation_spec(args.config), resume=args.resume, dry_run=args.dry_run,
        cases=args.case, methods=args.method, seeds=args.seed,
    )


__all__ = ["main"]
