"""Stable shared-runner adapter for the NucleoBench task."""

from __future__ import annotations

import argparse
import json

from ldm_tts.contracts import LDMTaskSpec

from tasks.nucleobench.core.cases import get_case
from tasks.nucleobench.core.task_spec import build_task_spec


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect the NucleoBench LDM task.")
    parser.add_argument("--case-id", default="malinois_k562")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def describe_ldm_task(args: argparse.Namespace) -> LDMTaskSpec:
    return build_task_spec(get_case(args.case_id))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.dry_run:
        raise SystemExit(
            "NucleoBench execution is not qualified yet; use --dry-run to inspect the contract."
        )
    case = get_case(args.case_id)
    task_spec = describe_ldm_task(args)
    print(
        json.dumps(
            {
                "task": "nucleobench",
                "case": case.to_dict(),
                "ldm_task_spec": task_spec.to_dict(),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
