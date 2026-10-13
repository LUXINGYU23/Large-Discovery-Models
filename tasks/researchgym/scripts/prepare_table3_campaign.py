#!/usr/bin/env python3
"""Write a no-model-call, immutable ResearchGym Table 3 campaign plan.

The command deliberately has no launch option.  It makes the exact 15-job
matrix, dollar caps, wall-clock caps, CMR selector, and suggested GPU waves
reviewable before any paid trajectory starts.  A campaign directory is
write-once; correcting a plan means creating a fresh directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tasks.researchgym.table3_protocol import (
    CMR_PRIMARY_ROWS,
    PROFILES,
    SEEDS,
    TASKS,
    ProtocolError,
    write_manifest,
)


def campaign_plan(*, profile_name: str, campaign_dir: Path, max_concurrent: int) -> dict[str, Any]:
    """Return the complete non-overlapping job matrix for one budget profile."""
    if max_concurrent <= 0:
        raise ProtocolError("max_concurrent must be positive")
    profile = PROFILES[profile_name]
    jobs: list[dict[str, Any]] = []
    for position, (task_id, task) in enumerate(
        (task_id, task) for task_id, task in TASKS.items() for _seed in SEEDS
    ):
        seed = SEEDS[position % len(SEEDS)]
        jobs.append({
            "job_id": f"{profile_name}--{task_id}--seed-{seed}",
            "task_id": task_id,
            "upstream_dir": task.upstream_dir,
            "seed": seed,
            "workspace": str(campaign_dir / "workspaces" / task_id / f"seed-{seed}"),
            "artifacts": str(campaign_dir / "artifacts" / task_id / f"seed-{seed}"),
            "api_budget_usd": profile.api_budget_usd,
            "wall_clock_hours": profile.wall_clock_hours,
            "suggested_wave": position // max_concurrent + 1,
            "suggested_gpu_slot": position % max_concurrent,
            "state": "pending_preflight",
        })
    return {
        "schema_version": 1,
        "profile": profile.name,
        "max_concurrent": max_concurrent,
        "jobs": jobs,
        "totals": {
            "jobs": len(jobs),
            "max_api_spend_usd": len(jobs) * profile.api_budget_usd,
            "max_gpu_hours": len(jobs) * profile.wall_clock_hours,
            "suggested_waves": (len(jobs) + max_concurrent - 1) // max_concurrent,
            "best_case_wall_clock_hours": ((len(jobs) + max_concurrent - 1) // max_concurrent)
            * profile.wall_clock_hours,
        },
    }


def prepare_campaign(*, campaign_dir: Path, profile_name: str, model: str, harness: str,
                     cmr_primary_row: str, external_access: str, max_concurrent: int,
                     cmr_table3_contract: dict[str, Any] | None = None) -> dict[str, Any]:
    """Create manifest and plan exactly once, without calling a model or grader."""
    campaign_dir = campaign_dir.resolve()
    manifest_path, plan_path = campaign_dir / "manifest.json", campaign_dir / "jobs.json"
    if manifest_path.exists() or plan_path.exists():
        raise ProtocolError(f"campaign already initialized: {campaign_dir}")
    campaign_dir.mkdir(parents=True, exist_ok=True)
    manifest = write_manifest(manifest_path, profile_name=profile_name, model=model, harness=harness,
                              cmr_primary_row=cmr_primary_row, external_access=external_access,
                              cmr_table3_contract=cmr_table3_contract)
    plan = campaign_plan(profile_name=profile_name, campaign_dir=campaign_dir, max_concurrent=max_concurrent)
    plan["manifest"] = str(manifest_path)
    plan["source_commit"] = manifest["source_commit"]
    plan["metric_selectors"] = manifest["metric_selectors"]
    plan["table3_eligibility"] = manifest["table3_eligibility"]
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return plan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", type=Path, required=True)
    parser.add_argument("--profile", choices=tuple(PROFILES), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--harness", default="LDM-CodeAgent")
    parser.add_argument("--cmr-primary-row", choices=CMR_PRIMARY_ROWS, required=True)
    parser.add_argument("--cmr-table3-contract", type=Path,
                        help="Author-provided JSON contract for the unpublished CMR Table 3 aggregation.")
    parser.add_argument("--external-access", choices=("disabled", "enabled"), default="disabled")
    parser.add_argument("--max-concurrent", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        contract = (json.loads(args.cmr_table3_contract.read_text(encoding="utf-8"))
                    if args.cmr_table3_contract else None)
        plan = prepare_campaign(campaign_dir=args.campaign_dir, profile_name=args.profile,
                                model=args.model, harness=args.harness,
                                cmr_primary_row=args.cmr_primary_row,
                                external_access=args.external_access,
                                max_concurrent=args.max_concurrent,
                                cmr_table3_contract=contract)
    except (OSError, json.JSONDecodeError, ProtocolError) as exc:
        raise SystemExit(f"Table 3 campaign rejected: {exc}") from exc
    print(json.dumps(plan["totals"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
