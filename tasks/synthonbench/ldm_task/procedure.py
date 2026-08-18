#!/usr/bin/env python3
"""Procedure adapter for the ``synthonbench`` task."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ldm_tts.contracts import (
    AcquisitionSpec,
    CandidateDomainSpec,
    LDMTaskSpec,
    ObjectiveSpec,
    ReservoirExpansionSpec,
    ReservoirSpec,
    ResponseSpaceSpec,
    ProposalSearchSpec,
    SurrogateSpaceSpec,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the synthonbench LDM task.")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--out-dir", type=Path, default=Path("runs/mock"))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def describe_ldm_task(args: argparse.Namespace) -> LDMTaskSpec:
    return LDMTaskSpec(
        task="synthonbench",
        candidate_domain=CandidateDomainSpec(
            name="replace_me",
            kind="replace_me",
            dimension=None,
            representation="Replace with the task candidate representation.",
        ),
        objectives=(
            ObjectiveSpec(
                name="objective",
                direction="maximize",
                description="Replace with the measured task objective.",
            ),
        ),
        response_spaces=(
            ResponseSpaceSpec(
                name="proposal",
                output_kind="json",
                description="Replace with the model response contract.",
            ),
        ),
        acquisition=AcquisitionSpec(
            name="mean",
            objective_names=("objective",),
            score_direction="maximize",
            selection_rule="Replace with the task selection rule.",
        ),
        reservoir=ReservoirSpec(
            name="candidate_reservoir",
            expansions=(
                ReservoirExpansionSpec(
                    name="direct_proposal",
                    action_kind="emit_candidate",
                    response_space="proposal",
                    produces_candidates=True,
                    description="Replace with the task's reservoir expansion action.",
                ),
            ),
            candidate_validator="Replace with the task candidate validator.",
            deduplication_key="Replace with the canonical candidate identity.",
        ),
        surrogate=SurrogateSpaceSpec(
            kind="none",
            representation="not configured in the draft scaffold",
            dimension_policy="none",
        ),
        proposal_search=ProposalSearchSpec(name="single_turn"),
        metadata={"mock": bool(args.mock)},
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.mock and not args.dry_run:
        raise SystemExit("Implement real task execution before running without --mock.")
    task_spec = describe_ldm_task(args)
    run_dir = args.out_dir
    payload = {
        "task": "synthonbench",
        "iterations": max(0, args.iterations),
        "mock": bool(args.mock),
        "ldm_task_spec": task_spec.to_dict(),
    }
    if args.mock and not args.dry_run:
        from ldm_tts.engine.run_store import unique_run_dir
        from tasks.synthonbench.core.mock_engine import run_mock_campaign

        run_dir = unique_run_dir(args.out_dir)
        result = run_mock_campaign(
            task_spec,
            iterations=max(0, args.iterations),
            run_dir=run_dir,
        )
        payload["engine_summary"] = result.summary
        payload["run_dir"] = str(run_dir.resolve())
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
