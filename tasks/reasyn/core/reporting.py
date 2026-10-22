"""ReaSyn campaign artifacts and benchmark-specific result metrics."""

from __future__ import annotations

import csv
import json

from ldm_tts.engine.reporting import load_successful_observations
from ldm_tts.engine.run_store import atomic_json_write
from .metrics import reconstruction_metrics, top_auc, top_mean


def write_campaign_report(args, runtime, target):
    if args.proposal_mode == "harness":
        kinds = ("harness", "policy_harness") if args.search_method == "ldm_harness_compiled" else ("harness",)
        pools = {}
        for kind in kinds:
            paths = [p
                for pattern in ("manifest.json", "main/manifest.json", "branches/round_*/manifest.json")
                for p in sorted((runtime.run_dir / kind).glob(pattern))]
            if not paths:
                pools[kind] = []
            for path in paths:
                pools[path.parent.relative_to(runtime.run_dir).as_posix()] = [path.relative_to(runtime.run_dir).as_posix()]
        atomic_json_write(runtime.run_dir / "harness_provenance.json", {"schema_version": 1, "pools": pools})
    checkpoint = runtime.run_dir / "checkpoint.json"
    observations = load_successful_observations(checkpoint) if checkpoint.exists() else []
    events = runtime.events()
    atomic_json_write(
        runtime.run_dir / "search_manifest.json",
        {"rounds": [e for e in events if e.get("event_type") == "reservoir_built"]},
    )
    atomic_json_write(
        runtime.run_dir / "selection_record.json",
        {
            "selections": [
                e for e in events if e.get("event_type") == "candidates_selected"
            ]
        },
    )
    rows = []
    for i, o in enumerate(observations):
        rows.append(
            {
                "evaluation": i + 1,
                "candidate_id": o["candidate"]["candidate_id"],
                "score": o["evaluation"]["metrics"][
                    "similarity" if target else "oracle_score"
                ],
            }
        )
    with (runtime.run_dir / "trajectory.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("evaluation", "candidate_id", "score")
        )
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "task": "reasyn",
        "benchmark": args.benchmark,
        "mock": args.mock,
        "qualification": "draft",
        "successful_evaluations": len(observations),
        "budget": json.loads((runtime.run_dir / "budget.json").read_text()),
    }
    if target:
        product_rows = []
        for o in observations:
            artifact = o["evaluation"]["artifacts"].get("projection")
            if artifact:
                data = json.loads((runtime.run_dir / artifact).read_text())
                product_rows.extend(
                    {**r, "target": target, "projection_query": r["target"]}
                    for r in data["rows"]
                )
        report["reconstruction_rows"] = product_rows
        report["metrics"] = reconstruction_metrics(
            [target], product_rows, mock=args.mock
        )
    else:
        cache_path = runtime.run_dir / "oracle_cache.json"
        entries = (
            json.loads(cache_path.read_text())["entries"] if cache_path.exists() else []
        )
        scores = [e["score"] for e in entries if e["status"] == "completed"]
        full = len(scores) == args.max_oracle_calls and len(entries) == len(scores)
        report.update(
            oracle=args.oracle,
            oracle_calls=len(entries),
            completed_oracle_calls=len(scores),
            oracle_budget_exhausted=full,
            official_budget=args.max_oracle_calls == 10000,
            metrics={
                "top10": top_mean(scores),
                "auc_top10_observed": top_auc(scores, max_calls=args.max_oracle_calls),
                "auc_top10": top_auc(scores, max_calls=args.max_oracle_calls)
                if full
                else None,
            },
        )
        # No automatic early-stop padding: an interrupted/pilot campaign is not
        # upstream scientific convergence. Aggregate only full-budget runs.
    atomic_json_write(runtime.run_dir / "result.json", report)
    return report
