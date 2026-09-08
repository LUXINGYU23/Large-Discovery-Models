"""Aggregate complete Iron Mind LDM campaigns into publication-ready tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from tasks.iron_mind.core.constants import OBJECTIVE_NAME, TASK_ID


TASK_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = TASK_ROOT / "resources" / "upstream_contract.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--suite", choices=("paper_v2", "public_union"), default="paper_v2")
    parser.add_argument("--expected-campaigns", type=int, default=20)
    parser.add_argument("--expected-evaluations", type=int, default=20)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _validate_args(args)
    dataset_ids = _suite_dataset_ids(args.suite)
    campaigns = _load_campaigns(args.runs_root, dataset_ids)
    _require_complete(campaigns, dataset_ids, args)
    summary = _summary_rows(campaigns, dataset_ids)
    trajectories = _trajectory_rows(campaigns, dataset_ids)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "dataset_summary.csv", summary)
    _write_csv(args.output_dir / "aggregate_trajectory.csv", trajectories)
    payload = {
        "schema_version": 1,
        "task": TASK_ID,
        "method": "ldm_tilted_ucb",
        "suite": args.suite,
        "expected_campaigns_per_dataset": args.expected_campaigns,
        "expected_evaluations_per_campaign": args.expected_evaluations,
        "datasets": summary,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "ok", "output_dir": str(args.output_dir)}, indent=2))
    return 0


def _load_campaigns(
    runs_root: Path, dataset_ids: tuple[str, ...]
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    identities: set[tuple[str, int]] = set()
    for result_path in sorted(runs_root.rglob("result.json")):
        run_dir = result_path.parent
        config = _read_object(run_dir / "config.json")
        dataset_id = config.get("dataset_id")
        if dataset_id not in dataset_ids:
            continue
        campaign_index = _integer(config.get("campaign_index"), "campaign_index")
        identity = (dataset_id, campaign_index)
        if identity in identities:
            raise ValueError(f"Duplicate campaign identity: {identity}")
        identities.add(identity)
        result = _read_object(result_path)
        grouped[dataset_id].append(
            {"campaign_index": campaign_index, "result": result, "run_dir": str(run_dir)}
        )
    return dict(grouped)


def _require_complete(
    campaigns: Mapping[str, list[dict[str, Any]]],
    dataset_ids: tuple[str, ...],
    args: argparse.Namespace,
) -> None:
    errors = []
    for dataset_id in dataset_ids:
        runs = campaigns.get(dataset_id, [])
        if len(runs) != args.expected_campaigns:
            errors.append(
                f"{dataset_id}: expected {args.expected_campaigns} campaigns, got {len(runs)}"
            )
        for run in runs:
            result = run["result"]
            if not result.get("finished"):
                errors.append(f"{dataset_id} campaign {run['campaign_index']}: not finished")
            if result.get("evaluation_count") != args.expected_evaluations:
                errors.append(
                    f"{dataset_id} campaign {run['campaign_index']}: "
                    f"expected {args.expected_evaluations} evaluations"
                )
    if errors and not args.allow_incomplete:
        raise ValueError("Incomplete benchmark:\n" + "\n".join(errors))


def _summary_rows(
    campaigns: Mapping[str, list[dict[str, Any]]], dataset_ids: tuple[str, ...]
) -> list[dict[str, Any]]:
    rows = []
    for dataset_id in dataset_ids:
        final_scores = [
            _best_score(run["result"])
            for run in campaigns.get(dataset_id, [])
            if run["result"].get("best_candidate") is not None
        ]
        rows.append(
            {
                "dataset_id": dataset_id,
                "campaign_count": len(campaigns.get(dataset_id, [])),
                "completed_campaign_count": sum(
                    bool(run["result"].get("finished"))
                    for run in campaigns.get(dataset_id, [])
                ),
                "mean_best_reaction_score": _mean(final_scores),
                "std_best_reaction_score": _std(final_scores),
                "median_best_reaction_score": _median(final_scores),
                "max_best_reaction_score": max(final_scores) if final_scores else None,
            }
        )
    return rows


def _trajectory_rows(
    campaigns: Mapping[str, list[dict[str, Any]]], dataset_ids: tuple[str, ...]
) -> list[dict[str, Any]]:
    rows = []
    for dataset_id in dataset_ids:
        by_step: dict[int, list[float]] = defaultdict(list)
        for run in campaigns.get(dataset_id, []):
            incumbent: float | None = None
            for evaluation in run["result"].get("evaluations", []):
                value = _finite(evaluation.get(OBJECTIVE_NAME), OBJECTIVE_NAME)
                incumbent = value if incumbent is None else max(incumbent, value)
                by_step[_integer(evaluation.get("iteration"), "iteration")].append(incumbent)
        for evaluation, values in sorted(by_step.items()):
            rows.append(
                {
                    "dataset_id": dataset_id,
                    "evaluation": evaluation,
                    "campaign_count": len(values),
                    "mean_best_reaction_score": _mean(values),
                    "std_best_reaction_score": _std(values),
                }
            )
    return rows


def _best_score(result: Mapping[str, Any]) -> float:
    best = result.get("best_candidate")
    if not isinstance(best, Mapping):
        raise ValueError("Completed campaign has no best_candidate object")
    return _finite(best.get(OBJECTIVE_NAME), OBJECTIVE_NAME)


def _suite_dataset_ids(suite: str) -> tuple[str, ...]:
    contract = _read_object(CONTRACT_PATH)
    values = contract.get("suites", {}).get(suite)
    if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
        raise ValueError(f"Invalid suite in upstream contract: {suite}")
    return tuple(values)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty aggregate table: {path.name}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _validate_args(args: argparse.Namespace) -> None:
    if not args.runs_root.is_dir():
        raise ValueError(f"Runs root does not exist: {args.runs_root}")
    if args.expected_campaigns < 1 or args.expected_evaluations < 1:
        raise ValueError("Expected campaign and evaluation counts must be positive")


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{label} must be finite")
    return numeric


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return value


def _mean(values: Iterable[float]) -> float | None:
    items = list(values)
    return statistics.fmean(items) if items else None


def _std(values: Iterable[float]) -> float | None:
    items = list(values)
    return statistics.stdev(items) if len(items) > 1 else (0.0 if items else None)


def _median(values: Iterable[float]) -> float | None:
    items = list(values)
    return statistics.median(items) if items else None


if __name__ == "__main__":
    raise SystemExit(main())
