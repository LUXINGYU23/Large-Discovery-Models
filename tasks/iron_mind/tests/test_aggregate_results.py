"""Tests for complete Iron Mind campaign aggregation."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from tasks.iron_mind.scripts.aggregate_official_results import main


def test_aggregator_emits_summary_and_incumbent_trajectory(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    for campaign_index, scores in enumerate(((1.0, 3.0), (2.0, 2.5))):
        _write_run(runs_root / f"run-{campaign_index}", campaign_index, scores)
    output = tmp_path / "aggregate"

    code = main([
        "--runs-root", str(runs_root),
        "--output-dir", str(output),
        "--suite", "paper_v2",
        "--expected-campaigns", "2",
        "--expected-evaluations", "2",
        "--allow-incomplete",
    ])

    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    rows = list(csv.DictReader((output / "aggregate_trajectory.csv").open(encoding="utf-8")))
    buchwald = next(item for item in summary["datasets"] if item["dataset_id"] == "buchwald_hartwig")
    assert code == 0
    assert summary["method"] == "ldm_tilted_ucb"
    assert buchwald["campaign_count"] == 2
    assert buchwald["mean_best_reaction_score"] == 2.75
    assert [row["mean_best_reaction_score"] for row in rows[:2]] == ["1.5", "2.75"]


def test_aggregator_rejects_duplicate_campaign_identity(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    _write_run(runs_root / "first", 0, (1.0,))
    _write_run(runs_root / "second", 0, (2.0,))

    with pytest.raises(ValueError, match="Duplicate campaign identity"):
        main([
            "--runs-root", str(runs_root),
            "--output-dir", str(tmp_path / "aggregate"),
            "--allow-incomplete",
        ])


def _write_run(run_dir: Path, campaign_index: int, scores: tuple[float, ...]) -> None:
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(
        json.dumps({"dataset_id": "buchwald_hartwig", "campaign_index": campaign_index}),
        encoding="utf-8",
    )
    evaluations = [
        {"iteration": index, "candidate_id": f"candidate-{index}", "reaction_score": score}
        for index, score in enumerate(scores, start=1)
    ]
    result = {
        "finished": True,
        "evaluation_count": len(scores),
        "evaluations": evaluations,
        "best_candidate": {"reaction_score": max(scores)},
    }
    (run_dir / "result.json").write_text(json.dumps(result), encoding="utf-8")
