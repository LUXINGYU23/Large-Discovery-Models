"""Direction-aware aggregation checks for quick comparisons."""

from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

from ldm_tts.quick_compare.reporting import _round_rows, _verdict


def test_minimization_verdict_treats_lower_ldm_values_as_better() -> None:
    rows = [
        {"case": "loss", "method": method, "seed": seed, "round_auc": value}
        for seed in (0, 1, 2)
        for method, value in (("ldm", 1.0), ("bo", 2.0), ("llm", 3.0))
    ]
    aggregates = [
        {
            "case": "loss",
            "method": method,
            "mean_round_auc": round_auc,
            "mean_final_best": final_best,
        }
        for method, round_auc, final_best in (
            ("ldm", 1.0, 0.5),
            ("bo", 2.0, 1.5),
            ("llm", 3.0, 2.5),
        )
    ]

    result = _verdict(rows, aggregates, "minimize")

    assert result["cases"] == [
        {"case": "loss", "verdict": "promising", "ldm_seed_wins": 3}
    ]


def test_harness_receives_its_own_baseline_verdict() -> None:
    values = {"ldm": 2.0, "harness": 1.0, "bo": 3.0, "llm": 4.0}
    rows = [
        {"case": "loss", "method": method, "seed": seed, "round_auc": value}
        for seed in (0, 1, 2)
        for method, value in values.items()
    ]
    aggregates = [
        {
            "case": "loss", "method": method,
            "mean_round_auc": value, "mean_final_best": value,
        }
        for method, value in values.items()
    ]

    result = _verdict(rows, aggregates, "minimize")

    assert result["cases"][0]["harness_verdict"] == "promising"
    assert result["cases"][0]["harness_seed_wins"] == 3


def test_evaluation_index_trajectory_uses_checkpoint_rounds_with_short_batches(
    tmp_path: Path,
) -> None:
    with (tmp_path / "trajectory.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("call_idx", "utility"))
        writer.writeheader()
        writer.writerows(
            {"call_idx": index, "utility": value}
            for index, value in enumerate((-5.0, -4.0, -3.0, -3.5, -2.0), start=1)
        )
    observations = [{"round_idx": value} for value in (0, 0, 1, 2, 2)]
    spec = SimpleNamespace(
        iterations=3,
        trajectory=SimpleNamespace(
            step_column="call_idx",
            step_kind="evaluation_index",
            objective_column="utility",
            direction="maximize",
        ),
    )

    rows = _round_rows(spec, tmp_path, "case", "harness", 0, observations)

    assert [row["round_evaluations"] for row in rows] == [2, 1, 2]
    assert [row["official_evaluations"] for row in rows] == [2, 3, 5]
    assert [row["best_so_far"] for row in rows] == [-4.0, -3.0, -2.0]
