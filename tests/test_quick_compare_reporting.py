"""Direction-aware aggregation checks for quick comparisons."""

from __future__ import annotations

from ldm_tts.quick_compare.reporting import _verdict


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
