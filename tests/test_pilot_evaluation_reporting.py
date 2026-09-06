"""Direction-aware aggregation checks for pilot evaluations."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ldm_tts.pilot_evaluation.config import load_pilot_evaluation_spec

from ldm_tts.pilot_evaluation.reporting import (
    METHOD_LABELS,
    _collect,
    _compiled_policy_records,
    _compiled_policy_summary,
    _expected_model_proposal_attempts,
    _integrity,
    _round_rows,
    _verdict,
    _write_csv,
)


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
    values = {
        "ldm": 2.0,
        "ldm_harness_compiled": 0.5,
        "harness": 1.0,
        "bo": 3.0,
        "llm": 4.0,
    }
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
    assert result["cases"][0]["ldm_harness_compiled_verdict"] == "promising"
    assert result["cases"][0]["ldm_harness_compiled_seed_wins"] == 3


def test_harness_methods_have_distinct_release_labels_and_budget_semantics() -> None:
    spec = SimpleNamespace(
        methods=("ldm_harness", "ldm_harness_compiled", "harness"),
        iterations=12,
        optimization_rounds=11,
    )
    common = {
        "case": "case",
        "seed": 0,
        "completed_rounds": 12,
        "budget_outer_iterations": 12,
        "candidate_ids_unique": True,
        "budget_llm_requests": 0,
        "initial_candidate_ids": ("shared",),
    }
    rows = [
        {
            **common,
            "method": "ldm_harness",
            "proposal_samples": 64,
            "evaluations_per_round": 16,
            "harness_candidates_per_session": 16,
            "budget_proposal_attempts": 44,
            "budget_harness_turns": 44,
        },
        {
            **common,
            "method": "harness",
            "proposal_samples": 16,
            "evaluations_per_round": 16,
            "harness_candidates_per_session": 16,
            "budget_proposal_attempts": 11,
            "budget_harness_turns": 11,
        },
        {
            **common,
            "method": "ldm_harness_compiled",
            "proposal_samples": 64,
            "evaluations_per_round": 16,
            "harness_candidates_per_session": 16,
            "budget_proposal_attempts": 44,
            "budget_harness_turns": 44,
            "budget_policy_harness_turns": 11,
            "policy_rounds": 11,
        },
    ]

    assert _integrity(spec, rows, [{}] * 36) == {"valid": True, "errors": []}
    assert METHOD_LABELS["ldm_harness"] == "LDM + Research Harness"
    assert METHOD_LABELS["ldm_harness_compiled"] == "Harness-Compiled LDM"
    assert METHOD_LABELS["harness"] == "Direct Research Harness"


@pytest.mark.parametrize("failed_usage", [None, {
    "providerCalls": 3, "toolCalls": {"bash": 2, "web_search": 1},
    "validationSubmissions": 2, "artifactBytes": 128,
}, {}])
def test_compiled_policy_reporting_uses_structured_selection_and_turn_records(
    tmp_path: Path,
    monkeypatch,
    failed_usage,
) -> None:
    monkeypatch.setenv("IRON_MIND_RUNS_ROOT", str(tmp_path))
    run_dir = tmp_path / "run"
    round_dir = run_dir / "policy_harness" / "rounds" / "round_001"
    turn_dir = run_dir / "policy_harness" / "turns" / "turn-1"
    round_dir.mkdir(parents=True)
    turn_dir.mkdir(parents=True)
    digest = "a" * 64
    artifact = "b" * 64
    submission = "c" * 64
    policy = {
        "epoch_id": "epoch_001",
        "artifact_sha256": artifact,
        "source": "artifact",
        "degraded": False,
        "stage": "explore",
        "alpha": 0.8,
        "eta": 2.0,
        "action": "replace",
        "status": "accepted",
        "prior_clip_count": 1,
        "validation_errors": [],
        "baseline_prediction_diagnostics": {
            "mean_mean": 0.1,
            "mean_std": 0.01,
            "acquisition_mean": 0.2,
            "acquisition_std": 0.02,
        },
        "compiled_prediction_diagnostics": {
            "mean_mean": 0.3,
            "mean_std": 0.03,
            "acquisition_mean": 0.4,
            "acquisition_std": 0.04,
        },
        "baseline_compiled_top10_overlap": 0.5,
        "harness_turn": {
            "turn_id": "turn-1",
            "session_id": "session-1",
            "usage": {
                "providerCalls": 3,
                "toolCalls": {"bash": 2, "web_search": 1},
                "validationSubmissions": 2,
                "artifactBytes": 128,
            },
        },
    }
    if failed_usage is not None:
        policy.pop("harness_turn")
        policy.update(action="fallback", status="runtime_fallback", degraded=True,
                      failed_harness_usage=failed_usage)
    (round_dir / "manifest.json").write_text(
        json.dumps({"input_sha256": digest}),
        encoding="utf-8",
    )
    (round_dir / "result.json").write_text(
        json.dumps(
            {
                **policy, "input_sha256": digest, "submission_sha256": submission,
            }
        ),
        encoding="utf-8",
    )
    (turn_dir / "turn_committed.json").write_text(
        json.dumps(
            {
                "turnId": "turn-1",
                "sessionId": "session-1",
                "submissionDigest": submission,
            }
        ),
        encoding="utf-8",
    )
    event = {
        "event_type": "candidates_selected",
        "iteration": 1,
        "payload": {
            "metadata": {
                "compiled_policy": policy,
                "base_probability_entropy": 0.5623351446,
                "probability_entropy": 0.5623351446,
                "probability_effective_sample_size": 1.6,
                "tilted_kl_from_q0": 0.5493061443,
                "q0_tilted_topk_overlap": 0.0,
                "base_selection": {
                    "mean_source": "compiled:artifact",
                    "prior_mean_clip_count": 1,
                    "residual_target_mean": 0.0,
                    "residual_target_std": 1.0,
                },
            },
        },
    }
    event_path = run_dir / "events.jsonl"
    event_path.write_text(json.dumps({"event_type": "run_started"}) + "\n" + json.dumps(event) + "\n")

    spec = load_pilot_evaluation_spec(
        Path(__file__).parents[1] / "config/pilot_evaluation/iron_mind.yaml"
    )
    records = _compiled_policy_records(run_dir, "case", 0, spec.policy_fields)
    summary = _compiled_policy_summary(records, spec.policy_mean_fields)

    assert records[0]["provider_request_count"] == (None if failed_usage == {} else 3)
    assert records[0]["tool_call_count"] == (None if failed_usage == {} else 3)
    assert records[0]["validation_submission_count"] == (None if failed_usage == {} else 2)
    assert records[0]["validation_failure_count"] == (
        None if failed_usage == {} else 1 if failed_usage is None else 2
    )
    assert records[0]["turn_committed"] == (failed_usage is None)
    assert records[0]["usage_complete"] == (failed_usage != {})
    assert records[0]["compiled_prediction_std"] == 0.03
    assert records[0]["compiled_acquisition_std"] == 0.04
    assert records[0]["tilted_kl_from_q0"] > 0
    assert records[0]["q0_tilted_topk_overlap"] == 0.0
    assert summary["policy_rounds"] == 1
    assert summary["policy_validation_failures"] == (0 if failed_usage == {} else 1 if failed_usage is None else 2)
    assert summary["policy_failed_turns"] == (failed_usage is not None)
    assert summary["policy_committed_turns"] == (failed_usage is None)
    assert summary["policy_usage_incomplete_rounds"] == (failed_usage == {})
    assert summary["policy_last_stage"] == "explore"

    event["payload"]["metadata"]["multiobjective"] = {
        "means": [1.0, -2.0], "per_objective": {"yield": 1.0, "cost": -2.0},
    }
    event_path.write_text(json.dumps(event) + "\n")
    records = _compiled_policy_records(run_dir, "case", 0, {
        "prior_vector": "multiobjective.means",
        "objective_metrics": "multiobjective.per_objective",
    })
    assert records[0]["prior_vector"] == [1.0, -2.0]
    assert "compiled_prediction_std" not in records[0]
    _write_csv(tmp_path / "policy.csv", records)
    with (tmp_path / "policy.csv").open(newline="") as handle:
        row = next(csv.DictReader(handle))
    assert json.loads(row["objective_metrics"]) == {"yield": 1.0, "cost": -2.0}
    with pytest.raises(ValueError, match="numeric scalar"):
        _compiled_policy_summary(records, ("prior_vector",))
    with pytest.raises(ValueError, match="built-in policy columns"):
        _compiled_policy_records(run_dir, "case", 0, {"round": "multiobjective.means"})


def test_model_proposal_budget_counts_minibatch_requests() -> None:
    ldm = {
        "method": "ldm",
        "proposal_samples": 64,
        "evaluations_per_round": 16,
        "proposal_candidates_per_request": 16,
    }
    llm = {
        "method": "llm",
        "proposal_samples": 16,
        "evaluations_per_round": 16,
        "proposal_candidates_per_request": 1,
    }

    assert _expected_model_proposal_attempts(ldm, 11) == 44
    assert _expected_model_proposal_attempts(llm, 11) == 176


@pytest.mark.parametrize("initial_count", [1, 2])
def test_report_counts_initialization_separately_from_optimization_batches(
    tmp_path: Path, initial_count: int,
) -> None:
    observations = [
        {"round_idx": round_idx, "candidate": {"candidate_id": str(index), "canonical_key": str(index)}}
        for index, round_idx in enumerate([0] * initial_count + [1, 1])
    ]
    for name, value in {
        "checkpoint": {"state": {"observations": observations}},
        "result": {},
        "config": {"evaluations_per_round": 2, "proposal_samples": 64},
        "campaign": {"contract_sha256": "contract"},
        "budget": {"counters": {}},
        "status": {"started_at_unix": 0, "updated_at_unix": 1},
    }.items():
        (tmp_path / f"{name}.json").write_text(json.dumps(value))
    with (tmp_path / "trajectory.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("round", "utility"))
        writer.writeheader()
        writer.writerows({"round": item["round_idx"], "utility": index}
                        for index, item in enumerate(observations))
    spec = SimpleNamespace(
        output_root=tmp_path, iterations=2, optimization_rounds=1, result_fields={},
        trajectory=SimpleNamespace(step_column="round", step_kind="round",
                                   objective_column="utility", direction="maximize"),
    )

    rows, trajectory, _ = _collect(spec, [("case/bo/seed_0", {"status": "completed", "run_dir": "."})])

    assert rows[0]["initial_candidate_ids"] == tuple(map(str, range(initial_count)))
    assert rows[0]["expected_evaluations"] == initial_count + 2
    assert rows[0]["evaluation_utilization"] == 1.0
    assert [item["round_evaluations"] for item in trajectory] == [initial_count, 2]


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
