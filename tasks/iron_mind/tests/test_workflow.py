"""Workflow-level contracts for the Iron Mind LDM campaign."""

from __future__ import annotations

import json
from pathlib import Path

from ldm_tts.registration.experiment import load_experiment_contract

from tasks.iron_mind.core.workflow import describe_ldm_task, main, parse_args


TASK_ROOT = Path(__file__).resolve().parents[1]


def test_describe_ldm_task_uses_the_default_configured_reservoir() -> None:
    args = parse_args(["--mock"])

    task_spec = describe_ldm_task(args)
    contract = load_experiment_contract(TASK_ROOT / "experiment.json")

    assert task_spec.task == "iron_mind"
    assert task_spec.candidate_domain.kind == "finite_reaction_conditions"
    assert task_spec.response_spaces[0].schema == {
        "type": "object",
        "additionalProperties": False,
        "required": ["dataset_id", "conditions"],
        "properties": {
            "dataset_id": {"type": "string"},
            "conditions": {"type": "object"},
        },
    }
    assert task_spec.reservoir.max_size == 64
    assert task_spec.proposal_search.breadth == 64
    assert task_spec.proposal_search.name == "parallel_independent_requests"
    assert task_spec.proposal_search.parameters == {"max_workers": 64}
    assert task_spec.metadata["reservoir_size"] == 64
    assert task_spec.metadata["proposal_max_workers"] == 64
    assert task_spec.metadata["proposal_transport"] == "openai_chat_completions_single_choice"
    assert task_spec.metadata["sampling_mode"] == "local_concurrent_independent_requests"
    assert task_spec.surrogate.dimension == 47
    assert task_spec.acquisition.objective_names == ("reaction_score",)
    assert task_spec.acquisition.parameters == {
        "base_beta": 1.0,
        "confidence_delta": 0.1,
        "kernel": "factor_ard_categorical_rbf",
    }
    assert contract.proposal_provider == {
        "kind": "model_endpoint",
        "requires_endpoint_preflight": True,
        "supports_collection": True,
    }
    assert contract.metrics["reported"][0]["name"] == "reaction_score"
    assert contract.metrics["optimized"][0]["name"] == "reaction_score"


def test_mock_campaign_runs_one_shared_engine_round(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    monkeypatch.setenv("LDM_DATA_COLLECTION_ENABLED", "1")

    code = main(
        [
            "--mock",
            "--proposal-mode",
            "callable",
            "--out-dir",
            str(tmp_path),
            "--run-name",
            "mock-round",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    run_dir = Path(payload["run_dir"])
    counters = json.loads((run_dir / "budget.json").read_text(encoding="utf-8"))["counters"]
    assert code == 0
    assert payload["engine_summary"]["successful_evaluation_count"] == 1
    assert counters == {
        "benchmark_jobs": 1,
        "expensive_evaluation_attempts": 1,
        "external_evaluations": 1,
        "llm_requests": 0,
        "outer_iterations": 1,
        "proposal_attempts": 64,
        "selected_candidates": 1,
        "successful_evaluations": 1,
        "valid_search_candidates": 64,
    }
    for name in (
        "ldm_data/ldm_ir.jsonl",
        "result.json",
        "trajectory.csv",
        "search_manifest.json",
        "selection_record.json",
        "evaluation_manifest.json",
    ):
        assert (run_dir / name).is_file()
    evaluation_manifest = json.loads(
        (run_dir / "evaluation_manifest.json").read_text(encoding="utf-8")
    )
    assert evaluation_manifest["artifacts"] == {
        "result": "result.json",
        "trajectory": "trajectory.csv",
    }


def test_describe_ldm_task_accepts_a_custom_reservoir_size() -> None:
    args = parse_args(["--mock", "--reservoir-size", "7"])

    task_spec = describe_ldm_task(args)

    assert task_spec.reservoir.max_size == 7
    assert task_spec.proposal_search.breadth == 7
    assert task_spec.proposal_search.parameters == {"max_workers": 64}
    assert task_spec.response_spaces[0].schema["required"] == [
        "dataset_id",
        "conditions",
    ]
