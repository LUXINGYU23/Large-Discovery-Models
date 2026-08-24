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
    assert (
        task_spec.response_spaces[0].parser
        == "tasks.iron_mind.core.proposal_parsing:parse_reaction_response"
    )
    assert task_spec.reservoir.max_size == 64
    assert task_spec.proposal_search.breadth == 64
    assert task_spec.proposal_search.name == "parallel_independent_requests"
    assert task_spec.proposal_search.parameters == {"max_workers": 64}
    assert task_spec.metadata["proposal_samples"] == 64
    assert task_spec.metadata["bo_pool_size"] == 32
    assert task_spec.metadata["proposal_max_workers"] == 64
    assert task_spec.metadata["proposal_transport"] == "openai_chat_completions_single_choice"
    assert task_spec.metadata["sampling_mode"] == "local_concurrent_independent_requests"
    assert task_spec.metadata["prompt_policy"] == "portfolio_v1"
    assert task_spec.metadata["prompt_version"] == "portfolio_v1"
    assert task_spec.surrogate.dimension == 47
    assert task_spec.acquisition.objective_names == ("reaction_score",)
    assert task_spec.acquisition.name == "ucb_tilted"
    assert task_spec.acquisition.score_direction == "sample"
    assert task_spec.acquisition.parameters == {
        "base_acquisition": "ucb",
        "base_acquisition_parameters": {
            "base_beta": 1.0,
            "confidence_delta": 0.1,
            "kernel": "factor_ard_categorical_rbf",
        },
        "base_measure": "empirical_proposal_frequency",
        "alpha_base_measure": 1.0,
        "eta_acquisition_tilt": 3.0,
        "normalization": "robust_z",
        "z_clip": 5.0,
        "sampling": "gumbel_top_k_without_replacement",
        "seed": 0,
        "pool_size": 32,
        "proposal_sample_count": 64,
    }
    assert contract.proposal_provider == {
        "kind": "hybrid",
        "requires_endpoint_preflight": False,
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
    events = (run_dir / "events.jsonl").read_text(encoding="utf-8")
    assert '"prompt_policy": "portfolio_v1"' in events
    assert '"selection_mode": "acquisition_tilted_sampling"' in events
    selection = json.loads((run_dir / "selection_record.json").read_text(encoding="utf-8"))
    selection_payload = selection["selections"][0]["payload"]
    selection_metadata = selection_payload["metadata"]
    assert len(selection_payload["predictions"]) == 32
    assert selection_metadata["proposal_samples_requested"] == 64
    assert selection_metadata["valid_proposal_occurrences"] == 64
    assert selection_metadata["unique_candidates_admitted"] == 64
    assert selection_metadata["bo_pool_size"] == 32
    assert len(selection_metadata["proposal_base_measure"]) == 64
    assert sum(
        item["proposal_q0_base_mass"]
        for item in selection_metadata["proposal_base_measure"]
    ) == 1.0


def test_describe_ldm_task_separates_proposal_samples_from_the_bo_pool() -> None:
    args = parse_args(["--mock", "--proposal-samples", "7", "--bo-pool-size", "3"])

    task_spec = describe_ldm_task(args)

    assert task_spec.reservoir.max_size == 7
    assert task_spec.proposal_search.breadth == 7
    assert task_spec.proposal_search.parameters == {"max_workers": 64}
    assert task_spec.acquisition.parameters["pool_size"] == 3
    assert task_spec.response_spaces[0].schema["required"] == [
        "dataset_id",
        "conditions",
    ]


def test_task_spec_declares_plain_bo_and_direct_llm_without_changing_the_engine() -> None:
    bo = describe_ldm_task(parse_args(["--mock", "--search-method", "bo", "--proposal-mode", "none"]))
    llm = describe_ldm_task(parse_args(["--mock", "--search-method", "llm"]))

    assert bo.proposal_search.name == "full_finite_domain_bo"
    assert bo.acquisition.name == "ucb"
    assert llm.proposal_search.name == "parallel_independent_direct_llm"
    assert llm.surrogate.kind == "none"
