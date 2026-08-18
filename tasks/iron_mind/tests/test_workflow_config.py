"""Configuration-level contracts for Iron Mind workflow entry points."""

from __future__ import annotations

import json
from pathlib import Path

from ldm_tts.cli.runner import build_plan, load_config
from ldm_tts.registration.experiment import load_experiment_contract


TASK_ROOT = Path(__file__).resolve().parents[1]


def test_real_tiny_config_matches_the_locked_experiment_profile() -> None:
    config_path = TASK_ROOT.parents[1] / "config" / "iron_mind" / "real_tiny.yaml"
    config = load_config(config_path)
    plan = build_plan(config, config_path)
    contract = load_experiment_contract(TASK_ROOT / "experiment.json")
    profile = contract.profile("real_tiny")

    assert plan["contract_profile"] == "real_tiny"
    assert profile.budget == {
        "outer_iterations": 1,
        "llm_requests": 1,
        "proposal_attempts": 1,
        "valid_search_candidates": 4,
        "selected_candidates": 1,
        "external_evaluations": 1,
        "expensive_evaluation_attempts": 1,
        "successful_evaluations": 1,
        "benchmark_jobs": 1,
    }
    assert config["args"]["proposal-mode"] == "openai"
    assert config["args"]["dataset-id"] == "buchwald_hartwig"
    assert config["args"]["reservoir-size"] == 4
    assert config["args"]["evaluations-per-round"] == 1
    assert config["args"]["acquisition-beta"] == 1.0
    assert (
        config["args"]["qualification-input"]
        == "tasks/iron_mind/resources/qualification_input.json"
    )
    assert config["env"] == {"LDM_DATA_COLLECTION_ENABLED": "1"}
    assert "LDM_LLM_API_KEY" not in json.dumps(config)


def test_mock_config_enables_collection_on_the_shared_ucb_path() -> None:
    config_path = TASK_ROOT.parents[1] / "config" / "iron_mind" / "mock.yaml"
    config = load_config(config_path)
    plan = build_plan(config, config_path)

    assert config["algorithm"] == "ucb"
    assert config["mode"] == "mock"
    assert config["args"] == {
        "mock": True,
        "proposal-mode": "callable",
        "dataset-id": "buchwald_hartwig",
        "iterations": 1,
        "reservoir-size": 4,
        "evaluations-per-round": 1,
        "acquisition-beta": 1.0,
    }
    assert config["env"] == {"LDM_DATA_COLLECTION_ENABLED": "1"}
    assert "--mock" in plan["argv"]
    assert "--proposal-mode" in plan["argv"]
