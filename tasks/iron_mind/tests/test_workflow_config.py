"""Configuration contracts for the public Iron Mind entry points."""

from __future__ import annotations

import json
from pathlib import Path

from ldm_tts.cli.runner import build_plan, load_config
from ldm_tts.registration.experiment import load_experiment_contract


TASK_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = TASK_ROOT.parents[1] / "config" / "iron_mind"


def test_real_smoke_config_is_portable_and_profile_locked(
    tmp_path: Path, monkeypatch
) -> None:
    data_root = tmp_path / "data"
    runs_root = tmp_path / "runs"
    monkeypatch.setenv("IRON_MIND_DATA_ROOT", str(data_root))
    monkeypatch.setenv("IRON_MIND_RUNS_ROOT", str(runs_root))
    config_path = CONFIG_ROOT / "real_smoke.yaml"
    config = load_config(config_path)
    plan = build_plan(config, config_path)
    contract = load_experiment_contract(TASK_ROOT / "experiment.json")
    profile = contract.profile("ldm_official_smoke")

    assert contract.qualification == "qualified"
    assert plan["contract_profile"] == "ldm_official_smoke"
    assert profile.budget["external_evaluations"] == 1
    assert config["args"]["proposal-mode"] == "openai"
    assert config["args"]["proposal-samples"] == 64
    assert config["args"]["bo-pool-size"] == 32
    assert config["args"]["proposal-max-workers"] == 64
    assert config["args"]["evaluations-per-round"] == 1
    assert config["args"]["llm-temperature"] == 0.7
    assert config["args"]["llm-max-tokens"] == 512
    assert config["args"]["prompt-policy"] == "portfolio_v1"
    assert str(data_root) in plan["argv"]
    out_dir = Path(plan["argv"][plan["argv"].index("--out-dir") + 1])
    assert out_dir == runs_root / "smoke"
    assert config["args"]["llm-url"] is None
    assert config["args"]["llm-model-name"] is None
    assert config["args"]["api-key"] is None
    assert "--api-key" not in plan["argv"]


def test_complete_ldm_profile_and_suites_lock_the_official_budget(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("IRON_MIND_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("IRON_MIND_RUNS_ROOT", str(tmp_path / "runs"))
    contract = load_experiment_contract(TASK_ROOT / "experiment.json")
    profile = contract.profile("ldm_official_20")
    paper = load_config(CONFIG_ROOT / "paper_v2_ldm_20x20.yaml")
    public = load_config(CONFIG_ROOT / "public_union_ldm_20x20.yaml")

    assert profile.budget["external_evaluations"] == 20
    assert "llm_requests" not in profile.budget
    assert "proposal_attempts" not in profile.budget
    assert contract.budget["llm_requests"] == 1280
    assert contract.budget["proposal_attempts"] == 1280
    assert "valid_search_candidates" not in profile.budget
    assert contract.budget["valid_search_candidates"] == 1280
    assert len(paper["experiments"]) == 6 * 20
    assert len(public["experiments"]) == 7 * 20
    assert all(entry["config"].startswith("ldm_20_") for entry in public["experiments"])

    dataset_configs = sorted(CONFIG_ROOT.glob("ldm_20_*.yaml"))
    assert len(dataset_configs) == 7
    for config_path in dataset_configs:
        config = load_config(config_path)
        plan = build_plan(config, config_path)
        serialized = json.dumps(config)
        assert plan["contract_profile"] == "ldm_official_20"
        assert config["args"]["iterations"] == 20
        assert config["args"]["proposal-samples"] == 64
        assert config["args"]["bo-pool-size"] == 32
        assert config["args"]["alpha"] == 1.0
        assert config["args"]["eta"] == 1.0
        assert config["args"]["proposal-max-workers"] == 64
        assert config["args"]["evaluations-per-round"] == 1
        assert config["args"]["llm-max-tokens"] == 512
        assert config["args"]["prompt-policy"] == "portfolio_v1"
        assert config["args"]["llm-url"] is None
        assert config["args"]["llm-model-name"] is None
        assert config["args"]["api-key"] is None
        assert "LLM_API_KEY" not in serialized
        assert "/mnt/data1/" not in serialized


def test_mock_config_enables_collection_on_the_shared_ucb_path() -> None:
    config_path = CONFIG_ROOT / "mock.yaml"
    config = load_config(config_path)
    plan = build_plan(config, config_path)

    assert config["algorithm"] == "ldm_tilted_ucb"
    assert config["mode"] == "mock"
    assert config["args"] == {
        "mock": True,
        "proposal-mode": "callable",
        "dataset-id": "buchwald_hartwig",
        "iterations": 1,
        "proposal-samples": 64,
        "bo-pool-size": 32,
        "proposal-max-workers": 64,
        "evaluations-per-round": 1,
        "acquisition-beta": 1.0,
        "alpha": 1.0,
        "eta": 1.0,
        "z-clip": 5.0,
        "prompt-policy": "portfolio_v1",
    }
    assert config["env"] == {"LDM_DATA_COLLECTION_ENABLED": "1"}
    assert "--mock" in plan["argv"]
    assert "--proposal-mode" in plan["argv"]


def test_prompt_baseline_smoke_has_its_own_locked_ablation_profile(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("IRON_MIND_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("IRON_MIND_RUNS_ROOT", str(tmp_path / "runs"))
    config_path = CONFIG_ROOT / "prompt_baseline_smoke.yaml"
    config = load_config(config_path)
    plan = build_plan(config, config_path)
    contract = load_experiment_contract(TASK_ROOT / "experiment.json")
    profile = contract.profile("ldm_prompt_baseline_smoke")

    assert plan["contract_profile"] == "ldm_prompt_baseline_smoke"
    assert profile.budget["external_evaluations"] == 1
    assert config["args"]["prompt-policy"] == "baseline_v1"
    assert "baseline_v1" in plan["argv"]


def test_profile_allows_custom_proposal_and_bo_pool_sizes(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("IRON_MIND_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("IRON_MIND_RUNS_ROOT", str(tmp_path / "runs"))
    config_path = CONFIG_ROOT / "real_smoke.yaml"
    config = load_config(config_path)
    config["args"] = {
        **config["args"],
        "proposal-samples": 8,
        "bo-pool-size": 4,
    }

    plan = build_plan(config, config_path)

    assert plan["contract_profile"] == "ldm_official_smoke"
    assert "--proposal-samples" in plan["argv"]
    assert "--bo-pool-size" in plan["argv"]
