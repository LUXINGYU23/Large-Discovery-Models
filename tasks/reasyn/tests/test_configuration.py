"""Public runner and pilot plans resolve to supported task behavior."""

from pathlib import Path

import pytest

from ldm_tts.cli.runner import build_plan, load_config
from ldm_tts.pilot_evaluation.config import load_pilot_evaluation_spec
from ldm_tts.pilot_evaluation.execution import _child_plan, _select_runs
from tasks.reasyn.core.workflow import parse_args
from tasks.reasyn.ldm_task.dependencies import check_dependencies


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIGS = sorted((REPO_ROOT / "config/reasyn").glob("*.yaml"))


@pytest.mark.parametrize("path", CONFIGS, ids=lambda path: path.stem)
def test_shipped_configuration_uses_supported_cli_and_contract(path, monkeypatch, tmp_path):
    monkeypatch.setenv("REASYN_BO_TARGETS", str(tmp_path / "targets.smi"))
    plan = build_plan(load_config(path), path)
    args = parse_args(plan["argv"])
    assert 1 <= args.evaluations_per_round <= args.bo_pool_size
    assert args.proposal_batch_size > 0
    if "harness" in args.search_method:
        assert args.proposal_mode == "harness"
        if args.search_method == "harness":
            assert args.harness_sessions == 1
            assert args.reservoir_size == args.evaluations_per_round
        else:
            assert args.harness_sessions >= 2


def test_pilot_plans_use_shared_seed_initialization_and_support_every_method(tmp_path, monkeypatch):
    pool = tmp_path / "targets.smi"
    pool.write_text("CCO\nCCN\nCCC\n")
    monkeypatch.setenv("REASYN_BO_TARGETS", str(pool))
    monkeypatch.setenv("REASYN_RUNS_ROOT", str(tmp_path / "runs"))
    spec = load_pilot_evaluation_spec(REPO_ROOT / "config/pilot_evaluation/reasyn.yaml")
    runs = _select_runs(spec, cases=None, methods=None, seeds=None)
    assert len(runs) == 18
    assert set(spec.methods) == {"ldm", "bo", "llm", "harness", "ldm_harness", "ldm_harness_compiled"}
    for run in runs:
        plan = _child_plan(spec, load_config(spec.base_config), run, resume=False)
        args = parse_args(plan["argv"])
        assert args.seed == run.seed
        assert args.run_name == f"seed_{run.seed}"
        assert args.initialization_mode == "shared_start"
        assert args.iterations == spec.optimization_rounds + 1 == 6
        assert args.evaluations_per_round == 2
        assert args.bo_targets_file == pool
        assert args.search_method == run.method


def _checks(argv, monkeypatch, **env):
    # Exercise task dependency choices without requiring released GPU assets.
    monkeypatch.setattr("tasks.reasyn.ldm_task.dependencies.importlib.util.find_spec", lambda name: object())
    for name in ("LLM_BASE_URL", "LLM_MODEL_NAME", "LDM_LLM_URL", "LDM_LLM_MODEL", "OPENAI_BASE_URL", "REASYN_BO_TARGETS"):
        monkeypatch.delenv(name, raising=False)
    return {item.name: item for item in check_dependencies({
        "task": "reasyn", "mode": "real", "argv": argv,
        "cwd": str(REPO_ROOT), "env_overrides": env,
    }, include_optional=False)}


def test_bo_dependency_check_requires_a_real_pool_but_no_provider(tmp_path, monkeypatch):
    pool = tmp_path / "targets.smi"
    pool.write_text("CCO\nCCN\n")
    checks = _checks(["--search-method", "bo", "--benchmark", "tdc"], monkeypatch, REASYN_BO_TARGETS=str(pool))
    assert checks["bo_target_pool"].status == "ok"
    assert "proposal_endpoint" not in checks
    missing = _checks(["--search-method", "bo", "--benchmark", "tdc"], monkeypatch)
    assert missing["bo_target_pool"].status == "fail"


def test_harness_dependencies_check_runtime_resources_and_mcp(tmp_path, monkeypatch):
    monkeypatch.setattr("tasks.reasyn.ldm_task.dependencies.shutil.which", lambda name: None)
    checks = _checks([
        "--search-method", "ldm_harness_compiled", "--proposal-mode", "none",
        "--harness-mcp-config", str(tmp_path / "missing.json"),
    ], monkeypatch, LLM_BASE_URL="http://localhost:8000/v1", LLM_MODEL_NAME="test-model")
    assert checks["proposal_endpoint"].status == "ok"
    assert checks["harness_container_runtime"].status == "fail"
    assert checks["harness_resources"].status == "ok"
    assert checks["harness_mcp_config"].status == "fail"


def test_mock_worker_check_is_lightweight_but_mock_harness_keeps_provider_boundary(monkeypatch):
    light = _checks(["--mock"], monkeypatch)
    assert set(light) == {"mock"}
    harness = _checks(["--mock", "--search-method", "harness"], monkeypatch)
    assert harness["proposal_endpoint"].status == "fail"
    assert "checkpoint_0" not in harness
