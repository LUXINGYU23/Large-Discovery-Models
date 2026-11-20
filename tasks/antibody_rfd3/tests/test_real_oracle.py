import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from tasks.antibody.core.absolut_oracle import AbsolutOracle
from tasks.antibody_rfd3.core.synchronous.evaluator import IsolatedAbsolutEvaluator
from tasks.antibody_rfd3.core.synchronous.workflow import Expander, parse_args, run
from tasks.antibody_rfd3.ldm_task.dependencies import check_dependencies


def test_resource_preflight_requires_cache(tmp_path, monkeypatch):
    binary = tmp_path / "src/bin/Absolut"
    binary.parent.mkdir(parents=True)
    binary.write_text("fixture")
    name = "abc-10-11-defStructures.txt"
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: SimpleNamespace(
        stdout="Pre-calculated structures are in " + name, stderr=""))
    oracle = AbsolutOracle(tmp_path, [tmp_path], tmp_path / "runs")
    with pytest.raises(FileNotFoundError, match="precomputation"):
        oracle.preflight("HER2", require_cache=True)
    (tmp_path / name).write_text("fixture")
    with pytest.raises(FileNotFoundError, match="interaction cache"):
        oracle.preflight("HER2", require_cache=True)
    (tmp_path / "abc+profileCompact.txt").write_text("fixture")
    assert oracle.preflight("HER2", require_cache=True)[0] == "1N8Z_C"
    assert not oracle.run_root.exists()


def test_dependency_hook_parses_runner_timeout_and_resources(tmp_path, monkeypatch):
    seen = []
    def preflight(oracle, target, require_cache=False):
        seen.append((oracle.timeout, oracle.resources, target, require_cache))
    monkeypatch.setattr(AbsolutOracle, "preflight", preflight)
    binary = tmp_path / "src/bin/Absolut"
    binary.parent.mkdir(parents=True)
    binary.touch(mode=0o755)
    plan = {"task": "antibody_rfd3", "cwd": str(tmp_path), "argv": [
        "--method", "fixed", "--oracle", "absolut", "--antigen", "HER2",
        "--absolut-path", str(tmp_path), "--absolut-timeout", "600",
        "--absolut-resources", str(tmp_path / "a"), "--absolut-resources", str(tmp_path / "b")]}
    results = check_dependencies(plan)
    assert next(r for r in results if r.name == "Antigen resources").status == "ok"
    assert seen == [(600, [tmp_path / "a", tmp_path / "b"], "HER2", True)]


def test_real_adapter_provenance_timeout_and_admission(tmp_path, monkeypatch):
    monkeypatch.setattr(AbsolutOracle, "preflight", lambda *a, **k: None)
    args = parse_args(["--oracle", "absolut", "--antigen", "HER2", "--absolut-path", str(tmp_path)])
    adapter = IsolatedAbsolutEvaluator(args, tmp_path / "campaign")
    calls = []
    def evaluate(oracle, sequences, target):
        calls.append(sequences)
        assert target == "1N8Z_C"
        return {"run_dir": str(oracle.run_root / "eval"), "binary_sha256": "abc",
                "results": [{"sequence": sequences[0], "absolut_energy": -50.0}]}
    monkeypatch.setattr(AbsolutOracle, "evaluate", evaluate)
    candidate = SimpleNamespace(candidate_id="candidate", payload={"sequence": "SNGKDGSPGVY"})
    result = adapter.evaluate(candidate)
    assert result.succeeded and result.metrics == {"absolut_energy": -50.0}
    assert result.metadata["binary_sha256"] == "abc"
    assert result.artifacts["result"].endswith("result.json")
    def timeout(*args):
        raise subprocess.TimeoutExpired("Absolut", 600)
    monkeypatch.setattr(AbsolutOracle, "evaluate", timeout)
    failure = adapter.evaluate(candidate)
    assert failure.status == "failed" and not failure.metrics
    assert failure.artifacts["oracle_root"] == result.artifacts["oracle_root"]
    candidate.payload["sequence"] += "A"
    invalid = adapter.evaluate(candidate)
    assert invalid.status == "failed" and "11 aa" in invalid.error
    assert len(calls) == 1


def test_expander_forwards_real_history(tmp_path):
    args = parse_args(["--oracle", "absolut", "--antigen", "PD-1"])
    worker = SimpleNamespace(evaluated=set(), remaining=lambda: {})
    expander = Expander(args, worker)
    artifact = str(tmp_path / "oracle/result.json")
    observation = SimpleNamespace(round_idx=0, candidate=SimpleNamespace(
        canonical_key="SNGKDGSPGVY", payload={"sequence": "SNGKDGSPGVY",
        "backbone_id": "job-00000", "design_id": "job-00000/0"}),
        evaluation=SimpleNamespace(succeeded=True, metrics={"absolut_energy": -70.0},
                                   artifacts={"result": artifact}))
    class Agent:
        def propose(self, round_idx, history, count):
            assert round_idx == 1 and count == 1
            assert history[0]["energy"] == -70.0
            assert history[0]["backbone_id"] == "job-00000"
            assert history[0]["result_artifacts"]["result"] == artifact
            return [{"sequence": "TSGGPGAPGTH"}]
    expander.agent = Agent()
    expander.expand(SimpleNamespace(round_idx=1, observations=[observation]))


def test_failed_oracle_consumes_attempt_without_success(tmp_path, monkeypatch):
    monkeypatch.setattr(AbsolutOracle, "preflight", lambda *a, **k: None)
    def timeout(*args):
        raise subprocess.TimeoutExpired("Absolut", 600)
    monkeypatch.setattr(AbsolutOracle, "evaluate", timeout)
    args = parse_args(["--mock", "--iterations", "1", "--antigen", "HER2",
                       "--absolut-path", str(tmp_path), "--out-dir", str(tmp_path / "run")])
    args.oracle = "absolut"
    assert run(args) == 1
    budget = json.loads((tmp_path / "run/budget.json").read_text())["counters"]
    assert budget["expensive_evaluation_attempts"] == 1
    assert budget.get("successful_evaluations", 0) == 0
    summary = json.loads((tmp_path / "run/summary.json").read_text())
    assert summary["failed_evaluation_count"] == 1
