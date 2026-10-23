"""Remote-only faults at actual Docker and shared campaign publication boundaries."""

import json
import os
from pathlib import Path

import pytest

from ldm_tts.contracts.evaluation import EvaluationPaused
from ldm_tts.engine.run_store import CampaignRuntime
from tasks.copper_inversion.core import execution, harness_runtime
from tasks.copper_inversion.core.evaluator import Evaluator
from tasks.copper_inversion.core.mock import MockClient
from tasks.copper_inversion.core.workflow import main

pytestmark = pytest.mark.skipif(
    not os.getenv("COPPER_TEST_IMAGE"), reason="requires remote Docker image"
)


@pytest.mark.parametrize("case", ["drillholes_3d", "drillholes_outcrops_3d"])
@pytest.mark.parametrize("boundary", ["solver", "scorer", "evaluator", "publication"])
def test_campaign_recovers_without_reexecution(tmp_path, monkeypatch, case, boundary):
    """Real admission/solver/scorer, fake researcher; interrupt the researched candidate."""
    starts, turns = [], []
    original_docker = execution.docker
    original_run = execution.Execution.run
    original_evaluate = Evaluator.evaluate
    original_checkpoint = CampaignRuntime.checkpoint
    original_turn = MockClient.run_turn
    completed = {"solver": 0, "scorer": 0, "evaluator": 0}
    interrupted = False

    def docker(*args, **kwargs):
        if args[0] == "start":
            starts.append(args[1])
        return original_docker(*args, **kwargs)

    def stop(phase):
        nonlocal interrupted
        completed[phase] += 1
        if boundary == phase and completed[phase] == 2 and not interrupted:
            interrupted = True
            raise EvaluationPaused("injected after " + phase)

    def run(self, *args, **kwargs):
        result = original_run(self, *args, **kwargs)
        if kwargs["phase"] in completed:
            stop(kwargs["phase"])
        return result

    def evaluate(self, candidate):
        result = original_evaluate(self, candidate)
        stop("evaluator")
        return result

    def checkpoint(self, state):
        nonlocal interrupted
        if boundary == "publication" and len(state.get("observations", [])) == 2 and not interrupted:
            interrupted = True
            raise EvaluationPaused("injected after publication before checkpoint")
        return original_checkpoint(self, state)

    def turn(self, requested, *args, **kwargs):
        turns.extend(t.turn_id for t in requested)
        return original_turn(self, requested, *args, **kwargs)

    monkeypatch.setattr(execution, "docker", docker)
    monkeypatch.setattr(execution.Execution, "run", run)
    monkeypatch.setattr(Evaluator, "evaluate", evaluate)
    monkeypatch.setattr(CampaignRuntime, "checkpoint", checkpoint)
    monkeypatch.setattr(MockClient, "run_turn", turn)
    monkeypatch.setattr(
        harness_runtime, "open_client",
        lambda args, root, problem, stack, account: MockClient(root / "harness"),
    )
    image = os.environ["COPPER_TEST_IMAGE"]
    args = ["--case-id", case, "--iterations", "1", "--out-dir", str(tmp_path),
            "--evaluator-image", image, "--sidecar-image", image]
    with pytest.raises(EvaluationPaused, match="injected"):
        main(args)
    assert interrupted
    assert main([*args, "--resume"]) == 0
    counters = json.loads((tmp_path / "budget.json").read_text())["counters"]
    assert turns == ["researcher_round_1"]
    assert len(starts) == len(set(starts)) == 8  # four interface + two solver + two scorer
    for name, count in {"interface_jobs": 4, "solver_jobs": 2, "scorer_jobs": 2,
                        "benchmark_jobs": 4, "harness_turns": 1,
                        "selected_candidates": 2, "external_evaluations": 2,
                        "expensive_evaluation_attempts": 2, "successful_evaluations": 2}.items():
        assert counters[name] == count, (name, counters)
    assert main([*args, "--resume"]) == 0
    assert json.loads((tmp_path / "budget.json").read_text())["counters"] == counters
    assert len(starts) == 8 and turns == ["researcher_round_1"]


def test_timeout_before_receipt_preserves_reason_and_runtime(tmp_path, monkeypatch):
    from tasks.copper_inversion.core.candidate import seal
    from tasks.copper_inversion.core.data import load_problem, save_problem

    inputs = tmp_path / "inputs"
    save_problem(load_problem("drillholes_3d", fixture=True), inputs)
    execution.atomic_json_write(inputs / "budget.json", {
        "seed": 47, "wall_time_seconds": 1, "cpu_count": 1, "memory_mb": 2048,
    })
    payload = seal(tmp_path, b"def solve(problem, budget, seed):\n    while True: pass\n")
    runner = execution.Execution(tmp_path / "jobs", os.environ["COPPER_TEST_IMAGE"])
    original_write = execution.atomic_json_write
    interrupted = False

    def write(path, data):
        nonlocal interrupted
        if Path(path).name == "receipt.json" and not interrupted:
            interrupted = True
            raise EvaluationPaused("before terminal receipt")
        return original_write(path, data)

    monkeypatch.setattr(execution, "atomic_json_write", write)
    kwargs = dict(phase="solver", inputs=inputs, timeout=1,
                  source=(tmp_path / payload["source_path"]).parent)
    with pytest.raises(EvaluationPaused, match="terminal receipt"):
        runner.run("timeout-recovery", **kwargs)
    job = next((tmp_path / "jobs").glob("*/job.json"))
    record = json.loads(job.read_text())
    assert record["stop_reason"] == "timeout"
    # Simulate long controller downtime without sleeping or changing Docker time.
    record["start_time"] -= 3600
    original_write(job, record)
    _, receipt = runner.run("timeout-recovery", **kwargs)
    assert receipt["reason"] == "timeout"
    assert 0 < receipt["wall_seconds"] < 30
    assert runner.run("timeout-recovery", **kwargs)[1] == receipt


@pytest.mark.parametrize("corruption", ["missing", "json", "nonfinite"])
def test_scorer_receipt_fault_is_infrastructure_pause(tmp_path, monkeypatch, corruption):
    from ldm_tts.contracts import RawProposal
    from tasks.copper_inversion.core.candidate import seal, Domain
    from tasks.copper_inversion.core.data import load_problem, RESOURCES

    runner = execution.Execution(tmp_path / "jobs", os.environ["COPPER_TEST_IMAGE"])
    payload = seal(tmp_path, (RESOURCES / "seeds/solver.py").read_bytes())
    candidate = Domain(tmp_path).admit(RawProposal(payload, "test"))
    original_run = runner.run

    def corrupt(*args, **kwargs):
        root, result = original_run(*args, **kwargs)
        if kwargs["phase"] == "scorer":
            receipt = root / "output/result.json"
            if corruption == "missing":
                receipt.unlink()
            else:
                receipt.write_text("{" if corruption == "json" else
                                   '{"status":"succeeded","metrics":{"loss_total":NaN}}')
        return root, result

    monkeypatch.setattr(runner, "run", corrupt)
    evaluator = Evaluator(tmp_path, runner, load_problem("drillholes_3d"), 10)
    with pytest.raises(EvaluationPaused, match="scorer receipt"):
        evaluator.evaluate(candidate)
    assert not list((tmp_path / "evaluations").glob("*/result.json"))


@pytest.mark.parametrize("fault", ["oom", "child", "accounting"])
def test_resources_and_cleanup_on_recovery(tmp_path, fault):
    from tasks.copper_inversion.core.candidate import seal
    from tasks.copper_inversion.core.data import load_problem, save_problem

    inputs = tmp_path / "inputs"
    save_problem(load_problem("drillholes_3d", fixture=True), inputs)
    execution.atomic_json_write(inputs / "budget.json", {
        "seed": 47, "wall_time_seconds": 3, "cpu_count": 1, "memory_mb": 128,
    })
    bodies = {
        "oom": "data = bytearray(512 * 1024 * 1024)\n    return problem['initial_level_set']",
        "child": "import subprocess, sys, time\n    subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n    time.sleep(60)",
        "accounting": "return problem['initial_level_set'].copy()",
    }
    payload = seal(tmp_path, ("def solve(problem, budget, seed):\n    " + bodies[fault] + "\n").encode())
    failed = False

    def account(*args, **kwargs):
        nonlocal failed
        if fault == "accounting" and not failed:
            failed = True
            raise EvaluationPaused("after receipt before accounting")

    runner = execution.Execution(tmp_path / "jobs", os.environ["COPPER_TEST_IMAGE"], account)
    kwargs = dict(phase="solver", inputs=inputs, timeout=3, memory_mb=128,
                  source=(tmp_path / payload["source_path"]).parent)
    if fault == "accounting":
        with pytest.raises(EvaluationPaused, match="before accounting"):
            runner.run(fault, **kwargs)
    _, result = runner.run(fault, **kwargs)
    if fault == "oom":
        assert result["oom"] and result["exit_code"] != 0
    elif fault == "child":
        assert result["reason"] == "timeout"
    else:
        assert result["exit_code"] == 0
    assert execution.docker("inspect", result["container"], check=False).returncode != 0
    assert runner.run(fault, **kwargs)[1] == result
