"""Remote Docker: trusted lifetime counters survive transient worker allocation."""
import os
import pytest
from tasks.copper_inversion.core.execution import Execution, authenticated_resources
from tasks.copper_inversion.core.candidate import seal, Domain
from tasks.copper_inversion.core.evaluator import Evaluator
from tasks.copper_inversion.core.data import load_problem
from ldm_tts.contracts import RawProposal


def test_forged_counter_is_not_authenticated():
    assert authenticated_resources('COPPER_RESOURCE_V1 {"payload":"{}","mac":"' + '0'*64 + '"}', '1'*64) is None


@pytest.mark.skipif(not os.getenv("COPPER_TEST_IMAGE"), reason="remote Docker image required")
def test_worker_cannot_forge_supervisor_and_descendant_peak_is_retained(tmp_path):
    source = b'''def solve(problem,budget,seed):
    import os, signal
    assert os.getuid() == 65534 and os.getgid() == 65534
    assert 'COPPER_RESOURCE_KEY' not in os.environ
    assert int(next(line.split()[1] for line in open('/proc/self/status') if line.startswith('CapEff:')),16) == 0
    try:
        open('/proc/1/environ','rb').read()
    except PermissionError:
        pass
    else:
        raise AssertionError('worker can read supervisor environment')
    try:
        os.kill(1, 0)
    except PermissionError:
        pass
    else:
        raise AssertionError('worker can signal supervisor')
    print('COPPER_RESOURCE_V1 {"payload":"{}","mac":"forged"}', flush=True)
    pid=os.fork()
    if pid == 0:
        spike=bytearray(96*1024*1024)
        del spike
        os._exit(0)
    assert os.waitpid(pid,0)[1] == 0
    return problem['initial_level_set'].copy()
'''
    payload = seal(tmp_path, source)
    runner = Execution(tmp_path / "jobs", os.environ["COPPER_TEST_IMAGE"])
    evaluator = Evaluator(tmp_path, runner, load_problem("drillholes_3d", fixture=True), 10)
    result = evaluator.evaluate(Domain(tmp_path).admit(RawProposal(payload, "test")))
    assert result.succeeded, result.error
    counters = result.metadata["workload_resources"]["solver"]
    assert counters["workload_complete"] and counters["exit_code"] == 0
    if os.getenv("COPPER_REQUIRE_PEAK"):
        assert 96*1024*1024 <= counters["peak_memory_bytes"] < 512*1024*1024
        assert result.resource_usage["solver_peak_memory_bytes"] == counters["peak_memory_bytes"]
    assert counters["cpu_seconds_through_reap"] > 0


@pytest.mark.skipif(not os.getenv("COPPER_TEST_IMAGE"), reason="remote Docker image required")
def test_lifetime_counters_survive_controller_disconnect(tmp_path, monkeypatch):
    import json
    from tasks.copper_inversion.core import execution
    from tasks.copper_inversion.core.data import save_problem
    from ldm_tts.engine.run_store import atomic_json_write
    from ldm_tts.contracts.evaluation import EvaluationPaused

    inputs = tmp_path / "inputs"
    save_problem(load_problem("drillholes_3d", fixture=True), inputs)
    atomic_json_write(inputs / "budget.json", dict(seed=47, wall_time_seconds=2, cpu_count=1, memory_mb=512))
    source = seal(tmp_path, b'def solve(problem,budget,seed):\n    spike=bytearray(96*1024*1024)\n    del spike\n    while True: pass\n')
    runner = Execution(tmp_path / "jobs", os.environ["COPPER_TEST_IMAGE"])
    original = execution.docker
    def disconnect(*args, **kwargs):
        result = original(*args, **kwargs)
        if args[0] == "start":
            raise EvaluationPaused("controller disconnected after dispatch")
        return result
    monkeypatch.setattr(execution, "docker", disconnect)
    kwargs = dict(phase="solver", inputs=inputs, source=(tmp_path / source["source_path"]).parent,
                  timeout=2, memory_mb=512)
    with pytest.raises(EvaluationPaused, match="disconnected"):
        runner.run("offline", **kwargs)
    record = json.loads(next((tmp_path / "jobs").glob("*/job.json")).read_text())
    assert original("wait", record["name"]).stdout.strip() == "137"
    monkeypatch.setattr(execution, "docker", original)
    _, result = runner.run("offline", **kwargs)
    assert result["reason"] == "timeout"
    assert result["resource_sampling"]["samples"] == 0
    assert result["workload_resources"]["workload_complete"]
    if os.getenv("COPPER_REQUIRE_PEAK"):
        assert result["workload_resources"]["peak_memory_bytes"] >= 96*1024*1024
