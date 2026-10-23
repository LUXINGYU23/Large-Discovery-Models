"""Explicit remote Docker tests: COPPER_TEST_IMAGE enables these integration checks."""

import json
import os
from pathlib import Path
import pytest
from tasks.copper_inversion.core.execution import Execution
from tasks.copper_inversion.core.candidate import seal, Domain
from tasks.copper_inversion.core.interface_checks import InterfaceChecks
from tasks.copper_inversion.core.evaluator import Evaluator
from tasks.copper_inversion.core.data import load_problem, RESOURCES
from ldm_tts.contracts import RawProposal

pytestmark = pytest.mark.skipif(
    not os.getenv("COPPER_TEST_IMAGE"), reason="requires remote Docker image"
)


@pytest.mark.parametrize(
    "case", ["drillholes_sketch_2d", "drillholes_3d", "drillholes_outcrops_3d"]
)
def test_real_seed_isolation_and_replay(tmp_path, case):
    calls = []
    e = Execution(
        tmp_path / "jobs",
        os.environ["COPPER_TEST_IMAGE"],
        lambda a, **k: calls.append((a, k)),
    )
    payload = seal(tmp_path, (RESOURCES / "seeds/solver.py").read_bytes())
    InterfaceChecks(tmp_path, e, case)(payload)
    candidate = Domain(tmp_path).admit(RawProposal(payload, "test"))
    evaluator = Evaluator(tmp_path, e, load_problem(case), 60)
    result = evaluator.evaluate(candidate)
    assert result.succeeded, result.error
    assert result.metrics["loss_total"] > 0
    preview = json.loads((tmp_path / result.artifacts["model_slices"]).read_text())
    assert preview["seed"] == 47
    assert len(preview["slices"]) == (1 if case.endswith("2d") else 3)
    from types import SimpleNamespace
    from tasks.copper_inversion.core.harness import serialize_history
    observation = SimpleNamespace(canonical_key=candidate.canonical_key, round_idx=0,
                                  candidate=candidate, evaluation=result, metrics=result.metrics)
    assert serialize_history(tmp_path, [observation])[0]["model_slices"] == preview
    count = len(calls)
    assert evaluator.evaluate(candidate).to_dict() == result.to_dict()
    assert len(calls) == count
    scorer = list((tmp_path / "jobs").glob("*/receipt.json"))[-1]
    assert scorer.exists()


@pytest.mark.parametrize(
    "body",
    [
        'return np.full(problem["shape"],np.nan)',
        'return np.full(problem["shape"],np.inf)',
        'return np.ones(problem["shape"])',
        'return -np.ones(problem["shape"])',
        'return np.zeros((2,3))',
        'return problem["initial_level_set"].astype(complex)',
        'return problem["initial_level_set"].tolist()',
        'problem["initial_level_set"].setflags(write=True); problem["initial_level_set"][0,0] += 1; return problem["initial_level_set"].copy()',
        'problem["observations"][0]["weight"] = 99; return problem["initial_level_set"].copy()',
        'problem["observations"][0]["mask"].setflags(write=True); return problem["initial_level_set"].copy()',
        'return np.ma.array(problem["initial_level_set"], mask=False)',
        "import os; os._exit(0)",
        'open("/input/problem.json","w").write("forged"); return problem["initial_level_set"]',
        'import os; os.symlink("/input/problem.npz","/output/model.npy"); os._exit(0)',
    ],
)
def test_rejects_bad_solver(tmp_path, body):
    e = Execution(tmp_path / "jobs", os.environ["COPPER_TEST_IMAGE"])
    source = (
        "import numpy as np\ndef solve(problem,budget,seed):\n    " + body + "\n"
    ).encode()
    payload = seal(tmp_path, source)
    with pytest.raises((ValueError, OSError)):
        InterfaceChecks(tmp_path, e, "drillholes_sketch_2d")(payload)


def test_import_exit_cannot_pass_interface(tmp_path):
    execution = Execution(tmp_path / "jobs", os.environ["COPPER_TEST_IMAGE"])
    payload = seal(tmp_path, b"import os\nos._exit(0)\ndef solve(problem,budget,seed): pass\n")
    with pytest.raises((ValueError, OSError)):
        InterfaceChecks(tmp_path, execution, "drillholes_3d")(payload)


def test_timeout_and_forged_score(tmp_path):
    from tasks.copper_inversion.core.data import save_problem
    from ldm_tts.engine.run_store import atomic_json_write

    e = Execution(tmp_path / "jobs", os.environ["COPPER_TEST_IMAGE"])
    p = load_problem("drillholes_sketch_2d", fixture=True)
    inputs = tmp_path / "input"
    save_problem(p, inputs)
    atomic_json_write(
        inputs / "budget.json",
        dict(seed=47, wall_time_seconds=1, cpu_count=1, memory_mb=2048),
    )
    payload = seal(tmp_path, b"def solve(problem,budget,seed):\n    while True: pass\n")
    _, r = e.run(
        "timeout",
        phase="solver",
        inputs=inputs,
        source=(tmp_path / payload["source_path"]).parent,
        timeout=1,
    )
    assert r["reason"] == "timeout"
    source = b'def solve(problem,budget,seed):\n    open("/output/result.json","w").write("{\\"loss_total\\": -999}")\n    return problem["initial_level_set"].copy()\n'
    payload = seal(tmp_path, source)
    candidate = Domain(tmp_path).admit(RawProposal(payload, "test"))
    evaluator = Evaluator(tmp_path, e, p, 10)
    result = evaluator.evaluate(candidate)
    assert result.succeeded and result.metrics["loss_total"] > 0
    # Simulate interruption after scorer completion, before candidate result publication.
    receipt = (
        tmp_path
        / "evaluations"
        / evaluator.evaluation_attempt_usage_key(candidate)
        / "result.json"
    )
    receipt.unlink()
    replay = evaluator.evaluate(candidate)
    assert replay.metrics == result.metrics


def test_worker_mount_excludes_task_assets_and_runs(tmp_path):
    e = Execution(tmp_path / 'jobs', os.environ['COPPER_TEST_IMAGE'])
    source = b'''from pathlib import Path
def solve(problem, budget, seed):
    assert not Path('/code/tasks/copper_inversion/resources').exists()
    assert not Path('/code/tasks/copper_inversion/runs').exists()
    return problem['initial_level_set'].copy()
'''
    payload = seal(tmp_path, source)
    InterfaceChecks(tmp_path, e, 'drillholes_sketch_2d')(payload)
    candidate = Domain(tmp_path).admit(RawProposal(payload, 'test'))
    result = Evaluator(tmp_path, e, load_problem('drillholes_sketch_2d'), 60).evaluate(candidate)
    assert result.succeeded
