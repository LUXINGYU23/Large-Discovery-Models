"""Evaluator lifecycle: process-tree timeouts, durable receipts, and coverage."""

import copy
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from ldm_tts.contracts import Candidate
from ldm_tts.contracts.evaluation import EVALUATION_ATTEMPT_RECEIPT_KEY
from tasks.researchgym.core import source as source_module
from tasks.researchgym.core.cases import CaseSpec, load_case, render
from tasks.researchgym.core.clock import CampaignClock
from tasks.researchgym.core.evaluator import ResearchGymEvaluator
from tasks.researchgym.core.execution import JobCommand, ProcessTreeRunner
from tasks.researchgym.core.source import SourceMismatchError, UpstreamCase


def test_timeout_terminates_background_descendants(tmp_path):
    marker = tmp_path / "late_marker"
    job = JobCommand("slow", ("bash", "-c", f"(sleep 1.5; touch {marker}) & sleep 30"))
    started = time.monotonic()
    outcome = ProcessTreeRunner(grace_seconds=1).run([job], cwd=tmp_path, log_dir=tmp_path / "logs",
                                                     deadline=time.monotonic() + 0.5)[0]
    assert outcome.status == "timed_out" and outcome.group_terminated
    assert time.monotonic() - started < 5
    time.sleep(2.0)
    assert not marker.exists()


def test_leader_exit_reaps_orphaned_group_and_unstarted_jobs_are_reported(tmp_path):
    marker = tmp_path / "orphan_marker"
    jobs = [JobCommand("leader", ("bash", "-c", f"(sleep 1.5; touch {marker}) & exit 0")),
            JobCommand("waits", ("bash", "-c", "sleep 30"))]
    outcomes = ProcessTreeRunner(max_parallel=1, grace_seconds=1).run(
        jobs, cwd=tmp_path, log_dir=tmp_path / "logs", deadline=time.monotonic() + 0.8)
    assert outcomes[0].status == "succeeded" and outcomes[0].group_terminated
    assert outcomes[1].status in {"timed_out", "not_started"}
    time.sleep(2.0)
    assert not marker.exists()


JOB = '''import json, sys, time
from pathlib import Path
import ldm_buffer
seed, out = int(sys.argv[1]), Path(sys.argv[2])
if getattr(ldm_buffer, "CRASH_SEED", None) == seed:
    raise SystemExit(3)
if getattr(ldm_buffer, "HANG", False):
    time.sleep(60)
out.mkdir(parents=True, exist_ok=True)
(out / f"seed{seed}.json").write_text(json.dumps({"seed": seed, "average_return": ldm_buffer.SCORE + seed}))
'''
GRADE = '''import json, sys
from pathlib import Path
rows = [json.loads(p.read_text()) for p in sorted(Path("results/ldm").glob("*.json"))]
values = [r["average_return"] for r in rows]
Path(sys.argv[1]).write_text(json.dumps({"environments": {"Cheetah-Run": {
    "average_return": {"mean": sum(values) / len(values), "std": 0.0, "count": len(values)},
    "samples": [{"seed": r["seed"], "average_return": r["average_return"]} for r in rows]}}}))
'''


@pytest.fixture
def synthetic_case(tmp_path, monkeypatch):
    task = tmp_path / "upstream" / "tasks/test/fake"
    task.mkdir(parents=True)
    (task / "job.py").write_text(JOB)
    (task / "grade.py").write_text(GRADE)
    files = {name: hashlib.sha256((task / name).read_bytes()).hexdigest() for name in ("job.py", "grade.py")}
    monkeypatch.setattr(source_module, "upstream_contract", lambda: {
        "source_url": "fixture", "commit": "f" * 40,
        "cases": {"improving_replay_buffers": {"task_path": "tasks/test/fake", "tree_sha256": "0" * 64, "files": files}}})
    raw = copy.deepcopy(load_case("improving_replay_buffers").raw)
    raw.update(upstream_task="tasks/test/fake", support_files=[], links=[], prepare_dirs=["results/{slot}"])
    raw["jobs"] = {"matrix": {"seed": [0, 1]}, "argv": ["{python}", "job.py", "{seed}", "results/{slot}"],
                   "expected_outputs": ["results/{slot}/seed{seed}.json"], "generated_files": []}
    raw["grader"] = {"argv": ["{python}", "grade.py", "{summary}"]}
    case = CaseSpec(raw)
    return case, UpstreamCase(tmp_path / "upstream", case), task


def evaluator(case, upstream, run_dir, timeout=20.0):
    clock = CampaignClock(run_dir / "clock.json", None)
    clock.start()
    return ResearchGymEvaluator(case=case, upstream=upstream, run_dir=run_dir, python=sys.executable, devices=(),
                                min_free_mib=0, evaluation_timeout=timeout, grader_timeout=20, clock=clock,
                                grace_seconds=1)


MINIMAL = """class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, capacity, device):
        self.items = []

    def add(self, obs, act, rew, next_obs, done):
        self.items.append(rew)

    def sample(self, batch_size):
        return self.items[:batch_size]

    def __len__(self):
        return len(self.items)
"""


def candidate(case, extra, name="c1"):
    program = MINIMAL + extra
    assert case.check_program(program) == []
    return Candidate(name, {"program": program}, name)


def test_evaluator_grades_full_coverage_and_replays_receipts(synthetic_case, tmp_path):
    case, upstream, _ = synthetic_case
    assert upstream.verify()["verified_files"] == 2
    ev = evaluator(case, upstream, tmp_path / "run")
    result = ev.evaluate(candidate(case, "\nSCORE = 100.0\n"))
    assert result.status == "succeeded", result.error
    assert result.metrics["rl_cheetah_run_return"] == 100.5
    assert result.metadata["official_grader"] and not result.metadata["official_protocol"]
    assert result.metadata[EVALUATION_ATTEMPT_RECEIPT_KEY] == "researchgym:c1"
    assert result.artifacts["grade_summary"] == "evaluations/c1/grade_summary.json"
    # A resumed process returns the terminal record instead of re-running.
    (tmp_path / "upstream/tasks/test/fake/job.py").write_text("raise SystemExit(9)\n")
    assert evaluator(case, upstream, tmp_path / "run").evaluate(candidate(case, "\nSCORE = 100.0\n")) == result


def test_crashed_cell_invalidates_candidate_and_interrupted_attempt_is_not_rerun(synthetic_case, tmp_path):
    case, upstream, _ = synthetic_case
    ev = evaluator(case, upstream, tmp_path / "run")
    crashed = ev.evaluate(candidate(case, "\nSCORE = 5.0\nCRASH_SEED = 1\n", name="c2"))
    assert crashed.status == "failed" and "1=failed" in crashed.error
    root = tmp_path / "run/evaluations/c3"
    root.mkdir(parents=True)
    (root / "started.json").write_text("{}")
    interrupted = ev.evaluate(candidate(case, "\nSCORE = 5.0\n", name="c3"))
    assert interrupted.status == "failed" and interrupted.metadata["interrupted"]


def test_candidate_that_rewrites_the_grader_is_invalid(synthetic_case, tmp_path):
    case, upstream, _ = synthetic_case
    tamper = "\nSCORE = 5.0\nimport pathlib\npathlib.Path('grade.py').write_text('print(1)')\n"
    result = evaluator(case, upstream, tmp_path / "run").evaluate(candidate(case, tamper, name="c5"))
    assert result.status == "invalid" and result.metadata["tampered_files"] == ["grade.py"]
    assert "grade_summary" not in result.artifacts


def test_evaluation_timeout_is_reported_without_orphans(synthetic_case, tmp_path):
    case, upstream, _ = synthetic_case
    result = evaluator(case, upstream, tmp_path / "run", timeout=1.0).evaluate(
        candidate(case, "\nSCORE = 5.0\nHANG = True\n", name="c4"))
    assert result.status == "timed_out" and "orphaned_process_groups" not in result.metadata
    jobs = json.loads((tmp_path / "run/evaluations/c4/jobs.json").read_text())["jobs"]
    assert {job["status"] for job in jobs} <= {"timed_out", "not_started"}


def test_modified_or_missing_pinned_files_are_rejected(synthetic_case):
    case, upstream, task = synthetic_case
    (task / "grade.py").write_text("print('changed')\n")
    with pytest.raises(SourceMismatchError, match="1 modified"):
        upstream.verify()


def pinned_checkout(case, destination):
    """Extract the case from Git objects at the pinned commit, ignoring working-tree edits."""
    root = Path.home() / "ResearchGym"
    commit = source_module.upstream_contract()["commit"]
    if not (root / ".git").exists() or subprocess.run(["git", "-C", str(root), "cat-file", "-e", commit],
                                                      capture_output=True).returncode:
        pytest.skip("pinned ResearchGym Git objects are not available")
    archive = subprocess.run(["git", "-C", str(root), "archive", commit, case.raw["upstream_task"]],
                             capture_output=True, check=True).stdout
    destination.mkdir()
    subprocess.run(["tar", "-x", "-C", str(destination)], input=archive, check=True)
    return destination


@pytest.mark.parametrize("case_id", ["time_series_explanation", "continual_learning", "cross_modal_retrieval",
                                     "improving_replay_buffers"])
def test_pinned_source_verifies_and_injects_slot_once(case_id, tmp_path):
    raw = copy.deepcopy(load_case(case_id).raw)
    raw["links"] = []  # data links are deployment inputs; this checks pinned code assembly only
    case = CaseSpec(raw)
    upstream = UpstreamCase(pinned_checkout(case, tmp_path / "upstream"), case)
    assert upstream.verify()["verified_files"] == len(upstream.pinned["files"])
    build = upstream.build_workspace(tmp_path / "ws", slot="ldm", program=case.seed_program())
    for injection in raw["injections"]:
        text = (tmp_path / "ws" / injection["path"]).read_text()
        block = render(injection["block"], {"slot": "ldm", "module": case.module_name})
        assert text.count(block) == 1 and text.count(injection["anchor"]) == 1
    for support in raw["support_files"]:
        assert (tmp_path / "ws" / support["dest"]).is_file()
    assert (tmp_path / "ws" / case.candidate_path).read_text() == case.seed_program()
    assert source_module.modified_files(tmp_path / "ws", build["file_digests"]) == []
