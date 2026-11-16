"""Parallel evaluation (WP10) through the real Execution, with a fake Docker CLI.

FakeDocker keeps a container registry and runs each "container" on the host:
single runs as `timeout -s KILL` subprocesses of the real v2 runner, the scorer
in-process through the real v2 scorer, batches through the real supervisor batch
loop. Wall times are fixed unless a run is killed, so receipts are reproducible.
No Docker daemon is used; tests marked `needs H20 phase` in the report cover that.
"""

import hashlib
import hmac
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ldm_tts.contracts import CandidateRejection, RawProposal
from ldm_tts.contracts.evaluation import EvaluationPaused
from ldm_tts.engine.run_store import BudgetLedger
from tasks.copper_inversion.core import execution, supervisor
from tasks.copper_inversion.core.candidate import seal
from tasks.copper_inversion.experiments.v2 import parallel_runtime
from tasks.copper_inversion.tests.test_rsi_service import FAILING, VALID, write_suite
from tasks.copper_inversion.v2 import scorer
from tasks.copper_inversion.v2.budget import ScientificSlots
from tasks.copper_inversion.v2.domain import CheckedDomain
from tasks.copper_inversion.v2.interface_checks import InterfaceChecks
from tasks.copper_inversion.v2.suite import SuiteEvaluator

REPO = Path(__file__).resolve().parents[3]
IMAGE = "sha256:" + "f" * 64
WALL = 1.25
INVALID = "import numpy as np\ndef solve(problem, budget, seed):\n    return np.ones(problem['shape'])\n"
SLEEPER = "import time\ndef solve(problem, budget, seed):\n    time.sleep(60)\n"
CRASH = "def solve(problem, budget, seed):\n    raise RuntimeError('crash')\n"
# Fails on one fixture only (index 4 has kilometre spacing).
PICKY = ("from tasks.copper_inversion.v2.baselines import initialize\n"
         "def solve(problem, budget, seed):\n"
         "    if problem['spacing'][0] > 100:\n        raise RuntimeError('scale')\n"
         "    return initialize(problem)\n")


class Result(SimpleNamespace):
    pass


class FakeDocker:
    """The docker CLI as Execution uses it; containers run on the host."""

    def __init__(self, scratch):
        self.scratch = Path(scratch)
        self.containers = {}
        self.calls = []
        self.timeouts = {}      # command -> number of timeouts to inject
        self.took_effect = set()  # commands whose timed-out call still happened
        self.lock = threading.Lock()

    def __call__(self, *args, check=True):
        args = [str(a) for a in args]
        with self.lock:
            self.calls.append(args)
            pending = self.timeouts.get(args[0], 0)
            if pending:
                self.timeouts[args[0]] = pending - 1
        if pending:
            if args[0] in self.took_effect:
                getattr(self, "_" + args[0])(args)
            raise execution.DockerTimeout("injected timeout")
        result = getattr(self, "_" + args[0].replace("-", "_"))(args)
        if check and result.returncode:
            raise EvaluationPaused("Docker operation failed: " + result.stderr)
        return result

    def _image(self, args):
        return Result(returncode=0, stdout=IMAGE + "\n", stderr="")

    def _ps(self, args):
        labels = [args[i + 1].removeprefix("label=") for i, a in enumerate(args) if a == "--filter"
                  and args[i + 1].startswith("label=")]
        rows = []
        for name, c in self.containers.items():
            if all(c["labels"].get(k) == v for k, v in (l.split("=", 1) for l in labels)):
                rows.append("\t".join([name, c["state"]["Status"], c["labels"].get("ldm4geo.job_root", "")]))
        return Result(returncode=0, stdout="".join(r + "\n" for r in rows), stderr="")

    def _inspect(self, args):
        c = self.containers.get(args[1])
        if c is None:
            return Result(returncode=1, stdout="[]", stderr="Error: No such object")
        return Result(returncode=0, stdout=json.dumps([{"Id": c["id"], "State": c["state"]}]), stderr="")

    def _create(self, args):
        name = args[args.index("--name") + 1]
        env = dict(args[i + 1].split("=", 1) for i, a in enumerate(args) if a == "-e")
        mounts = {}
        labels = {}
        for i, a in enumerate(args):
            if a == "--mount":
                fields = dict(f.split("=", 1) for f in args[i + 1].split(",") if "=" in f)
                mounts[fields["dst"]] = fields["src"]
            if a == "--label":
                key, value = args[i + 1].split("=", 1)
                labels[key] = value
        tail = args[args.index(IMAGE) + 1:]
        self.containers[name] = dict(id=hashlib.sha256(name.encode()).hexdigest(), env=env, mounts=mounts,
                                     labels=labels, tail=tail, log="",
                                     state=dict(Status="created", Running=False, Pid=0, ExitCode=0,
                                                OOMKilled=False, StartedAt="0001-01-01T00:00:00Z",
                                                FinishedAt="0001-01-01T00:00:00Z"))
        return Result(returncode=0, stdout=name + "\n", stderr="")

    def _start(self, args):
        c = self.containers[args[1]]
        root = self.scratch / ("fs-" + args[1])
        for dst, src in c["mounts"].items():
            link = root / dst.lstrip("/")
            link.parent.mkdir(parents=True, exist_ok=True)
            if not link.exists():
                link.symlink_to(src)
        (root / "tmp").mkdir(exist_ok=True)
        key = c["env"]["COPPER_RESOURCE_KEY"]
        env = {k: v for k, v in c["env"].items() if k != "COPPER_RESOURCE_KEY"}
        env.update(PATH=os.environ.get("PATH", "/usr/bin:/bin"), PYTHONPATH=str(REPO),
                   COPPER_SOURCE=str(root / "submitted/solver.py"))
        cgroup = root / "cgroup"
        cgroup.mkdir(exist_ok=True)
        (cgroup / "memory.events").write_text("oom_kill 0\n")
        assert c["tail"][:3] == ["python", "-m", "tasks.copper_inversion.core.supervisor"]
        rest = c["tail"][3:]
        payload = {"schema_version": 1, "workload_complete": True, "peak_memory_bytes": None,
                   "cpu_seconds_through_reap": 0.5, "oom_kill_count": 0}
        killed = False
        if rest[0] == "--batch":
            seconds, count, module = rest[1:]
            runs = supervisor.batch(seconds, int(count), module, env, cases=str(root / "cases"),
                                    outputs=c["mounts"]["/output"], tmp=str(root / "tmp"), cgroup=str(cgroup),
                                    user=None, as_owner=False, kill_all=False)
            for run in runs:  # reproducible receipts: fixed wall unless killed
                if not run["exit_code"] == 137:
                    run["worker_elapsed_seconds"] = WALL
            code = 0
            payload.update(runs=runs, exit_code=0)
        else:
            seconds, module = rest
            if module.endswith("scorer"):
                try:
                    scorer.main(root)
                    code = 0
                except Exception:
                    code = 1
            else:
                env.update(COPPER_INPUT=str(root / "input"), COPPER_OUTPUT=c["mounts"]["/output"],
                           COPPER_WORKLOAD_DEADLINE_MONOTONIC=str(time.monotonic() + float(seconds)))
                code = subprocess.run(["timeout", "--signal=KILL", seconds, sys.executable, "-m", module],
                                      env=env, capture_output=True).returncode
                code = code if code >= 0 else 128 - code
                killed = code == 137
            payload["exit_code"] = code
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        mac = hmac.new(bytes.fromhex(key), encoded.encode(), hashlib.sha256).hexdigest()
        c["log"] = supervisor.PREFIX + json.dumps({"payload": encoded, "mac": mac}) + "\n"
        wall = float(seconds) + 0.5 if killed else WALL
        start = int(time.time())  # whole seconds: the parsed wall time is exact
        iso = lambda t: datetime.fromtimestamp(t, timezone.utc).isoformat().replace("+00:00", "Z")
        c["state"].update(Status="exited", Running=False, ExitCode=code, StartedAt=iso(start),
                          FinishedAt=iso(start + wall))
        return Result(returncode=0, stdout=args[1] + "\n", stderr="")

    def _logs(self, args):
        return Result(returncode=0, stdout=self.containers[args[1]]["log"], stderr="")

    def _kill(self, args):
        return Result(returncode=0, stdout="", stderr="")

    def _rm(self, args):
        name = args[-1]
        if self.containers.pop(name, None) is None:
            return Result(returncode=1, stdout="", stderr="Error: No such container: " + name)
        return Result(returncode=0, stdout=name + "\n", stderr="")


@pytest.fixture
def docker(tmp_path, monkeypatch):
    fake = FakeDocker(tmp_path / "docker")
    fake.scratch.mkdir()
    monkeypatch.setattr(execution, "docker", fake)
    monkeypatch.setattr(execution.Execution, "sleep", staticmethod(lambda seconds: None))
    monkeypatch.setattr(execution.Execution, "labels", {})
    monkeypatch.setattr(SuiteEvaluator, "case_workers", 4)
    yield fake
    # Container cleanup (H20 had 73,879 stopped containers): nothing may be left.
    assert fake.containers == {}, sorted(fake.containers)


def candidate(root, body):
    payload = seal(root, body.encode())
    return SimpleNamespace(candidate_id=payload["sha256"], canonical_key=payload["sha256"], payload=payload)


def evaluation(root, manifest, body, seconds=10, **options):
    root.mkdir(parents=True, exist_ok=True)
    ex = execution.Execution(root / "jobs", IMAGE)
    evaluator = SuiteEvaluator(root, ex, manifest, seconds=seconds, **options)
    c = candidate(root, body)
    result = evaluator.evaluate(c)
    key = evaluator.evaluation_attempt_usage_key(c)
    folder = root / "evaluations" / key
    files = {str(p.relative_to(folder)): p.read_bytes() for p in sorted(folder.rglob("*"))
             if p.is_file() and p.name in ("receipt.json", "result.json", "private-case-results.json")}
    return key, result, files


def models(root):
    return sorted(hashlib.sha256(p.read_bytes()).hexdigest() for p in (root / "evaluations").rglob("model/model.npy"))


# AT-40 merged scoring ----------------------------------------------------------

@pytest.mark.parametrize("objective", ["confirmation", "oracle_iou"])
def test_at40_merged_scoring_is_bit_identical(tmp_path, docker, monkeypatch, objective):
    manifest = write_suite(tmp_path / "release", "train", bodies=3)
    options = {} if objective == "confirmation" else {"objective": objective}
    outcomes = {}
    for merged in (True, False):
        monkeypatch.setattr(execution.Execution, "merged", merged)
        before = len([c for c in docker.calls if c[0] == "create"])
        outcomes[merged] = [evaluation(tmp_path / f"{body[:12]}-{merged}", manifest, body, **options)
                            for body in (VALID, INVALID, FAILING)]
        creates = len([c for c in docker.calls if c[0] == "create"]) - before
        # 3 + 3 + 3 solves; scoring: one container per evaluation with models, or one per model.
        assert creates == (9 + 1 + 1 if merged else 9 + 3 + 3)
    for (k1, r1, f1), (k2, r2, f2) in zip(outcomes[True], outcomes[False]):
        assert k1 == k2 and f1 == f2
        assert json.dumps(r1.to_dict(), sort_keys=True) == json.dumps(r2.to_dict(), sort_keys=True)
    valid = outcomes[True][0][2]["private-case-results.json"]
    assert models(tmp_path / f"{VALID[:12]}-True") == models(tmp_path / f"{VALID[:12]}-False")
    rows = json.loads(valid)
    assert all(r["status"] == "succeeded" and r["jobs"] == 2 for r in rows)
    # T4's judge-aware selection reads the objective on oracle receipts.
    assert all(("objective" in r) == (objective == "oracle_iou") for r in rows)


def test_merged_scorer_main_matches_single_case_main(tmp_path):
    """The two scorer entry points, on the same files, give the same per-case dicts."""
    manifest = write_suite(tmp_path / "release", "train", bodies=2)
    cases = json.loads(manifest.read_text())["cases"]
    from tasks.copper_inversion.v2.baselines import initialize
    from tasks.copper_inversion.v2.data import read_problem
    merged = tmp_path / "merged"
    singles = []
    for index, case in enumerate(cases):
        problem = read_problem(manifest.parent / case["public"])
        model = tmp_path / f"model{index}"
        model.mkdir()
        np.save(model / "model.npy", np.asarray(initialize(problem), dtype=float))
        single = tmp_path / f"single{index}"
        (single / "output").mkdir(parents=True)
        for name, src in (("input", manifest.parent / case["public"]), ("model", model),
                          ("truth", manifest.parent / case["truth"])):
            (single / name).symlink_to(src)
            (merged / "cases" / str(index)).mkdir(parents=True, exist_ok=True)
            (merged / "cases" / str(index) / name).symlink_to(src)
        scorer.main(single)
        singles.append(json.loads((single / "output/result.json").read_text()))
    (merged / "output").mkdir()
    scorer.main(merged)
    assert json.loads((merged / "output/results.json").read_text()) == singles


# AT-41 merged interface checks -------------------------------------------------

def admit(root, bodies, merged, monkeypatch, parallel=False):
    monkeypatch.setattr(execution.Execution, "merged", merged)
    root.mkdir(parents=True, exist_ok=True)
    ex = execution.Execution(root / "jobs", IMAGE)
    domain = CheckedDomain(root, InterfaceChecks(root, ex))
    def one(body):
        result = domain.admit(RawProposal(seal(root, body.encode()), "test"))
        return result.message if isinstance(result, CandidateRejection) else "accepted"
    if parallel:
        with ThreadPoolExecutor(max_workers=len(bodies)) as pool:
            return list(pool.map(one, bodies))
    return [one(body) for body in bodies]


def test_at41_merged_interface_reports_the_same_rejections(tmp_path, docker, monkeypatch):
    bodies = [VALID, SLEEPER, INVALID, CRASH, PICKY]
    merged = admit(tmp_path / "merged", bodies, True, monkeypatch)
    single = admit(tmp_path / "single", bodies, False, monkeypatch)
    assert merged == single
    assert merged[0] == "accepted"
    assert merged[1] == "public interface public-interface-0 failed: timeout"
    assert merged[3] == "public interface public-interface-0 failed: 1"
    assert merged[4] == "public interface public-interface-4 failed: 1"
    assert merged[2] != "accepted" and "public interface" not in merged[2]
    # One container per candidate, six attested runs each.
    receipts = [json.loads(p.read_text()) for p in (tmp_path / "merged/jobs").glob("*/receipt.json")]
    assert len(receipts) == len(bodies)
    assert all([r["index"] for r in x["workload_resources"]["runs"]] == list(range(6)) for x in receipts)
    assert len(list((tmp_path / "single/jobs").glob("*/receipt.json"))) > len(bodies)


def test_batch_runs_are_judged_like_single_containers():
    runs = [dict(index=0, exit_code=0, worker_elapsed_seconds=1., oom_kill_count=0, output_limit=False),
            dict(index=1, exit_code=137, worker_elapsed_seconds=5.2, oom_kill_count=0, output_limit=False),
            dict(index=2, exit_code=137, worker_elapsed_seconds=1.1, oom_kill_count=1, output_limit=False),
            dict(index=3, exit_code=137, worker_elapsed_seconds=0.4, oom_kill_count=0, output_limit=True)]
    judged = execution.batch_runs({"runs": runs}, 4, 5)
    assert [j["reason"] for j in judged] == [None, "timeout", None, "output_limit"]
    assert [j["oom"] for j in judged] == [False, False, True, False]
    assert execution.batch_runs(None, 4, 5) is None
    with pytest.raises(EvaluationPaused):
        execution.batch_runs({"runs": runs[:3]}, 4, 5)


def test_supervisor_batch_isolates_runs_and_enforces_output_limits(tmp_path):
    """The real batch loop, unprivileged: each run sees only its case; strays are cleared."""
    module = tmp_path / "mods" / "probe_worker.py"
    module.parent.mkdir()
    module.write_text(
        "import os, pathlib\n"
        "out = pathlib.Path(os.environ['COPPER_OUTPUT'])\n"
        "i = int(out.name)\n"
        "(out / 'case.txt').write_text(pathlib.Path(os.environ['COPPER_INPUT'], 'id').read_text())\n"
        "(out.parent / f'stray{i}').write_text('x')\n"
        "if i == 1:\n    [(out / f'f{k}').write_text('y') for k in range(40)]\n")
    cases, outputs, tmp, cgroup = (tmp_path / n for n in ("cases", "out", "tmp", "cg"))
    for path in (outputs, tmp, cgroup):
        path.mkdir()
    (cgroup / "memory.events").write_text("oom_kill 0\n")
    for i in range(3):
        (cases / str(i) / "input").mkdir(parents=True)
        (cases / str(i) / "input" / "id").write_text(f"case-{i}")
    (tmp / "left").write_text("z")
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "PYTHONPATH": str(module.parent)}
    runs = supervisor.batch("5", 3, "probe_worker", env, cases=str(cases), outputs=str(outputs), tmp=str(tmp),
                            cgroup=str(cgroup), user=None, as_owner=False, kill_all=False)
    assert [r["exit_code"] for r in runs] == [0, 0, 0]
    assert [r["output_limit"] for r in runs] == [False, True, False]
    assert sorted(p.name for p in outputs.iterdir()) == ["0", "1", "2"]
    assert [(outputs / str(i) / "case.txt").read_text() for i in range(3)] == ["case-0", "case-1", "case-2"]
    assert not list(tmp.iterdir())
    assert all(not os.access(outputs / str(i), os.W_OK) or os.geteuid() == 0 for i in range(3))


# AT-42 case concurrency is scheduling only ------------------------------------

def test_at42_case_concurrency_does_not_change_results(tmp_path, docker, monkeypatch):
    manifest = write_suite(tmp_path / "release", "train", bodies=4, packs=1, seeds=(47, 48))
    outcomes = []
    for workers in (1, 16):
        monkeypatch.setattr(SuiteEvaluator, "case_workers", workers)
        key, result, files = evaluation(tmp_path / f"w{workers}", manifest, VALID)
        context = SuiteEvaluator(tmp_path / f"w{workers}", execution.Execution(tmp_path / "j", IMAGE),
                                 manifest, seconds=10).context
        outcomes.append((key, json.dumps(result.to_dict(), sort_keys=True), files, context))
    assert outcomes[0] == outcomes[1]
    assert "case_workers" not in outcomes[0][3]


def test_policy_sets_case_concurrency_per_tier():
    policy = {"case_workers": {"S": 12, "F": 16}}
    assert parallel_runtime.case_workers(policy, "S") == 12
    assert parallel_runtime.case_workers(policy, "F") == 16
    assert parallel_runtime.case_workers(policy, None) == 16
    assert parallel_runtime.case_workers({}, "S") == 4
    with pytest.raises(ValueError):
        parallel_runtime.case_workers({"case_workers": 0})


# AT-43 parallel admission ------------------------------------------------------

def test_at43_concurrent_admission_equals_serial_and_inputs_are_written_once(tmp_path, docker, monkeypatch):
    from tasks.copper_inversion.v2 import interface_checks
    writes = []
    original = interface_checks.save_problem
    monkeypatch.setattr(interface_checks, "save_problem",
                        lambda problem, folder: (writes.append(str(folder)), original(problem, folder))[1])
    bodies = [VALID.replace("initialize(problem)", f"initialize(problem)  # {i}") if i % 4 else CRASH + f"# {i}\n"
              for i in range(16)]
    serial = admit(tmp_path / "serial", bodies, True, monkeypatch)
    hashes = lambda root: {str(p.relative_to(root)): (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
                           for p in sorted((root / "interface-inputs").rglob("*")) if p.is_file()}
    writes.clear()
    concurrent = admit(tmp_path / "parallel", bodies, True, monkeypatch, parallel=True)
    assert concurrent == serial
    assert len(writes) == 6
    first = hashes(tmp_path / "parallel")
    assert len(first) == 18
    assert {k: v[0] for k, v in first.items()} == {k: v[0] for k, v in hashes(tmp_path / "serial").items()}
    again = admit(tmp_path / "parallel", bodies[:4], True, monkeypatch, parallel=True)
    assert again == serial[:4] and hashes(tmp_path / "parallel") == first and len(writes) == 6


def test_parallel_service_has_no_global_admission_lock(tmp_path, monkeypatch):
    from tasks.copper_inversion.v2 import service
    def init(self, config):
        self.ledger = BudgetLedger(limits={"algorithm_feedback_slots": 3}, path=tmp_path / "budget.json")
        self.slots = ScientificSlots(self.ledger.consume_many)
        self.domain = SimpleNamespace(admit="original")
        self.evaluator = SimpleNamespace(execution=SimpleNamespace(account=None))
    monkeypatch.setattr(service.EvaluatorService, "__init__", init)
    backend = parallel_runtime.ParallelService({})
    assert backend.domain.admit == "original" and not hasattr(backend, "_admission_lock")


# AT-45 start retry, then global pause ------------------------------------------

def policy_for(root, **extra):
    queue = root / "queue"
    (queue / "startup").mkdir(parents=True)
    for i in range(4):
        (queue / f"{i:03}.lock").touch()
        (queue / "startup" / f"{i:03}.lock").touch()
    policy = dict(queue_root=str(queue), global_jobs=4, interface_slots=0, jobs_per_process=4,
                  startup_slots=4, start_retry_seconds=[5, 15, 45], case_workers={"S": 12, "F": 16}, **extra)
    policy["policy_sha256"] = execution.digest(policy)
    return policy


@pytest.fixture
def installed(tmp_path, monkeypatch):
    monkeypatch.setattr(execution.Execution, "run", execution.Execution.run)
    monkeypatch.setattr(execution.Execution, "start_retry_seconds", (5, 15, 45))
    def install(policy, tier=None):
        queue = parallel_runtime.install(policy, tier)
        monkeypatch.setattr(execution.Execution, "sleep", staticmethod(lambda s: delays.append(s)))
        return queue
    delays = []
    install.delays = delays
    return install


def solve_once(root, identity="job"):
    inputs = root / "inputs"
    if not inputs.exists():
        from tasks.copper_inversion.tests.test_v2_science import fixture
        from tasks.copper_inversion.v2.data import save_problem
        save_problem(fixture(), inputs)
        (inputs / "budget.json").write_text(json.dumps(dict(seed=47, wall_time_seconds=10, cpu_count=2, memory_mb=4096)))
    source = root / "src"
    source.mkdir(exist_ok=True)
    (source / "solver.py").write_text(VALID)
    return execution.Execution(root / "jobs", IMAGE).run(identity, phase="solver", inputs=inputs, timeout=10,
                                                         cpus=2, memory_mb=4096, source=source, science_module="v2")


@pytest.mark.parametrize("command", ["create", "start"])
def test_at45_one_start_timeout_is_retried_without_pause(tmp_path, docker, installed, command):
    policy = policy_for(tmp_path)
    installed(policy, "S")
    assert SuiteEvaluator.case_workers == 12
    docker.timeouts[command] = 1
    job, receipt = solve_once(tmp_path)
    assert receipt["exit_code"] == 0
    assert not (Path(policy["queue_root"]) / "PAUSED.json").exists()
    record = json.loads((tmp_path / "jobs/scheduling" / f"{job.name}.json").read_text())
    assert [r["command"] for r in record["start_retries"]] == [command]
    assert [r["delay_seconds"] for r in record["start_retries"]] == [5] and installed.delays == [5]
    assert record["slots_in_use_at_start"] == 1 and record["global_jobs"] == 4
    assert "queue_wait_seconds" in record and record["phase"] == "solver"
    # The scientific receipt holds no scheduling information.
    assert not {"start_retries", "slots_in_use_at_start"} & set(receipt)


def test_at45_timed_out_command_that_took_effect_is_not_repeated(tmp_path, docker, installed):
    installed(policy_for(tmp_path))
    docker.timeouts["create"] = 1
    docker.took_effect.add("create")
    job, receipt = solve_once(tmp_path)
    assert receipt["exit_code"] == 0
    assert [c[0] for c in docker.calls].count("create") == 1
    assert json.loads((job / "job.json").read_text()).get("start_retries") is None


def test_at45_four_timeouts_pause_every_run(tmp_path, docker, installed):
    policy = policy_for(tmp_path)
    installed(policy)
    docker.timeouts["create"] = 4
    with pytest.raises(EvaluationPaused, match="timed out after 3 retries"):
        solve_once(tmp_path)
    assert installed.delays == [5, 15, 45]
    assert (Path(policy["queue_root"]) / "PAUSED.json").exists()
    with pytest.raises(EvaluationPaused, match="shared scheduler paused"):
        solve_once(tmp_path, "other")


def test_slots_in_use_counts_other_holders(tmp_path):
    for i in range(4):
        (tmp_path / f"{i:03}.lock").touch()
    queue = parallel_runtime.JobQueue(tmp_path, 4, 4)
    seen = []
    with queue.acquire():
        with queue.acquire():
            seen.append(queue.slots_in_use_at_start)
        seen.append(queue.slots_in_use_at_start)
        assert queue.in_use() == 1
    assert seen == [2, 2] and queue.in_use() == 0


# Container cleanup -------------------------------------------------------------

def test_every_container_is_removed_on_every_outcome(tmp_path, docker, monkeypatch):
    """Success, timeout, crash, output limit, scorer, batch, replay: nothing stays behind."""
    manifest = write_suite(tmp_path / "release", "train", bodies=2)
    for body in (VALID, INVALID, FAILING, SLEEPER.replace("60", "30")):
        evaluation(tmp_path / hashlib.sha256(body.encode()).hexdigest()[:8], manifest, body, seconds=2)
        assert docker.containers == {}
    admit(tmp_path / "admit", [VALID, CRASH], True, monkeypatch)
    assert docker.containers == {}
    # Controller lost after start: the container stays for reconciliation, then the
    # replay finishes the job and removes it.
    original = docker._start
    def lost(args):
        original(args)
        raise EvaluationPaused("controller disconnected after dispatch")
    monkeypatch.setattr(docker, "_start", lost)
    with pytest.raises(EvaluationPaused):
        solve_once(tmp_path / "lost")
    assert len(docker.containers) == 1
    monkeypatch.setattr(docker, "_start", original)
    solve_once(tmp_path / "lost")
    assert docker.containers == {}
    # Replays of finished jobs remove a leftover container of the same name.
    rm = [c for c in docker.calls if c[0] == "rm"]
    assert rm and all(c[1] == "-f" for c in rm)


def test_failed_removal_is_listed_not_fatal(tmp_path, docker, monkeypatch):
    original = docker._rm
    monkeypatch.setattr(docker, "_rm", lambda args: Result(returncode=1, stdout="", stderr="daemon busy"))
    job, receipt = solve_once(tmp_path)
    assert receipt["exit_code"] == 0
    pending = [json.loads(l) for l in (tmp_path / "jobs/cleanup-pending.jsonl").read_text().splitlines()]
    assert pending[0]["name"] == receipt["container"]
    monkeypatch.setattr(docker, "_rm", original)
    docker._rm(["rm", "-f", receipt["container"]])


def test_containers_carry_reaper_labels(tmp_path, docker, installed):
    policy = policy_for(tmp_path)
    installed(policy)
    original = docker._rm
    names = []
    docker._rm = lambda args: (names.append(dict(docker.containers[args[-1]]["labels"])), original(args))[1]
    job, _ = solve_once(tmp_path)
    labels = names[0]
    assert labels["ldm4geo.managed"] == "1"
    assert labels["ldm4geo.queue"] == parallel_runtime.container_labels(policy)["ldm4geo.queue"]
    assert labels["ldm4geo.job_root"] == str(job)
