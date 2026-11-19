"""Batch launcher, reconciliation and archive (WP9, WP10 items 6-7) without Docker.

The launcher's operating-system actions go through a fake host: services and
controllers are scripted processes. Two real threaded risk-only services run
in-process over the fake Docker CLI of test_rsi_parallel.
"""

import hashlib
import json
import shutil
import tarfile
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from tasks.copper_inversion.core import execution
from tasks.copper_inversion.core.execution import digest
from tasks.copper_inversion.experiments.rsi import archive, launch
from tasks.copper_inversion.experiments.v2 import parallel_runtime, qualify_load
from tasks.copper_inversion.rsi import protocol as protocols
from tasks.copper_inversion.rsi.controller import Config, Controller
from tasks.copper_inversion.tests.test_rsi_parallel import (FakeDocker, IMAGE, VALID, docker, installed,  # noqa: F401
                                                            policy_for)
from tasks.copper_inversion.tests.test_rsi_service import (LoopbackClient, RecordingAgent, make_service,
                                                           scripted_select, write_suite)

EXAMPLE = Path(__file__).resolve().parents[1] / "resources/rsi/protocol.example.json"
VALUES = {"global_jobs": 8, "interface_slots": 2, "startup_slots": 2, "jobs_per_process": 8,
          "case_workers": {"S": 12, "F": 16}, "batch_workers": 16, "start_retry_seconds": [5, 15, 45],
          "max_concurrent_runs": {"S": 16, "F": 6}, "max_agent_sessions": 64}


def protocol():
    return json.loads(EXAMPLE.read_text())


def plan(runs, experiment="M1", tier="S", **extra):
    return {"schema": launch.SCHEMA, "name": "b1", "protocol": str(EXAMPLE), "experiment": experiment,
            "tier": tier, "runs": [dict(arm=a, judge=j, seed=s) for a, j, s in runs], **extra}


# AT-46 batches are balanced --------------------------------------------------------

def test_at46_unbalanced_batches_are_refused():
    proto, policy = protocol(), dict(VALUES)
    good = plan([(a, "standard", s) for a in ("ldm4geo", "accumulated") for s in (105, 106)])
    summary = launch.check_batch(good, proto, policy)
    assert summary["seeds"] == [105, 106] and summary["arms"] == ["accumulated", "ldm4geo"]
    unequal_count = plan([("ldm4geo", "standard", 105), ("ldm4geo", "standard", 106), ("accumulated", "standard", 105)])
    unequal_seeds = plan([("ldm4geo", "standard", 105), ("accumulated", "standard", 106)])
    missing_cell = plan([("ldm4geo", "standard", 201), ("accumulated", "oracle", 201)], experiment="E3")
    for bad in (unequal_count, unequal_seeds, missing_cell):
        with pytest.raises(launch.PlanError):
            launch.check_batch(bad, proto, policy)
    e3 = plan([(a, j, s) for a in ("ldm4geo", "independent_sampling") for j in ("standard", "oracle")
               for s in (201, 202)], experiment="E3")
    assert launch.check_batch(e3, proto, policy)["judges"] == ["oracle", "standard"]


def test_continuation_batches_balance_modes_and_replicates():
    proto = protocol()
    modes = {"history": "ldm4geo", "conversation": "accumulated", "best_program": "restart_best"}
    def runs(replicates):
        return [{"arm": arm, "judge": "standard", "seed": 101, "continuation": "/runs/M1/accumulated-s101/checkpoints/round_10",
                 "mode": mode, "replicate": r} for mode, arm in modes.items() for r in replicates[mode]]
    good = {**plan([], experiment="E1"), "runs": runs({m: (1, 2) for m in modes})}
    summary = launch.check_batch(good, proto, VALUES)
    assert "ck101-history-r2" in summary["runs"] and summary["seeds"] == [101]
    bad = {**plan([], experiment="E1"), "runs": runs({**{m: (1, 2) for m in modes}, "history": (1,)})}
    with pytest.raises(launch.PlanError, match="differ"):
        launch.check_batch(bad, proto, VALUES)
    planted = [dict(r, planted=True) for r in runs({"history": (1,), "conversation": (1,), "best_program": ()})]
    assert launch.run_id("E2", planted[0]) == "ck101-history-planted-r1"


def test_batches_follow_the_protocol_and_the_policy():
    proto = protocol()
    with pytest.raises(launch.PlanError, match="seed table"):
        launch.check_batch(plan([("ldm4geo", "standard", 999)]), proto, VALUES)
    with pytest.raises(launch.PlanError, match="arm is not defined"):
        launch.check_batch(plan([("nonexistent", "standard", 101)]), proto, VALUES)
    with pytest.raises(launch.PlanError, match="twice"):
        launch.check_batch(plan([("ldm4geo", "standard", 101)] * 2), proto, VALUES)
    big = plan([(a, "standard", s) for a in ("ldm4geo", "accumulated") for s in range(101, 105)])
    with pytest.raises(launch.PlanError, match="max_concurrent_runs"):
        launch.check_batch(big, proto, dict(VALUES, max_concurrent_runs={"S": 6}))
    with pytest.raises(launch.PlanError, match="agent sessions"):
        launch.check_batch(big, proto, dict(VALUES, max_agent_sessions=16))
    smoke = plan([("ldm4geo", "standard", 7)], experiment="smoke")  # no smoke seed table
    assert launch.check_batch(smoke, proto, VALUES)["runs"] == ["ldm4geo-s7"]


def test_batch_plan_schema(tmp_path):
    path = tmp_path / "b.json"
    for bad in (dict(plan([("ldm4geo", "standard", 101)]), schema="other"),
                dict(plan([("ldm4geo", "standard", 101)]), name="../x"),
                dict(plan([("ldm4geo", "standard", 101)]), prefixes=[0]),
                {**plan([]), "runs": [{"arm": "ldm4geo", "judge": "standard", "seed": 1, "method": "x"}]}):
        path.write_text(json.dumps(bad))
        with pytest.raises(launch.PlanError):
            launch.load_batch(path)


def test_formal_batches_freeze_the_frozen_prefixes():
    """B4/B11/C3: prefixes and checkpoints come from the G3-frozen analysis config."""
    seeds = [("ldm4geo", "standard", 101), ("accumulated", "standard", 101)]
    launch.check_frozen_prefixes(plan(seeds, prefixes=["full", 5, 10], checkpoint_round=10))
    launch.check_frozen_prefixes(plan(seeds, experiment="E2"))
    for bad in (plan(seeds), plan(seeds, prefixes=["full", 5, 10, 15]),
                plan(seeds, experiment="E3", checkpoint_round=10),
                plan(seeds, prefixes=["full", 5, 10], checkpoint_round=12)):
        with pytest.raises(launch.PlanError):
            launch.check_frozen_prefixes(bad)


# Policy, qualification, pause and reaping ----------------------------------------

def test_policy_is_hashed_and_pinned_to_the_runtime(tmp_path):
    policy = launch.make_policy(VALUES, tmp_path / "queue", tmp_path / "policy.json")
    assert len(list((tmp_path / "queue").glob("*.lock"))) == 8
    assert len(list((tmp_path / "queue/startup").glob("*.lock"))) == 2
    assert launch.load_policy(tmp_path / "policy.json") == policy
    changed = dict(policy, global_jobs=128)
    (tmp_path / "changed.json").write_text(json.dumps(changed))
    with pytest.raises(launch.PlanError, match="changed"):
        launch.load_policy(tmp_path / "changed.json")
    runtime = tmp_path / "runtime.py"
    runtime.write_text("# other runtime\n")
    with pytest.raises(launch.PlanError, match="runtime"):
        launch.load_policy(tmp_path / "policy.json", runtime=runtime)


def test_at44_qualification_verdict_and_launcher_gate(tmp_path):
    idle = [1000.] * 8
    assert qualify_load.verdict(idle, [960.] * 95 + [900.] * 5)["passed"]
    assert not qualify_load.verdict(idle, [940.] * 100)["passed"]                # median below 95%
    assert not qualify_load.verdict(idle, [990.] * 90 + [800.] * 10)["passed"]   # worst 5% below 85%
    record = tmp_path / "qualification.json"
    record.write_text(json.dumps({"status": "passed", "global_jobs": 8}))
    sha = hashlib.sha256(record.read_bytes()).hexdigest()
    site = {"load_qualification": str(record)}
    assert launch.check_qualification({}, VALUES, "smoke") is None
    with pytest.raises(launch.PlanError, match="need a load qualification"):
        launch.check_qualification({}, VALUES, "M1")
    assert launch.check_qualification(site, dict(VALUES, load_qualification_sha256=sha), "M1") == "passed"
    with pytest.raises(launch.PlanError, match="slot count"):
        launch.check_qualification(site, dict(VALUES, global_jobs=96, load_qualification_sha256=sha), "M1")


def test_unpause_keeps_the_faults(tmp_path):
    queue = parallel_runtime.JobQueue(tmp_path, 1, 1)
    assert launch.unpause(tmp_path) == {"status": "not_paused"}
    queue.pause(RuntimeError("container start timed out after 3 retries"))
    other = threading.Thread(target=queue.pause, args=(RuntimeError("second fault from another worker"),))
    other.start()
    other.join()
    result = launch.unpause(tmp_path)
    assert result["status"] == "unpaused" and len(result["faults"]) == 2
    assert not (tmp_path / "PAUSED.json").exists()
    kept = Path(result["kept"])
    assert {p.name for p in kept.iterdir()} == set(result["faults"]) | {"unpaused.json"}


def test_reap_removes_only_finished_jobs(tmp_path):
    fake = FakeDocker(tmp_path)
    label = parallel_runtime.container_labels({"queue_root": str(tmp_path / "queue")})["ldm4geo.queue"]
    for name, status, receipt in (("done", "exited", True), ("lost", "exited", False), ("busy", "running", True),
                                  ("other", "exited", True)):
        job = tmp_path / name
        job.mkdir()
        if receipt:
            (job / "receipt.json").write_text("{}")
        fake.containers[name] = {"labels": {"ldm4geo.queue": label if name != "other" else "x",
                                            "ldm4geo.job_root": str(job)}, "state": {"Status": status}}
    result = launch.reap(tmp_path / "queue", docker=fake)
    assert result["removed"] == 1 and sorted(k["name"] for k in result["kept"]) == ["busy", "lost"]
    assert sorted(fake.containers) == ["busy", "lost", "other"]


# AT-37 two independent services; a scheduling pause recovers ---------------------

def test_at37_two_services_are_independent_and_recover_from_a_pause(tmp_path, docker, installed):
    policy = policy_for(tmp_path)
    queue = installed(policy, "S")
    manifest = write_suite(tmp_path / "release", "train", bodies=2)
    services = []
    for name in ("a", "b"):
        config = {"manifest": str(manifest), "root": str(tmp_path / name), "image": IMAGE, "seconds": 10,
                  "slots": 13, "caller_uid": 0, "socket": str(tmp_path / (name + ".sock"))}
        services.append(parallel_runtime.risk_service(config))
    a, b = services
    assert a.public_contract != b.public_contract and a.audit["sentinel"] != b.audit["sentinel"]
    request = lambda s, body: s.dispatch({"operation": "evaluate", "source": body, "contract_digest": s.public_contract})
    first = request(a, VALID)
    assert first["status"] == "valid"
    assert a.ledger.counters["algorithm_feedback_slots"] == 1 and b.ledger.counters.get("algorithm_feedback_slots", 0) == 0
    queue.pause(RuntimeError("injected infrastructure fault"))
    other = VALID + "# b\n"
    assert request(b, other) == {"status": "paused"}
    assert request(a, VALID + "# a\n") == {"status": "paused"}
    assert launch.unpause(policy["queue_root"])["status"] == "unpaused"
    assert request(b, other)["status"] == "valid"
    assert request(a, VALID + "# a\n")["status"] == "valid"
    # Pauses are not charged twice; each service kept its own budget and receipts.
    assert a.ledger.counters["algorithm_feedback_slots"] == 2 and b.ledger.counters["algorithm_feedback_slots"] == 1
    assert hashlib.sha256(other.encode()).hexdigest() not in {p.name for p in (tmp_path / "a/candidates").iterdir()}


class FakeProcess:
    def __init__(self, pid, code=None):
        self.pid, self.returncode = pid, code

    def poll(self):
        return self.returncode

    def wait(self):
        return self.returncode


class FakeHost:
    """Scripted services and controllers; records every spawn."""

    def __init__(self, root, outcomes):
        self.root, self.outcomes, self.spawned, self.live, self.next = Path(root), outcomes, [], set(), 100

    def account(self, user):
        return {"evaluator": (991, 991, [991]), "client": (1001, 1001, [1001])}[user]

    def chown(self, path, uid, gid):
        pass

    def alive(self, pid):
        return pid in self.live

    def stop(self, pid):
        self.live.discard(pid)

    def sleep(self, seconds):
        pass

    def spawn(self, user, argv, *, env, log, cwd):
        self.next += 1
        self.spawned.append((user, argv))
        mode = argv[argv.index(launch.RUNTIME_MODULE) + 1:]
        mode = next(m for m in mode if m in ("rsi-service", "rsi", "rsi-freeze"))
        if mode == "rsi-service":
            config = json.loads(Path(argv[argv.index("--config") + 1]).read_text())
            Path(config["socket"]).touch()
            policy = json.loads(Path(argv[argv.index("--policy") + 1]).read_text())
            with Path(log).open("a") as stream:
                stream.write(json.dumps({"status": "listening", "pid": self.next, "case_workers": 12,
                                         "contract_digest": digest(config), "policy": policy["policy_sha256"]}) + "\n")
            self.live.add(self.next)
            return FakeProcess(self.next)
        if mode == "rsi":
            run = Path(argv[argv.index("--out-dir") + 1])
            outcome = self.outcomes[run.name].pop(0)
            run.mkdir(parents=True, exist_ok=True)
            (run / "run-status.json").write_text(json.dumps(
                {"status": "completed", "stop_reason": "completed"} if outcome == "ok" else {"status": "paused"}))
            if outcome == "paused":
                parallel_runtime.JobQueue(self.queue, 1, 1).pause(RuntimeError("injected"))
            return FakeProcess(self.next, 0 if outcome == "ok" else 1)
        run_id = argv[argv.index("--run") + 1]
        selection = self.root / "private" / argv[argv.index("--protocol-id") + 1] / "selection" / argv[
            argv.index("--experiment") + 1] / run_id
        for prefix in argv[argv.index("--prefix") + 1::2]:
            folder = selection / (prefix if prefix == "full" else f"round_{int(prefix):02d}")
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "prefix.json").write_text("{}")
        return FakeProcess(self.next, 0)


def test_runtime_hands_the_tier_to_the_controller(tmp_path, monkeypatch):
    """The runtime parser also defines --tier; on H20 it swallowed the controller's tier."""
    policy = launch.make_policy({k: v for k, v in VALUES.items()}, tmp_path / "queue", tmp_path / "policy.json")
    seen = []
    from tasks.copper_inversion.rsi import workflow
    monkeypatch.setattr(workflow, "main", lambda argv: seen.append(argv) or 0)
    monkeypatch.setattr("sys.argv", ["runtime", "--policy", str(tmp_path / "policy.json"), "rsi", "--protocol", "p.json",
                                     "--tier", "S", "--seed", "8"])
    with pytest.raises(SystemExit):
        parallel_runtime.main()
    (argv,) = seen
    assert argv[argv.index("--tier") + 1] == "S" and argv.count("--tier") == 1 and policy["global_jobs"] == 8


def test_at37_launcher_starts_two_smoke_runs_and_resumes_after_a_pause(tmp_path, monkeypatch):
    from tasks.copper_inversion.rsi import workflow
    monkeypatch.setattr(workflow, "prepare", lambda args: None)
    monkeypatch.setattr(archive, "conditions", lambda run, service: {"engine": "ok", "budget": "ok", "blind": {
        "review_required": 0, "review_sha256": None}})
    reaped = []
    monkeypatch.setattr(launch, "reap", lambda queue: reaped.append(queue) or {"removed": 0, "kept": []})
    root = tmp_path / "active"
    policy = launch.make_policy(VALUES, tmp_path / "queue", tmp_path / "policy.json")
    site = {"root": str(root), "release": str(tmp_path / "release"), "source": str(tmp_path), "service_python": "py-e",
            "client_python": "py-c", "service_user": "evaluator", "client_user": "client", "image": IMAGE,
            "sidecar_image": "sidecar@sha256", "embedding_model_path": "/models/e", "harness_cache_dir": "/cache",
            "api_key_file": "/keys/k", "policy": str(tmp_path / "policy.json"),
            "public_catalogs": {"S": "/catalogs/S.json"}}
    batch = tmp_path / "batch.json"
    batch.write_text(json.dumps(plan([("ldm4geo", "standard", 7), ("accumulated", "standard", 7)],
                                     experiment="smoke", prefixes=["full", 2])))
    host = FakeHost(root, {"ldm4geo-s7": ["ok"], "accumulated-s7": ["paused", "ok"]})
    host.queue = policy["queue_root"]
    launcher = launch.Launcher(site, batch, host=host)
    with pytest.raises(RuntimeError, match="accumulated-s7"):
        launcher.start()
    state = launcher.status()
    assert state["runs"]["ldm4geo-s7"]["status"] == "finished"
    assert state["runs"]["accumulated-s7"]["status"] == "needs_reconciliation" and state["paused"]
    services = [state["runs"][n]["service"] for n in ("ldm4geo-s7", "accumulated-s7")]
    assert services[0]["socket"] != services[1]["socket"]
    assert services[0]["contract_digest"] != services[1]["contract_digest"]
    configs = [json.loads((launcher.paths(n)["service"] / "service-config.json").read_text())
               for n in ("ldm4geo-s7", "accumulated-s7")]
    assert configs[0]["root"] != configs[1]["root"] and configs[0]["tier"] == "S"
    assert {c["caller_uid"] for c in configs} == {1001}
    assert [u for u, _ in host.spawned] == ["evaluator", "evaluator", "client", "client"]
    assert all("rsi-service" in a for _, a in host.spawned[:2]) and all(" rsi " in " ".join(a) for _, a in host.spawned[2:])
    controller = host.spawned[2][1]
    assert controller[controller.index("--out-dir") + 1] == str(root / "runs" / launcher.pid / "smoke" / "ldm4geo-s7")
    assert "--resume" not in controller and "--api-key-file" in controller
    # A paused scheduler must be reconciled and unpaused before resume.
    with pytest.raises(launch.PlanError, match="paused"):
        launcher.resume()
    launch.unpause(policy["queue_root"])
    state = launch.Launcher(site, batch, host=host).resume()
    assert all(entry["status"] == "finished" for entry in state["runs"].values())
    assert host.spawned[-1][1][-1] == "--resume" and len(host.spawned) == 5
    assert state["runs"]["accumulated-s7"]["resumes"] == 1
    state = launch.Launcher(site, batch, host=host).finish()
    freezes = [a for _, a in host.spawned[5:]]
    assert len(freezes) == 2 and all(u == "evaluator" for u, _ in host.spawned[5:])
    assert all("rsi-freeze" in a and a[a.index("--prefix") + 1::2] == ["full", "2"] for a in freezes)
    listing = json.loads(Path(state["frozen_list"]).read_text())
    assert listing["solvers"] == [{"run_id": r, "prefix": p} for r in ("accumulated-s7", "ldm4geo-s7")
                                  for p in ("full", "round_02")]
    assert host.live == set() and reaped == [policy["queue_root"]] and state["stage"] == "frozen_and_validated"


def test_one_batch_holds_the_machine(tmp_path):
    site = {"root": str(tmp_path / "active"), "release": "r", "source": "s", "service_python": "p",
            "client_python": "p", "service_user": "evaluator", "client_user": "client", "image": IMAGE,
            "sidecar_image": "x", "embedding_model_path": "m", "harness_cache_dir": "c", "api_key_file": "k",
            "policy": str(tmp_path / "policy.json"), "public_catalogs": {"S": "x"}}
    launch.make_policy(VALUES, tmp_path / "queue", tmp_path / "policy.json")
    batch = tmp_path / "batch.json"
    batch.write_text(json.dumps(plan([("ldm4geo", "standard", 7)], experiment="smoke")))
    first, second = launch.Launcher(site, batch), launch.Launcher(site, batch)
    with first.machine():
        with pytest.raises(launch.PlanError, match="one at a time"):
            with second.machine():
                pass
    with second.machine():
        pass


# AT-38 archive ---------------------------------------------------------------------

def finished_run(tmp_path):
    """A scripted run in the contracts §1.1 layout, sealed, with a frozen prefix."""
    pid, exp, name = "rsi-dev-0123456789abcdef", "smoke", "ldm4geo-s7"
    root = tmp_path / "active"
    manifest = write_suite(tmp_path / "release", "train")
    service = make_service(tmp_path, manifest, name=f"active/private/{pid}/services/{exp}/{name}")
    run = root / "runs" / pid / exp / name
    run.parent.mkdir(parents=True)
    config = Config("ldm4geo", 7, rounds=3, proposals_per_branch=2, batch=4, reset_period=2, token_cap_input=0)
    identity = {"run_id": name, "experiment": exp, "protocol_id": pid, "code_commit": "abc", "synthetic_test": True}
    Controller(run, config, RecordingAgent(), LoopbackClient(run, service), scripted_select,
               "Fixed task brief: minimize the reported risk.", identity).run()
    service.seal_training()
    selection = root / "private" / pid / "selection" / exp / name / "full"
    selection.mkdir(parents=True)
    (selection / "prefix.json").write_text(json.dumps({"prefix": "full"}))
    (selection / "frozen.json").write_text("{}")
    return root, pid, exp, name


def test_at38_archive_equals_the_active_copy_and_lists_every_file(tmp_path):
    root, pid, exp, name = finished_run(tmp_path)
    store = tmp_path / "archive"
    report = archive.create(root, store, pid, exp, name, protocol=EXAMPLE)
    assert report["status"] == "passed"
    assert {p: v["active_copy"] for p, v in report["packages"].items()} == {"public": "identical", "private": "identical"}
    target = store / pid / exp / name
    record = json.loads((target / "archive-manifest.json").read_text())
    assert record["run_manifest"]["run_id"] == name and record["protocol"] == protocol()
    assert record["conditions"]["budget"]["status"] == "passed" and record["conditions"]["briefs"]["status"] == "passed"
    paths = launch.layout(root, pid, exp, name)
    for package, sources in (("public", [paths["run"]]), ("private", [paths["service"], paths["run_selection"]])):
        files = [p for source in sources for p in source.rglob("*") if p.is_file() and not p.is_symlink()]
        summary = record["packages"][package]
        assert summary["files"] == len(files) and summary["content_bytes"] == sum(p.stat().st_size for p in files)
        assert summary["sha256"] == hashlib.sha256((target / summary["file"]).read_bytes()).hexdigest()
    # Raw provider requests are archived.
    with tarfile.open(target / "public.tar.gz") as package:
        assert any(n.endswith(".request.bin") for n in package.getnames())
    # Self-describing after a move off the active area.
    moved = tmp_path / "elsewhere"
    shutil.copytree(target, moved)
    assert archive.verify(moved)["status"] == "passed"
    assert archive.create(root, store, pid, exp, name)["status"] == "passed"  # idempotent
    # Unpacked read-only for the analysis recompute (T8 reads <exp>/<run_id>/).
    unpacked = Path(archive.unpack(moved, tmp_path / "unpacked")["public"])
    assert unpacked == tmp_path / "unpacked" / exp / name
    active = {str(q.relative_to(paths["run"])): q.read_bytes() for q in paths["run"].rglob("*") if q.is_file()}
    assert {str(q.relative_to(unpacked)): q.read_bytes() for q in unpacked.rglob("*") if q.is_file()} == active
    # Any difference is refused.
    (paths["run"] / "result.json").write_text("{}")
    with pytest.raises(ValueError, match="active public copy differs"):
        archive.verify(target, root=root)
    body = bytearray((moved / "private.tar.gz").read_bytes())
    body[-1] ^= 1
    (moved / "private.tar.gz").write_bytes(bytes(body))
    with pytest.raises(ValueError, match="private package differs"):
        archive.verify(moved)
    assert archive.verify(moved, packages=("public",))["status"] == "passed"


def test_archive_refuses_unfinished_or_unreconciled_runs(tmp_path):
    root, pid, exp, name = finished_run(tmp_path)
    paths = launch.layout(root, pid, exp, name)
    ledger = paths["run"] / "ledger.jsonl"
    lines = ledger.read_text().splitlines(keepends=True)
    ledger.write_text("".join(lines[:-1]))
    with pytest.raises(ValueError):
        archive.create(root, tmp_path / "archive", pid, exp, name)
    ledger.write_text("".join(lines))
    shutil.rmtree(paths["run_selection"])
    with pytest.raises(ValueError, match="frozen"):
        archive.create(root, tmp_path / "archive", pid, exp, name)
    assert not (tmp_path / "archive" / pid / exp / name / "archive-manifest.json").exists()
