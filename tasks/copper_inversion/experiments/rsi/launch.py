"""Batch launcher: one evaluator service and one controller per run, a batch at a time.

Run as root on the authorized host. Every value comes from three files:

- the batch (run plan): `{"schema": "rsi-batch-v1", "name", "protocol", "experiment", "tier",
  "runs": [{"arm", "judge", "seed"}, ...], "prefixes": ["full", 5, 10], "checkpoint_round": 10}`;
  an E1/E2 run adds `"continuation": "<source run>/checkpoints/round_<t>", "mode", "replicate"`
  and, for E2, `"planted": true`; its service imports the checkpoint's ledger prefix;
- the site: accounts, interpreters, release, images, catalogs, key file, policy (see SITE_KEYS);
- the scheduling policy written by `policy` (parallelism.md §6).

Rules enforced before anything starts: the protocol defines every arm, judge, tier
and seed (the entry point's own checks run for every run); every arm x judge cell
of the batch has the same number of runs and the same seeds (rule 3, AT-46);
the batch fits max_concurrent_runs and max_agent_sessions; one batch holds the
machine at a time, so the full tier always runs alone (rule 4); formal
experiments need a passed load qualification matching the policy (AT-44).

Each service runs as the evaluator account (`parallel_runtime rsi-service`, the
threaded risk-only service); each controller as the client account
(`parallel_runtime rsi`). After the runs: reconciliation (engine status, slot
budgets, blind audit, brief rebuild), training seal check, freeze and validation
through T7's freeze as the evaluator account, the frozen-solver list for the later
unseal, service shutdown and container reaping. Nothing here unseals.

    launch.py policy --values V.json --queue-root Q --out policy.json
    launch.py check|start|resume|finish|status --site site.json --batch batch.json
    launch.py unpause --site site.json
    launch.py reap --site site.json
"""

import argparse
import fcntl
import hashlib
import json
import os
import pwd
import re
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from ldm_tts.engine.run_store import atomic_json_write
from tasks.copper_inversion.core import execution
from tasks.copper_inversion.core.execution import digest
from tasks.copper_inversion.experiments.rsi.freeze import FULL, layout
from tasks.copper_inversion.experiments.rsi.unseal import REVIEW_FILE
from tasks.copper_inversion.rsi import protocol as protocols

SCHEMA = "rsi-batch-v1"
SITE_KEYS = frozenset({"root", "release", "source", "service_python", "client_python", "service_user",
                       "client_user", "image", "sidecar_image", "embedding_model_path", "harness_cache_dir",
                       "api_key_file", "policy", "public_catalogs"})
SITE_OPTIONAL = frozenset({"embedding_devices", "load_qualification", "archive_root", "poll_seconds",
                           "service_start_seconds"})
RUN_KEYS = frozenset({"arm", "judge", "seed"})
CONTINUATION_KEYS = frozenset({"continuation", "mode", "replicate", "planted"})
POLICY_REQUIRED = ("queue_root", "global_jobs", "interface_slots", "startup_slots", "jobs_per_process")
RUNTIME = Path(__file__).resolve().parents[1] / "v2" / "parallel_runtime.py"
RUNTIME_MODULE = "tasks.copper_inversion.experiments.v2.parallel_runtime"
NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")


class PlanError(ValueError):
    """The batch plan is refused before anything starts."""


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# Plan ------------------------------------------------------------------------------

def load_batch(path):
    batch = json.loads(Path(path).read_text())
    if not isinstance(batch, dict) or batch.get("schema") != SCHEMA:
        raise PlanError("not an rsi batch plan")
    if set(batch) - {"schema", "name", "protocol", "experiment", "tier", "runs", "prefixes", "checkpoint_round"} or not {
            "name", "protocol", "experiment", "tier", "runs"} <= set(batch):
        raise PlanError("batch plan fields differ from " + SCHEMA)
    if not isinstance(batch["name"], str) or NAME.fullmatch(batch["name"]) is None:
        raise PlanError("invalid batch name")
    if not isinstance(batch["runs"], list) or not batch["runs"]:
        raise PlanError("a batch needs runs")
    for run in batch["runs"]:
        if not isinstance(run, dict) or not RUN_KEYS <= set(run) <= RUN_KEYS | CONTINUATION_KEYS or type(run["seed"]) is not int:
            raise PlanError("each run is {arm, judge, seed}, plus {continuation, mode, replicate[, planted]} for E1/E2")
        if (set(run) & CONTINUATION_KEYS) and not {"continuation", "mode", "replicate"} <= set(run):
            raise PlanError("a continuation run names its checkpoint, mode and replicate")
    if batch.get("checkpoint_round") is not None and (type(batch["checkpoint_round"]) is not int or batch["checkpoint_round"] < 1):
        raise PlanError("checkpoint_round is a positive round")
    prefixes = batch.get("prefixes", [FULL])
    if not prefixes or any(p != FULL and (type(p) is not int or p < 1) for p in prefixes) or len(set(prefixes)) != len(prefixes):
        raise PlanError("prefixes are 'full' or positive rounds, without repeats")
    return batch


def run_id(experiment, run):
    from tasks.copper_inversion.rsi import workflow
    if "continuation" in run:
        return workflow.continuation_run_id(run["seed"], run["mode"], run["replicate"], run.get("planted", False))
    return workflow.run_id(experiment, run["arm"], run["judge"], run["seed"])


def check_batch(batch, protocol, policy):
    """Refuse an unbalanced or oversized batch (parallelism.md §5 rules 3-4, §6)."""
    experiment, tier = batch["experiment"], batch["tier"]
    if experiment not in protocols.EXPERIMENTS:
        raise PlanError("unknown experiment")
    if tier not in protocol.get("tiers", {}):
        raise PlanError("tier is not defined by the protocol")
    seeds = protocol["seeds"].get(experiment)
    allowed = None if seeds is None else set(seeds) | set(protocol["seeds"].get(experiment + "_reserve", []))
    for run in batch["runs"]:
        if run["arm"] not in protocol["arms"]:
            raise PlanError("arm is not defined by the protocol: " + str(run["arm"]))
        if run["judge"] not in protocol["judges"]:
            raise PlanError("judge is not defined by the protocol: " + str(run["judge"]))
        if allowed is not None and run["seed"] not in allowed:
            raise PlanError(f"seed {run['seed']} is not in the protocol seed table for {experiment}")
    if not protocols.is_development(protocol):
        check_frozen_prefixes(batch)
    names = [run_id(experiment, run) for run in batch["runs"]]
    if len(set(names)) != len(names):
        raise PlanError("a run appears twice in the batch")
    # A cell is one treatment: arm, continuation mode and planting, under one judge.
    cells = {}
    for run in batch["runs"]:
        treatment = (run["arm"], run.get("mode"), bool(run.get("planted", False)))
        cells.setdefault((treatment, run["judge"]), []).append((run["seed"], run.get("replicate")))
    treatments = sorted({t for t, _ in cells}, key=str)
    arms = sorted({t[0] for t in treatments})
    judges = sorted({judge for _, judge in cells})
    if set(cells) != {(t, judge) for t in treatments for judge in judges}:
        raise PlanError("every arm must run under every judge of the batch")
    seed_sets = {cell: sorted(values, key=str) for cell, values in cells.items()}
    if len({tuple(values) for values in seed_sets.values()}) != 1:
        label = lambda t, j: "/".join(str(x) for x in (*t, j) if x not in (None, False))
        raise PlanError("arms differ in run count or seeds within the batch: "
                        + json.dumps({label(t, j): [s for s, _ in v] for (t, j), v in sorted(seed_sets.items(), key=str)}))
    limit = policy.get("max_concurrent_runs", {})
    limit = limit.get(tier) if isinstance(limit, dict) else limit
    if limit is not None and len(names) > limit:
        raise PlanError(f"batch of {len(names)} runs exceeds max_concurrent_runs {limit} for tier {tier}")
    sessions = sum(protocol["arms"][run["arm"]]["branches"] for run in batch["runs"])
    if policy.get("max_agent_sessions") is not None and sessions > policy["max_agent_sessions"]:
        raise PlanError(f"batch needs {sessions} agent sessions, policy allows {policy['max_agent_sessions']}")
    return {"experiment": experiment, "tier": tier, "runs": names, "arms": arms, "judges": judges,
            "seeds": sorted({s for s, _ in next(iter(seed_sets.values()))}), "agent_sessions": sessions}


def check_frozen_prefixes(batch):
    """A formal batch freezes exactly the prefixes the G3-frozen analysis config names for its
    experiment, and only M1 batches take checkpoints (at the config's checkpoint round)."""
    from tasks.copper_inversion.experiments.rsi.analysis.bundle import DEFAULTS
    expected = DEFAULTS["freeze_prefixes"].get(batch["experiment"])
    if expected is not None and sorted(batch.get("prefixes", [FULL]), key=str) != sorted(expected, key=str):
        raise PlanError(f"{batch['experiment']} batches freeze prefixes {expected} (analysis config)")
    ck = batch.get("checkpoint_round")
    if ck is not None and (batch["experiment"] != DEFAULTS["checkpoint_experiment"] or ck != DEFAULTS["checkpoint_round"]):
        raise PlanError(f"only {DEFAULTS['checkpoint_experiment']} batches take checkpoints, at round "
                        f"{DEFAULTS['checkpoint_round']}")


# Policy ----------------------------------------------------------------------------

def make_policy(values, queue_root, out, runtime=RUNTIME):
    """Create the queue lock files and write the hashed scheduling policy."""
    values = dict(values)
    if set(values) & {"queue_root", "runtime_sha256", "policy_sha256"}:
        raise ValueError("queue_root and hashes are filled in here")
    missing = [k for k in POLICY_REQUIRED[1:] if k not in values]
    if missing:
        raise ValueError("policy values miss " + ", ".join(missing))
    queue = Path(queue_root).resolve()
    (queue / "startup").mkdir(parents=True, exist_ok=True)
    for folder, count in ((queue, values["global_jobs"]), (queue / "startup", values["startup_slots"])):
        folder.chmod(0o777)
        for index in range(count):
            lock = folder / f"{index:03}.lock"
            lock.touch()
            lock.chmod(0o666)
    policy = dict(values, queue_root=str(queue), runtime_sha256=_sha(runtime))
    policy["policy_sha256"] = digest(policy)
    atomic_json_write(out, policy)
    Path(out).chmod(0o644)
    return policy


def load_policy(path, runtime=RUNTIME):
    policy = json.loads(Path(path).read_text())
    if any(k not in policy for k in POLICY_REQUIRED + ("runtime_sha256", "policy_sha256")):
        raise PlanError("scheduling policy is incomplete")
    if digest({k: v for k, v in policy.items() if k != "policy_sha256"}) != policy["policy_sha256"]:
        raise PlanError("scheduling policy changed after it was hashed")
    if _sha(runtime) != policy["runtime_sha256"]:
        raise PlanError("scheduling runtime changed after the policy was written")
    return policy


def check_qualification(site, policy, experiment):
    """AT-44: formal experiments run only at a qualified slot count."""
    if experiment in protocols.DEVELOPMENT_EXPERIMENTS:
        return None
    path = site.get("load_qualification")
    if path is None or policy.get("load_qualification_sha256") is None:
        raise PlanError("formal experiments need a load qualification named by the policy")
    if _sha(path) != policy["load_qualification_sha256"]:
        raise PlanError("load qualification record differs from the policy")
    record = json.loads(Path(path).read_text())
    if record.get("status") != "passed" or record.get("global_jobs") != policy["global_jobs"]:
        raise PlanError("load qualification did not pass at the policy's slot count")
    return record["status"]


def unpause(queue_root):
    """Clear a global pause after reconciliation; the faults are kept, never deleted."""
    queue = Path(queue_root)
    faults = sorted(queue.glob("fault-*.json")) + sorted((queue / "startup").glob("fault-*.json"))
    paused = [p for p in (queue / "PAUSED.json", queue / "startup" / "PAUSED.json") if p.exists()]
    if not paused and not faults:
        return {"status": "not_paused"}
    kept = queue / "faults" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    kept.mkdir(parents=True)
    moved = []
    for path in paused + faults:
        name = ("startup-" if path.parent.name == "startup" else "") + path.name
        if path.name == "PAUSED.json":
            path.unlink()  # a hard link to one of the fault files
        else:
            path.rename(kept / name)
            moved.append(name)
    atomic_json_write(kept / "unpaused.json", {"time": _now(), "faults": moved})
    return {"status": "unpaused", "faults": moved, "kept": str(kept)}


def reap(queue_root, docker=None):
    """Remove stopped containers of this queue whose job has a receipt; list the rest."""
    docker = docker or execution.docker
    label = "ldm4geo.queue=" + execution.digest(str(Path(queue_root).resolve()))[:16]
    listing = docker("ps", "-a", "--filter", "label=" + label, "--format",
                     '{{.Names}}\t{{.State}}\t{{.Label "ldm4geo.job_root"}}').stdout
    removed, kept = [], []
    for line in listing.splitlines():
        if not line.strip():
            continue
        name, state, job_root = (line.split("\t") + ["", ""])[:3]
        finished = state.lower() in ("exited", "created", "dead")
        if finished and job_root and (Path(job_root) / "receipt.json").is_file():
            docker("rm", "-f", name, check=False)
            removed.append(name)
        else:
            # Running, or stopped without a receipt: evidence for reconciliation.
            kept.append({"name": name, "state": state, "job_root": job_root})
    return {"removed": len(removed), "kept": kept}


# Host --------------------------------------------------------------------------------

class Host:
    """Operating-system actions; tests substitute a fake."""

    def account(self, user):
        entry = pwd.getpwnam(user)
        return entry.pw_uid, entry.pw_gid, os.getgrouplist(user, entry.pw_gid)

    def spawn(self, user, argv, *, env, log, cwd):
        uid, gid, groups = self.account(user)
        with Path(log).open("ab") as stream:
            return subprocess.Popen(argv, cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=stream,
                                    stderr=subprocess.STDOUT, user=uid, group=gid, extra_groups=groups,
                                    start_new_session=True)

    def chown(self, path, uid, gid):
        os.chown(path, uid, gid)

    def alive(self, pid):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    def stop(self, pid):
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass

    def sleep(self, seconds):
        time.sleep(seconds)


# Launcher ----------------------------------------------------------------------------

class Launcher:
    def __init__(self, site, batch, host=None):
        self.site = json.loads(Path(site).read_text()) if not isinstance(site, dict) else dict(site)
        if not SITE_KEYS <= set(self.site) <= SITE_KEYS | SITE_OPTIONAL:
            raise PlanError("site fields differ: need " + ", ".join(sorted(SITE_KEYS)))
        self.batch_path = Path(batch).resolve()
        self.batch = load_batch(self.batch_path)
        self.protocol_path = Path(self.batch["protocol"]).resolve()
        self.protocol = protocols.load(self.protocol_path)
        self.pid = protocols.protocol_id(self.protocol)
        self.policy_path = Path(self.site["policy"]).resolve()
        self.policy = load_policy(self.policy_path)
        self.host = host or Host()
        self.root = Path(self.site["root"]).resolve()
        self.experiment, self.tier = self.batch["experiment"], self.batch["tier"]
        self.home = self.root / "launcher" / self.batch["name"]
        self.state_path = self.home / "state.json"

    # paths and checks
    def runs(self):
        return [(run_id(self.experiment, run), run) for run in self.batch["runs"]]

    def paths(self, name):
        paths = layout(self.root, self.pid, self.experiment, name)
        paths["socket_dir"] = self.root / "rpc" / digest([self.pid, self.experiment, name])[:12]
        return paths

    def check(self):
        summary = check_batch(self.batch, self.protocol, self.policy)
        summary["load_qualification"] = check_qualification(self.site, self.policy, self.experiment)
        image = self.site["image"]
        if not image.startswith("sha256:") or len(image) != 71:
            raise PlanError("pin the evaluator image digest")
        frozen = self.protocol["hashes"].get("evaluator_image")
        if frozen is not None and frozen != image:
            raise PlanError("evaluator image differs from the frozen protocol digest")
        if self.tier not in self.site["public_catalogs"]:
            raise PlanError("site names no public catalog for tier " + self.tier)
        from tasks.copper_inversion.rsi import workflow
        for name, run in self.runs():
            # The entry point's own refusals (seed table, freeze, hashes, run name) before any start.
            workflow.prepare(workflow.parse_args(self.controller_argv(name, run, {"socket": "unused",
                             "contract_digest": "0" * 64, "service_uid": 0}, resume=False)[6:]))
        return summary

    def manifest_path(self, run):
        judge = self.protocol["judges"][run["judge"]]
        return Path(self.site["release"]) / judge.get("train_manifest", self.protocol["tiers"][self.tier]["train_manifest"])

    def env(self):
        env = {k: v for k, v in os.environ.items() if k != "LLM_API_KEY"}
        env.update(PYTHONPATH=str(self.site["source"]), PYTHONDONTWRITEBYTECODE="1",
                   OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
        return env

    # services
    def start_service(self, name, run):
        """Evaluator-account service, permission model of rsi/launch_service.py."""
        paths = self.paths(name)
        private, sockets = paths["service"], paths["socket_dir"]
        uid, gid, _ = self.host.account(self.site["service_user"])
        client_uid, client_gid, _ = self.host.account(self.site["client_user"])
        if uid == client_uid:
            raise PlanError("research and evaluator accounts must differ")
        config_path = private / "service-config.json"
        socket_path = sockets / "e.sock"
        if len(os.fsencode(socket_path)) >= 104:
            raise PlanError("Unix socket path is too long")
        if not config_path.exists():
            for folder in (self.root / "private" / self.pid, private.parent):
                if not folder.exists():
                    folder.mkdir(parents=True, mode=0o700)
                    self.host.chown(folder, uid, gid)
            private.mkdir(mode=0o700)
            self.host.chown(private, uid, gid)
            sockets.mkdir(parents=True, mode=0o750)
            self.host.chown(sockets, uid, client_gid)
            tier = self.protocol["tiers"][self.tier]
            config = {"manifest": str(self.manifest_path(run).resolve()), "root": str(private),
                      "image": self.site["image"], "seconds": tier["solver_seconds"], "slots": tier["slots"],
                      "caller_uid": client_uid, "socket": str(socket_path),
                      "objective": self.protocol["judges"][run["judge"]]["objective"], "tier": self.tier}
            if "continuation" in run:
                config["import"] = self.import_spec(run, private, uid, gid)
            config_path.write_text(json.dumps(config, sort_keys=True))
            self.host.chown(config_path, uid, gid)
            config_path.chmod(0o600)
            for folder in (private / "tmp", private / "tmp" / "docker"):
                folder.mkdir(mode=0o700)
                self.host.chown(folder, uid, gid)
        elif socket_path.exists():
            # Restart of a stopped service: its old socket has no listener left.
            socket_path.unlink()
        env = {**self.env(), "TMPDIR": str(private / "tmp"), "DOCKER_CONFIG": str(private / "tmp" / "docker")}
        log = private / "service.log"
        argv = [self.site["service_python"], "-m", RUNTIME_MODULE, "--policy", str(self.policy_path),
                "--tier", self.tier, "rsi-service", "--config", str(config_path)]
        process = self.host.spawn(self.site["service_user"], argv, env=env, log=log, cwd=self.site["source"])
        # An import re-projects the whole prefix before the service listens.
        deadline = time.monotonic() + self.site.get("service_start_seconds", 600 if "continuation" in run else 120)
        ready = None
        while ready is None:
            if socket_path.exists() and log.exists():
                for line in log.read_text().splitlines():
                    if line.startswith("{") and '"listening"' in line:
                        row = json.loads(line)
                        if row.get("pid") == process.pid:
                            ready = row
            if ready is None:
                if process.poll() is not None or time.monotonic() >= deadline:
                    if process.poll() is None:
                        self.host.stop(process.pid)
                    raise RuntimeError("evaluator service did not start: " + name)
                self.host.sleep(0.2)
        if ready["policy"] != self.policy["policy_sha256"]:
            raise RuntimeError("service runs a different scheduling policy: " + name)
        self.host.chown(socket_path, uid, client_gid)
        socket_path.chmod(0o660)
        deployment = {"socket": str(socket_path), "contract_digest": ready["contract_digest"], "service_uid": uid,
                      "pid": process.pid, "case_workers": ready.get("case_workers")}
        (sockets / "deployment.json").write_text(json.dumps(deployment, sort_keys=True) + "\n")
        return deployment

    def import_spec(self, run, private, uid, gid):
        """Continuation service: import the source run's ledger prefix (T4/T6 import hook)."""
        checkpoint = Path(run["continuation"]).resolve()
        source_run = checkpoint.parent.parent
        source = json.loads((source_run / "manifest.json").read_text())
        if source.get("protocol_id") != self.pid:
            raise PlanError("checkpoint belongs to another protocol")
        source_service = layout(self.root, self.pid, source["experiment"], source["run_id"])["service"]
        if not (source_service / "identity.json").is_file():
            raise PlanError("source service directory of the checkpoint is missing")
        prefix = (checkpoint / "ledger-prefix.jsonl").read_bytes()
        copy = private / "import-ledger-prefix.jsonl"
        copy.write_bytes(prefix)
        self.host.chown(copy, uid, gid)
        copy.chmod(0o400)
        return {"source_root": str(source_service), "ledger_prefix": str(copy),
                "ledger_prefix_sha256": hashlib.sha256(prefix).hexdigest()}

    # controllers
    def controller_argv(self, name, run, deployment, resume):
        devices = self.site.get("embedding_devices") or ["cuda:0"]
        index = [n for n, _ in self.runs()].index(name)
        argv = [self.site["client_python"], "-m", RUNTIME_MODULE, "--policy", str(self.policy_path), "rsi",
                "--protocol", str(self.protocol_path), "--experiment", self.experiment, "--arm", run["arm"],
                "--tier", self.tier, "--judge", run["judge"], "--seed", str(run["seed"]),
                "--out-dir", str(self.paths(name)["run"]),
                "--public-catalog", str(self.site["public_catalogs"][self.tier]),
                "--evaluation-socket", deployment["socket"], "--evaluation-contract", deployment["contract_digest"],
                "--evaluation-service-uid", str(deployment["service_uid"]),
                "--embedding-model-path", str(self.site["embedding_model_path"]),
                "--embedding-device", devices[index % len(devices)],
                "--sidecar-image", self.site["sidecar_image"], "--harness-cache-dir", str(self.site["harness_cache_dir"]),
                "--api-key-file", str(self.site["api_key_file"])]
        if "continuation" in run:
            argv += ["--continuation", str(run["continuation"]), "--continuation-mode", run["mode"],
                     "--replicate", str(run["replicate"])] + (["--planted"] if run.get("planted") else [])
        elif self.batch.get("checkpoint_round") is not None:
            argv += ["--checkpoint-round", str(self.batch["checkpoint_round"])]
        return argv + (["--resume"] if resume else [])

    def start_controller(self, name, run, deployment, resume):
        paths = self.paths(name)
        client_uid, client_gid, _ = self.host.account(self.site["client_user"])
        if not paths["runs"].exists():
            paths["runs"].mkdir(parents=True)
            for folder in [paths["runs"], *paths["runs"].parents]:
                if folder == self.root:
                    break
                self.host.chown(folder, client_uid, client_gid)
        if not resume and paths["run"].exists():
            raise PlanError("run directory already exists: " + name)
        log = paths["runs"] / f"{name}.controller.log"
        process = self.host.spawn(self.site["client_user"], self.controller_argv(name, run, deployment, resume),
                                  env=self.env(), log=log, cwd=self.site["source"])
        return process

    # state
    def load_state(self):
        state = json.loads(self.state_path.read_text())
        if state["batch_sha256"] != _sha(self.batch_path) or state["policy_sha256"] != self.policy["policy_sha256"]:
            raise PlanError("batch plan or policy changed since the batch started")
        return state

    def save_state(self, state):
        state["updated_at"] = _now()
        atomic_json_write(self.state_path, state)
        self.state_path.chmod(0o644)

    @contextmanager
    def machine(self):
        """One batch holds the machine (rules 3-4) while a command runs."""
        (self.root / "launcher").mkdir(parents=True, exist_ok=True)
        with (self.root / "launcher" / "machine.lock").open("a+") as stream:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise PlanError("another batch holds the machine; batches run one at a time") from None
            stream.seek(0)
            stream.truncate()
            stream.write(json.dumps({"batch": self.batch["name"], "pid": os.getpid(), "since": _now()}))
            stream.flush()
            yield

    # commands
    def start(self):
        summary = self.check()
        with self.machine():
            return self._start(summary)

    def _start(self, summary):
        if self.state_path.exists():
            raise PlanError("batch already started; use resume")
        self.home.mkdir(parents=True, exist_ok=True)
        state = {"schema": "rsi-batch-state-v1", "batch": self.batch["name"], "batch_sha256": _sha(self.batch_path),
                 "protocol_id": self.pid, "policy_sha256": self.policy["policy_sha256"], "started_at": _now(),
                 "summary": summary, "runs": {}}
        processes = {}
        # Services first (interleaved by plan order), then controllers, so no arm starts ahead.
        for name, run in self.runs():
            state["runs"][name] = {"run": run, "service": self.start_service(name, run), "status": "starting"}
            self.save_state(state)
        for name, run in self.runs():
            processes[name] = self.start_controller(name, run, state["runs"][name]["service"], resume=False)
            state["runs"][name].update(controller_pid=processes[name].pid, status="running")
            self.save_state(state)
        return self.monitor(state, processes)

    def resume(self):
        """After `unpause`: restart dead services and every unfinished controller with --resume."""
        if (Path(self.policy["queue_root"]) / "PAUSED.json").exists():
            raise PlanError("the scheduler is paused; reconcile, then unpause before resume")
        with self.machine():
            return self._resume()

    def _resume(self):
        state = self.load_state()
        processes = {}
        for name, run in self.runs():
            entry = state["runs"][name]
            if entry["status"] == "finished":
                continue
            if not self.host.alive(entry["service"]["pid"]):
                restarted = self.start_service(name, run)
                if restarted["contract_digest"] != entry["service"]["contract_digest"]:
                    raise RuntimeError("restarted service measures a different contract: " + name)
                entry["service"] = restarted
            if entry.get("controller_pid") and self.host.alive(entry["controller_pid"]):
                raise PlanError("controller still running: " + name)
            # A controller that never wrote its run directory starts fresh.
            processes[name] = self.start_controller(name, run, entry["service"],
                                                    resume=self.paths(name)["run"].exists())
            entry.update(controller_pid=processes[name].pid, status="running",
                         resumes=entry.get("resumes", 0) + 1)
            self.save_state(state)
        return self.monitor(state, processes)

    def monitor(self, state, processes):
        queue = Path(self.policy["queue_root"])
        while True:
            running = [name for name, p in processes.items() if p.poll() is None]
            state["paused"] = (queue / "PAUSED.json").exists()
            for name, process in processes.items():
                if process.poll() is not None and state["runs"][name]["status"] == "running":
                    status_path = self.paths(name)["run"] / "run-status.json"
                    status = json.loads(status_path.read_text()) if status_path.exists() else {}
                    ok = process.returncode == 0 and status.get("status") in ("completed", "stopped")
                    state["runs"][name].update(status="finished" if ok else "needs_reconciliation",
                                               exit_code=process.returncode, run_status=status)
            self.save_state(state)
            if not running:
                break
            self.host.sleep(self.site.get("poll_seconds", 15))
        failed = sorted(n for n, e in state["runs"].items() if e["status"] != "finished")
        state["stage"] = "runs_finished" if not failed else "needs_reconciliation"
        self.save_state(state)
        if failed:
            # Services stay up; reconcile, `unpause`, then `resume` (global pause semantics).
            raise RuntimeError("runs need reconciliation before resume: " + ", ".join(failed))
        return state

    def finish(self):
        """Reconcile, check the seal, freeze and validate every run; never unseal."""
        with self.machine():
            return self._finish()

    def _finish(self):
        from tasks.copper_inversion.experiments.rsi.archive import conditions
        state = self.load_state()
        unfinished = sorted(n for n, e in state["runs"].items() if e["status"] != "finished")
        if unfinished:
            raise PlanError("unfinished runs: " + ", ".join(unfinished))
        reconciliation = {}
        for name, _ in self.runs():
            paths = self.paths(name)
            reconciliation[name] = conditions(paths["run"], paths["service"])
            blind = reconciliation[name]["blind"]
            if blind["review_required"]:
                # unseal.py refuses this run until a reviewer signs off this exact hit list.
                atomic_json_write(paths["service"] / REVIEW_FILE, {
                    "run_id": name, "review_sha256": blind["review_sha256"], "hits": blind["review_required"]})
        state["reconciliation"] = reconciliation
        state["stage"] = "reconciled"
        self.save_state(state)
        prefixes = [str(p) for p in self.batch.get("prefixes", [FULL])]
        processes = {}
        for name, _ in self.runs():
            argv = [self.site["service_python"], "-m", RUNTIME_MODULE, "--policy", str(self.policy_path),
                    "--tier", self.tier, "rsi-freeze", "--root", str(self.root), "--protocol-id", self.pid,
                    "--experiment", self.experiment, "--run", name, "--release", str(self.site["release"]),
                    "--protocol", str(self.protocol_path)]
            for prefix in prefixes:
                argv += ["--prefix", prefix]
            log = self.paths(name)["service"] / "freeze.log"
            processes[name] = self.host.spawn(self.site["service_user"], argv, env=self.env(), log=log,
                                              cwd=self.site["source"])
        codes = {name: p.wait() for name, p in processes.items()}
        if any(codes.values()):
            state["stage"] = "freeze_failed"
            state["freeze_exit_codes"] = codes
            self.save_state(state)
            raise RuntimeError("freeze or validation failed; inspect the private freeze.log files")
        state["frozen_list"] = str(self.frozen_list())
        for entry in state["runs"].values():
            self.host.stop(entry["service"]["pid"])
        state["containers"] = reap(self.policy["queue_root"])
        state["stage"] = "frozen_and_validated"
        state["finished_at"] = _now()
        self.save_state(state)
        return state

    def frozen_list(self):
        """T7's unseal input: every frozen run of the experiment with its prefixes."""
        paths = layout(self.root, self.pid, self.experiment)
        solvers = [{"run_id": run.name, "prefix": prefix.name}
                   for run in sorted(p for p in paths["selection"].iterdir() if p.is_dir())
                   for prefix in sorted(p for p in run.iterdir() if (p / "prefix.json").is_file())]
        target = self.root / "private" / self.pid / "frozen-lists" / f"{self.experiment}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        atomic_json_write(target, {"experiment": self.experiment, "protocol_id": self.pid, "solvers": solvers})
        uid, gid, _ = self.host.account(self.site["service_user"])
        for path in (target.parent, target):
            self.host.chown(path, uid, gid)
        return target

    def status(self):
        return self.load_state() if self.state_path.exists() else {"status": "not_started"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("policy", "check", "start", "resume", "finish", "status", "unpause", "reap"))
    parser.add_argument("--site", type=Path)
    parser.add_argument("--batch", type=Path)
    parser.add_argument("--values", type=Path, help="policy: scheduling values (parallelism.md §6)")
    parser.add_argument("--queue-root", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    if args.command == "policy":
        result = make_policy(json.loads(args.values.read_text()), args.queue_root, args.out)
    elif args.command in ("unpause", "reap"):
        site = json.loads(args.site.read_text())
        queue = load_policy(site["policy"])["queue_root"]
        result = unpause(queue) if args.command == "unpause" else reap(queue)
    else:
        if args.command in ("start", "resume", "finish") and os.geteuid() != 0:
            raise PermissionError("the batch launcher runs as the trusted operator")
        launcher = Launcher(args.site, args.batch)
        result = getattr(launcher, args.command)()
    print(json.dumps(result, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
