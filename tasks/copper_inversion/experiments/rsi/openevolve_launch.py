"""Batch launcher for the OpenEvolve external comparison arm (4 seeds, shared queue).

Mirrors ``experiments/rsi/launch.py`` without requiring the Pi sidecar or code-embedding
GPU model:
- Starts one isolated ``RiskOnlyService`` per seed under ``service_user`` via
  ``parallel_runtime rsi-service`` so all evaluations share ``$R/queue-g0`` limits;
- Spawns one ``openevolve_arm`` process per seed under ``client_user`` with ``nice -n 10``,
  BLAS/OMP thread caps ``= 1``, and ``parallel_evaluations <= 16``;
- Enforces China DeepSeek off-peak hours (refuses Mon-Fri 09:00-12:00 and 14:00-18:00 UTC+8
  unless ``--allow-peak`` is passed explicitly);
- On ``finish``: reconciles public/private slot budgets, runs the blind-channel audit,
  freezes the top-5 training finalists and runs standard validation selection
  (``parallel_runtime rsi-freeze``), stops training services, reaps containers, and writes
  both ``frozen-lists/<experiment>.json`` and one-shot ``REFOE`` inputs
  (``refoe-inputs/references.json`` + ``refoe-selections.json``) for ``unseal.py`` and
  ``export_outcomes.py``. Never unseals sealed test splits.

    openevolve_launch.py check|start|resume|finish|status|unseal-cmd --site site.json --batch batch.json
"""

import argparse
import fcntl
import json
import os
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ldm_tts.engine.run_store import atomic_json_write
from tasks.copper_inversion.core.execution import digest
from tasks.copper_inversion.experiments.rsi import q1
from tasks.copper_inversion.experiments.rsi.freeze import FULL, layout
from tasks.copper_inversion.experiments.rsi.launch import (
    MIN_FREE_DISK_BYTES,
    NAME,
    RUNTIME_MODULE,
    Host,
    PlanError,
    _now,
    _readable,
    _sha,
    check_qualification,
    load_policy,
    reap,
)
from tasks.copper_inversion.experiments.rsi.openevolve_arm import (
    DEFAULT_API_TYPE,
    DEFAULT_LLM_MAX_TOKENS,
    MAX_PARALLEL_EVALUATIONS,
    THREAD_ENV_CAPS,
)
from tasks.copper_inversion.experiments.rsi.unseal import MULTI_CANDIDATE_REFERENCE_EXPERIMENTS, REVIEW_FILE
from tasks.copper_inversion.rsi import audit, protocol as protocols

SCHEMA = "rsi-oe-batch-v1"
SITE_KEYS = frozenset({
    "root", "release", "source", "service_python", "openevolve_python",
    "service_user", "client_user", "image", "api_key_file", "policy", "public_catalogs",
})
SITE_OPTIONAL = frozenset({
    "load_qualification", "archive_root", "poll_seconds", "service_start_seconds",
    "machine_lock", "strip_proxy_env",
})
BATCH_KEYS = frozenset({"schema", "name", "protocol", "experiment", "tier", "seeds"})
BATCH_OPTIONAL = frozenset({
    "judge", "reference_experiment", "parallel_evaluations", "api_type",
    "full_rewrite_only", "full_rewrite_probability", "llm_max_tokens",
    "slots", "target_evaluations", "max_llm_calls", "fixture_only",
})
CHINA_TZ = timezone(timedelta(hours=8))
PROXY_ENV_KEYS = frozenset({
    "http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
})


def is_china_offpeak(moment=None):
    """True outside Mon-Fri 09:00-12:00 and 14:00-18:00 China Standard Time (UTC+8)."""
    dt = (moment or datetime.now(timezone.utc)).astimezone(CHINA_TZ)
    if dt.weekday() >= 5:
        return True
    hour = dt.hour
    if 9 <= hour < 12 or 14 <= hour < 18:
        return False
    return True


def require_offpeak(moment=None, *, allow_peak=False):
    if allow_peak or is_china_offpeak(moment):
        return
    dt = (moment or datetime.now(timezone.utc)).astimezone(CHINA_TZ)
    raise PlanError(
        f"DeepSeek peak hours in China ({dt.strftime('%Y-%m-%d %H:%M CST')}); "
        "framework batches run off-peak only (outside Mon-Fri 09:00-12:00 and 14:00-18:00 CST)"
    )


def oe_run_id(experiment, judge, seed):
    return f"openevolve-{judge}-s{seed}" if experiment == "E3" or judge != "standard" else f"openevolve-s{seed}"


def load_oe_batch(path):
    batch = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(batch, dict) or batch.get("schema") != SCHEMA:
        raise PlanError("not an OpenEvolve batch plan: expected schema " + SCHEMA)
    if not BATCH_KEYS <= set(batch) <= BATCH_KEYS | BATCH_OPTIONAL:
        raise PlanError("batch fields differ from " + SCHEMA)
    if not isinstance(batch["name"], str) or NAME.fullmatch(batch["name"]) is None:
        raise PlanError("invalid batch name")
    seeds = batch["seeds"]
    if not isinstance(seeds, list) or not seeds or any(type(s) is not int or s < 0 for s in seeds):
        raise PlanError("seeds must be a non-empty list of non-negative integers")
    if len(set(seeds)) != len(seeds):
        raise PlanError("duplicate seed in batch")
    par = batch.get("parallel_evaluations", 8)
    if type(par) is not int or not 1 <= par <= MAX_PARALLEL_EVALUATIONS:
        raise PlanError(f"parallel_evaluations must be an int in 1..{MAX_PARALLEL_EVALUATIONS}")
    ref_exp = batch.get("reference_experiment", "REFOE")
    if ref_exp not in MULTI_CANDIDATE_REFERENCE_EXPERIMENTS:
        raise PlanError(f"reference_experiment must be one of {MULTI_CANDIDATE_REFERENCE_EXPERIMENTS}")
    return batch


class OpenEvolveLauncher:
    def __init__(self, site, batch, host=None):
        self.site = json.loads(Path(site).read_text(encoding="utf-8")) if not isinstance(site, dict) else dict(site)
        if not SITE_KEYS <= set(self.site) <= SITE_KEYS | SITE_OPTIONAL:
            raise PlanError("site fields differ: need " + ", ".join(sorted(SITE_KEYS)))
        self.batch_path = Path(batch).resolve()
        self.batch = load_oe_batch(self.batch_path)
        self.protocol_path = Path(self.batch["protocol"]).resolve()
        self.protocol = protocols.load(self.protocol_path)
        self.pid = protocols.protocol_id(self.protocol)
        self.policy_path = Path(self.site["policy"]).resolve()
        self.policy = load_policy(self.policy_path)
        self.host = host or Host()
        self.root = Path(self.site["root"]).resolve()
        self.experiment = self.batch["experiment"]
        self.tier = self.batch["tier"]
        self.judge = self.batch.get("judge", "standard")
        self.reference_experiment = self.batch.get("reference_experiment", "REFOE")
        self.home = self.root / "launcher" / self.batch["name"]
        self.state_path = self.home / "state.json"

    def runs(self):
        return [(oe_run_id(self.experiment, self.judge, s), int(s)) for s in self.batch["seeds"]]

    def paths(self, name):
        paths = layout(self.root, self.pid, self.experiment, name)
        paths["socket_dir"] = self.root / "rpc" / digest([self.pid, self.experiment, name])[:12]
        return paths

    def service_slots(self):
        return int(self.protocol["tiers"][self.tier]["slots"])

    def slots_target(self):
        override = self.batch.get("target_evaluations", self.batch.get("slots"))
        if override is not None:
            if self.experiment not in protocols.DEVELOPMENT_EXPERIMENTS:
                raise PlanError("target_evaluations override is permitted only for smoke/rehearsal batches")
            if type(override) is not int or not 1 <= override <= self.service_slots():
                raise PlanError(f"target_evaluations override must lie in 1..{self.service_slots()}")
            return override
        return self.service_slots()

    def env(self):
        strip_proxy = self.site.get("strip_proxy_env", True)
        env = {
            k: v for k, v in os.environ.items()
            if k != "LLM_API_KEY" and (not strip_proxy or k not in PROXY_ENV_KEYS)
        }
        env.update(
            PYTHONPATH=str(self.site["source"]),
            PYTHONDONTWRITEBYTECODE="1",
            **THREAD_ENV_CAPS,
        )
        return env

    def manifest_path(self):
        judge_spec = self.protocol["judges"][self.judge]
        return Path(self.site["release"]) / judge_spec.get(
            "train_manifest", self.protocol["tiers"][self.tier]["train_manifest"]
        )

    def preflight(self):
        free = self.host.free_bytes(self.root)
        if free < MIN_FREE_DISK_BYTES:
            raise PlanError(
                f"insufficient free disk space on {self.root}: "
                f"{free / (1024 ** 3):.1f} GiB free, require >= {MIN_FREE_DISK_BYTES // (1024 ** 3)} GiB"
            )
        model = self.protocol["model"]
        probe = self.host.probe_provider(model["base_url"], model["name"], self.site["api_key_file"])
        return {"free_gib": round(free / (1024 ** 3), 1), "provider_probe": probe}

    def check(self):
        if self.experiment not in protocols.EXPERIMENTS:
            raise PlanError("unknown experiment: " + str(self.experiment))
        if self.tier not in self.protocol.get("tiers", {}):
            raise PlanError("tier is not defined by the protocol: " + str(self.tier))
        if self.judge not in self.protocol.get("judges", {}):
            raise PlanError("judge is not defined by the protocol: " + str(self.judge))
        isolation = protocols.case_isolation(self.protocol)
        if isolation["case_feedback"] != "none":
            raise PlanError("OpenEvolve requires aggregate-only case_isolation.case_feedback == 'none'")
        allowed = self.protocol["seeds"].get(self.experiment)
        if allowed is not None:
            allowed_set = set(allowed) | set(self.protocol["seeds"].get(self.experiment + "_reserve", []))
            for _, seed in self.runs():
                if seed not in allowed_set:
                    raise PlanError(f"seed {seed} is not in protocol seeds.{self.experiment}")
        if self.experiment not in protocols.DEVELOPMENT_EXPERIMENTS:
            protocols.verify_analysis_reference(self.protocol)
            protocols.require_frozen(self.protocol_path, self.protocol)
        qual = check_qualification(self.site, self.policy, self.experiment)
        limit = self.policy.get("max_concurrent_runs", {})
        limit = limit.get(self.tier) if isinstance(limit, dict) else limit
        if limit is not None and len(self.runs()) > limit:
            raise PlanError(f"batch of {len(self.runs())} runs exceeds max_concurrent_runs {limit} for tier {self.tier}")
        image = self.site["image"]
        if not image.startswith("sha256:") or len(image) != 71:
            raise PlanError("pin the evaluator image digest")
        frozen_img = self.protocol["hashes"].get("evaluator_image")
        if frozen_img is not None and frozen_img != image:
            raise PlanError("evaluator image differs from the frozen protocol digest")
        if self.tier not in self.site["public_catalogs"]:
            raise PlanError("site names no public catalog for tier " + self.tier)
        if not Path(self.site["public_catalogs"][self.tier]).is_file():
            raise PlanError("public catalog missing: " + str(self.site["public_catalogs"][self.tier]))
        if not self.manifest_path().is_file():
            raise PlanError("train manifest missing: " + str(self.manifest_path()))
        if not Path(self.site["openevolve_python"]).is_file():
            raise PlanError("openevolve_python missing: " + str(self.site["openevolve_python"]))
        preflight = self.preflight()
        return {
            "protocol_id": self.pid,
            "experiment": self.experiment,
            "reference_experiment": self.reference_experiment,
            "tier": self.tier,
            "judge": self.judge,
            "slots": self.slots_target(),
            "parallel_evaluations": self.batch.get("parallel_evaluations", 8),
            "runs": [n for n, _ in self.runs()],
            "seeds": [s for _, s in self.runs()],
            "china_offpeak_now": is_china_offpeak(),
            "load_qualification": qual,
            "preflight": preflight,
        }

    @contextmanager
    def machine(self):
        (self.root / "launcher").mkdir(parents=True, exist_ok=True)
        lock_name = self.site.get("machine_lock", "machine-oe.lock")
        with (self.root / "launcher" / lock_name).open("a+", encoding="utf-8") as stream:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise PlanError(f"another batch holds {lock_name}; batches run one at a time") from None
            stream.seek(0)
            stream.truncate()
            stream.write(json.dumps({"batch": self.batch["name"], "pid": os.getpid(), "since": _now()}))
            stream.flush()
            yield

    def load_state(self):
        state = json.loads(self.state_path.read_text(encoding="utf-8"))
        if state["batch_sha256"] != _sha(self.batch_path) or state["policy_sha256"] != self.policy["policy_sha256"]:
            raise PlanError("batch plan or policy changed since the batch started")
        return state

    def save_state(self, state):
        state["updated_at"] = _now()
        atomic_json_write(self.state_path, state)
        self.state_path.chmod(0o644)

    def start_service(self, name):
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
        slots = self.service_slots()
        if not config_path.exists():
            for folder in (self.root / "private" / self.pid, private.parent):
                if not folder.exists():
                    folder.mkdir(parents=True, mode=0o700)
                    self.host.chown(folder, uid, gid)
            private.mkdir(mode=0o700)
            self.host.chown(private, uid, gid)
            sockets.mkdir(parents=True, mode=0o750)
            self.host.chown(sockets, uid, client_gid)
            tier_spec = self.protocol["tiers"][self.tier]
            judge_spec = self.protocol["judges"][self.judge]
            config = {
                "manifest": str(self.manifest_path().resolve()),
                "root": str(private),
                "image": self.site["image"],
                "seconds": tier_spec["solver_seconds"],
                "slots": slots,
                "caller_uid": client_uid,
                "socket": str(socket_path),
                "objective": judge_spec["objective"],
                "tier": self.tier,
            }
            if "resample" in judge_spec:
                config["resample"] = dict(judge_spec["resample"])
            if "confirmation_packs" in judge_spec:
                packs = judge_spec["confirmation_packs"]
                config["confirmation_packs"] = list(packs) if isinstance(packs, list) else packs
            if self.protocol["loop"]["finalists"] != 3:
                config["finalists"] = self.protocol["loop"]["finalists"]
            if "case_isolation" in self.protocol:
                config["case_isolation"] = protocols.case_isolation(self.protocol)
            config_path.write_text(json.dumps(config, sort_keys=True))
            self.host.chown(config_path, uid, gid)
            config_path.chmod(0o600)
            for folder in (private / "tmp", private / "tmp" / "docker"):
                folder.mkdir(mode=0o700)
                self.host.chown(folder, uid, gid)
        elif socket_path.exists():
            socket_path.unlink()
        env = {**self.env(), "TMPDIR": str(private / "tmp"), "DOCKER_CONFIG": str(private / "tmp" / "docker")}
        log = private / "service.log"
        argv = [
            "nice", "-n", "10",
            self.site["service_python"], "-m", RUNTIME_MODULE,
            "--policy", str(self.policy_path), "--tier", self.tier,
            "rsi-service", "--config", str(config_path),
        ]
        process = self.host.spawn(self.site["service_user"], argv, env=env, log=log, cwd=self.site["source"])
        deadline = (self.site.get("service_start_seconds", 120))
        waited = 0.0
        ready = None
        while ready is None:
            if socket_path.exists() and log.exists():
                for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
                    if line.startswith("{") and '"listening"' in line:
                        row = json.loads(line)
                        if row.get("pid") == process.pid:
                            ready = row
            if ready is None:
                if process.poll() is not None or waited >= deadline:
                    if process.poll() is None:
                        self.host.stop(process.pid)
                    raise RuntimeError("evaluator service did not start: " + name)
                self.host.sleep(0.2)
                waited += 0.2
        if ready["policy"] != self.policy["policy_sha256"]:
            raise RuntimeError("service runs a different scheduling policy: " + name)
        self.host.chown(socket_path, uid, client_gid)
        socket_path.chmod(0o660)
        deployment = {
            "socket": str(socket_path),
            "contract_digest": ready["contract_digest"],
            "service_uid": uid,
            "pid": process.pid,
            "case_workers": ready.get("case_workers"),
        }
        (sockets / "deployment.json").write_text(json.dumps(deployment, sort_keys=True) + "\n")
        return deployment

    def controller_argv(self, name, seed, deployment, *, resume=False):
        svc_slots = self.service_slots()
        target_evals = self.slots_target()
        par = self.batch.get("parallel_evaluations", 8)
        api_type = self.batch.get("api_type", DEFAULT_API_TYPE)
        max_tokens = self.batch.get("llm_max_tokens", DEFAULT_LLM_MAX_TOKENS)
        argv = [
            "nice", "-n", "10",
            self.site["openevolve_python"], "-m",
            "tasks.copper_inversion.experiments.rsi.openevolve_arm",
            "--protocol", str(self.protocol_path),
            "--protocol-id", self.pid,
            "--experiment", self.experiment,
            "--tier", self.tier,
            "--judge", self.judge,
            "--seed", str(seed),
            "--out-dir", str(self.paths(name)["run"]),
            "--public-catalog", str(self.site["public_catalogs"][self.tier]),
            "--evaluation-socket", deployment["socket"],
            "--evaluation-contract", deployment["contract_digest"],
            "--evaluation-service-uid", str(deployment["service_uid"]),
            "--api-key-file", str(self.site["api_key_file"]),
            "--slots", str(svc_slots),
            "--target-evaluations", str(target_evals),
            "--parallel-evaluations", str(par),
            "--api-type", api_type,
            "--llm-max-tokens", str(max_tokens),
        ]
        if self.batch.get("max_llm_calls") is not None:
            argv += ["--max-llm-calls", str(self.batch["max_llm_calls"])]
        if self.batch.get("full_rewrite_only"):
            argv.append("--full-rewrite-only")
        elif "full_rewrite_probability" in self.batch:
            argv += ["--full-rewrite-probability", str(self.batch["full_rewrite_probability"])]
        if self.batch.get("fixture_only"):
            argv.append("--fixture-only")
        if resume:
            argv.append("--resume")
        return argv

    def start_controller(self, name, seed, deployment, *, resume=False):
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
        return self.host.spawn(
            self.site["client_user"],
            self.controller_argv(name, seed, deployment, resume=resume),
            env=self.env(),
            log=log,
            cwd=self.site["source"],
        )

    def start(self, *, allow_peak=False):
        require_offpeak(allow_peak=allow_peak)
        summary = self.check()
        with self.machine():
            if self.state_path.exists():
                raise PlanError("batch already started; use resume")
            self.home.mkdir(parents=True, exist_ok=True)
            state = {
                "schema": "rsi-oe-batch-state-v1",
                "batch": self.batch["name"],
                "batch_sha256": _sha(self.batch_path),
                "protocol_id": self.pid,
                "policy_sha256": self.policy["policy_sha256"],
                "started_at": _now(),
                "summary": summary,
                "runs": {},
            }
            processes = {}
            for name, seed in self.runs():
                state["runs"][name] = {
                    "seed": seed,
                    "service": self.start_service(name),
                    "status": "starting",
                }
                self.save_state(state)
            for name, seed in self.runs():
                proc = self.start_controller(name, seed, state["runs"][name]["service"], resume=False)
                processes[name] = proc
                state["runs"][name].update(controller_pid=proc.pid, status="running")
                self.save_state(state)
            return self.monitor(state, processes)

    def resume(self, *, allow_peak=False):
        require_offpeak(allow_peak=allow_peak)
        if (Path(self.policy["queue_root"]) / "PAUSED.json").exists():
            raise PlanError("the scheduler is paused; reconcile, then unpause before resume")
        self.preflight()
        with self.machine():
            state = self.load_state()
            processes = {}
            for name, seed in self.runs():
                entry = state["runs"][name]
                if entry["status"] == "finished":
                    continue
                if not self.host.alive(entry["service"]["pid"]):
                    restarted = self.start_service(name)
                    if restarted["contract_digest"] != entry["service"]["contract_digest"]:
                        raise RuntimeError("restarted service measures a different contract: " + name)
                    entry["service"] = restarted
                if entry.get("controller_pid") and self.host.alive(entry["controller_pid"]):
                    raise PlanError("controller still running: " + name)
                processes[name] = self.start_controller(
                    name, seed, entry["service"], resume=self.paths(name)["run"].exists()
                )
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
                    run_status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
                    ok = process.returncode == 0 and run_status.get("status") in ("completed", "stopped")
                    state["runs"][name].update(
                        status="finished" if ok else "needs_reconciliation",
                        exit_code=process.returncode,
                        run_status=run_status,
                    )
            self.save_state(state)
            if not running:
                break
            self.host.sleep(self.site.get("poll_seconds", 15))
        failed = sorted(n for n, e in state["runs"].items() if e["status"] != "finished")
        state["stage"] = "runs_finished" if not failed else "needs_reconciliation"
        self.save_state(state)
        if failed:
            raise RuntimeError("runs need reconciliation before resume: " + ", ".join(failed))
        return state

    def finish(self):
        """Reconcile budgets, scan blind channel, freeze top-5 & validate on standard validation, write REFOE inputs."""
        with self.machine():
            state = self.load_state()
            unfinished = sorted(n for n, e in state["runs"].items() if e["status"] != "finished")
            if unfinished:
                raise PlanError("unfinished runs: " + ", ".join(unfinished))
            expected_slots = self.slots_target()
            uid, gid, _ = self.host.account(self.site["service_user"])
            reconciliation = {}
            for name, _ in self.runs():
                paths = self.paths(name)
                run_dir, service_dir = paths["run"], paths["service"]
                budget_res = audit.reconcile_budget(run_dir, service_dir)
                if budget_res["charged_sources"] != expected_slots:
                    raise ValueError(
                        f"{name}: expected exactly {expected_slots} charged evaluations, got {budget_res['charged_sources']}"
                    )
                blind = audit.scan_blind(
                    run_dir, service_dir, require_requests=False, key_file=self.site["api_key_file"]
                )
                if blind["review_required"]:
                    atomic_json_write(service_dir / REVIEW_FILE, {
                        "run_id": name, "review_sha256": blind["review_sha256"], "hits": blind["review_required"],
                    })
                    self.host.chown(service_dir / REVIEW_FILE, uid, gid)
                reconciliation[name] = {
                    "budget": budget_res,
                    "blind": {k: blind[k] for k in (
                        "status", "files_checked", "marker_hits", "iou_hits_by_kind",
                        "leaks", "review_required", "review_sha256",
                    )},
                }
                _readable(run_dir)
            state["reconciliation"] = reconciliation
            state["stage"] = "reconciled"
            self.save_state(state)

            processes = {}
            for name, _ in self.runs():
                argv = [
                    "nice", "-n", "10",
                    self.site["service_python"], "-m", RUNTIME_MODULE,
                    "--policy", str(self.policy_path), "--tier", self.tier,
                    "rsi-freeze", "--root", str(self.root), "--protocol-id", self.pid,
                    "--experiment", self.experiment, "--run", name,
                    "--release", str(self.site["release"]), "--protocol", str(self.protocol_path),
                    "--prefix", FULL,
                ]
                log = self.paths(name)["service"] / "freeze.log"
                processes[name] = self.host.spawn(
                    self.site["service_user"], argv, env=self.env(), log=log, cwd=self.site["source"]
                )
            codes = {name: p.wait() for name, p in processes.items()}
            if any(codes.values()):
                state["stage"] = "freeze_failed"
                state["freeze_exit_codes"] = codes
                self.save_state(state)
                raise RuntimeError("freeze or validation failed; inspect private freeze.log files")
            state.pop("freeze_exit_codes", None)

            state["frozen_list"] = str(self._write_frozen_list(uid, gid))
            ref_files = self._write_reference_inputs(uid, gid)
            state["reference_unseal_inputs"] = ref_files
            for entry in state["runs"].values():
                self.host.stop(entry["service"]["pid"])
            state["containers"] = reap(self.policy["queue_root"])
            state["stage"] = "frozen_and_validated"
            state["finished_at"] = _now()
            self.save_state(state)
            return state

    def _write_frozen_list(self, uid, gid):
        paths = layout(self.root, self.pid, self.experiment)
        solvers = [
            {"run_id": name, "prefix": FULL}
            for name, _ in self.runs()
            if (paths["selection"] / name / FULL / "prefix.json").is_file()
        ]
        target = self.root / "private" / self.pid / "frozen-lists" / f"{self.experiment}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        atomic_json_write(target, {"experiment": self.experiment, "protocol_id": self.pid, "solvers": solvers})
        for p in (target.parent, target):
            self.host.chown(p, uid, gid)
        return target

    def _write_reference_inputs(self, uid, gid):
        ref_label = self.reference_experiment.lower()
        ref_dir = self.root / "private" / self.pid / f"{ref_label}-inputs"
        ref_dir.mkdir(parents=True, exist_ok=True)
        refs = {}
        selections = {}
        for name, seed in self.runs():
            paths = self.paths(name)
            sel_target = paths["run_selection"] / FULL
            report = json.loads((sel_target / "selection.json").read_text(encoding="utf-8"))
            selected_sha = report["selected_sha256"]
            src_file = paths["run"] / "candidates" / selected_sha / "solver.py"
            refs[name] = {"source": str(src_file.resolve()), "sha256": selected_sha, "seed": int(seed)}
            selections[name] = str(sel_target.resolve())
        refs_path = ref_dir / "references.json"
        sels_path = ref_dir / "selections.json"
        atomic_json_write(refs_path, refs)
        atomic_json_write(sels_path, selections)
        q1.references(refs_path)
        for p in (ref_dir, refs_path, sels_path):
            self.host.chown(p, uid, gid)
        return {
            "reference_experiment": self.reference_experiment,
            "references": str(refs_path),
            "selections": str(sels_path),
        }

    def unseal_commands(self):
        ref_label = self.reference_experiment.lower()
        priv = self.root / "private" / self.pid
        refs_path = priv / f"{ref_label}-inputs" / "references.json"
        sels_path = priv / f"{ref_label}-inputs" / "selections.json"
        out_path = priv / f"outcomes-{self.reference_experiment}.json"
        env_prefix = "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 nice -n 10"
        py = self.site["service_python"]
        src = self.site["source"]
        return {
            "reference_experiment": self.reference_experiment,
            "unseal": (
                f"sudo -u {self.site['service_user']} env PYTHONPATH={src} {env_prefix} "
                f"{py} -m tasks.copper_inversion.experiments.rsi.unseal "
                f"--root {self.root} --experiment {self.reference_experiment} --protocol-id {self.pid} "
                f"--references {refs_path} --selections {sels_path} --ref-workers 4"
            ),
            "export": (
                f"sudo -u {self.site['service_user']} env PYTHONPATH={src} {env_prefix} "
                f"{py} -m tasks.copper_inversion.experiments.rsi.export_outcomes "
                f"--root {self.root} --protocol-id {self.pid} --experiment {self.reference_experiment} "
                f"--references {sels_path} --output {out_path}"
            ),
        }

    def status(self):
        return self.load_state() if self.state_path.exists() else {"status": "not_started"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("check", "start", "resume", "finish", "status", "unseal-cmd"))
    parser.add_argument("--site", type=Path, required=True)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--allow-peak", action="store_true",
                        help="Override China DeepSeek peak-hour guard (for smoke/emergency use only)")
    args = parser.parse_args(argv)
    if args.command in ("start", "resume", "finish") and os.geteuid() != 0:
        raise PermissionError("the batch launcher runs as the trusted operator (root)")
    launcher = OpenEvolveLauncher(args.site, args.batch)
    if args.command == "start":
        result = launcher.start(allow_peak=args.allow_peak)
    elif args.command == "resume":
        result = launcher.resume(allow_peak=args.allow_peak)
    elif args.command == "unseal-cmd":
        result = launcher.unseal_commands()
    else:
        result = getattr(launcher, args.command)()
    print(json.dumps(result, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
