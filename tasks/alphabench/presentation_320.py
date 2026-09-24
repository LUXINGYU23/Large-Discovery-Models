"""Launch the frozen CSI300 five-method, four-seed 320-evaluation comparison."""

import argparse
from dataclasses import replace
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from urllib.request import urlopen

from ldm_tts.engine.run_store import atomic_json_write
from tasks.alphabench.core.initialization import run_initialization
from tasks.alphabench.core.protocol import T3Protocol, digest
from tasks.alphabench.core.seed_bundle import verify_seed_bundle
from tasks.alphabench.ldm_task.procedure import describe_ldm_task, parse_args


REPO = Path(__file__).resolve().parents[2]
DATA_ROOT = Path("/mnt/data1/Large-Discovery-Models")
STUDY = DATA_ROOT / "runs/alphabench-t3/presentation-320-20260924"
SOURCE = DATA_ROOT / "data/alphabench/AlphaBench"
DATA = DATA_ROOT / "data/alphabench/manifests/qlib_csi300.json"
KEY = DATA_ROOT / "secrets/alphabench-t3/deepseek.key"
ORACLE_CONFIG = REPO / "tasks/alphabench/resources/oracle_configs/partial_csi300_qlib_pilot.json"
ORACLE_URL = "http://127.0.0.1:19780"
SEEDS = (42, 43, 3407, 10086)
METHODS = ("ldm_harness_compiled", "harness", "alphabench_cot", "alphabench_tot", "alphabench_ea")
PYTHON = REPO / "tasks/alphabench/.venv/bin/python"
ORACLE_PYTHON = REPO / "tasks/alphabench/environments/qlib/.venv/bin/python"


def frozen(path, payload):
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != payload:
            raise ValueError(f"frozen file changed: {path}")
    else:
        atomic_json_write(path, payload)


def budgets(method, *, initialization=False):
    if initialization:
        return {"model_requests": 0, "proposal_attempts": 0, "dynamic_checks": 0, "lint_checks": 0,
                "initialization_evaluations": 42, "validation_evaluations": 42,
                "test_evaluations": 0, "analysis_jobs": 0, "quality_checks": 0,
                "oracle_job_slots": 1000, "benchmark_jobs": 1000, "policy_turns": 0,
                "harness_turns": 0}
    sessions = 4 if method == "ldm_harness_compiled" else 1
    return {"model_requests": 2400, "proposal_attempts": 600,
            "dynamic_checks": 15000, "lint_checks": 0, "initialization_evaluations": 0,
            "validation_evaluations": 400, "test_evaluations": 50, "analysis_jobs": 2,
            "quality_checks": 15000, "oracle_job_slots": 50000, "benchmark_jobs": 50000,
            "policy_turns": 20 if method == "ldm_harness_compiled" else 0,
            "harness_turns": 20 * sessions if method in {"harness", "ldm_harness_compiled"} else 0}


def protocol(method, seed, *, initialization=False):
    base = T3Protocol(**json.loads((REPO / "tasks/alphabench/resources/protocols/partial_csi300_cot_pilot.json")
                                   .read_text(encoding="utf-8")))
    native = {
        "alphabench_cot": {"oracle_workers": 4, "enable_reason": True, "accept_threshold": 0.0,
                            "workers": 16},
        "alphabench_tot": {"oracle_workers": 4, "enable_reason": True, "accept_threshold": 0.0,
                            "workers": 4, "N": 6, "top_k": 3},
        "alphabench_ea": {"oracle_workers": 4, "enable_reason": True, "accept_threshold": 0.0,
                           "N": 16, "mutation_rate": .5, "crossover_rate": .5,
                           "pool_size": 30, "seeds_top_k": 12},
    }
    return replace(base, method=method, random_seed=seed, rounds=20, evaluations=320, batch_size=16,
                   sessions=4 if method == "ldm_harness_compiled" else 1,
                   candidates_per_session=32 if method == "ldm_harness_compiled" else 16,
                   bo_pool_size=64 if method == "ldm_harness_compiled" else None,
                   native_parameters=native.get(method, {}), init_mode="alpha158",
                   alpha158_groups=("kbar", "price", "rolling"), cold_seed_count=0,
                   factor_select_n=50, harness_surrogate_query=False,
                   budgets=budgets(method, initialization=initialization))


def proto_path(method, seed, *, initialization=False):
    section = "initialization" if initialization else "protocols"
    return STUDY / section / f"{method}-seed_{seed}.json"


def run_dir(method, seed):
    return STUDY / "runs" / method / f"seed_{seed}"


def bundle_dir(seed):
    return STUDY / "initializations" / f"seed_{seed}" / "initialization"


def prepare():
    if not DATA.is_file() or not SOURCE.is_dir() or not KEY.is_file():
        raise FileNotFoundError("frozen data, source, or provider credential is missing")
    version = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    for seed in SEEDS:
        frozen(proto_path("alphabench_cot", seed, initialization=True),
               protocol("alphabench_cot", seed, initialization=True).to_dict())
        for method in METHODS:
            frozen(proto_path(method, seed), protocol(method, seed).to_dict())
    frozen(STUDY / "manifest.json", {"repository_commit": version, "data_digest": digest(json.loads(DATA.read_text())),
           "source_root": str(SOURCE), "oracle_config_digest": digest(json.loads(ORACLE_CONFIG.read_text())),
           "seeds": list(SEEDS), "methods": list(METHODS), "rounds": 20, "new_evaluations": 320,
           "initial_factors": 42, "comparison_policy": "partial_comparison"})
    print(json.dumps({"study": str(STUDY), "campaigns": len(SEEDS) * len(METHODS), "commit": version}))


def processes():
    path = STUDY / "processes.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def spawn(name, command, log_path, *, python=PYTHON, resume=False):
    registry = processes()
    previous = registry.get(name)
    if previous and not resume:
        raise ValueError(f"process already launched: {name}; inspect and resume its existing run")
    if resume:
        if previous is None:
            raise ValueError(f"process was not launched: {name}")
        try:
            os.kill(previous["pid"], 0)
        except ProcessLookupError:
            pass
        else:
            raise ValueError(f"existing process is still alive: {name}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update(LDM_DATA_COLLECTION_ENABLED="1", TMPDIR=str(DATA_ROOT / "tmp"),
                       UV_CACHE_DIR=str(DATA_ROOT / "cache/uv"))
    code_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    with log_path.open("ab") as log:
        child = subprocess.Popen([str(python), *command], cwd=REPO, env=environment,
                                 stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
    attempts = previous.get("prior_attempts", []) if previous else []
    if previous:
        attempts = [*attempts, {"pid": previous["pid"], "started_unix": previous["started_unix"]}]
    registry[name] = {"pid": child.pid, "log": str(log_path), "started_unix": time.time(),
                      "prior_attempts": attempts, "code_commit": code_commit}
    atomic_json_write(STUDY / "processes.json", registry)
    return child.pid


def health():
    with urlopen(ORACLE_URL + "/t3/health", timeout=60) as response:
        return json.load(response)


def start_oracle():
    try:
        state = health()
    except OSError:
        spawn("oracle", ["-m", "tasks.alphabench.core.oracle_service", "--config", str(ORACLE_CONFIG),
                         "--root", str(STUDY / "oracle"), "--port", "19780"],
              STUDY / "logs/oracle.log", python=ORACLE_PYTHON, resume="oracle" in processes())
        for _ in range(30):
            time.sleep(1)
            try:
                state = health()
                break
            except OSError:
                continue
        else:
            raise RuntimeError("Oracle did not become healthy; inspect logs/oracle.log")
    expected = json.loads(ORACLE_CONFIG.read_text(encoding="utf-8"))
    if any(state[key] != expected[key] for key in ("backend", "market", "data_digest", "environment_digest")):
        raise ValueError("Oracle health differs from the frozen data contract")
    print(json.dumps({"oracle": ORACLE_URL, "config_digest": state["config_digest"]}))


def initialize_one(seed):
    path = proto_path("alphabench_cot", seed, initialization=True)
    args = parse_args(["--protocol-file", str(path), "--data-manifest", str(DATA),
                       "--upstream-root", str(SOURCE), "--oracle-url", ORACLE_URL])
    run_initialization(T3Protocol(**json.loads(path.read_text(encoding="utf-8"))), args,
                       describe_ldm_task(args), None, bundle_dir(seed).parent)
    print(json.dumps({"seed": seed, "bundle": str(bundle_dir(seed)), "status": "completed"}))


def initialize():
    health()
    for seed in SEEDS:
        spawn(f"initialization:seed_{seed}", ["-m", "tasks.alphabench.presentation_320", "init-one",
                                           "--seed", str(seed)], STUDY / f"logs/initialization-seed_{seed}.log")
    print(json.dumps({"initializations_started": len(SEEDS)}))


def launch():
    health()
    for seed in SEEDS:
        for method in METHODS:
            verify_seed_bundle(bundle_dir(seed), protocol(method, seed), mock=False)
    for seed in SEEDS:
        for method in METHODS:
            name = f"{method}:seed_{seed}"
            spawn(name, campaign_command(method, seed), STUDY / f"logs/{method}-seed_{seed}.log")
    print(json.dumps({"campaigns_started": len(SEEDS) * len(METHODS), "study": str(STUDY)}))


def campaign_command(method, seed, *, resume=False):
    return ["-m", "tasks.alphabench.ldm_task.procedure",
            "--protocol-file", str(proto_path(method, seed)),
            "--data-manifest", str(DATA), "--upstream-root", str(SOURCE),
            "--oracle-url", ORACLE_URL, "--api-key-file", str(KEY),
            "--initialization-bundle", str(bundle_dir(seed)),
            "--resume-run" if resume else "--out-dir", str(run_dir(method, seed))]


def paused_run(method, seed):
    if method not in METHODS or seed not in SEEDS:
        raise ValueError("unknown study method or seed")
    name = f"{method}:seed_{seed}"
    entry = processes().get(name)
    if entry is None:
        raise ValueError("campaign was not launched")
    try:
        os.kill(entry["pid"], 0)
    except ProcessLookupError:
        pass
    else:
        raise ValueError("campaign process is still alive")
    directory = run_dir(method, seed)
    if not json.loads((directory / "status.json").read_text())["status"].startswith("paused_"):
        raise ValueError("campaign is not paused")
    if json.loads((directory / "protocol.json").read_text()) != json.loads(proto_path(method, seed).read_text()):
        raise ValueError("campaign protocol differs from the frozen study")
    return directory


def resume(method, seed):
    paused_run(method, seed)
    health()
    name = f"{method}:seed_{seed}"
    pid = spawn(name, campaign_command(method, seed, resume=True),
                STUDY / f"logs/{method}-seed_{seed}.log", resume=True)
    print(json.dumps({"resumed": name, "pid": pid, "run_dir": str(run_dir(method, seed))}))


def resolve_model(method, seed):
    directory = paused_run(method, seed)
    receipts = []
    for path in (directory / "private/model").glob("*.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record["state"] == "dispatch_intent":
            receipts.append((path, record))
    if len(receipts) != 1:
        raise ValueError("expected exactly one unknown model response")
    path, receipt = receipts[0]
    if (digest(receipt["identity"]) != path.stem or digest(receipt["request"]) != receipt["request_digest"]
            or receipt["request"]["protocol"] != protocol(method, seed).identity):
        raise ValueError("unknown model receipt failed identity verification")
    marker = directory / "private/model_recovery" / path.name
    frozen(marker, {"action": "discard_unknown_response_and_retry_once",
                    "original_receipt_digest": digest(receipt)})
    print(json.dumps({"authorized_one_counted_model_retry": f"{method}:seed_{seed}",
                      "unknown_receipt": path.stem}))


def resolve_worker(request_id):
    from tasks.alphabench.core.oracle_service import OracleService

    if not re.fullmatch(r"[a-f0-9]{64}", request_id):
        raise ValueError("invalid worker request identity")
    service = OracleService(ORACLE_CONFIG, STUDY / "oracle")
    receipt = service.reconciled(request_id)
    directory = STUDY / "oracle/jobs" / request_id
    request_path = directory / "request.json"
    request = json.loads(request_path.read_text(encoding="utf-8"))
    permit = json.loads((directory / "permit.json").read_text(encoding="utf-8"))
    if (receipt is None or receipt["state"] != "dispatch_intent" or receipt["request"] != request
            or request["request_id"] != request_id or request["phase"] not in {"check", "search"}
            or request["operation"] != ("check" if request["phase"] == "check" else "evaluate")
            or permit != {"request_digest": digest(request), "job_id": request_id + ":0"}
            or (directory / "response.json").exists()):
        raise ValueError("worker request is not an unresolved expression crash")
    log_path = directory / "worker.log"
    log = log_path.read_bytes()
    error = "'numpy.float64' object has no attribute 'name'"
    if b"AttributeError: " + error.encode() not in log:
        raise ValueError("worker log does not prove the known Qlib expression failure")
    active = subprocess.check_output(["ps", "-eo", "args"], text=True)
    if any("tasks.alphabench.core.oracle_worker" in line and request_id in line for line in active.splitlines()):
        raise ValueError("worker is still running")
    response = {"request_id": request_id, "success": False,
                "error": "Qlib rejected the factor expression: " + error,
                "metrics": {}, "daily": [], "scores": [], "portfolio": None,
                "elapsed_seconds": max(0.0, log_path.stat().st_mtime - request_path.stat().st_mtime),
                "elapsed_estimated_from_files": True,
                "jobs": [{"job_id": request_id + ":0", "operation": request["operation"]}]}
    if request["phase"] == "check":
        response["check_kind"] = T3Protocol(**request["protocol"]).check_kind
    frozen(directory / "recovery.json", {"request_digest": digest(request),
        "worker_log_sha256": sha256(log).hexdigest(), "response_digest": digest(response)})
    atomic_json_write(directory / "response.json", response)
    if service.reconciled(request_id)["state"] != "completed":
        raise RuntimeError("worker failure did not reconcile")
    print(json.dumps({"reconciled_failed_worker": request_id}))


def status():
    rows = {}
    for name, entry in processes().items():
        try:
            os.kill(entry["pid"], 0)
            alive = True
        except ProcessLookupError:
            alive = False
        method, _, seed = name.partition(":seed_")
        directory = bundle_dir(int(seed)) if method == "initialization" else (
            run_dir(method, int(seed)) if seed else None)
        status_path = directory / "status.json" if directory else None
        recorded = json.loads(status_path.read_text(encoding="utf-8"))["status"] if status_path and status_path.exists() else None
        budget_path = directory / "budget.json" if directory else None
        counters = json.loads(budget_path.read_text(encoding="utf-8"))["counters"] if budget_path and budget_path.exists() else {}
        rows[name] = {"pid": entry["pid"], "alive": alive, "status": recorded,
                      "evaluations": counters.get("external_evaluations"),
                      "rounds": counters.get("outer_iterations"),
                      "model_requests": counters.get("model_requests"), "log": entry["log"]}
    print(json.dumps(rows, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "start-oracle", "initialize", "init-one", "launch",
                                           "resume", "resolve-model", "resolve-worker", "status"))
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--request-id")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare()
    elif args.action == "start-oracle":
        start_oracle()
    elif args.action == "initialize":
        initialize()
    elif args.action == "init-one":
        if args.seed is None:
            parser.error("init-one requires --seed")
        initialize_one(args.seed)
    elif args.action == "launch":
        launch()
    elif args.action in {"resume", "resolve-model"}:
        if args.method is None or args.seed is None:
            parser.error(args.action + " requires --method and --seed")
        (resume if args.action == "resume" else resolve_model)(args.method, args.seed)
    elif args.action == "resolve-worker":
        if args.request_id is None:
            parser.error("resolve-worker requires --request-id")
        resolve_worker(args.request_id)
    else:
        status()


if __name__ == "__main__":
    main()
