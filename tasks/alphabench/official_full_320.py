"""Independent, measured 320-search cohort for the three AlphaBench T3 methods."""

import argparse
from dataclasses import replace
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import time
from urllib.request import urlopen

from ldm_tts.engine.run_store import atomic_json_write
from tasks.alphabench.core.protocol import digest
from tasks.alphabench.core.seed_bundle import initialization_contract, verify_seed_bundle
from tasks.alphabench.presentation_320 import protocol as original_protocol


REPO = Path(__file__).resolve().parents[2]
ROOT = Path("/mnt/data1/Large-Discovery-Models")
STUDY = ROOT / "runs/alphabench-t3/official-full-320-20260927"
ORIGINAL = ROOT / "runs/alphabench-t3/presentation-320-20260924"
SOURCE = ROOT / "data/alphabench/AlphaBench"
DATA = ROOT / "data/alphabench/manifests/qlib_csi300.json"
KEY = ROOT / "secrets/alphabench-t3/deepseek.key"
PYTHON = ROOT / "envs/shared-8b1f38b5/bin/python"
ORACLE_URL = "http://127.0.0.1:19780"
METHODS = ("alphabench_cot", "alphabench_tot", "alphabench_ea")
SEEDS = (42, 43, 3407, 10086)


def protocol(method, seed):
    value = original_protocol(method, seed)
    if method == "alphabench_cot":
        value = replace(value, budgets={**value.budgets, "proposal_attempts": 1200})
    return value


def freeze(path, payload):
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != payload:
            raise ValueError(f"frozen study file changed: {path}")
    else:
        atomic_json_write(path, payload)


def bundle(seed):
    return ORIGINAL / "initializations" / f"seed_{seed}" / "initialization"


def run_dir(method, seed):
    return STUDY / "runs" / method / f"seed_{seed}"


def protocol_path(method, seed):
    return STUDY / "protocols" / f"{method}-seed_{seed}.json"


def health():
    with urlopen(ORACLE_URL + "/t3/health", timeout=60) as response:
        state = json.load(response)
    manifest = json.loads((ORIGINAL / "manifest.json").read_text(encoding="utf-8"))
    if (state["data_digest"] != manifest["data_digest"] or
            state["config_digest"] != manifest["oracle_config_digest"]):
        raise ValueError("Oracle data/config differs from the original comparison")
    return state


def prepare():
    if not all(path.is_file() for path in (DATA, KEY, PYTHON)) or not SOURCE.is_dir():
        raise FileNotFoundError("the frozen data, provider file, Python, or pinned source is missing")
    service = health()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    rows = {}
    for seed in SEEDS:
        first = protocol(METHODS[0], seed)
        verify_seed_bundle(bundle(seed), first, mock=False)
        for method in METHODS:
            value = protocol(method, seed)
            if initialization_contract(value) != initialization_contract(first):
                raise ValueError("methods do not share the frozen initialization contract")
            payload = value.to_dict()
            freeze(protocol_path(method, seed), payload)
            rows[f"{method}:seed_{seed}"] = {
                "protocol_sha256": sha256(protocol_path(method, seed).read_bytes()).hexdigest(),
                "original_protocol_sha256": sha256((ORIGINAL / "protocols" /
                    f"{method}-seed_{seed}.json").read_bytes()).hexdigest(),
                "initialization_bundle": str(bundle(seed)),
            }
    freeze(STUDY / "manifest.json", {
        "code_commit": commit, "methods": list(METHODS), "seeds": list(SEEDS),
        "rounds": 20, "search_evaluations": 320, "test_limit": 50,
        "data_digest": digest(json.loads(DATA.read_text(encoding="utf-8"))),
        "oracle_config_digest": service["config_digest"],
        "source_root": str(SOURCE), "original_study": str(ORIGINAL),
        "adaptations": ["retry empty CoT rounds", "top up unique EA offspring",
                        "measure a ToT tail batch and settle every paid search attempt"],
        "proposal_attempt_limit": {"alphabench_cot": 1200, "alphabench_tot": 600,
                                   "alphabench_ea": 600},
        "campaigns": rows,
    })
    print(json.dumps({"study": str(STUDY), "commit": commit, "campaigns": len(rows)}))


def launch():
    prepare()
    registry_path = STUDY / "processes.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8")) if registry_path.exists() else {}
    environment = os.environ.copy()
    environment.update(LDM_DATA_COLLECTION_ENABLED="1", TMPDIR=str(ROOT / "tmp"),
                       UV_CACHE_DIR=str(ROOT / "cache/uv"))
    for seed in SEEDS:
        for method in METHODS:
            name = f"{method}:seed_{seed}"
            if name in registry:
                continue
            directory = run_dir(method, seed)
            if directory.exists():
                raise ValueError(f"unregistered campaign directory requires inspection: {directory}")
            log = STUDY / "logs" / f"{method}-seed_{seed}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            command = [str(PYTHON), "-m", "tasks.alphabench.ldm_task.procedure",
                       "--protocol-file", str(protocol_path(method, seed)),
                       "--data-manifest", str(DATA), "--upstream-root", str(SOURCE),
                       "--oracle-url", ORACLE_URL, "--api-key-file", str(KEY),
                       "--initialization-bundle", str(bundle(seed)),
                       "--out-dir", str(directory)]
            with log.open("ab") as stream:
                child = subprocess.Popen(command, cwd=REPO, env=environment, stdin=subprocess.DEVNULL,
                                         stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            registry[name] = {"pid": child.pid, "log": str(log), "started_unix": time.time()}
            atomic_json_write(registry_path, registry)
    print(json.dumps({"launched": len(registry), "study": str(STUDY)}))


def status():
    registry_path = STUDY / "processes.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8")) if registry_path.exists() else {}
    rows = {}
    for name, item in registry.items():
        method, seed = name.split(":seed_")
        directory = run_dir(method, int(seed))
        state = directory / "status.json"
        budget = directory / "budget.json"
        state = json.loads(state.read_text(encoding="utf-8"))["status"] if state.exists() else None
        counters = json.loads(budget.read_text(encoding="utf-8"))["counters"] if budget.exists() else {}
        try:
            os.kill(item["pid"], 0)
            alive = True
        except ProcessLookupError:
            alive = False
        rows[name] = {"status": state, "alive": alive,
                      "search": counters.get("expensive_evaluation_attempts", 0),
                      "test": counters.get("test_evaluations", 0),
                      "proposal_attempts": counters.get("proposal_attempts", 0),
                      "model_requests": counters.get("model_requests", 0),
                      "final_report": (directory / "report_manifest.json").exists()}
    print(json.dumps(rows, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "launch", "status"))
    action = parser.parse_args().action
    {"prepare": prepare, "launch": launch, "status": status}[action]()


if __name__ == "__main__":
    main()
