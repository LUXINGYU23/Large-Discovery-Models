#!/usr/bin/env python3
"""Run the four paired H20 antibody smoke cases with the async task entry.

The historical effective configurations are the source of the task arguments.
The first case generates a new shared initialization; the other three replay it.
Start this script from the checkout to validate, not from the historical checkout.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import secrets
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request


REPO = Path(__file__).resolve().parents[4]
ROOT = Path("/Aidata/shared/ldm/antibody-poc")
ACTIVE_ROOT = Path("/Aidata/shared/ldm")
BRIDGE = "172.17.0.1"
MODEL = "Qwen3-8B"
VLLM = Path("/home/zsgpu/vllm-env/bin/vllm")
WEIGHTS = Path("/Aidata/shared/rsi-models/Qwen3-8B")
CAMPAIGN_PYTHON = ROOT / "env/antibody_campaign/bin/python"
ANTBO_PYTHON = ROOT / "env/antbo_native/bin/python"
ANTBO_HOME = ROOT / "env/sources/HEBO-77636146/AntBO"
BASELINES = REPO / "config/antibody_rfd3/smoke-baseline"
CASES = (
    ("ldm_harness", "qwen3-ldm_harness_3.json"),
    ("harness", "qwen3-harness.json"),
    ("fixed", "paired-fixed.json"),
    ("antbo", "paired-antbo.json"),
)


def now():
    return datetime.now(timezone.utc).isoformat()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="New directory under /Aidata/shared/ldm for runs, logs, and summary")
    parser.add_argument("--sidecar-image", required=True,
                        help="New protocol-compatible Pi image; the historical image is not changed")
    parser.add_argument("--gpu", type=int, choices=(0, 1, 2, 3), default=0)
    parser.add_argument("--startup-timeout", type=int, default=300)
    parser.add_argument("--case-timeout", type=int, default=1800)
    args = parser.parse_args()
    if not args.sidecar_image.strip() or min(args.startup_timeout, args.case_timeout) < 1:
        parser.error("The image must be nonempty and both timeouts must be positive")
    args.output_dir = args.output_dir.expanduser().resolve()
    if ACTIVE_ROOT not in args.output_dir.parents:
        parser.error("The output directory must be below /Aidata/shared/ldm")
    return args


def scrub(value, private_values):
    value = str(value)
    for private in private_values:
        value = value.replace(private, "[redacted]")
    return value


def private_environment_values(env, key):
    values = {key}
    for name, value in env.items():
        if len(value) >= 8 and any(part in name.upper() for part in ("KEY", "TOKEN", "SECRET", "PASSWORD")):
            values.add(value)
    return sorted(values, key=len, reverse=True)


def log_pipe(stream, target, private_values):
    for line in iter(stream.readline, ""):
        target.write(scrub(line, private_values))
        target.flush()


def start_logged(command, env, log_path, private_values):
    log = log_path.open("w", encoding="utf-8")
    try:
        process = subprocess.Popen(command, cwd=REPO, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, errors="replace",
                                   bufsize=1, start_new_session=True)
    except BaseException:
        log.close()
        raise
    thread = threading.Thread(target=log_pipe, args=(process.stdout, log, private_values), daemon=True)
    thread.start()
    return process, thread, log


def finish_log(process, thread, log):
    thread.join(timeout=5)
    if not thread.is_alive():
        process.stdout.close()
        log.close()


def descendant_groups(root_pid):
    """Find only process groups led by a descendant of our child process."""
    try:
        output = subprocess.check_output(["ps", "-eo", "pid=,ppid=,pgid="], text=True)
    except (OSError, subprocess.CalledProcessError):
        return {root_pid}
    rows = []
    for line in output.splitlines():
        try:
            rows.append(tuple(map(int, line.split())))
        except ValueError:
            continue
    descendants = {root_pid}
    while True:
        found = {pid for pid, ppid, _ in rows if ppid in descendants} - descendants
        if not found:
            break
        descendants.update(found)
    own_group = os.getpgrp()
    return {pgid for pid, _, pgid in rows if pid in descendants and pid == pgid and pgid != own_group} | {root_pid}


def stop_process(process):
    if process.poll() is not None:
        return
    groups = descendant_groups(process.pid)
    for group in sorted(groups, reverse=True):
        try:
            os.killpg(group, signal.SIGTERM)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass
    for group in sorted(groups, reverse=True):
        try:
            os.killpg(group, signal.SIGKILL)
        except ProcessLookupError:
            pass
    process.wait()


def save_summary(path, summary):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def prepare_cases(args, cpus):
    anchor = args.output_dir / "ldm_harness"
    cases = []
    for method, filename in CASES:
        source = BASELINES / filename
        content = source.read_bytes()
        original = json.loads(content)
        if original["method"] != method:
            raise ValueError(f"Unexpected method in {source}")
        if method in ("ldm_harness", "harness") and args.sidecar_image == original["sidecar_image"]:
            raise ValueError("Use the new history-capable sidecar image, not the historical image tag")
        effective = dict(original)
        effective["out_dir"] = str(args.output_dir / method)
        if method != "ldm_harness":
            effective["initialization_run"] = str(anchor)
        if method in ("ldm_harness", "harness"):
            effective["sidecar_image"] = args.sidecar_image
        effective["gpu"] = args.gpu
        effective["concurrency"] = 1
        changes = {k: {"baseline": original.get(k), "effective": v}
                   for k, v in effective.items() if k not in original or original[k] != v}
        python = ANTBO_PYTHON if method == "antbo" else CAMPAIGN_PYTHON
        command = ["taskset", "-c", cpus, str(python), "scripts/run_ldm_tts.py",
                   f"config/antibody_rfd3/{method}.yaml"]
        for key, value in effective.items():
            command.extend(("--set", f"args.{key.replace('_', '-')}={json.dumps(value)}"))
        cases.append({"method": method, "baseline": str(source),
                      "baseline_sha256": hashlib.sha256(content).hexdigest(),
                      "baseline_args": original, "effective_args": effective,
                      "changes": changes, "command": command, "status": "pending",
                      "log": str(args.output_dir / "logs" / f"{method}.log")})
    return cases


def run_case(case, env, timeout, private_values, save):
    case["status"] = "running"
    case["started_at"] = now()
    save()
    print(f"Starting {case['method']}; log: {case['log']}", flush=True)
    started = time.monotonic()
    # A null config value is omitted by the YAML runner; remove environment
    # defaults too, so it remains null in the effective task arguments.
    case_env = dict(env)
    case_env["PYTHONPATH"] = str(REPO)
    if case["effective_args"]["guest_cache"] is None:
        case_env.pop("LDM_GONDOLIN_CACHE", None)
    if case["effective_args"]["absolut_path"] is None:
        case_env.pop("ABSOLUT_PATH", None)
    try:
        process, thread, log = start_logged(case["command"], case_env, Path(case["log"]), private_values)
    except BaseException as exc:
        case["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        case["finished_at"] = now()
        save()
        raise
    try:
        try:
            case["exit_code"] = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            case["status"] = "timed_out"
            stop_process(process)
            raise RuntimeError(f"{case['method']} exceeded {timeout} seconds")
        if case["exit_code"]:
            case["status"] = "failed"
            raise RuntimeError(f"{case['method']} exited with status {case['exit_code']}")
        run_dir = Path(case["effective_args"]["out_dir"])
        if not run_dir.is_dir():
            case["status"] = "failed"
            raise RuntimeError(f"{case['method']} reported success without {run_dir}")
        case["run_dir"] = str(run_dir)
        case["artifacts"] = [name for name in ("summary.json", "status.json", "budget.json", "events.jsonl",
                                             "controller.sqlite3", "experiment-service.sqlite3", "async-runtime.json")
                             if (run_dir / name).exists()]
        case["status"] = "completed"
    except BaseException as exc:
        if case["status"] == "running":
            case["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        raise
    finally:
        if process.poll() is None:
            stop_process(process)
        finish_log(process, thread, log)
        case["finished_at"] = now()
        case["elapsed_seconds"] = round(time.monotonic() - started, 2)
        save()


def validate_anchor(path, baseline):
    job = path / "design_jobs/job-00000"
    receipt = json.loads((job / "receipt.json").read_text())
    expected = {"status": "succeeded", "round": 0, "action": "design_h3", "mock_generator": False,
                "count": baseline["samples_per_call"], "temperature": baseline["temperature"],
                "seed": baseline["seed"]}
    if any(receipt.get(name) != value for name, value in expected.items()):
        raise RuntimeError("The new initialization receipt does not match the baseline")
    for name in ("config.resolved.json", "generation.json", "designs.json"):
        if not (job / "output" / name).is_file():
            raise RuntimeError(f"The new initialization is missing {name}")


def main():
    args = parse_args()
    for path in (VLLM, WEIGHTS, CAMPAIGN_PYTHON, ANTBO_PYTHON, ANTBO_HOME / "bo/optimizer.py"):
        if not path.exists():
            raise RuntimeError(f"Required H20 asset is missing: {path}")
    cpus = ",".join(map(str, sorted(os.sched_getaffinity(0))[:2]))
    cases = prepare_cases(args, cpus)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "logs").mkdir()
    summary_path = args.output_dir / "acceptance-summary.json"
    summary = {"status": "running", "started_at": now(), "checkout": str(REPO),
               "sidecar_image": args.sidecar_image, "gpu": args.gpu,
               "concurrency": 1, "case_timeout_seconds": args.case_timeout,
               "startup_timeout_seconds": args.startup_timeout, "cases": cases}
    save = lambda: save_summary(summary_path, summary)
    save()
    key = secrets.token_urlsafe(24)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), OMP_NUM_THREADS="2",
               OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2", VLLM_API_KEY=key,
               LLM_API_KEY=key, LLM_MODEL_NAME=MODEL, HF_HUB_OFFLINE="1",
               TOKENIZERS_PARALLELISM="false", NO_PROXY=f"{BRIDGE},localhost,127.0.0.1",
               no_proxy=f"{BRIDGE},localhost,127.0.0.1", LDM_GONDOLIN_CACHE=str(ROOT / "cache/gondolin"),
               LDM_PI_IMAGE=args.sidecar_image, VLLM_NO_USAGE_STATS="1", VLLM_WORKER_MULTIPROC_METHOD="spawn",
               VLLM_CACHE_ROOT=str(ROOT / "cache/vllm"), ANTBO_HOME=str(ANTBO_HOME))
    private_values = private_environment_values(env, key)
    server = listener = thread = server_log = bridge_log = None
    try:
        with socket.socket() as sock:
            sock.bind((BRIDGE, 0))
            port = sock.getsockname()[1]
        upstream = f"http://{BRIDGE}:{port}/v1"
        command = ["taskset", "-c", cpus, str(VLLM), "serve", str(WEIGHTS), "--host", BRIDGE,
                   "--port", str(port), "--served-model-name", MODEL, "--max-model-len", "16384",
                   "--max-num-seqs", "1", "--max-num-batched-tokens", "1024",
                   "--gpu-memory-utilization", "0.27", "--enforce-eager", "--enable-auto-tool-choice",
                   "--tool-call-parser", "hermes", "--reasoning-parser", "qwen3", "--generation-config", "vllm"]
        summary["vllm"] = {"command": command, "log": str(args.output_dir / "logs/vllm.log")}
        save()
        server, thread, server_log = start_logged(command, env, Path(summary["vllm"]["log"]), private_values)
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        deadline = time.monotonic() + args.startup_timeout
        while time.monotonic() < deadline:
            if server.poll() is not None:
                raise RuntimeError("vLLM exited during startup; inspect its log")
            try:
                request = urllib.request.Request(upstream + "/models", headers={"Authorization": "Bearer " + key})
                with opener.open(request, timeout=3) as response:
                    models = json.load(response).get("data", [])
                if any(row.get("id") == MODEL for row in models):
                    break
            except (OSError, ValueError, urllib.error.URLError):
                pass
            time.sleep(2)
        else:
            raise RuntimeError("vLLM startup timed out; inspect its log")
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import qwen_responses_bridge as bridge

        bridge_log = (args.output_dir / "logs/bridge.log").open("w", encoding="utf-8")
        def bridge_print(*values, **_kwargs):
            bridge_log.write(scrub(" ".join(map(str, values)), private_values) + "\n")
            bridge_log.flush()
        bridge.print = bridge_print
        listener = bridge.start(BRIDGE, key, upstream)
        env["LLM_BASE_URL"] = f"http://{BRIDGE}:{listener.server_port}/v1"
        summary["vllm"]["model"] = MODEL
        summary["vllm"]["bridge_url"] = env["LLM_BASE_URL"]
        save()
        try:
            run_case(cases[0], env, args.case_timeout, private_values, save)
            try:
                validate_anchor(Path(cases[0]["run_dir"]), cases[0]["baseline_args"])
            except Exception:
                cases[0]["status"] = "invalid_initialization"
                save()
                raise
            run_case(cases[1], env, args.case_timeout, private_values, save)
        finally:
            # The fixed baseline does not allow GPU sharing; release vLLM first.
            stop_process(server)
            finish_log(server, thread, server_log)
            server = None
            if listener is not None:
                listener.shutdown()
                listener.server_close()
                listener = None
            bridge_log.close()
            bridge_log = None
        for case in cases[2:]:
            run_case(case, env, args.case_timeout, private_values, save)
        summary["status"] = "completed"
        print(f"All four cases completed; summary: {summary_path}", flush=True)
        return 0
    except BaseException as exc:
        summary["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        summary["error"] = scrub(f"{type(exc).__name__}: {exc}", private_values)
        print(f"Acceptance {summary['status']}; see {summary_path}", file=sys.stderr, flush=True)
        return 130 if isinstance(exc, KeyboardInterrupt) else 1
    finally:
        if server is not None:
            stop_process(server)
            finish_log(server, thread, server_log)
        if listener is not None:
            listener.shutdown()
            listener.server_close()
        if bridge_log is not None:
            bridge_log.close()
        summary["finished_at"] = now()
        save()


if __name__ == "__main__":
    def interrupted(_signum, _frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    raise SystemExit(main())
