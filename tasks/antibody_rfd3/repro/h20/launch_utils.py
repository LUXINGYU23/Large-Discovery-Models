"""H20 process supervision and redacted launch records."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time


REPO = Path(__file__).resolve().parents[4]
ROOT = Path("/Aidata/shared/ldm/antibody-poc")
ACTIVE_ROOT = Path("/Aidata/shared/ldm")
CAMPAIGN_PYTHON = ROOT / "env/antibody_campaign/bin/python"


def now():
    return datetime.now(timezone.utc).isoformat()


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
