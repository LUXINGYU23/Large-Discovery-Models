"""Container PID 1: isolate the worker identity and attest post-workload counters.

Only this trusted process retains SETUID/SETGID/KILL. The worker has a different
UID and no capabilities; it cannot read PID 1's environment or forge its MAC.
"""
import hashlib
import hmac
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

PREFIX = "COPPER_RESOURCE_V1 "


def main():
    if os.getpid() != 1 or os.getuid() != 0:
        raise RuntimeError("supervisor must be root PID 1 in a private container PID namespace")
    key = os.environ["COPPER_RESOURCE_KEY"]
    env = {k: v for k, v in os.environ.items() if k != "COPPER_RESOURCE_KEY"}
    seconds, module = sys.argv[1:]
    started = time.monotonic()
    env['COPPER_WORKLOAD_DEADLINE_MONOTONIC'] = str(started + float(seconds))
    worker = subprocess.Popen(
        ["timeout", "--signal=KILL", seconds, sys.executable, "-m", module],
        env=env, user=65534, group=65534, extra_groups=[], start_new_session=True,
    )
    code = worker.wait()
    # PID 1 is excluded by Linux kill(-1). Stop detached descendants too, then
    # reap them before reading the hierarchical lifetime high-water mark.
    try:
        os.kill(-1, signal.SIGKILL)
    except ProcessLookupError:
        pass
    while True:
        try:
            os.waitpid(-1, 0)
        except ChildProcessError:
            break
    group = Path("/sys/fs/cgroup")
    peak = group / "memory.peak"
    cpu = dict(line.split() for line in (group / "cpu.stat").read_text().splitlines())
    events = dict(line.split() for line in (group / "memory.events").read_text().splitlines())
    payload = {
        "schema_version": 1,
        "scope": "container workload and supervisor through final descendant reap",
        "workload_complete": True,
        "peak_memory_bytes": int(peak.read_text()) if peak.exists() else None,
        "cpu_seconds_through_reap": int(cpu["usage_usec"]) / 1e6,
        "oom_kill_count": int(events.get("oom_kill", 0)),
        "exit_code": code if code >= 0 else 128 - code,
        "worker_elapsed_seconds": time.monotonic() - started,
        "worker_budget_seconds": float(seconds),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    mac = hmac.new(bytes.fromhex(key), encoded.encode(), hashlib.sha256).hexdigest()
    print(PREFIX + json.dumps({"payload": encoded, "mac": mac}), flush=True)
    sys.exit(payload["exit_code"])


if __name__ == "__main__":
    main()
