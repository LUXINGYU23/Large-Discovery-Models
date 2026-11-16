"""Container PID 1: isolate the worker identity and attest post-workload counters.

Only this trusted process retains SETUID/SETGID/KILL. The worker has a different
UID and no capabilities; it cannot read PID 1's environment or forge its MAC.

Batch mode (`--batch SECONDS COUNT MODULE`) runs COUNT workers one after another,
worker i on /cases/i/input writing /output/i, each under its own deadline. PID 1
judges and attests every run; a run cannot see or change another run's output.
"""
import hashlib
import hmac
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

PREFIX = "COPPER_RESOURCE_V1 "
# Per-run output limits, equal to the host's limits for a single-run container.
MAX_OUTPUT_FILES, MAX_OUTPUT_BYTES = 32, 4 * 1024 * 1024


def output_usage(root, skip=()):
    """Files and bytes below root, never following links, outside the frozen runs."""
    count = size = 0
    for base, dirs, files in os.walk(root, followlinks=False):
        if Path(base) == Path(root):
            dirs[:] = [d for d in dirs if d not in skip]
        for name in files:
            count += 1
            try:
                size += (Path(base) / name).lstat().st_size
            except FileNotFoundError:
                pass
    return count, size


def _copy_out(source, target):
    """Copy a run's files and links (never following one) into a new read-only directory."""
    target.mkdir()
    for item in sorted(source.iterdir()):
        if item.is_symlink():
            os.symlink(os.readlink(item), target / item.name)
        elif item.is_file():
            try:
                descriptor = os.open(item, os.O_RDONLY | os.O_NOFOLLOW)
            except OSError:
                continue  # unreadable to the host as well: the model counts as missing
            with os.fdopen(descriptor, "rb") as stream:
                body = stream.read(MAX_OUTPUT_BYTES + 1)
            (target / item.name).write_bytes(body)
            (target / item.name).chmod(0o444)
    target.chmod(0o555)


def freeze_output(outputs, index, as_owner=True):
    """Replace /output/<i> by a read-only copy owned by the host owner of /output.

    PID 1 holds no CHOWN/FOWNER/DAC capability, so it cannot take files away from
    the worker. Instead a child switches to the uid owning the /output mount and
    writes the copy; later runs (uid 65534) can neither change nor remove it, and
    the evaluator account on the host can still delete it.
    """
    outputs = Path(outputs)
    run, spent = outputs / str(index), outputs / f".run{index}"
    run.rename(spent)
    if as_owner:
        info = outputs.stat()
        child = os.fork()
        if child == 0:
            try:
                os.setgroups([])
                os.setgid(info.st_gid)
                os.setuid(info.st_uid)
                _copy_out(spent, run)
                os._exit(0)
            except BaseException:
                os._exit(1)
        _, status = os.waitpid(child, 0)
        if status:
            raise RuntimeError("could not freeze a batch run's output")
    else:
        _copy_out(spent, run)
    shutil.rmtree(spent, ignore_errors=True)


def clear_children(path, keep=()):
    """Best effort: without capabilities some worker-owned leftovers may stay."""
    for item in Path(path).iterdir():
        if item.name in keep:
            continue
        try:
            if item.is_dir() and not item.is_symlink():
                shutil.rmtree(item, ignore_errors=True)
            else:
                item.unlink(missing_ok=True)
        except OSError:
            pass


def reap():
    # PID 1 is excluded by Linux kill(-1). Stop detached descendants too.
    try:
        os.kill(-1, signal.SIGKILL)
    except ProcessLookupError:
        pass
    while True:
        try:
            os.waitpid(-1, 0)
        except ChildProcessError:
            break


def oom_kills(cgroup="/sys/fs/cgroup"):
    events = dict(line.split() for line in (Path(cgroup) / "memory.events").read_text().splitlines())
    return int(events.get("oom_kill", 0))


def batch(seconds, count, module, env, *, cases="/cases", outputs="/output", tmp="/tmp",
          cgroup="/sys/fs/cgroup", user=65534, as_owner=True, kill_all=True):
    """One run per case, in order; returns the per-run attestation rows.

    The keyword paths and identities are the container's; tests run the same
    logic unprivileged on a temporary tree (user None, as_owner/kill_all False).
    Output limits apply to what each run adds under /output.
    """
    runs, frozen = [], []
    identity = {} if user is None else dict(user=user, group=user, extra_groups=[])
    for index in range(count):
        output = Path(outputs) / str(index)
        output.mkdir()
        output.chmod(0o777)  # PID 1 owns it; the worker writes into it
        base_files, base_size = output_usage(outputs)
        before = oom_kills(cgroup)
        started = time.monotonic()
        worker = subprocess.Popen(
            ["timeout", "--signal=KILL", seconds, sys.executable, "-m", module],
            env={**env, "COPPER_WORKLOAD_DEADLINE_MONOTONIC": str(started + float(seconds)),
                 "COPPER_INPUT": f"{cases}/{index}/input", "COPPER_OUTPUT": str(output)},
            start_new_session=True, **identity,
        )
        limited = False
        while True:
            try:
                code = worker.wait(timeout=0.2)
                break
            except subprocess.TimeoutExpired:
                files, size = output_usage(outputs)
                if not limited and (files - base_files > MAX_OUTPUT_FILES or size - base_size > MAX_OUTPUT_BYTES):
                    limited = True
                    os.killpg(worker.pid, signal.SIGKILL)
        elapsed = time.monotonic() - started
        if kill_all:
            reap()
        else:
            try:
                os.killpg(worker.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        files, size = output_usage(outputs)
        limited = limited or files - base_files > MAX_OUTPUT_FILES or size - base_size > MAX_OUTPUT_BYTES
        runs.append({"index": index, "exit_code": code if code >= 0 else 128 - code,
                     "worker_elapsed_seconds": elapsed, "worker_budget_seconds": float(seconds),
                     "oom_kill_count": oom_kills(cgroup) - before, "output_limit": limited})
        # Strays outside the run's own directory and in /tmp should not reach the next run.
        frozen.append(str(index))
        freeze_output(outputs, index, as_owner)
        clear_children(outputs, keep=frozen)
        clear_children(tmp)
    return runs


def main():
    if os.getpid() != 1 or os.getuid() != 0:
        raise RuntimeError("supervisor must be root PID 1 in a private container PID namespace")
    key = os.environ["COPPER_RESOURCE_KEY"]
    env = {k: v for k, v in os.environ.items() if k != "COPPER_RESOURCE_KEY"}
    if sys.argv[1] == "--batch":
        return main_batch(key, env, *sys.argv[2:])
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
    attest(key, payload)
    sys.exit(payload["exit_code"])


def attest(key, payload):
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    mac = hmac.new(bytes.fromhex(key), encoded.encode(), hashlib.sha256).hexdigest()
    print(PREFIX + json.dumps({"payload": encoded, "mac": mac}), flush=True)


def main_batch(key, env, seconds, count, module):
    started = time.monotonic()
    runs = batch(seconds, int(count), module, env)
    group = Path("/sys/fs/cgroup")
    peak = group / "memory.peak"
    cpu = dict(line.split() for line in (group / "cpu.stat").read_text().splitlines())
    payload = {
        "schema_version": 1,
        "scope": "container workload and supervisor through final descendant reap",
        "workload_complete": True,
        "peak_memory_bytes": int(peak.read_text()) if peak.exists() else None,
        "cpu_seconds_through_reap": int(cpu["usage_usec"]) / 1e6,
        "oom_kill_count": oom_kills(),
        "exit_code": 0,
        "worker_elapsed_seconds": time.monotonic() - started,
        "worker_budget_seconds": float(seconds) * int(count),
        "runs": runs,
    }
    attest(key, payload)
    sys.exit(0)


if __name__ == "__main__":
    main()
