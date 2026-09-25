"""Evaluator subprocess lifecycle: one process group per job, killed as a tree.

A timed-out or interrupted evaluation terminates every process group it
started and waits until the groups are empty before the next candidate may
use the workspace or devices. Jobs are scheduled onto an explicit device list.
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

POLL_SECONDS = 0.2


@dataclass(frozen=True)
class JobCommand:
    job_id: str
    argv: tuple[str, ...]
    env: dict[str, str] = field(default_factory=dict)


@dataclass
class JobOutcome:
    job_id: str
    status: str  # succeeded, failed, timed_out, not_started, interrupted
    returncode: int | None
    device: str | None
    log: str
    elapsed_seconds: float = 0.0
    group_terminated: bool = True

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


class _Running:
    def __init__(self, job: JobCommand, process: subprocess.Popen, device: str | None, log: Path, handle) -> None:
        self.job, self.process, self.device, self.log, self.handle = job, process, device, log, handle
        self.started = time.monotonic()


def _group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def terminate_group(process: subprocess.Popen, *, grace_seconds: float) -> bool:
    """SIGTERM then SIGKILL the whole group; True when no member survives."""
    pgid = process.pid
    for sig, wait in ((signal.SIGTERM, grace_seconds), (signal.SIGKILL, grace_seconds)):
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + wait
        while time.monotonic() < deadline:
            process.poll()
            if process.returncode is not None and not _group_alive(pgid):
                return True
            time.sleep(POLL_SECONDS)
    process.poll()
    return process.returncode is not None and not _group_alive(pgid)


class ProcessTreeRunner:
    def __init__(self, *, devices: tuple[str, ...] = (), max_parallel: int | None = None,
                 grace_seconds: float = 15.0, base_env: dict[str, str] | None = None) -> None:
        self.devices = tuple(devices)
        self.max_parallel = max_parallel or max(1, len(self.devices))
        self.grace_seconds = grace_seconds
        self.base_env = dict(os.environ if base_env is None else base_env)

    def run(self, jobs: list[JobCommand], *, cwd: Path, log_dir: Path, deadline: float,
            tick=None) -> list[JobOutcome]:
        """Run all jobs before the monotonic deadline; never leave a group behind."""
        log_dir.mkdir(parents=True, exist_ok=True)
        pending = list(jobs)
        running: list[_Running] = []
        outcomes: dict[str, JobOutcome] = {}
        free = list(self.devices) if self.devices else [None] * self.max_parallel
        try:
            while pending or running:
                while pending and free and len(running) < self.max_parallel and time.monotonic() < deadline:
                    job = pending.pop(0)
                    device = free.pop(0)
                    running.append(self._start(job, device, cwd, log_dir))
                for item in list(running):
                    code = item.process.poll()
                    if code is None:
                        continue
                    # The leader exited; its descendants in the group must not outlive it.
                    clean = terminate_group(item.process, grace_seconds=self.grace_seconds) \
                        if _group_alive(item.process.pid) else True
                    item.handle.close()
                    outcomes[item.job.job_id] = JobOutcome(
                        item.job.job_id, "succeeded" if code == 0 and clean else "failed", code,
                        item.device, str(item.log), time.monotonic() - item.started, clean)
                    running.remove(item)
                    free.append(item.device)
                if time.monotonic() >= deadline:
                    for item in running:
                        clean = terminate_group(item.process, grace_seconds=self.grace_seconds)
                        item.handle.close()
                        outcomes[item.job.job_id] = JobOutcome(
                            item.job.job_id, "timed_out", item.process.returncode, item.device,
                            str(item.log), time.monotonic() - item.started, clean)
                    running = []
                    for job in pending:
                        outcomes[job.job_id] = JobOutcome(job.job_id, "not_started", None, None, "")
                    pending = []
                    break
                if tick is not None:
                    tick()
                if running:
                    time.sleep(POLL_SECONDS)
        except BaseException:
            for item in running:
                clean = terminate_group(item.process, grace_seconds=self.grace_seconds)
                item.handle.close()
                outcomes[item.job.job_id] = JobOutcome(
                    item.job.job_id, "interrupted", item.process.returncode, item.device,
                    str(item.log), time.monotonic() - item.started, clean)
            raise
        return [outcomes[job.job_id] for job in jobs]

    def _start(self, job: JobCommand, device: str | None, cwd: Path, log_dir: Path) -> _Running:
        log = log_dir / f"{job.job_id}.log"
        handle = log.open("w", encoding="utf-8")
        env = {**self.base_env, **job.env}
        if device is not None:
            env["CUDA_VISIBLE_DEVICES"] = device
        process = subprocess.Popen(
            list(job.argv), cwd=str(cwd), env=env, stdout=handle, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, start_new_session=True,
        )
        return _Running(job, process, device, log, handle)


def free_devices(devices: tuple[str, ...], min_free_mib: int) -> tuple[str, ...]:
    """Keep devices with enough free memory; nvidia-smi absence is an explicit error."""
    if min_free_mib <= 0 or not devices:
        return devices
    try:
        output = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=30,
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"cannot query device memory with nvidia-smi: {exc}") from exc
    available = {}
    for line in output.splitlines():
        index, _, free = line.partition(",")
        available[index.strip()] = int(free.strip() or 0)
    return tuple(device for device in devices if available.get(device, 0) >= min_free_mib)
