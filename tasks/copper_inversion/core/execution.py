"""Task-local Docker jobs; shared Campaign owns evaluations and their publication.

Containers have stable identities, no network/credentials, bounded resources and
read-only code/input. The independent scorer never mounts candidate source.
"""

import hashlib
import hmac
import json
import os
from pathlib import Path
import subprocess
import time
import threading
import shutil
import secrets
from datetime import datetime
from ldm_tts.engine.run_store import atomic_json_write
from ldm_tts.contracts.evaluation import EvaluationPaused
from .geometry import read_model, MAX_MODEL_BYTES

TASK = Path(__file__).resolve().parents[1]


def authenticated_resources(log, key):
    """Ignore candidate stdout, including forged resource records."""
    from .supervisor import PREFIX
    for line in reversed(log.splitlines()):
        if not line.startswith(PREFIX):
            continue
        try:
            envelope = json.loads(line[len(PREFIX):])
            encoded, mac = envelope["payload"], envelope["mac"]
            expected = hmac.new(bytes.fromhex(key), encoded.encode(), hashlib.sha256).hexdigest()
            if hmac.compare_digest(expected, mac):
                return json.loads(encoded)
        except (ValueError, KeyError, TypeError, AttributeError):
            continue
    return None


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def source_identity():
    return digest(
        {
            str(p.relative_to(TASK)): hashlib.sha256(p.read_bytes()).hexdigest()
            for folder in ("core", "v2", "resources")
            for p in sorted((TASK / folder).rglob("*"))
            if p.is_file()
            and "__pycache__" not in p.parts
            and p.suffix != ".pyc"
            and p.name != "qualification_evidence.json"
            and "evidence" not in p.parts
        }
    )


def docker(*args, check=True):
    try:
        r = subprocess.run(
            ["docker", *map(str, args)], capture_output=True, text=True, timeout=40
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        raise EvaluationPaused("Docker command outcome unavailable; reconcile the job before retry") from exc
    if check and r.returncode:
        raise EvaluationPaused("Docker operation failed: " + r.stderr[-700:])
    return r


def sample_resources(record, pid):
    """Host-owned cgroup v2 samples, including all container descendants.

    memory.current is sampled, not a lifetime high-water mark. CPU is cumulative
    through the last available sample. Never substitute zero for missing data.
    """
    try:
        membership = (Path("/proc") / str(pid) / "cgroup").read_text()
        relative = next(line[3:] for line in membership.splitlines() if line.startswith("0::"))
        if record["container_id"] not in relative:
            raise ValueError("PID no longer belongs to the dispatched container")
        group = Path("/sys/fs/cgroup") / relative.lstrip("/")
        memory = int((group / "memory.current").read_text())
        cpu = dict(line.split() for line in (group / "cpu.stat").read_text().splitlines())
        seconds = int(cpu["usage_usec"]) / 1e6
    except (OSError, ValueError, KeyError, StopIteration):
        # Short-lived/exited containers may have already lost their cgroup.
        record.setdefault("resource_samples", 0)
        return
    record["resource_samples"] = record.get("resource_samples", 0) + 1
    record["sampled_peak_memory_bytes"] = max(record.get("sampled_peak_memory_bytes", 0), memory)
    record["cpu_seconds_observed"] = max(record.get("cpu_seconds_observed", 0), seconds)
    record["last_resource_sample_unix"] = time.time()


class Execution:
    def __init__(self, root, image, account=None):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.image = docker(
            "image", "inspect", image, "--format", "{{.Id}}"
        ).stdout.strip()
        self.account = account
        self._account_lock = threading.Lock()

    def run(
        self,
        identity,
        *,
        phase,
        inputs,
        timeout,
        cpus=1,
        memory_mb=2048,
        source=None,
        model=None,
        science_module="core",
        truth=None,
    ):
        if science_module not in {"core", "v2"}:
            raise ValueError("unknown scientific contract")
        if truth is not None and (phase != "scorer" or science_module != "v2" or source is not None):
            raise ValueError("hidden truth is restricted to the independent v2 scorer")
        if phase == "scorer" and source is not None:
            raise ValueError("independent scorer cannot mount candidate source")
        truth_digest = None
        if truth is not None:
            truth_digest = hashlib.sha256((Path(truth) / "truth.npy").read_bytes()).hexdigest()
        key = digest(
            {
                "identity": identity,
                "science_module": science_module,
                "truth_digest": truth_digest,
                "phase": phase,
                "image": self.image,
                "timeout": timeout,
                "cpus": cpus,
                "memory_mb": memory_mb,
                "implementation": source_identity(),
            }
        )
        root = self.root / key
        root.mkdir(exist_ok=True)
        receipt = root / "receipt.json"
        job = root / "job.json"
        out = root / "output"
        out.mkdir(exist_ok=True)
        # The unprivileged worker is a different UID from PID 1 and the host.
        out.chmod(0o1777)
        # Only public problem/budget files are mounted here. Atomic JSON writes
        # default to 0600. Keep host-owner write access for recovery; the worker
        # receives a read-only bind mount and can only read these public files.
        Path(inputs).chmod(0o755)
        for path in Path(inputs).iterdir():
            path.chmod(0o644)
        name = "ldm-copper-" + digest(str(self.root))[:10] + "-" + key[:20]
        if receipt.exists():
            result = json.loads(receipt.read_text())
            self._account(key, result, phase)
            docker("rm", name, check=False)
            if result.get('reason') == 'supervisor_timeout':
                raise EvaluationPaused('container supervisor exceeded infrastructure grace; reconcile recorded job')
            return root, result
        if job.exists():
            record = json.loads(job.read_text())
            if record["key"] != key:
                raise EvaluationPaused("job identity changed")
        else:
            record = {
                "key": key,
                "name": name,
                "created": time.time(),
                "started": False,
                "resource_key": secrets.token_hex(32),
            }
            atomic_json_write(job, record)
        inspection = docker("inspect", name, check=False)
        if inspection.returncode:
            if record["started"]:
                raise EvaluationPaused(
                    "dispatched container missing; reconcile before retry"
                )
            command = [
                "create",
                "--name",
                name,
                "--network",
                "none",
                "--read-only",
                "--cap-drop",
                "ALL",
                "--cap-add", "SETUID",
                "--cap-add", "SETGID",
                "--cap-add", "KILL",
                "--security-opt",
                "no-new-privileges",
                "--pids-limit",
                "64",
                "--cpus",
                str(cpus),
                "--memory",
                f"{memory_mb}m",
                "--memory-swap",
                f"{memory_mb}m",
                "--user",
                "0:0",
                "--tmpfs",
                "/tmp:rw,nosuid,size=128m",
                "--ulimit",
                "fsize=2097152:2097152",
                "--log-opt",
                "max-size=1m",
                "--log-opt",
                "max-file=1",
                "-e",
                "PYTHONDONTWRITEBYTECODE=1",
                "-e",
                "OPENBLAS_NUM_THREADS=1",
                "-e",
                "OMP_NUM_THREADS=1",
                "-e",
                "PYTHONPATH=/code",
                "-e",
                "COPPER_RESOURCE_KEY=" + record["resource_key"],
                "-w",
                "/tmp",
            ]
            mounts = [
                (inputs, "/input", True),
                (TASK / "core", "/code/tasks/copper_inversion/core", True),
                (TASK / "v2", "/code/tasks/copper_inversion/v2", True),
                (out, "/output", False),
            ]
            if truth is not None:
                mounts.append((truth, "/truth", True))
            if source:
                mounts.append((source, "/submitted", True))
            if model:
                mounts.append((model, "/model", True))
            for a, b, ro in mounts:
                command += [
                    "--mount",
                    f"type=bind,src={Path(a).resolve()},dst={b}"
                    + (",readonly" if ro else ""),
                ]
            command += [
                self.image,
                "python",
                "-m",
                "tasks.copper_inversion.core.supervisor",
                str(timeout),
                f"tasks.copper_inversion.{science_module}."
                + ("scorer" if phase == "scorer" else "runner"),
            ]
            docker(*command)
        inspected = json.loads(docker("inspect", name).stdout)[0]
        if record.get("container_id") not in (None, inspected["Id"]):
            raise EvaluationPaused("container identity changed after dispatch")
        record["container_id"] = inspected["Id"]
        state = inspected["State"]
        if state["Status"] == "created":
            record.update(started=True, start_time=time.time())
            atomic_json_write(job, record)
            docker("start", name)
        elif not record["started"]:
            raise EvaluationPaused("container state differs from pre-dispatch receipt")
        # Workload timeout remains inside the container; allow only a bounded
        # finalization interval for descendant reap and authenticated counters.
        # Docker admission/boot and PID-1 finalization are infrastructure time.
        # The worker's independent coreutils deadline still enforces exactly
        # `timeout`, even if this controller disappears. Do not shorten it
        # because docker start was queued on a busy daemon.
        state = json.loads(docker("inspect", name).stdout)[0]["State"]
        container_started = datetime.fromisoformat(state["StartedAt"].replace("Z", "+00:00")).timestamp()
        deadline = container_started + timeout + 30
        reason = record.get("stop_reason")
        while True:
            state = json.loads(docker("inspect", name).stdout)[0]["State"]
            if not state["Running"]:
                break
            sample_resources(record, state["Pid"])
            atomic_json_write(job, record)
            # Filesize limits apply per-file; also stop candidates that create many files.
            count = 0
            size = 0
            for base, dirs, files in os.walk(out, followlinks=False):
                for f in files:
                    count += 1
                    try:
                        size += (Path(base) / f).lstat().st_size
                    except FileNotFoundError:
                        pass
                    if count > 32 or size > 4 * 1024 * 1024:
                        break
                if count > 32 or size > 4 * 1024 * 1024:
                    break
            if time.time() > deadline or count > 32 or size > 4 * 1024 * 1024:
                reason = "supervisor_timeout" if time.time() > deadline else "output_limit"
                # Persist the controller's decision before killing: a restart must
                # retain the timeout classification even if no receipt was written.
                record["stop_reason"] = reason
                atomic_json_write(job, record)
                docker("kill", name, check=False)
                break
            time.sleep(0.2)
        state = json.loads(docker("inspect", name).stdout)[0]["State"]
        if state["Running"]:
            raise EvaluationPaused("container did not stop")
        wall_seconds = max(
            0,
            datetime.fromisoformat(state["FinishedAt"].replace("Z", "+00:00")).timestamp()
            - datetime.fromisoformat(state["StartedAt"].replace("Z", "+00:00")).timestamp(),
        )
        # The in-container coreutils deadline remains active while the controller
        # is offline. A terminal kill at its deadline is still a timeout on replay.
        log = docker("logs", name, check=False)
        (root / "log.txt").write_text((log.stdout + log.stderr)[-1024 * 1024 :])
        resources = authenticated_resources(log.stdout, record["resource_key"])
        oom = state["OOMKilled"] or bool(resources and resources["oom_kill_count"])
        if reason is None and not oom and state["ExitCode"] == 137 and wall_seconds >= timeout:
            reason = "timeout"
        if reason is not None:
            record["stop_reason"] = reason
            atomic_json_write(job, record)
        if resources is None and not (oom or reason):
            raise EvaluationPaused("authenticated workload counters unavailable")
        result = {
            "exit_code": state["ExitCode"],
            "oom": oom,
            "reason": reason,
            # Docker's terminal timestamps exclude controller downtime on replay.
            "wall_seconds": wall_seconds,
            "container": name,
            "container_id": record["container_id"],
            "key": key,
            "image": self.image,
            "cpus": cpus,
            "workload_resources": resources,
            "resource_sampling": {
                "method": "host_cgroup_v2_polling",
                "target_interval_seconds": 0.2,
                "samples": record.get("resource_samples", 0),
                "sampled_peak_memory_bytes": record.get("sampled_peak_memory_bytes"),
                "cpu_seconds_observed": record.get("cpu_seconds_observed"),
                "last_sample_unix": record.get("last_resource_sample_unix"),
                "complete_lifetime_measurement": False,
            },
        }
        atomic_json_write(receipt, result)
        self._account(key, result, phase)
        docker("rm", name, check=False)
        if reason == 'supervisor_timeout':
            raise EvaluationPaused('container supervisor exceeded infrastructure grace; reconcile recorded job')
        return root, result

    def _account(self, key, result, phase):
        with self._account_lock:
            self._record_account(key, result, phase)

    def _record_account(self, key, result, phase):
        if self.account:
            counts = {f"{phase}_jobs": 1, f"{phase}_wall_seconds": result["wall_seconds"]}
            sampling = result["resource_sampling"]
            workload = result["workload_resources"]
            if workload is not None:
                counts[f"{phase}_cpu_seconds_through_reap"] = workload["cpu_seconds_through_reap"]
            if sampling["cpu_seconds_observed"] is not None:
                counts[f"{phase}_cpu_seconds_observed"] = sampling["cpu_seconds_observed"]
            self.account(
                counts,
                usage_key="copper:job:" + key,
            )


def freeze_model(path, destination, shape):
    # All worker processes have stopped before this function. Never follow a link.
    read_model(path, shape)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / "model.npy"
    data = Path(path).read_bytes()
    if len(data) > MAX_MODEL_BYTES:
        raise ValueError("oversized model")
    sha = hashlib.sha256(data).hexdigest()
    if target.exists() and target.read_bytes() != data:
        raise EvaluationPaused("frozen model changed")
    if not target.exists():
        target.write_bytes(data)
    target.chmod(0o444)
    return sha
