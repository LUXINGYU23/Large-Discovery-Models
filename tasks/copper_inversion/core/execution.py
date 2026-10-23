"""Task-local Docker jobs; shared Campaign owns evaluations and their publication.

Containers have stable identities, no network/credentials, bounded resources and
read-only code/input. The independent scorer never mounts candidate source.
"""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import shutil
from datetime import datetime
from ldm_tts.engine.run_store import atomic_json_write
from ldm_tts.contracts.evaluation import EvaluationPaused
from .geometry import read_model, MAX_MODEL_BYTES

TASK = Path(__file__).resolve().parents[1]


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def source_identity():
    return digest(
        {
            str(p.relative_to(TASK)): hashlib.sha256(p.read_bytes()).hexdigest()
            for folder in ("core", "resources")
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


class Execution:
    def __init__(self, root, image, account=None):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.image = docker(
            "image", "inspect", image, "--format", "{{.Id}}"
        ).stdout.strip()
        self.account = account

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
    ):
        key = digest(
            {
                "identity": identity,
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
        name = "ldm-copper-" + digest(str(self.root))[:10] + "-" + key[:20]
        if receipt.exists():
            result = json.loads(receipt.read_text())
            self._account(key, result, phase)
            docker("rm", name, check=False)
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
                f"{os.getuid()}:{os.getgid()}",
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
                "-w",
                "/tmp",
            ]
            mounts = [
                (inputs, "/input", True),
                (TASK / "core", "/code/tasks/copper_inversion/core", True),
                (out, "/output", False),
            ]
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
                "tasks.copper_inversion.core."
                + ("scorer" if phase == "scorer" else "runner"),
            ]
            docker(*command)
        state = json.loads(docker("inspect", name).stdout)[0]["State"]
        if state["Status"] == "created":
            record.update(started=True, start_time=time.time())
            atomic_json_write(job, record)
            docker("start", name)
        elif not record["started"]:
            raise EvaluationPaused("container state differs from pre-dispatch receipt")
        deadline = record["start_time"] + timeout
        reason = record.get("stop_reason")
        while True:
            state = json.loads(docker("inspect", name).stdout)[0]["State"]
            if not state["Running"]:
                break
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
                reason = "timeout" if time.time() > deadline else "output_limit"
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
        log = docker("logs", name, check=False)
        (root / "log.txt").write_text((log.stdout + log.stderr)[-1024 * 1024 :])
        result = {
            "exit_code": state["ExitCode"],
            "oom": state["OOMKilled"],
            "reason": reason,
            # Docker's terminal timestamps exclude controller downtime on replay.
            "wall_seconds": max(
                0,
                datetime.fromisoformat(state["FinishedAt"].replace("Z", "+00:00")).timestamp()
                - datetime.fromisoformat(state["StartedAt"].replace("Z", "+00:00")).timestamp(),
            ),
            "container": name,
            "key": key,
            "image": self.image,
            "cpus": cpus,
        }
        atomic_json_write(receipt, result)
        self._account(key, result, phase)
        docker("rm", name, check=False)
        return root, result

    def _account(self, key, result, phase):
        if self.account:
            self.account(
                {f"{phase}_jobs": 1, f"{phase}_wall_seconds": result["wall_seconds"]},
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
