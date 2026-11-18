"""Sequence evaluation contracts for the independent experiment service."""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import stat
import subprocess
import sys
import time
import traceback
import uuid

from experiment_service import (Artifact, AttemptStatus, BlockingExecutor, DispatchUncertain,
                                ExecutorCapabilities)
from ldm_tts.async_runtime import ExperimentRequest, MappedCodec

KIND = "antibody_rfd3.energy"
VERSION = "1"
ENERGY_UTILITY = "energy_utility"
ABSOLUT_TIMEOUT_SECONDS = 1800
STOP_GRACE_SECONDS = 5
LOCK_PREFIX = "ldm-absolut-v1"


def history_projection(metric):
    return {"version": 1, "fields": [
        {"name": "sequence", "type": "string", "source": "candidate.payload.sequence", "filterable": True},
        {"name": "backbone_id", "type": "string", "source": "candidate.payload.backbone_id", "filterable": True},
        {"name": "energy", "type": "number", "source": f"evaluation.metrics.{metric}",
         "filterable": True, "sortable": True, "direction": "minimize"},
    ], "artifacts": []}


class AntibodyCodec(MappedCodec):
    version = "2"

    def __init__(self, antigen, oracle, *, utility=False):
        self.antigen, self.oracle = antigen, oracle
        self.utility = utility
        self.metric = "mock_energy" if oracle == "mock" else "absolut_energy"
        self.name = f"antibody-energy[{antigen}:{oracle}:" + ("negative-energy-v1" if utility else "raw") + "]"

    def to_request(self, candidate):
        limits = {"max_runtime_seconds": ABSOLUT_TIMEOUT_SECONDS} if self.oracle == "absolut" else {}
        return ExperimentRequest(KIND, VERSION, {"sequence": candidate.payload["sequence"],
                                                "antigen": self.antigen, "oracle": self.oracle}, limits)

    def metrics(self, candidate, result):
        energy = float(result[self.metric])
        metrics = {self.metric: energy}
        if self.utility:
            metrics[ENERGY_UTILITY] = -energy
        return metrics


def mock_energy(antigen, sequence):
    digest = hashlib.sha256(f"mock-v1/{antigen}/{sequence}".encode()).digest()
    return -int.from_bytes(digest[:4], "big") / 2**32


def _validate_spec(spec, antigen, oracle):
    from tasks.antibody.core.engine_adapters import _sequence_violations
    if set(spec) != {"sequence", "antigen", "oracle"} or spec.get("antigen") != antigen or spec.get("oracle") != oracle:
        return ["candidate spec does not match the configured antigen and oracle"]
    sequence = spec.get("sequence")
    if not isinstance(sequence, str):
        return ["sequence must be a string"]
    return _sequence_violations(sequence, 11)


def make_executor(context):
    options = dict(context.options)
    unknown = set(options) - {"workers", "antigen", "oracle", "absolut_path", "work_dir"}
    if unknown:
        raise ValueError(f"unknown antibody executor options: {sorted(unknown)}")
    antigen, oracle = options["antigen"], options["oracle"]
    if oracle not in ("mock", "absolut"):
        raise ValueError("unknown antibody oracle")
    validate = lambda spec: _validate_spec(spec, antigen, oracle)
    if oracle == "mock":
        return BlockingExecutor(KIND, VERSION, lambda spec: {"mock_energy": mock_energy(antigen, spec["sequence"])},
                                validate=validate, workers=int(options.get("workers", 1)), retry_safe=True)
    if not options.get("absolut_path"):
        raise ValueError("Absolut path must contain src/bin/Absolut")
    base = Path(options["absolut_path"]).resolve()
    if not (base / "src/bin/Absolut").is_file():
        raise ValueError("Absolut path must contain src/bin/Absolut")
    if not options.get("work_dir"):
        raise ValueError("Absolut executor needs a campaign-local work_dir")
    return AbsolutExecutor(base, Path(options["work_dir"]), antigen, validate)


def _write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, allow_nan=False, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _boot_id():
    return Path("/proc/sys/kernel/random/boot_id").read_text(encoding="utf-8").strip()


def _proc_stat(pid):
    try:
        text = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        fields = text[text.rfind(")") + 2:].split()
        return {"state": fields[0], "group": int(fields[2]), "start": int(fields[19])}
    except (FileNotFoundError, ProcessLookupError):
        return None


def _group_alive(identity):
    """Check a Linux process group without mistaking a reused PID for this attempt."""
    if identity["boot_id"] != _boot_id():
        return False
    pid = identity["pid"]
    leader = _proc_stat(pid)
    if leader is not None and leader["start"] != identity["start"]:
        return False
    if leader is not None and leader["state"] != "Z":
        return True
    # The leader can exit before a child; never release the lock or acknowledge
    # cancellation until every live member of its group has stopped.
    for entry in Path("/proc").iterdir():
        if entry.name.isdigit():
            member = _proc_stat(entry.name)
            if member is not None and member["group"] == pid and member["state"] != "Z":
                return True
    return False


def _identity(pid):
    stat = _proc_stat(pid)
    if stat is None:
        raise RuntimeError("worker disappeared before its identity could be recorded")
    return {"pid": int(pid), "start": stat["start"], "boot_id": _boot_id()}


@contextmanager
def _lock_guard(path):
    with Path(str(path) + ".guard").open("a") as guard:
        fcntl.flock(guard, fcntl.LOCK_EX)
        yield


def _lock_identity(target):
    parts = target.split(":")
    if len(parts) != 5 or parts[0] != LOCK_PREFIX:
        return None
    try:
        return {"boot_id": parts[1], "pid": int(parts[2]), "start": int(parts[3])}
    except ValueError:
        return None


def _remove_stale_lock(path, expected=None):
    """Remove only a dead worker's own managed lock; never remove legacy locks."""
    path = Path(path)
    with _lock_guard(path):
        try:
            target = os.readlink(path)
        except (FileNotFoundError, OSError):
            return False
        owner = _lock_identity(target)
        if owner is None or (expected is not None and owner != expected) or _group_alive(owner):
            return False
        path.unlink()
        return True


def _lock_path(base, antigen):
    return Path(base) / f".antbo_llm_acq_{antigen}.lock"


def _assets(base, antigen):
    """Validate the run's immutable, explicitly listed Absolut inputs once."""
    if not re.fullmatch(r"[A-Za-z0-9_]+", antigen):
        raise ValueError("invalid Absolut antigen name")
    manifest_path = base / "assets.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("Absolut assets.json is required")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("antigen") != antigen:
        raise ValueError("Absolut asset manifest version or antigen does not match")
    records = manifest.get("assets")
    if not isinstance(records, list) or not records:
        raise ValueError("Absolut asset manifest must contain an asset list")
    names = set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("invalid Absolut asset record")
        name, size, checksum = record.get("name"), record.get("bytes"), record.get("sha256")
        if not isinstance(name, str) or not name:
            raise ValueError("invalid Absolut asset name")
        relative = Path(name)
        if (relative.is_absolute() or ".." in relative.parts or relative.as_posix() != name or
                (relative.parent != Path(".") and name != "src/bin/Absolut") or name in names):
            raise ValueError(f"invalid or duplicate Absolut asset name: {name}")
        if (name == "assets.json" or name.startswith(f".antbo_llm_acq_{antigen}.lock") or
                name.startswith(f"TempCDR3_{antigen}_") or name.startswith(f"TempBindingsFor{antigen}_") or
                name.startswith(f"{antigen}FinalBindings_")):
            raise ValueError(f"Absolut asset conflicts with a runtime output: {name}")
        if (not isinstance(size, int) or isinstance(size, bool) or size < 0 or
                not isinstance(checksum, str) or not re.fullmatch(r"[0-9a-f]{64}", checksum)):
            raise ValueError(f"invalid Absolut asset metadata: {name}")
        path = base / relative
        part = path
        while part != base:
            if part.is_symlink():
                raise ValueError(f"Absolut assets must not contain symlinks: {name}")
            part = part.parent
        try:
            info = path.stat()
        except FileNotFoundError as exc:
            raise ValueError(f"missing Absolut asset: {name}") from exc
        if (not stat.S_ISREG(info.st_mode) or info.st_size != size or info.st_mode & 0o222 or
                os.access(path, os.W_OK)):
            raise ValueError(f"Absolut asset must be a read-only regular file of the recorded size: {name}")
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != checksum:
            raise ValueError(f"Absolut asset checksum differs: {name}")
        names.add(name)
    if "src/bin/Absolut" not in names or not os.access(base / "src/bin/Absolut", os.X_OK):
        raise ValueError("Absolut manifest must include an executable src/bin/Absolut")
    # The native repertoire command otherwise regenerates caches in its working
    # directory. Require the matching structure, detailed and compact inputs.
    compact = (re.fullmatch(r"(.+)\+(.+)-10-11-(.+)Compact\.txt", name) for name in names)
    if ("selfFoldingsL=10minI=11Compact.txt" not in names or not any(
            match and f"{match.group(1)}-10-11-{match.group(3)}Structures.txt" in names and
            f"{match.group(1)}+{match.group(2)}-10-11-{match.group(3)}.txt" in names
            for match in compact)):
        raise ValueError("Absolut manifest does not contain a complete precomputed cache")
    return tuple(sorted(names))


class AbsolutExecutor:
    """External process groups with durable references and confirmed cancellation."""

    kind, kind_version = KIND, VERSION
    # Attempts use isolated workspaces and immutable inputs, so a confirmed
    # infrastructure failure can be retried without repeating external effects.
    capabilities = ExecutorCapabilities(cancel_running="cooperative", dispatch="lookup", retry_safe=True,
                                        in_process=False, cancel_idempotent=True)

    def __init__(self, base, work_dir, antigen, validate):
        self.base, self.work_dir, self.antigen, self._validate = Path(base).resolve(), work_dir.resolve(), antigen, validate
        if self.work_dir == self.base or self.work_dir.is_relative_to(self.base):
            raise ValueError("Absolut attempt directory must be outside the input library")
        self.assets = _assets(self.base, antigen)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.processes = {}

    def validate(self, spec):
        return self._validate(spec)

    @staticmethod
    def _ref(experiment_id, index):
        return hashlib.sha256(json.dumps([experiment_id, index], separators=(",", ":")).encode()).hexdigest()

    def start(self, attempt):
        ref = self._ref(attempt.experiment_id, attempt.attempt_index)
        directory = self.work_dir / ref
        try:
            directory.mkdir()
        except FileExistsError:
            if self.find(attempt.experiment_id, attempt.attempt_index):
                return ref
            raise DispatchUncertain("attempt directory exists but its launch was not confirmed")
        process = None
        try:
            workspace = directory / "absolut"
            workspace.mkdir()
            for name in self.assets:
                target = workspace / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.symlink_to(self.base / name)
            _write_json(directory / "request.json", {**dict(attempt.spec), "absolut_path": str(self.base),
                                                       "attempt_id": uuid.uuid4().hex})
            env = {key: value for key, value in os.environ.items()
                   if not any(key.upper().endswith(suffix) for suffix in ("_KEY", "_TOKEN", "_SECRET", "_PASSWORD"))}
            with (directory / "worker.log").open("ab") as log:
                process = subprocess.Popen([sys.executable, "-m", "tasks.antibody_rfd3.core.experiments",
                                            "absolut", str(directory)], stdout=log, stderr=subprocess.STDOUT,
                                           env=env, start_new_session=True)
        except Exception as exc:
            if process is not None:
                self.processes[ref] = process
                raise DispatchUncertain("worker launched but dispatch confirmation failed") from exc
            # Nothing was launched, so cleanup cannot affect a live worker.
            shutil.rmtree(directory)
            raise
        self.processes[ref] = process
        return ref

    def find(self, experiment_id, attempt_index):
        ref = self._ref(experiment_id, attempt_index)
        directory = self.work_dir / ref
        # An intent is a stable reference even if the worker has not written
        # its identity yet. poll() reports that uncertainty without relaunching.
        return ref if ref in self.processes or (directory / "request.json").is_file() else None

    def _info(self, ref):
        path = self.work_dir / ref / "worker.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None

    def _signal(self, ref, sig):
        process = self.processes.get(ref)
        identity = self._info(ref)
        if identity is None:
            if process is None or process.poll() is not None:
                raise RuntimeError("worker identity is not known; cancellation is unconfirmed")
            pid = process.pid  # An unreaped Popen child cannot have its PID reused.
        else:
            if not _group_alive(identity):
                return
            pid = identity["pid"]
        try:
            os.killpg(pid, sig)
        except ProcessLookupError:
            pass

    def cancel(self, ref):
        status = self.poll(ref)
        if status.terminal:
            return
        directory = self.work_dir / ref
        if not directory.is_dir():
            raise RuntimeError("unknown attempt; cancellation is unconfirmed")
        cancel = directory / "cancel.json"
        if not cancel.exists():
            _write_json(cancel, {"requested_at": time.monotonic()})
        self._signal(ref, signal.SIGTERM)

    def poll(self, ref):
        directory = self.work_dir / ref
        if not directory.is_dir():
            return AttemptStatus.unreachable("unknown executor reference")
        process = self.processes.get(ref)
        exit_code = process.poll() if process is not None else None
        identity = self._info(ref)
        if identity is None:
            if process is not None and exit_code is not None:
                # The worker records its identity before it can launch Absolut.
                if (directory / "cancel.json").exists():
                    return AttemptStatus.stopped()
                return AttemptStatus.failed("infrastructure", f"worker exited before recording its identity; see {directory / 'worker.log'}")
            return AttemptStatus.active("starting") if process is not None else AttemptStatus.unreachable("worker launch unconfirmed")
        alive = _group_alive(identity)
        cancel = directory / "cancel.json"
        if alive and cancel.exists():
            requested = json.loads(cancel.read_text(encoding="utf-8"))["requested_at"]
            if time.monotonic() - requested >= STOP_GRACE_SECONDS:
                self._signal(ref, signal.SIGKILL)
        if alive:
            return AttemptStatus.active("stopping" if cancel.exists() else "running")
        _remove_stale_lock(_lock_path(directory / "absolut", self.antigen), expected=identity)
        result_path = directory / "result.json"
        if result_path.is_file():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if result["state"] == "succeeded":
                return AttemptStatus.succeeded(result["metrics"], artifacts=(
                    Artifact("absolut.log", "text/plain", path=str(directory / "worker.log")),))
            if result["state"] == "stopped":
                return AttemptStatus.stopped()
            return AttemptStatus.failed(result.get("failure_class", "execution_error"),
                                        f"{result.get('message', '')}; see {directory / 'worker.log'}")
        if cancel.exists():
            return AttemptStatus.stopped()
        return AttemptStatus.failed("infrastructure", f"worker exited without a result; see {directory / 'worker.log'}")

    def close(self):
        # Recovered attempts have no Popen handle in this ES process. They are
        # still ours and must receive the same confirmed shutdown treatment.
        refs = sorted(set(self.processes) | {path.name for path in self.work_dir.iterdir()
                      if path.is_dir() and (path / "worker.json").is_file()})
        for ref in refs:
            if not self.poll(ref).terminal:
                self.cancel(ref)
        deadline = time.monotonic() + STOP_GRACE_SECONDS + 5
        while refs and time.monotonic() < deadline:
            refs = [ref for ref in refs if not self.poll(ref).terminal]
            if refs:
                time.sleep(0.05)
        if refs:
            raise RuntimeError("Absolut worker termination was not confirmed")


class _Stopped(BaseException):
    pass


def _run_absolut(directory):
    """Run the unchanged scientific evaluator; write JSON separately from its logs."""
    directory = Path(directory)
    identity = _identity(os.getpid())
    _write_json(directory / "worker.json", identity)

    def stop(_signum, _frame):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        raise _Stopped()
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        from tasks.antibody.core.engine_adapters import AntibodyCandidateDomain, AntibodyEvaluator
        from tasks.antibody.core.ldm_light.ldm_acq import AbsolutEvaluator
        from ldm_tts.contracts import RawProposal

        class ManagedAbsolut(AbsolutEvaluator):
            def _acquire_lock(self, lock_dir, timeout_s=600):
                path = Path(lock_dir)
                self.lock_target = f"{LOCK_PREFIX}:{identity['boot_id']}:{identity['pid']}:{identity['start']}:{uuid.uuid4().hex}"
                deadline = time.monotonic() + timeout_s
                while True:
                    _remove_stale_lock(path)
                    with _lock_guard(path):
                        try:
                            os.symlink(self.lock_target, path)
                            return
                        except FileExistsError:
                            pass
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"Timed out waiting for Absolut lock: {path}")
                    time.sleep(0.2)

            def _release_lock(self, lock_dir):
                # The evaluator's finally block can run while a subprocess is
                # still exiting. The ES removes our symlink only after the
                # entire recorded process group is confirmed dead.
                pass

        request = json.loads((directory / "request.json").read_text(encoding="utf-8"))
        cpus = sorted(os.sched_getaffinity(0))
        evaluator = AntibodyEvaluator(ManagedAbsolut({"antigen": request["antigen"], "process": 1,
            "startTask": cpus[0], "cpu_affinity": cpus, "path": str(directory / "absolut")},
            request["attempt_id"]))
        candidate = AntibodyCandidateDomain(11).admit(RawProposal({"sequence": request["sequence"]}, "experiment"))
        result = evaluator.evaluate(candidate)
        if not result.succeeded:
            raise RuntimeError(result.error)
        outcome = {"state": "succeeded", "metrics": result.metrics}
    except _Stopped:
        outcome = {"state": "stopped"}
    except Exception as exc:
        traceback.print_exc()
        outcome = {"state": "failed", "failure_class": "timeout" if isinstance(exc, TimeoutError) else "execution_error",
                   "message": f"{type(exc).__name__}: {exc}"[:4000]}
    _write_json(directory / "result.json", outcome)


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "absolut":
        raise SystemExit("expected: absolut <attempt-directory>")
    _run_absolut(sys.argv[2])
