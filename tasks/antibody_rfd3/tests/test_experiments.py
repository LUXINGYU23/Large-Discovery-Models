"""Contract checks with fake subprocesses; no Absolut installation is needed."""

import hashlib
import json
from pathlib import Path
import signal
import threading
from types import SimpleNamespace

import pytest

from experiment_service import AttemptContext
from tasks.antibody_rfd3.core import experiments as exp


SPEC = {"sequence": "SNGKDGSPGVY", "antigen": "1ADQ_A", "oracle": "absolut"}


class FakeProcess:
    def __init__(self, pid=12345):
        self.pid = pid
        self.returncode = None

    def poll(self):
        return self.returncode


def prepare_assets(base):
    files = {
        "src/bin/Absolut": b"fake executable",
        "struct+sequence-10-11-forbidden.txt": b"details",
        "struct+sequence-10-11-forbiddenCompact.txt": b"compact",
        "struct-10-11-forbiddenStructures.txt": b"structures",
        "selfFoldingsL=10minI=11Compact.txt": b"0\n",
    }
    records = []
    for name, data in files.items():
        path = base / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        path.chmod(0o555 if name == "src/bin/Absolut" else 0o444)
        records.append({"name": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    exp._write_json(base / "assets.json", {"schema_version": 1, "antigen": SPEC["antigen"], "assets": records})
    return files


def executor(tmp_path, monkeypatch):
    source = tmp_path / "inputs"
    source.mkdir()
    prepare_assets(source)
    process = FakeProcess()
    popens = []

    def spawn(command, **kwargs):
        popens.append((command, kwargs))
        return process

    monkeypatch.setattr(exp.subprocess, "Popen", spawn)
    state = {"alive": True}
    monkeypatch.setattr(exp, "_group_alive", lambda _identity: state["alive"])
    worker = exp.AbsolutExecutor(source, tmp_path / "attempts", SPEC["antigen"], lambda _spec: [])
    context = AttemptContext("campaign.d1.s0", 1, "campaign", exp.KIND, exp.VERSION, SPEC, "digest", {})
    ref = worker.start(context)
    directory = worker.work_dir / ref
    exp._write_json(directory / "worker.json", {"pid": process.pid, "start": 77, "boot_id": "boot"})
    return worker, ref, directory, process, state, popens


def test_mock_remains_sequence_based_and_does_not_start_a_process(tmp_path, monkeypatch):
    monkeypatch.setattr(exp.subprocess, "Popen", lambda *_a, **_k: pytest.fail("unexpected subprocess"))
    context = SimpleNamespace(options={"workers": 1, "antigen": "1ADQ_A", "oracle": "mock",
                                       "absolut_path": None, "work_dir": str(tmp_path)})
    worker = exp.make_executor(context)
    mock_spec = {**SPEC, "oracle": "mock"}
    assert not worker.validate(mock_spec)
    assert worker._function(mock_spec) == {"mock_energy": exp.mock_energy("1ADQ_A", SPEC["sequence"])}
    worker.close()


def test_codec_sets_real_timeout_without_changing_mock():
    candidate = SimpleNamespace(payload={"sequence": SPEC["sequence"]})
    real = exp.AntibodyCodec("1ADQ_A", "absolut").to_request(candidate)
    mock = exp.AntibodyCodec("1ADQ_A", "mock").to_request(candidate)
    assert real.limits == {"max_runtime_seconds": exp.ABSOLUT_TIMEOUT_SECONDS}
    assert mock.limits == {}
    assert real.spec == SPEC


def test_native_command_uses_explicit_cpu_affinity(tmp_path, monkeypatch):
    from tasks.antibody.core.ldm_light import ldm_acq

    commands = []

    def run(command, **_kwargs):
        commands.append(command)
        Path(f"{SPEC['antigen']}FinalBindings_Process_1_Of_1.txt").write_text(
            "antigen\nID_slide_Variant\tEnergy\n1_00a\t-12.5\n")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(ldm_acq.subprocess, "run", run)
    tool = ldm_acq.AbsolutEvaluator({"antigen": SPEC["antigen"], "process": 1,
                                   "startTask": 2, "cpu_affinity": [2, 4, 6, 8], "path": str(tmp_path)}, "test")
    values, sequences = tool.energy(ldm_acq.np.zeros((1, 11), dtype=ldm_acq.np.int32))
    assert commands[0][:3] == ["taskset", "-c", "2,4,6,8"]
    assert list(values) == [-12.5]
    assert sequences == ["A" * 11]


def test_result_is_separate_from_noisy_stdout_and_recoverable(tmp_path, monkeypatch):
    worker, ref, directory, process, state, popens = executor(tmp_path, monkeypatch)
    assert popens[0][1]["start_new_session"] is True
    assert worker.poll(ref).state == "active"
    (directory / "worker.log").write_text("arbitrary scientific library stdout\n", encoding="utf-8")
    exp._write_json(directory / "result.json", {"state": "succeeded", "metrics": {"absolut_energy": -74.7}})
    process.returncode = 0
    state["alive"] = False
    result = worker.poll(ref)
    assert result.state == "succeeded"
    assert result.result == {"absolut_energy": -74.7}
    assert result.artifacts[0].path == str(directory / "worker.log")
    restarted = exp.AbsolutExecutor(worker.base, worker.work_dir, SPEC["antigen"], lambda _spec: [])
    assert restarted.find("campaign.d1.s0", 1) == ref
    assert restarted.poll(ref) == result


def test_cancel_waits_for_process_group_and_escalates(tmp_path, monkeypatch):
    worker, ref, directory, process, state, _ = executor(tmp_path, monkeypatch)
    clock = {"now": 10.0}
    signals = []
    monkeypatch.setattr(exp.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(exp.os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    worker.cancel(ref)
    assert signals == [(process.pid, signal.SIGTERM)]
    assert worker.poll(ref).state == "active"  # request is not a stop confirmation
    clock["now"] += exp.STOP_GRACE_SECONDS + 1
    assert worker.poll(ref).state == "active"
    assert signals[-1] == (process.pid, signal.SIGKILL)
    process.returncode = -signal.SIGKILL
    state["alive"] = False
    assert worker.poll(ref).state == "stopped"
    assert json.loads((directory / "cancel.json").read_text())["requested_at"] == 10.0


def test_cancel_before_worker_identity_is_recorded(tmp_path, monkeypatch):
    worker, ref, directory, process, _state, _ = executor(tmp_path, monkeypatch)
    (directory / "worker.json").unlink()
    monkeypatch.setattr(exp.os, "killpg", lambda _pid, _sig: None)
    worker.cancel(ref)
    process.returncode = -signal.SIGTERM
    assert worker.poll(ref).state == "stopped"


def test_close_cancels_a_recovered_worker_without_a_popen_handle(tmp_path, monkeypatch):
    original, ref, directory, process, state, _ = executor(tmp_path, monkeypatch)
    recovered = exp.AbsolutExecutor(original.base, original.work_dir, SPEC["antigen"], lambda _spec: [])
    signals = []

    def stop(pid, sig):
        signals.append((pid, sig))
        state["alive"] = False

    monkeypatch.setattr(exp.os, "killpg", stop)
    recovered.close()
    assert signals == [(process.pid, signal.SIGTERM)]
    assert (directory / "cancel.json").is_file()
    assert recovered.poll(ref).state == "stopped"


def test_recovered_intent_waits_for_identity_without_relaunch(tmp_path, monkeypatch):
    original, ref, directory, _process, _state, popens = executor(tmp_path, monkeypatch)
    (directory / "worker.json").unlink()
    recovered = exp.AbsolutExecutor(original.base, original.work_dir, SPEC["antigen"], lambda _spec: [])
    assert recovered.find("campaign.d1.s0", 1) == ref
    assert recovered.poll(ref).state == "unreachable"
    assert len(popens) == 1


def test_failure_class_and_missing_result_are_recorded(tmp_path, monkeypatch):
    worker, ref, directory, process, state, _ = executor(tmp_path, monkeypatch)
    process.returncode = 1
    state["alive"] = False
    assert worker.poll(ref).failure_class == "infrastructure"
    exp._write_json(directory / "result.json", {"state": "failed", "failure_class": "timeout", "message": "lock wait"})
    status = worker.poll(ref)
    assert status.state == "failed"
    assert status.failure_class == "timeout"


def test_dead_managed_lock_is_removed_but_legacy_or_live_locks_are_not(tmp_path, monkeypatch):
    path = tmp_path / ".antbo_llm_acq_1ADQ_A.lock"
    owner = {"boot_id": "boot", "pid": 12345, "start": 77}
    target = "ldm-absolut-v1:boot:12345:77:unique"
    path.symlink_to(target)
    monkeypatch.setattr(exp, "_group_alive", lambda _owner: True)
    assert not exp._remove_stale_lock(path, expected=owner)
    monkeypatch.setattr(exp, "_group_alive", lambda _owner: False)
    assert not exp._remove_stale_lock(path, expected={**owner, "pid": 999})
    assert exp._remove_stale_lock(path, expected=owner)
    path.mkdir()
    assert not exp._remove_stale_lock(path)
    assert path.is_dir()


def test_worker_keeps_lock_after_evaluator_cleanup_until_group_exit(tmp_path, monkeypatch):
    from tasks.antibody.core import engine_adapters
    from tasks.antibody.core.ldm_light import ldm_acq

    directory = tmp_path / "attempt"
    directory.mkdir()
    (directory / "absolut").mkdir()
    exp._write_json(directory / "request.json", {**SPEC, "absolut_path": str(tmp_path), "attempt_id": "a"})
    identity = {"boot_id": "boot", "pid": 12345, "start": 77}
    monkeypatch.setattr(exp, "_identity", lambda _pid: identity)
    monkeypatch.setattr(exp, "_group_alive", lambda _owner: True)
    monkeypatch.setattr(exp.signal, "signal", lambda *_args: None)

    class FakeAbsolut:
        def __init__(self, bbox, _run_id):
            self.bbox = bbox

    class FakeEvaluator:
        def __init__(self, tool):
            self.tool = tool

        def evaluate(self, _candidate):
            lock = exp._lock_path(Path(self.tool.bbox["path"]), SPEC["antigen"])
            self.tool._acquire_lock(str(lock))
            self.tool._release_lock(str(lock))
            assert lock.is_symlink()
            return SimpleNamespace(succeeded=True, metrics={"absolut_energy": -74.7})

    monkeypatch.setattr(ldm_acq, "AbsolutEvaluator", FakeAbsolut)
    monkeypatch.setattr(engine_adapters, "AntibodyEvaluator", FakeEvaluator)
    monkeypatch.setattr(engine_adapters, "AntibodyCandidateDomain", lambda _length: SimpleNamespace(admit=lambda _raw: object()))
    exp._run_absolut(directory)
    assert json.loads((directory / "result.json").read_text())["state"] == "succeeded"
    lock = exp._lock_path(directory / "absolut", SPEC["antigen"])
    assert lock.is_symlink()
    monkeypatch.setattr(exp, "_group_alive", lambda _owner: False)
    assert exp._remove_stale_lock(lock, expected=identity)


def test_manifest_rejects_mutable_missing_or_incorrect_inputs(tmp_path):
    files = prepare_assets(tmp_path)
    exp._assets(tmp_path, SPEC["antigen"])
    data = tmp_path / "struct+sequence-10-11-forbiddenCompact.txt"
    data.chmod(0o644)
    with pytest.raises(ValueError, match="read-only"):
        exp._assets(tmp_path, SPEC["antigen"])
    data.write_bytes(b"corrupt")
    data.chmod(0o444)
    with pytest.raises(ValueError, match="checksum differs"):
        exp._assets(tmp_path, SPEC["antigen"])
    data.unlink()
    with pytest.raises(ValueError, match="missing"):
        exp._assets(tmp_path, SPEC["antigen"])
    data.symlink_to(tmp_path / "struct-10-11-forbiddenStructures.txt")
    with pytest.raises(ValueError, match="symlinks"):
        exp._assets(tmp_path, SPEC["antigen"])
    assert (tmp_path / "struct-10-11-forbiddenStructures.txt").read_bytes() == files[
        "struct-10-11-forbiddenStructures.txt"]


def test_manifest_rejects_incomplete_cache_and_runtime_output(tmp_path):
    prepare_assets(tmp_path)
    manifest_path = tmp_path / "assets.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["assets"] = [item for item in manifest["assets"] if not item["name"].endswith("Structures.txt")]
    exp._write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="complete precomputed cache"):
        exp._assets(tmp_path, SPEC["antigen"])
    manifest["assets"].append({"name": f"{SPEC['antigen']}FinalBindings_Process_1_Of_1.txt",
                               "bytes": 0, "sha256": hashlib.sha256(b"").hexdigest()})
    exp._write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="conflicts with a runtime output"):
        exp._assets(tmp_path, SPEC["antigen"])


def test_failed_launch_cleans_only_attempt_and_keeps_inputs(tmp_path, monkeypatch):
    source = tmp_path / "inputs"
    source.mkdir()
    files = prepare_assets(source)
    worker = exp.AbsolutExecutor(source, tmp_path / "attempts", SPEC["antigen"], lambda _spec: [])
    monkeypatch.setattr(exp.subprocess, "Popen", lambda *_a, **_k: (_ for _ in ()).throw(OSError("spawn failed")))
    attempt = AttemptContext("campaign.d1.s0", 1, "campaign", exp.KIND, exp.VERSION, SPEC, "digest", {})
    with pytest.raises(OSError, match="spawn failed"):
        worker.start(attempt)
    assert not list(worker.work_dir.iterdir())
    assert all((source / name).read_bytes() == contents for name, contents in files.items())


def test_attempts_cannot_be_stored_in_input_library(tmp_path):
    prepare_assets(tmp_path)
    with pytest.raises(ValueError, match="outside the input library"):
        exp.AbsolutExecutor(tmp_path, tmp_path / "attempts", SPEC["antigen"], lambda _spec: [])


def test_isolated_workers_overlap_and_produce_independent_results(tmp_path, monkeypatch):
    from tasks.antibody.core import engine_adapters
    from tasks.antibody.core.ldm_light import ldm_acq

    worker, first_ref, first_dir, _process, _state, _ = executor(tmp_path, monkeypatch)
    second_spec = {**SPEC, "sequence": "TNGKDGSPGVY"}
    attempt = AttemptContext("campaign.d1.s1", 1, "campaign", exp.KIND, exp.VERSION, second_spec, "digest", {})
    second_ref = worker.start(attempt)
    second_dir = worker.work_dir / second_ref
    assert first_ref != second_ref
    for directory in (first_dir, second_dir):
        assert all((directory / "absolut" / name).is_symlink() for name in worker.assets)
        assert all((directory / "absolut" / name).resolve() == worker.base / name for name in worker.assets)
    identity = {"boot_id": "boot", "pid": 12345, "start": 77}
    monkeypatch.setattr(exp, "_identity", lambda _pid: identity)
    monkeypatch.setattr(exp.signal, "signal", lambda *_args: None)
    monkeypatch.setattr(exp.os, "sched_getaffinity", lambda _pid: {2, 4, 6, 8, 10, 12, 14, 16})
    barrier = threading.Barrier(2)
    seen_affinity = []

    class FakeAbsolut:
        def __init__(self, bbox, _run_id):
            self.bbox = bbox

    class FakeEvaluator:
        def __init__(self, tool):
            self.tool = tool

        def evaluate(self, candidate):
            bbox = self.tool.bbox
            seen_affinity.append(bbox["cpu_affinity"])
            path = Path(bbox["path"])
            self.tool._acquire_lock(str(exp._lock_path(path, SPEC["antigen"])))
            barrier.wait(timeout=5)
            (path / f"{SPEC['antigen']}FinalBindings_Process_1_Of_1.txt").write_text(candidate.payload["sequence"])
            return SimpleNamespace(succeeded=True, metrics={"absolut_energy": -10.0 if
                candidate.payload["sequence"] == SPEC["sequence"] else -20.0})

    monkeypatch.setattr(ldm_acq, "AbsolutEvaluator", FakeAbsolut)
    monkeypatch.setattr(engine_adapters, "AntibodyEvaluator", FakeEvaluator)
    monkeypatch.setattr(engine_adapters, "AntibodyCandidateDomain",
                        lambda _length: SimpleNamespace(admit=lambda raw: raw))
    threads = [threading.Thread(target=exp._run_absolut, args=(directory,))
               for directory in (first_dir, second_dir)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
        assert not thread.is_alive()
    assert seen_affinity == [[2, 4, 6, 8, 10, 12, 14, 16]] * 2
    for directory, sequence, energy in ((first_dir, SPEC["sequence"], -10.0),
                                         (second_dir, second_spec["sequence"], -20.0)):
        assert json.loads((directory / "result.json").read_text())["metrics"] == {"absolut_energy": energy}
        assert (directory / "absolut" / f"{SPEC['antigen']}FinalBindings_Process_1_Of_1.txt").read_text() == sequence
    assert not exp._lock_path(worker.base, SPEC["antigen"]).exists()


def test_recovery_releases_only_dead_attempt_lock(tmp_path, monkeypatch):
    worker, ref, directory, process, state, _ = executor(tmp_path, monkeypatch)
    lock = exp._lock_path(directory / "absolut", SPEC["antigen"])
    lock.symlink_to(f"{exp.LOCK_PREFIX}:boot:{process.pid}:77:unique")
    source_before = {name: (worker.base / name).read_bytes() for name in worker.assets}
    recovered = exp.AbsolutExecutor(worker.base, worker.work_dir, SPEC["antigen"], lambda _spec: [])
    assert recovered.poll(ref).state == "active"
    assert lock.is_symlink()
    monkeypatch.setattr(exp.os, "killpg", lambda _pid, _signal: None)
    recovered.cancel(ref)
    assert lock.is_symlink()
    state["alive"] = False
    assert recovered.poll(ref).state == "stopped"
    assert not lock.is_symlink()
    assert all((worker.base / name).read_bytes() == data for name, data in source_before.items())
    assert not exp._lock_path(worker.base, SPEC["antigen"]).exists()
