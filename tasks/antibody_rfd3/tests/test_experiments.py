"""Contract checks with fake subprocesses; no Absolut installation is needed."""

import json
import signal
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


def executor(tmp_path, monkeypatch):
    process = FakeProcess()
    popens = []

    def spawn(command, **kwargs):
        popens.append((command, kwargs))
        return process

    monkeypatch.setattr(exp.subprocess, "Popen", spawn)
    monkeypatch.setattr(exp, "_remove_stale_lock", lambda *_args, **_kwargs: False)
    state = {"alive": True}
    monkeypatch.setattr(exp, "_group_alive", lambda _identity: state["alive"])
    worker = exp.AbsolutExecutor(tmp_path, tmp_path / "attempts", SPEC["antigen"], lambda _spec: [])
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
    restarted = exp.AbsolutExecutor(tmp_path, worker.work_dir, SPEC["antigen"], lambda _spec: [])
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
    recovered = exp.AbsolutExecutor(tmp_path, original.work_dir, SPEC["antigen"], lambda _spec: [])
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
    recovered = exp.AbsolutExecutor(tmp_path, original.work_dir, SPEC["antigen"], lambda _spec: [])
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
    exp._write_json(directory / "request.json", {**SPEC, "absolut_path": str(tmp_path), "attempt_id": "a"})
    identity = {"boot_id": "boot", "pid": 12345, "start": 77}
    monkeypatch.setattr(exp, "_identity", lambda _pid: identity)
    monkeypatch.setattr(exp, "_group_alive", lambda _owner: True)
    monkeypatch.setattr(exp.signal, "signal", lambda *_args: None)

    class FakeAbsolut:
        def __init__(self, _bbox, _run_id):
            pass

    class FakeEvaluator:
        def __init__(self, tool):
            self.tool = tool

        def evaluate(self, _candidate):
            lock = exp._lock_path(tmp_path, SPEC["antigen"])
            self.tool._acquire_lock(str(lock))
            self.tool._release_lock(str(lock))
            assert lock.is_symlink()
            return SimpleNamespace(succeeded=True, metrics={"absolut_energy": -74.7})

    monkeypatch.setattr(ldm_acq, "AbsolutEvaluator", FakeAbsolut)
    monkeypatch.setattr(engine_adapters, "AntibodyEvaluator", FakeEvaluator)
    monkeypatch.setattr(engine_adapters, "AntibodyCandidateDomain", lambda _length: SimpleNamespace(admit=lambda _raw: object()))
    exp._run_absolut(directory)
    assert json.loads((directory / "result.json").read_text())["state"] == "succeeded"
    lock = exp._lock_path(tmp_path, SPEC["antigen"])
    assert lock.is_symlink()
    monkeypatch.setattr(exp, "_group_alive", lambda _owner: False)
    assert exp._remove_stale_lock(lock, expected=identity)
