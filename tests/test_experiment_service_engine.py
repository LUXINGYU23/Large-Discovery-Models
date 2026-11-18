"""Scheduling, recovery (design 4.5), simulation, conformance and entry points of the ES."""
import json
import os
import random
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import pytest

from experiment_service import (AttemptStatus, BlockingExecutor, ExecutorCapabilities, ExperimentFailure, ExperimentService,
                                ServiceAlreadyRunning, SimulatedExecutor, VirtualClock)
from experiment_service.conformance import check_executor
from experiment_service.manifest import load_task_executors
from experiment_service.simulation import distribution
from experiment_service.testing import request, running_service

ROOT = Path(__file__).resolve().parents[1]


def submit(service, eid, x, campaign="c1", kind="sim.objective", limits=None):
    status, resource = service.submit(eid, {"campaign_id": campaign, "kind": kind, "kind_version": "1",
                                            "spec": {"x": x}, "limits": limits or {}})
    assert status == 201
    return resource


def open_service(path, executors, clock, **options):
    options.setdefault("pools", {"default": 4})
    return ExperimentService(path / "es.sqlite3", executors, clock=clock, **options)


def running_units(service):
    return sum(r["units"] for r in service.store.rows("SELECT units FROM experiments WHERE state='RUNNING'"))


def test_pool_capacity_and_units_are_respected(tmp_path):
    clock = VirtualClock()
    small = SimulatedExecutor(clock, kind="sim.small", duration=("uniform", 1, 5), seed=1)
    big = SimulatedExecutor(clock, kind="sim.big", units=2, duration=("uniform", 1, 5), seed=2)
    service = open_service(tmp_path, [small, big], clock, pools={"default": 3})
    for i in range(8):
        submit(service, f"e{i}", i, campaign=f"c{i % 2}", kind="sim.big" if i % 3 == 0 else "sim.small")
    peak = 0
    service.run_until_idle()
    while True:
        peak = max(peak, running_units(service))
        assert running_units(service) <= 3
        if not service.advance():
            break
    assert peak == 3
    assert {service.get(f"e{i}")["state"] for i in range(8)} == {"SUCCEEDED"}


def test_round_robin_across_campaigns_and_fifo_within(tmp_path):
    clock = VirtualClock()
    executor = SimulatedExecutor(clock, duration=1.0)
    service = open_service(tmp_path, [executor], clock, pools={"default": 1})
    for name in ("a1", "a2", "a3", "a4"):
        submit(service, name, int(name[1]), campaign="A")
    for name in ("b1", "b2"):
        submit(service, name, int(name[1]), campaign="B")
    service.run_simulation()
    assert [eid for eid, _ in executor.launches] == ["a1", "b1", "a2", "b2", "a3", "a4"]


def test_max_running_per_campaign_and_independent_pools(tmp_path):
    clock = VirtualClock()
    cpu = SimulatedExecutor(clock, kind="sim.cpu", pool="cpu")
    gpu = SimulatedExecutor(clock, kind="sim.gpu", pool="gpu")
    service = open_service(tmp_path, [cpu, gpu], clock, pools={"cpu": 4, "gpu": 1}, max_running_per_campaign=2)
    for i in range(3):
        submit(service, f"a{i}", i, campaign="A", kind="sim.cpu")
    submit(service, "b0", 0, campaign="B", kind="sim.cpu")
    submit(service, "g0", 0, campaign="C", kind="sim.gpu")
    submit(service, "g1", 1, campaign="D", kind="sim.gpu")
    service.run_until_idle()
    running = {r["experiment_id"] for r in service.store.rows("SELECT experiment_id FROM experiments WHERE state='RUNNING'")}
    assert running == {"a0", "a1", "b0", "g0"}
    with pytest.raises(ValueError, match="undeclared pool"):
        open_service(tmp_path / "x", [SimulatedExecutor(clock, pool="tpu")], clock)
    with pytest.raises(ValueError, match="more units"):
        open_service(tmp_path / "y", [SimulatedExecutor(clock, units=5)], clock)


@pytest.mark.parametrize("cancel_running, state, finished", [("cooperative", "FAILED", 6.0), ("none", "SUCCEEDED", 20.0)])
def test_max_runtime_is_enforced_through_the_executor(tmp_path, cancel_running, state, finished):
    clock = VirtualClock()
    service = open_service(tmp_path, [SimulatedExecutor(clock, duration=20.0, cancel_running=cancel_running)], clock)
    submit(service, "t", 1, limits={"max_runtime_seconds": 5})
    service.run_simulation()
    final = service.get("t")
    assert final["state"] == state and service.store.row("SELECT finished_at FROM experiments")["finished_at"] == finished
    if state == "FAILED":
        assert final["failure"]["class"] == "timeout"

def test_second_process_cannot_recover_live_attempts(tmp_path):
    clock = VirtualClock()
    executor = SimulatedExecutor(clock, in_process=True, retry_safe=True)
    db = tmp_path / "es.sqlite3"
    service = ExperimentService(db, [executor])
    alias = tmp_path / "alias.sqlite3"
    alias.symlink_to(db)
    config = tmp_path / "deploy.json"
    config.write_text(json.dumps({"database": str(alias), "port": 0, "tasks": []}))
    cmd = [sys.executable, "-m", "experiment_service", "--config", str(config)]
    try:
        submit(service, "active", 1)
        submit(service, "waiting", 2)
        service.run_until_idle()
        result = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, timeout=10)
        assert result.returncode == 1 and "already running" in result.stderr
        assert not result.stdout
        assert service.get("active")["state"] == "RUNNING"
        assert service.get("waiting")["state"] == "QUEUED"
        assert executor.launches == [("active", 1)]
        with pytest.raises(ServiceAlreadyRunning):
            ExperimentService(alias)
    finally:
        service.close()
    ExperimentService(alias).close()


class CancelTimesOut(SimulatedExecutor):
    def __init__(self, clock, *, cancel_idempotent, stop_on_first_send=False):
        super().__init__(clock, duration=30, cancel_delay=2)
        self.capabilities = replace(self.capabilities, cancel_idempotent=cancel_idempotent)
        self.stop_on_first_send = stop_on_first_send
        self.cancel_calls = []

    def cancel(self, ref):
        self.cancel_calls.append(ref)
        if len(self.cancel_calls) == 1:
            if self.stop_on_first_send:
                super().cancel(ref)
            raise TimeoutError("lost stop acknowledgement")
        super().cancel(ref)


@pytest.mark.parametrize("runtime_limit", [False, True])
def test_cancel_timeout_retries_after_restart_and_waits_for_stop(tmp_path, runtime_limit):
    clock = VirtualClock()
    executor = CancelTimesOut(clock, cancel_idempotent=True)
    options = {"pools": {"default": 1}}
    db = tmp_path / "es.sqlite3"
    path = "/v1/experiments/active"
    with running_service([executor], clock=clock, database=db, **options) as es:
        submit(es.service, "active", 1, limits={"max_runtime_seconds": 1} if runtime_limit else None)
        submit(es.service, "waiting", 2)
        es.service.run_until_idle()
        clock.advance(1)
        if not runtime_limit:
            status, _, resource = request(es.base_url, "POST", path + ":cancel", {"campaign_id": "c1"})
            assert status == 202 and resource["state"] == "RUNNING"
        es.service.run_until_idle()
        assert es.service.get("active")["condition"] == "cancel_unconfirmed"
        assert es.service.get("waiting")["state"] == "QUEUED"

    with running_service([executor], clock=clock, database=db, **options) as es:
        assert request(es.base_url, "GET", path)[2]["condition"] == "cancel_unconfirmed"
        es.service.advance(4.0)
        assert executor.cancel_calls == ["sim:active#1"]
        assert es.service.next_event_time() == 6.0
        es.service.advance()
        assert executor.cancel_calls == ["sim:active#1"] * 2
        pending = request(es.base_url, "GET", path)[2]
        assert (pending["state"], pending["condition"]) == ("RUNNING", "cancel_pending")
        assert es.service.get("waiting")["state"] == "QUEUED"
        es.service.advance()
        final = request(es.base_url, "GET", path)[2]
        assert final["condition"] is None
        if runtime_limit:
            assert final["state"] == "FAILED" and final["failure"]["class"] == "timeout"
        else:
            assert final["state"] == "CANCELED"
        assert executor.launches == [("active", 1), ("waiting", 1)]


@pytest.mark.parametrize("cancel_idempotent", [False, True])
def test_cancel_ack_lost_but_poll_confirms_stop_without_resending(tmp_path, cancel_idempotent):
    clock = VirtualClock()
    executor = CancelTimesOut(clock, cancel_idempotent=cancel_idempotent, stop_on_first_send=True)
    service = open_service(tmp_path, [executor], clock)
    try:
        submit(service, "active", 1)
        service.run_until_idle()
        service.cancel("active")
        service.run_until_idle()
        assert service.get("active")["condition"] == "cancel_unconfirmed"
        service.advance(10)
        assert service.get("active")["state"] == "CANCELED"
        assert executor.cancel_calls == ["sim:active#1"]
    finally:
        service.close()


def test_non_idempotent_cancel_timeout_never_resends_and_keeps_late_result(tmp_path):
    clock = VirtualClock()
    executor = CancelTimesOut(clock, cancel_idempotent=False)
    service = open_service(tmp_path, [executor], clock)
    try:
        submit(service, "active", 1)
        service.run_until_idle()
        service.cancel("active")
        service.run_until_idle()
    finally:
        service.close()
    service = open_service(tmp_path, [executor], clock)
    try:
        for _ in range(3):
            assert service.cancel("active")[0] == 202
            service.advance(6)
            assert service.get("active")["condition"] == "cancel_unconfirmed"
        service.advance()
        assert service.get("active")["state"] == "SUCCEEDED"
        assert executor.cancel_calls == ["sim:active#1"]
    finally:
        service.close()


def test_cancel_during_poll_prevents_retrying_a_failed_attempt(tmp_path):
    clock = VirtualClock()

    class FailsDuringCancel(SimulatedExecutor):
        def poll(self, ref):
            # The scheduler already read the attempt without a cancellation request.
            assert service.cancel("active")[0] == 202
            return AttemptStatus.failed("infrastructure", "executor failed after cancel")

    executor = FailsDuringCancel(clock, retry_safe=True)
    service = open_service(tmp_path, [executor], clock)
    try:
        submit(service, "active", 1)
        service.run_until_idle()
        assert service.get("active")["state"] == "FAILED"
        assert executor.launches == [("active", 1)]
    finally:
        service.close()


@pytest.mark.parametrize("retry_safe", [True, False])
def test_cancel_during_poll_of_a_stopped_attempt_is_honored(tmp_path, retry_safe):
    clock = VirtualClock()

    class StopsDuringCancel(SimulatedExecutor):
        def poll(self, ref):
            # The scheduler has read the attempt, but the stop and the HTTP
            # cancellation both arrive before the poll completes.
            assert service.cancel("active")[0] == 202
            return AttemptStatus.stopped()

    executor = StopsDuringCancel(clock, retry_safe=retry_safe)
    service = open_service(tmp_path, [executor], clock)
    try:
        submit(service, "active", 1)
        service.run_until_idle()
        final = service.get("active")
        assert final["state"] == "CANCELED" and final["failure"] is None
        assert final["attempts"][0]["outcome"] == {"state": "stopped"}
        assert executor.launches == [("active", 1)]
    finally:
        service.close()


def test_unexpected_stop_discovered_after_offline_deadline_is_retried(tmp_path):
    clock = VirtualClock()
    executor = SimulatedExecutor(clock, duration=100, cancel_delay=0, retry_safe=True)
    service = open_service(tmp_path, [executor], clock)
    submit(service, "active", 1, limits={"max_runtime_seconds": 5})
    service.run_until_idle()
    executor.cancel("sim:active#1")  # Out-of-band stop, not an ES cancel request.
    service.close()

    clock.advance(10)
    service = open_service(tmp_path, [executor], clock)
    try:
        service.tick()
        active = service.get("active")
        assert active["state"] == "RUNNING" and executor.launches == [("active", 1), ("active", 2)]
        assert active["attempts"][0]["outcome"] == {
            "state": "stopped", "failure_class": "infrastructure",
            "message": "executor stopped the attempt without a request"}
    finally:
        service.close()


@pytest.mark.parametrize("retry_safe", [True, False])
def test_infrastructure_failures_retry_only_when_declared(tmp_path, retry_safe):
    clock = VirtualClock()
    executor = SimulatedExecutor(clock, retry_safe=retry_safe,
                                 failures=lambda spec, attempt: "infrastructure" if attempt == 1 else None)
    service = open_service(tmp_path, [executor], clock)
    submit(service, "r", 1)
    service.run_simulation()
    final = service.get("r")
    if retry_safe:
        assert final["state"] == "SUCCEEDED" and [a["index"] for a in final["attempts"]] == [1, 2]
        assert final["attempts"][0]["outcome"]["failure_class"] == "infrastructure"
    else:
        assert final["state"] == "FAILED" and final["failure"]["class"] == "infrastructure"
        assert len(final["attempts"]) == 1


def test_failure_classes_from_the_executor(tmp_path):
    clock = VirtualClock()
    executor = SimulatedExecutor(clock, retry_safe=True, failures=lambda spec, attempt: ["infeasible", "execution_error"][spec["x"]])
    service = open_service(tmp_path, [executor], clock)
    submit(service, "a", 0)
    submit(service, "b", 1)
    service.run_simulation()
    assert service.get("a")["failure"]["class"] == "infeasible" and service.get("b")["failure"]["class"] == "execution_error"
    assert len(executor.launches) == 2


def test_unreachable_is_a_condition_not_a_terminal_state(tmp_path):
    clock = VirtualClock()
    executor = SimulatedExecutor(clock, duration=5.0)
    service = open_service(tmp_path, [executor], clock)
    submit(service, "u", 1)
    service.run_until_idle()
    executor.set_unreachable()
    service.advance(50.0)
    unreachable = service.get("u")
    assert unreachable["state"] == "RUNNING" and unreachable["condition"] == "unreachable"
    executor.set_unreachable(unreachable=False)
    service.run_until_idle()
    assert service.get("u")["state"] == "SUCCEEDED" and service.get("u")["condition"] is None


@pytest.mark.parametrize("retry_safe", [False, True])
def test_restart_fails_or_retries_in_process_attempts(tmp_path, retry_safe):
    clock = VirtualClock()
    executor = SimulatedExecutor(clock, in_process=True, retry_safe=retry_safe)
    service = open_service(tmp_path, [executor], clock)
    submit(service, "p", 1)
    service.run_until_idle()
    service.close()
    service = open_service(tmp_path, [executor], clock)
    restarted = service.get("p")
    if retry_safe:
        assert restarted["state"] == "RUNNING" and len(restarted["attempts"]) == 2
        service.run_simulation()
        assert service.get("p")["state"] == "SUCCEEDED" and len(executor.launches) == 2
    else:
        assert restarted["state"] == "FAILED" and restarted["failure"]["class"] == "infrastructure"
        assert restarted["started_at"] is not None


class Crash(BaseException):
    """Stands for the ES process dying inside ``start``."""


class CrashingExecutor(SimulatedExecutor):
    def __init__(self, clock, launch_before_crash, **options):
        super().__init__(clock, **options)
        self.crash, self.launch_before_crash = True, launch_before_crash

    def start(self, attempt):
        if self.crash:
            self.crash = False
            if self.launch_before_crash:
                super().start(attempt)
            raise Crash()
        return super().start(attempt)


@pytest.mark.parametrize("dispatch, launched, expected", [
    ("at_most_once", False, "dispatch_unconfirmed"), ("at_most_once", True, "dispatch_unconfirmed"),
    ("lookup", True, "SUCCEEDED"), ("lookup", False, "dispatch_unconfirmed"),
    ("idempotent", True, "SUCCEEDED"), ("idempotent", False, "SUCCEEDED")])
def test_restart_after_crash_inside_start_follows_dispatch_capability(tmp_path, dispatch, launched, expected):
    clock = VirtualClock()
    executor = CrashingExecutor(clock, launched, dispatch=dispatch)
    service = open_service(tmp_path, [executor], clock)
    submit(service, "x", 1)
    with pytest.raises(Crash):
        service.tick()
    assert service.store.row("SELECT state FROM attempts")["state"] == "dispatching"
    service.close()
    service = open_service(tmp_path, [executor], clock)
    service.run_simulation()
    final = service.get("x")
    assert len(executor.launches) == int(launched or dispatch == "idempotent")  # never a second physical launch
    if expected == "SUCCEEDED":
        assert final["state"] == "SUCCEEDED"
    else:
        assert final["state"] == "RUNNING" and final["condition"] == expected
        abandoned = service.abandon("x", "operator checked the backend")
        assert abandoned["state"] == "FAILED" and abandoned["failure"]["class"] == "infrastructure"


@pytest.mark.parametrize("dispatch", ["at_most_once", "lookup", "idempotent"])
def test_uncertain_dispatch_at_runtime(tmp_path, dispatch):
    clock = VirtualClock()
    executor = SimulatedExecutor(clock, dispatch=dispatch, dispatch_uncertain=1.0, launch_on_uncertain=1.0)
    service = open_service(tmp_path, [executor], clock)
    submit(service, "u", 1)
    service.run_simulation()
    final = service.get("u")
    assert len(executor.launches) == 1
    if dispatch == "at_most_once":
        assert final["condition"] == "dispatch_unconfirmed" and final["state"] == "RUNNING"
        service.adopt("u", "sim:u#1")
        service.run_simulation()
    assert service.get("u")["state"] == "SUCCEEDED"


def test_missing_executor_after_restart_and_definite_start_failure(tmp_path):
    clock = VirtualClock()
    executor = SimulatedExecutor(clock)
    service = open_service(tmp_path, [executor], clock, pools={"default": 1})
    submit(service, "run", 1)
    submit(service, "wait", 2)
    service.run_until_idle()
    service.close()
    service = open_service(tmp_path, [], clock, pools={"default": 1})
    service.run_until_idle()
    assert service.get("run")["condition"] == "executor_unavailable" and service.get("wait")["state"] == "QUEUED"
    service.close()
    service = open_service(tmp_path, [executor], clock, pools={"default": 1})
    service.run_simulation()
    assert service.get("run")["state"] == service.get("wait")["state"] == "SUCCEEDED"

    class Broken(SimulatedExecutor):
        def start(self, attempt):
            raise RuntimeError("backend refused")

    service = open_service(tmp_path / "b", [Broken(clock)], clock)
    submit(service, "b", 1)
    service.run_until_idle()
    assert service.get("b")["state"] == "FAILED" and "backend refused" in service.get("b")["failure"]["message"]


def scenario(tmp_path, seed):
    clock = VirtualClock()
    executor = SimulatedExecutor(clock, seed=seed, duration=("heavy_tail", 2.0, 1.5),
                                 failures={"infeasible": 0.2, "execution_error": 0.1}, cancel_running="cooperative")
    service = open_service(tmp_path, [executor], clock, pools={"default": 3})
    for i in range(12):
        submit(service, f"s{i}", i % 7, campaign=f"c{i % 3}")
    service.advance(1.0)
    service.cancel("s11")
    service.cancel("s1")
    service.run_simulation()
    rows = service.store.rows("SELECT experiment_id, state, revision, started_at, finished_at, result, failure "
                              "FROM experiments ORDER BY experiment_id")
    service.close()
    return rows


def test_simulation_is_deterministic(tmp_path):
    first, again, other = scenario(tmp_path / "a", 7), scenario(tmp_path / "b", 7), scenario(tmp_path / "c", 8)
    assert first == again and first != other
    assert {row["state"] for row in first} >= {"SUCCEEDED", "CANCELED"}
    clock = VirtualClock()
    executor = SimulatedExecutor(clock, seed=7, duration=("exponential", 10.0))
    assert executor.plan({"x": 3}) == executor.plan({"x": 3}) == SimulatedExecutor(VirtualClock(), seed=7,
                                                                                  duration=("exponential", 10.0)).plan({"x": 3})
    service = open_service(tmp_path / "d", [executor], clock)
    submit(service, "same-a", 3, campaign="arm-a")
    submit(service, "same-b", 3, campaign="arm-b")
    service.run_simulation()
    runtimes = {service.get(e)["usage"]["runtime_seconds"] for e in ("same-a", "same-b")}
    assert runtimes == {executor.plan({"x": 3})["duration"]}


def test_duration_distributions():
    rng = random.Random(0)
    assert distribution(4)(rng) == 4.0
    assert all(2 <= distribution(("uniform", 2, 3))(rng) <= 3 for _ in range(200))
    assert all(distribution(["heavy_tail", 1.5, 1.2])(rng) >= 1.5 for _ in range(200))
    mean = sum(distribution(("exponential", 5.0))(rng) for _ in range(4000)) / 4000
    assert 4.5 < mean < 5.5
    assert distribution(("lognormal", 3.0, 0.1))(rng) > 0
    with pytest.raises(ValueError):
        distribution(("gamma", 1))


@pytest.mark.parametrize("options", [{"cancel_running": "cooperative"}, {"cancel_running": "immediate"},
                                     {"cancel_running": "none"}, {"dispatch": "idempotent"}, {"dispatch": "lookup"}])
def test_simulated_executor_is_conformant(options):
    executor = SimulatedExecutor(VirtualClock(), validate=lambda s: [] if "x" in s else ["x missing"], **options)
    assert check_executor(executor, {"x": 1}, invalid_spec={"y": 1}) == []


def test_blocking_executor_conformance_and_problem_detection():
    blocking = BlockingExecutor("blk", "1", lambda spec: {"objective": spec["x"] * 2})
    assert check_executor(blocking, {"x": 1}) == []
    blocking.close()

    class Regressing(SimulatedExecutor):
        def poll(self, ref):
            status = super().poll(ref)
            self.flip = not getattr(self, "flip", False)
            return status if not status.terminal or self.flip else type(status).active()

    class Forgetful(SimulatedExecutor):
        def find(self, experiment_id, attempt_index):
            return None

    regressing = check_executor(Regressing(VirtualClock()), {"x": 1})
    assert any("after a terminal status" in p for p in regressing)
    assert any("find" in p for p in check_executor(Forgetful(VirtualClock(), dispatch="lookup"), {"x": 1}))
    assert any("lookup" in p for p in check_executor(type("NoFind", (), {
        "kind": "k", "kind_version": "1", "capabilities": ExecutorCapabilities(dispatch="lookup"),
        "validate": lambda self, s: [], "start": lambda self, a: "r", "poll": lambda self, r: None,
        "cancel": lambda self, r: None})(), {}))


def test_blocking_executor_over_http_with_background_scheduler():
    def evaluate(spec):
        if spec["x"] < 0:
            raise ExperimentFailure("infeasible", "negative")
        if spec["x"] == 13:
            raise RuntimeError("unlucky")
        time.sleep(0.05)
        return {"objective": spec["x"] * 2}

    executor = BlockingExecutor("blk", "1", evaluate, workers=2)
    with running_service([executor], pools={"default": 2}, autorun=True) as es:
        for x in (4, -1, 13):
            payload = {"campaign_id": "c", "kind": "blk", "kind_version": "1", "spec": {"x": x}}
            assert request(es.base_url, "PUT", f"/v1/experiments/b{x + 1}", payload)[0] == 201
        deadline, states = time.monotonic() + 20, {}
        while time.monotonic() < deadline:
            states = {x: request(es.base_url, "GET", f"/v1/experiments/b{x + 1}")[2] for x in (4, -1, 13)}
            if all(s["state"] in ("SUCCEEDED", "FAILED") for s in states.values()):
                break
            time.sleep(0.02)
    executor.close()
    assert states[4]["state"] == "SUCCEEDED" and states[4]["result"] == {"objective": 8}
    assert states[-1]["failure"]["class"] == "infeasible"
    assert states[13]["failure"] == {"class": "execution_error", "message": "RuntimeError: unlucky"}


class ClosingExecutor(SimulatedExecutor):
    def __init__(self, clock, *, closes, fail=False, **options):
        super().__init__(clock, **options)
        self.closes, self.fail = closes, fail

    def close(self):
        self.closes.append(self.kind)
        if self.fail:
            raise RuntimeError("close boom")


def test_service_close_calls_optional_executor_close_once_and_survives_errors(tmp_path, caplog):
    clock, closes = VirtualClock(), []
    first = ClosingExecutor(clock, closes=closes, kind="k.first", fail=True)
    second = ClosingExecutor(clock, closes=closes, kind="k.second")
    plain = SimulatedExecutor(clock, kind="k.plain")  # no close(): nothing to call
    assert not hasattr(plain, "close")
    service = open_service(tmp_path, [first, second, plain], clock)
    service.executors["k.alias"] = second  # one object serving two kinds is closed once
    with caplog.at_level("ERROR", logger="experiment_service.service"):
        service.close()
        service.close()  # idempotent: no second round of close calls
    assert sorted(closes) == ["k.first", "k.second"]
    assert service.store.closed
    assert any("k.first" in r.getMessage() and r.exc_info for r in caplog.records)


def test_service_close_stops_blocking_executor_threads(tmp_path):
    import threading

    blocking = BlockingExecutor("blk", "1", lambda spec: {"objective": spec["x"]}, workers=2)
    service = ExperimentService(tmp_path / "es.sqlite3", [blocking], pools={"default": 2})
    for i in range(2):
        submit(service, f"b{i}", i, kind="blk")
    service.run_until_idle()
    deadline = time.monotonic() + 10
    while service.get("b0")["state"] != "SUCCEEDED" or service.get("b1")["state"] != "SUCCEEDED":
        assert time.monotonic() < deadline
        time.sleep(0.01)
        service.tick()
    assert any(t.name.startswith("es-blk") for t in threading.enumerate())
    service.close()
    deadline = time.monotonic() + 10
    while any(t.name.startswith("es-blk") for t in threading.enumerate()):
        assert time.monotonic() < deadline, "executor threads survived service.close()"
        time.sleep(0.01)
    with pytest.raises(RuntimeError):
        blocking.start(type("A", (), {"experiment_id": "late", "attempt_index": 1, "spec": {"x": 1}})())
    blocking.close()  # closing twice is harmless


def test_service_package_is_independent_of_ldm_tts():
    code = ("import sys, experiment_service, experiment_service.testing, experiment_service.conformance, "
            "experiment_service.manifest, experiment_service.__main__\n"
            "assert not [m for m in sys.modules if m.split('.')[0] in ('ldm_tts', 'tasks')], sys.modules")
    subprocess.run([sys.executable, "-c", code], check=True, cwd=ROOT, env=dict(os.environ, PYTHONPATH=str(ROOT)))


def write_task(tmp_path, executors):
    task = tmp_path / "task" / "task.json"
    task.parent.mkdir(parents=True, exist_ok=True)
    task.write_text(json.dumps({"schema_version": 1, "task_id": "demo", "async": {"executors": executors}}))
    return task


def test_manifest_loading(tmp_path):
    (tmp_path / "es_demo_hooks.py").write_text(
        "from experiment_service import SimulatedExecutor\n"
        "def make_executor(context):\n"
        "    return SimulatedExecutor(context.clock, kind=context.kind, **context.options)\n"
        "def make_wrong(context):\n"
        "    return SimulatedExecutor(context.clock, kind='other')\n")
    sys.path.insert(0, str(tmp_path))
    try:
        task = write_task(tmp_path, {"demo.a": "es_demo_hooks:make_executor",
                                     "demo.b": "experiment_service.simulation:simulated_executor_factory"})
        clock = VirtualClock()
        executors = load_task_executors(task, clock=clock, options={"demo.b": {"duration": ["uniform", 1, 2]}})
        assert [e.kind for e in executors] == ["demo.a", "demo.b"] and executors[1].clock is clock
        assert 1 <= executors[1].plan({"x": 1})["duration"] <= 2
        with pytest.raises(ValueError, match="manifest says"):
            load_task_executors(write_task(tmp_path / "bad", {"demo.c": "es_demo_hooks:make_wrong"}), clock=clock)
        with pytest.raises(ValueError, match="module:object"):
            load_task_executors(write_task(tmp_path / "bad2", {"demo.c": "no_colon"}), clock=clock)
    finally:
        sys.path.remove(str(tmp_path))


def test_command_line_service(tmp_path):
    (tmp_path / "es_cli_hooks.py").write_text(
        "from experiment_service import SimulatedExecutor\n"
        "def make(context):\n"
        "    return SimulatedExecutor(context.clock, kind='cli.sim', duration=0.05)\n")
    task = write_task(tmp_path, {"cli.sim": "es_cli_hooks:make"})
    config = tmp_path / "deploy.json"
    config.write_text(json.dumps({"database": "state/es.sqlite3", "port": 0, "pools": {"default": 2},
                                  "tasks": [str(task.relative_to(tmp_path))], "python_path": ["."],
                                  "poll_interval_seconds": 0.02}))
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    process = subprocess.Popen([sys.executable, "-m", "experiment_service", "--config", str(config)],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env, cwd=tmp_path)
    try:
        line = process.stdout.readline()
        assert line.startswith("experiment service listening on http://"), process.stderr.read()
        base_url = line.split()[-1]
        payload = {"campaign_id": "c", "kind": "cli.sim", "kind_version": "1", "spec": {"x": 1}}
        assert request(base_url, "PUT", "/v1/experiments/cli1", payload)[0] == 201
        deadline, state = time.monotonic() + 20, None
        while time.monotonic() < deadline and state != "SUCCEEDED":
            state = request(base_url, "GET", "/v1/experiments/cli1")[2]["state"]
            time.sleep(0.02)
        assert state == "SUCCEEDED" and (tmp_path / "state" / "es.sqlite3").exists()
    finally:
        process.terminate()
        process.wait(timeout=20)
