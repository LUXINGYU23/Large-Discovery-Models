"""Executor conformance checks (design 8.1); tasks run these before plugging an executor in.

    problems = check_executor(executor, {"x": 42}, invalid_spec={"x": "bad"})
    assert problems == []

Checked: the declared surface; ``validate`` accepts the sample, rejects the invalid spec and is
deterministic; ``start`` returns a reference promptly; ``poll`` reports valid statuses and never
leaves a terminal status; ``cancel`` after completion changes nothing; ``idempotent`` dispatch
returns the same reference; ``lookup`` finds the attempt and nothing else; ``immediate`` cancel
stops at once and ``cooperative`` cancel ends the attempt.
"""
from __future__ import annotations

import time

from .clock import VirtualClock
from .executors import AttemptContext, AttemptStatus, executor_problems
from .service import digest


def _context(executor, experiment_id, spec, index=1):
    return AttemptContext(experiment_id, index, "conformance", executor.kind, executor.kind_version, dict(spec),
                          digest(spec), {})


def _default_advance(executor):
    clock = getattr(executor, "clock", None)
    if isinstance(clock, VirtualClock) and callable(getattr(executor, "next_event_time", None)):
        return lambda: clock.advance_to(executor.next_event_time() or clock.now() + 1.0)
    return lambda: time.sleep(0.01)


def _drive(executor, ref, advance, timeout, problems, label):
    deadline, terminal = time.monotonic() + timeout, None
    while time.monotonic() < deadline:
        status = executor.poll(ref)
        if not isinstance(status, AttemptStatus):
            problems.append(f"{label}: poll returned {type(status).__name__}, not AttemptStatus")
            return None
        if terminal is not None and status != terminal:
            problems.append(f"{label}: status changed after a terminal status ({terminal.state} -> {status.state})")
            return None
        if status.terminal:
            if terminal is not None:
                return status
            terminal = status
            continue
        advance()
    problems.append(f"{label}: attempt did not reach a terminal status within {timeout}s")
    return None


def check_executor(executor, spec, *, invalid_spec=None, advance=None, timeout=10.0, start_timeout=1.0) -> list[str]:
    problems = executor_problems(executor)
    if problems:
        return problems
    advance = advance or _default_advance(executor)
    caps = executor.capabilities
    first, second = list(executor.validate(spec)), list(executor.validate(spec))
    if first:
        problems.append(f"validate rejects the sample spec: {first}")
    if first != second:
        problems.append("validate is not deterministic")
    if invalid_spec is not None and not list(executor.validate(invalid_spec)):
        problems.append("validate accepts the invalid spec")

    context = _context(executor, "conformance-run", spec)
    began = time.monotonic()
    ref = executor.start(context)
    if time.monotonic() - began > start_timeout:
        problems.append("start blocked; it must return before the experiment ends")
    if not isinstance(ref, str) or not ref:
        return problems + ["start must return a non-empty string reference"]
    final = _drive(executor, ref, advance, timeout, problems, "run")
    if final is not None:
        executor.cancel(ref)
        if executor.poll(ref) != final:
            problems.append("cancel after a terminal status changed the outcome")
    if caps.dispatch == "idempotent" and executor.start(context) != ref:
        problems.append("dispatch='idempotent' but a repeated start returned a different reference")
    if caps.dispatch == "lookup":
        if executor.find(context.experiment_id, 1) != ref:
            problems.append("dispatch='lookup' but find did not return the started attempt")
        if executor.find("conformance-absent", 1) is not None:
            problems.append("find returned a reference for an attempt that was never started")

    if caps.cancel_running != "none":
        ref = executor.start(_context(executor, "conformance-cancel", spec))
        executor.cancel(ref)
        if caps.cancel_running == "immediate" and not executor.poll(ref).terminal:
            problems.append("cancel_running='immediate' but the attempt did not stop at once")
        _drive(executor, ref, advance, timeout, problems, "cancel")
    return problems


def assert_conformant(executor, spec, **options):
    problems = check_executor(executor, spec, **options)
    if problems:
        raise AssertionError("executor is not conformant:\n- " + "\n- ".join(problems))
