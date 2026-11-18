"""Executor protocol (design 3.9), attempt status values and the blocking-function adapter."""
from __future__ import annotations

import concurrent.futures
import threading
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Mapping, Protocol, Sequence, runtime_checkable

CANCEL_RUNNING = ("none", "cooperative", "immediate")
DISPATCH = ("at_most_once", "lookup", "idempotent")
FAILURE_CLASSES = ("infeasible", "execution_error", "timeout", "infrastructure")
ATTEMPT_STATES = ("active", "succeeded", "failed", "stopped", "unreachable")
ATTEMPT_TERMINAL = ("succeeded", "failed", "stopped")


@dataclass(frozen=True)
class ExecutorCapabilities:
    """Declared, never probed. ``in_process`` means attempts die with the ES process,
    so after a restart the ES knows they are no longer running (design 3.9)."""
    cancel_running: str = "none"
    dispatch: str = "at_most_once"
    retry_safe: bool = False
    pool: str = "default"
    units: int = 1
    in_process: bool = False
    cancel_idempotent: bool = False

    def __post_init__(self):
        if self.cancel_running not in CANCEL_RUNNING:
            raise ValueError(f"cancel_running must be one of {CANCEL_RUNNING}")
        if self.dispatch not in DISPATCH:
            raise ValueError(f"dispatch must be one of {DISPATCH}")
        if any(type(flag) is not bool for flag in (self.retry_safe, self.in_process, self.cancel_idempotent)):
            raise ValueError("retry_safe, in_process and cancel_idempotent must be booleans")
        if not isinstance(self.pool, str) or not self.pool:
            raise ValueError("pool must be a non-empty string")
        if type(self.units) is not int or self.units < 1:
            raise ValueError("units must be a positive integer")

    def to_json(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class AttemptContext:
    experiment_id: str
    attempt_index: int
    campaign_id: str
    kind: str
    kind_version: str
    spec: Mapping[str, Any]
    spec_digest: str
    limits: Mapping[str, Any]


@dataclass(frozen=True)
class Artifact:
    """An output file. Give either ``data`` or a ``path`` the ES copies at the terminal transition."""
    name: str
    media_type: str = "application/octet-stream"
    data: bytes | None = None
    path: str | None = None


@dataclass(frozen=True)
class AttemptStatus:
    state: str
    result: Mapping[str, Any] | None = None
    artifacts: tuple = ()
    usage: Mapping[str, Any] = field(default_factory=dict)
    failure_class: str | None = None
    message: str | None = None
    phase: str | None = None

    def __post_init__(self):
        if self.state not in ATTEMPT_STATES:
            raise ValueError(f"attempt state must be one of {ATTEMPT_STATES}")

    @property
    def terminal(self) -> bool:
        return self.state in ATTEMPT_TERMINAL

    @classmethod
    def active(cls, phase=None):
        return cls("active", phase=phase)

    @classmethod
    def succeeded(cls, result, artifacts=(), usage=None):
        return cls("succeeded", result=result, artifacts=tuple(artifacts), usage=dict(usage or {}))

    @classmethod
    def failed(cls, failure_class, message=None, usage=None):
        return cls("failed", failure_class=failure_class, message=message, usage=dict(usage or {}))

    @classmethod
    def stopped(cls, usage=None):
        return cls("stopped", usage=dict(usage or {}))

    @classmethod
    def unreachable(cls, message=None):
        return cls("unreachable", message=message)


class DispatchUncertain(Exception):
    """Raised by ``start`` when it cannot tell whether the attempt was launched (e.g. a timeout)."""


class ExperimentFailure(Exception):
    """Raised by blocking functions to report a classified failure, e.g. ``infeasible``."""

    def __init__(self, failure_class: str, message: str = ""):
        if failure_class not in FAILURE_CLASSES:
            raise ValueError(f"failure class must be one of {FAILURE_CLASSES}")
        super().__init__(message)
        self.failure_class = failure_class
        self.message = message


@runtime_checkable
class ExperimentExecutor(Protocol):
    kind: str
    kind_version: str
    capabilities: ExecutorCapabilities

    def validate(self, spec: Mapping[str, Any]) -> Sequence[Any]:
        """Pure and fast; a non-empty result makes the PUT a 422."""

    def start(self, attempt: AttemptContext) -> str:
        """Launch one attempt and return its executor reference without waiting for the outcome."""

    def poll(self, executor_ref: str) -> AttemptStatus:
        """Current attempt status."""

    def cancel(self, executor_ref: str) -> None:
        """Return after acknowledging the stop request; raise if delivery is uncertain.

        Only a terminal poll confirms the outcome. With cancel_idempotent, repeating the
        request for the same stable reference is safe, including after service restart.
        """

    def find(self, experiment_id: str, attempt_index: int) -> str | None:
        """Optional (required for ``dispatch='lookup'``): recover an attempt by its labels."""

    def close(self) -> None:
        """Optional: release resources (threads, connections). ``ExperimentService.close()`` calls it once
        per loaded executor object, after the scheduler has stopped; exceptions are logged, not raised.
        In-process work must finish before this returns. A restarted ES needs fresh executors."""


def executor_problems(executor) -> list[str]:
    """Static checks of the declared surface; behaviour is checked by ``conformance``."""
    problems = []
    for name in ("kind", "kind_version"):
        value = getattr(executor, name, None)
        if not isinstance(value, str) or not value:
            problems.append(f"{name} must be a non-empty string")
    if not isinstance(getattr(executor, "capabilities", None), ExecutorCapabilities):
        problems.append("capabilities must be an ExecutorCapabilities")
    for name in ("validate", "start", "poll", "cancel"):
        if not callable(getattr(executor, name, None)):
            problems.append(f"{name} must be callable")
    caps = getattr(executor, "capabilities", None)
    if isinstance(caps, ExecutorCapabilities) and caps.dispatch == "lookup" and not callable(getattr(executor, "find", None)):
        problems.append("dispatch='lookup' requires find")
    return problems


def is_executor(value) -> bool:
    return not isinstance(value, type) and all(callable(getattr(value, n, None)) for n in ("start", "poll", "cancel")) \
        and hasattr(value, "kind")


class BlockingExecutor:
    """Adapts a blocking ``function(spec) -> result mapping`` to the executor protocol.

    Calls run on worker threads inside the ES process, so the attempts are declared
    ``in_process``: after an ES restart they are recorded as ``FAILED / infrastructure``
    (or retried when ``retry_safe``). Running calls cannot be interrupted
    (``cancel_running='none'``); a call that has not begun yet is dropped by ``cancel``.
    The function may raise ``ExperimentFailure`` or return an ``AttemptStatus``.
    """

    def __init__(self, kind: str, kind_version: str, function: Callable[[Mapping[str, Any]], Any], *,
                 validate: Callable[[Mapping[str, Any]], Sequence[Any]] | None = None, pool: str = "default",
                 units: int = 1, retry_safe: bool = False, workers: int = 4, schema: Mapping | None = None):
        self.kind, self.kind_version, self.schema = kind, kind_version, schema
        self.capabilities = ExecutorCapabilities(cancel_running="none", dispatch="at_most_once", retry_safe=retry_safe,
                                                 pool=pool, units=units, in_process=True, cancel_idempotent=True)
        self._function, self._validate = function, validate
        self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=workers, thread_name_prefix=f"es-{kind}")
        self._futures: dict[str, concurrent.futures.Future] = {}
        self._lock = threading.Lock()

    def validate(self, spec):
        return list(self._validate(spec)) if self._validate else []

    def start(self, attempt):
        ref = f"{attempt.experiment_id}#{attempt.attempt_index}"
        with self._lock:
            if ref not in self._futures:
                self._futures[ref] = self._pool.submit(self._run, dict(attempt.spec))
        return ref

    def _run(self, spec):
        try:
            out = self._function(spec)
        except ExperimentFailure as exc:
            return AttemptStatus.failed(exc.failure_class, exc.message)
        except Exception as exc:  # noqa: BLE001 - any task error is an execution error of this attempt
            return AttemptStatus.failed("execution_error", f"{type(exc).__name__}: {exc}")
        return out if isinstance(out, AttemptStatus) else AttemptStatus.succeeded(out)

    def poll(self, executor_ref):
        with self._lock:
            future = self._futures.get(executor_ref)
        if future is None:
            return AttemptStatus.unreachable("unknown executor reference")
        if future.cancelled():
            return AttemptStatus.stopped()
        return future.result() if future.done() else AttemptStatus.active()

    def cancel(self, executor_ref):
        with self._lock:
            future = self._futures.get(executor_ref)
        if future is not None:
            future.cancel()

    def find(self, experiment_id, attempt_index):
        ref = f"{experiment_id}#{attempt_index}"
        with self._lock:
            return ref if ref in self._futures else None

    def close(self):
        """Drop queued calls and wait for running calls before the service releases its lock."""
        self._pool.shutdown(wait=True, cancel_futures=True)
