"""Fault injection (design 8.2): named crash points and a programmable transport proxy.

``CrashPlan`` is passed to the store/controller; every named point calls ``check(name)``.
``SimulatedCrash`` derives from ``BaseException`` so no ``except Exception`` in the controller can
swallow it - it behaves like the process dying at that instant. ``FaultyTransport`` wraps the ES
client's transport and can drop requests or responses, hold a request back and deliver it later
(the original arriving after its retry, W5/W7), duplicate it, synthesize 429/5xx bursts and partition.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field

from .es_client import TransportError

# Every transaction the controller commits; each has ``<name>.before_commit`` and ``<name>.after_commit``.
TRANSACTIONS = (
    "freeze", "startup", "intent", "ack", "retry", "reject", "ingest", "anti_entropy", "schedule", "reachability",
    "cancel_request", "cancel_ack", "write_off", "stop", "close", "pause", "resume", "resolve", "retry_rejected",
    "selection",
)
# Transactions added by controller-pacing (research lane, reservoir, plans, drain/abort).
PACING_TRANSACTIONS = ("generation_start", "generation_end", "lane", "proposals", "plan", "drain")
# Transactions added by policy-prior (re-application of a frozen policy outside a generation's readiness).
POLICY_TRANSACTIONS = ("prior_application",)
# Points between an HTTP answer and the transaction that records it (W4/W6-style windows).
SEND_POINTS = ("put.after_send", "cancel.after_send")
# Inside a research job: the panel outcome is saved, the policy turn has not run yet.
POLICY_POINTS = ("policy_turn.after_panel",)
CRASH_POINTS = tuple(f"{name}.{when}" for name in TRANSACTIONS + PACING_TRANSACTIONS + POLICY_TRANSACTIONS
                     for when in ("before_commit", "after_commit")) + SEND_POINTS + POLICY_POINTS


def hook_points(name: str, kind: str) -> tuple[str, ...]:
    """Crash points of one hook: effect hooks crash between effect and cursor (``after_effect``)."""
    points = (f"hook.{name}.before_commit", f"hook.{name}.after_commit")
    return ((f"hook.{name}.after_effect",) if kind == "effect" else ()) + points


class SimulatedCrash(BaseException):
    def __init__(self, point: str):
        super().__init__(f"simulated crash at {point}")
        self.point = point


class CrashPlan:
    """Crash at the ``occurrence``-th hit of ``point`` (``None``: only record hits)."""

    def __init__(self, point: str | None = None, occurrence: int = 1):
        self.point, self.occurrence = point, occurrence
        self.hits: dict[str, int] = {}
        self.fired = False

    def check(self, name: str) -> None:
        self.hits[name] = self.hits.get(name, 0) + 1
        if not self.fired and name == self.point and self.hits[name] == self.occurrence:
            self.fired = True
            raise SimulatedCrash(name)


@dataclass
class Fault:
    """One scripted fault, applied to the next ``times`` matching requests.

    ``kind``: ``drop_request`` (never sent, caller times out), ``drop_response`` (sent, answer lost),
    ``delay`` (held back, caller times out; ``FaultyTransport.release()`` delivers it later),
    ``duplicate`` (sent twice, second answer returned) or ``status`` (synthesized ``status`` reply,
    not sent; e.g. 429 with ``headers={"Retry-After": "3"}`` or 503).
    """

    kind: str
    method: str | None = None
    path_contains: str | None = None
    times: int = 1
    status: int = 503
    headers: dict = field(default_factory=dict)

    def matches(self, method, url):
        return self.times > 0 and (self.method is None or self.method == method) \
            and (self.path_contains is None or self.path_contains in url)


class FaultyTransport:
    def __init__(self, inner):
        self.inner = inner
        self.faults: list[Fault] = []
        self.held: list[tuple] = []
        self.partitioned = False
        self.log: list[tuple[str, str, str]] = []
        self._lock = threading.Lock()

    def add(self, kind, **options) -> Fault:
        fault = Fault(kind, **options)
        with self._lock:
            self.faults.append(fault)
        return fault

    def clear(self):
        with self._lock:
            self.faults.clear()

    def release(self) -> list:
        """Deliver held-back requests now (answers are discarded: their sender gave up long ago)."""
        with self._lock:
            held, self.held = self.held, []
        results = []
        for method, url, body, headers, timeout in held:
            try:
                results.append(self.inner(method, url, body, headers, timeout)[0])
            except TransportError as exc:
                results.append(exc)
        return results

    def __call__(self, method, url, body, headers, timeout):
        with self._lock:
            if self.partitioned:
                self.log.append(("partition", method, url))
                raise TransportError(f"simulated partition: {method} {url}")
            fault = next((f for f in self.faults if f.matches(method, url)), None)
            if fault is not None:
                fault.times -= 1
                self.log.append((fault.kind, method, url))
                if fault.kind == "delay":
                    self.held.append((method, url, body, headers, timeout))
        if fault is None:
            return self.inner(method, url, body, headers, timeout)
        if fault.kind in ("drop_request", "delay"):
            raise TransportError(f"simulated timeout ({fault.kind}): {method} {url}")
        if fault.kind == "status":
            return fault.status, {k.lower(): v for k, v in fault.headers.items()}, \
                b'{"title":"simulated","status":%d}' % fault.status
        reply = self.inner(method, url, body, headers, timeout)
        if fault.kind == "drop_response":
            raise TransportError(f"simulated lost response: {method} {url}")
        if fault.kind == "duplicate":
            return self.inner(method, url, body, headers, timeout)
        return reply
