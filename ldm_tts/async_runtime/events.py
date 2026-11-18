"""Events and hooks (design 5.9): durable per-hook cursors, at-least-once delivery, effect at most once.

Three hook kinds:

* ``WakeHook`` (built-in: outbox, pacing): "look at the state again". Losing one is harmless
  because the next step rescans everything; the cursor is still kept.
* ``EffectHook`` (observers, exporters, notifiers): the side effect runs first, keyed by
  ``event.seq`` for idempotency, then the cursor advances in its own transaction - never the
  other way round. Crash point ``hook.<name>.after_effect`` sits between the two.
* ``ActionHook`` (task action hooks): a pure function of ``(event, view)`` returning action
  requests; the requests, their verdicts and the cursor advance are one transaction
  (``hook_outputs``), so a replay reads the record instead of running the hook again.

Order: an event goes to every built-in hook before any task hook (task hooks never run past
the slowest built-in cursor), then to task hooks in declaration order.
"""
from __future__ import annotations

import json
import logging
import threading
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from .store import Store
from .wire import canonical

log = logging.getLogger(__name__)
TASK_ACTIONS = frozenset({"request_research", "notify_agents", "request_pause"})


@dataclass(frozen=True)
class Event:
    seq: int
    type: str
    subject_id: str | None
    payload: Mapping[str, Any]
    created_at: float


def matches(patterns: Iterable[str], event_type: str) -> bool:
    for pattern in patterns:
        if pattern == "*" or pattern == event_type:
            return True
        if pattern.endswith(".*") and event_type.startswith(pattern[:-1]):
            return True
    return False


@dataclass
class WakeHook:
    name: str
    subscribe: tuple[str, ...]
    callback: Callable[[Event], None]
    builtin: bool = True
    kind = "wake"


@dataclass
class EffectHook:
    """``effect(event, view)`` must be idempotent per ``event.seq``."""

    name: str
    subscribe: tuple[str, ...]
    effect: Callable[[Event, Any], None]
    builtin: bool = False
    kind = "effect"


@dataclass
class ActionHook:
    """``decide(event, view) -> [{"action": ..., ...}]``; must be pure."""

    name: str
    subscribe: tuple[str, ...]
    decide: Callable[[Event, Any], Sequence[Mapping[str, Any]]]
    actions: frozenset[str] = frozenset()
    version: str = "1"
    on_failure: str | None = None  # None: RuntimeOptions.action_hook_failure
    builtin: bool = False
    kind = "action"


@dataclass(frozen=True)
class ActionRequest:
    request_id: str  # "<hook>:<event seq>:<index>" - stable across redelivery
    hook: str
    action: str
    params: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Verdict:
    accepted: bool
    reason: str = ""


class ActionAdjudicator(Protocol):
    """Decides each task action inside the hook's transaction with the same guards as pacing."""

    def adjudicate(self, db, ledger, hook: ActionHook, event: Event, request: ActionRequest) -> Verdict: ...


class CoreAdjudicator:
    """controller-core policy: undeclared actions are refused; ``request_pause`` is always accepted;
    ``request_research`` and ``notify_agents`` need the research lane (controller-pacing) and are
    refused with ``lane_unavailable`` until a later package replaces this adjudicator."""

    def adjudicate(self, db, ledger, hook, event, request):
        if request.action not in TASK_ACTIONS:
            return Verdict(False, "unknown_action")
        if request.action not in hook.actions:
            return Verdict(False, "undeclared_action")
        if request.action == "request_pause":
            ledger.pause(db, f"hook.{hook.name}:{request.params.get('reason', '')}")
            return Verdict(True, "paused")
        return Verdict(False, "lane_unavailable")


def task_hooks(manifest) -> list:
    """Wrap an ``AsyncManifest``'s hook declarations: observers become effect hooks, the rest action hooks.
    A loaded hook is either callable ``(event, view)`` or has ``on_event(event, view)``."""
    hooks = []
    for declaration in manifest.hooks:
        target = manifest.load_task_hook(declaration.name)
        function = getattr(target, "on_event", target)
        if declaration.is_action_hook:
            hooks.append(ActionHook(declaration.name, tuple(declaration.subscribe), function,
                                    frozenset(declaration.actions), declaration.version))
        else:
            hooks.append(EffectHook(declaration.name, tuple(declaration.subscribe), function))
    return hooks


class HookDispatcher:
    def __init__(self, store: Store, ledger, hooks: Sequence, *, view: Callable[[], Any], adjudicator=None,
                 max_attempts: int = 3, action_failure: str = "pause", timeout: float | None = None,
                 backoff: Callable[[int], float] | None = None):
        names = [hook.name for hook in hooks]
        if len(set(names)) != len(names):
            raise ValueError("hook names must be unique")
        self.store, self.ledger, self.view = store, ledger, view
        self.builtin = [hook for hook in hooks if hook.builtin]
        self.task = [hook for hook in hooks if not hook.builtin]
        self.adjudicator = adjudicator or CoreAdjudicator()
        self.max_attempts, self.action_failure, self.timeout = max_attempts, action_failure, timeout
        self.backoff = backoff or (lambda attempts: 0.0)

    @property
    def hooks(self):
        return self.builtin + self.task

    def cursor(self, name: str) -> int:
        return self.store.value("SELECT last_seq FROM hook_cursors WHERE hook=?", (name,), 0)

    def events(self, after: int, upto: int | None = None, limit: int = 500) -> list[Event]:
        sql, args = "SELECT * FROM events WHERE seq > ?", [after]
        if upto is not None:
            sql, args = sql + " AND seq <= ?", args + [upto]
        rows = self.store.rows(sql + " ORDER BY seq LIMIT ?", (*args, limit))
        return [Event(r["seq"], r["type"], r["subject_id"], json.loads(r["payload_json"]), r["created_at"]) for r in rows]

    def dispatch(self, limit: int = 500) -> bool:
        """Deliver committed events; returns True if any cursor moved or any delivery was attempted."""
        progressed = False
        for hook in self.builtin:
            progressed = self._run(hook, None, limit) or progressed
        ceiling = min((self.cursor(h.name) for h in self.builtin), default=None)
        for hook in self.task:
            progressed = self._run(hook, ceiling, limit) or progressed
        return progressed

    def _set_cursor(self, hook, seq: int, db=None) -> None:
        sql = """INSERT INTO hook_cursors VALUES (?,?) ON CONFLICT (hook) DO UPDATE SET last_seq=excluded.last_seq
            WHERE excluded.last_seq > hook_cursors.last_seq"""
        if db is not None:
            db.execute(sql, (hook.name, seq))
            return
        with self.store.tx(f"hook.{hook.name}") as tx:
            tx.execute(sql, (hook.name, seq))

    def _retry_due(self, hook, event) -> bool:
        """A failed delivery is retried only after its backoff (review item 5); ``next_due`` includes the time."""
        due = self.store.value("SELECT next_attempt_at FROM hook_failures WHERE hook=? AND event_seq=? AND gave_up=0",
                               (hook.name, event.seq))
        return due is None or due <= self.ledger.clock.now()

    def _run(self, hook, ceiling, limit) -> bool:
        cursor = self.cursor(hook.name)
        events = self.events(cursor, ceiling, limit)
        if not events:
            return False
        last, attempted = cursor, False
        for event in events:
            if not matches(hook.subscribe, event.type):
                last = event.seq
                continue
            if hook.kind == "wake":
                hook.callback(event)
                last = event.seq
                continue
            if not self._retry_due(hook, event):
                break
            attempted = True
            if hook.kind == "effect":
                if not self._effect(hook, event):
                    break
            elif not self._action(hook, event):
                break
            last = event.seq
        moved = last > cursor and last > self.cursor(hook.name)
        if moved:
            self._set_cursor(hook, last)
        return moved or attempted

    def _failure(self, hook, event, exc) -> bool:
        """Record a failed delivery; returns True when the hook gave up on this event."""
        error = f"{type(exc).__name__}: {exc}"
        log.warning("hook %s failed on event %s: %s", hook.name, event.seq, error)
        with self.store.tx(f"hook.{hook.name}.failure") as db:
            db.execute("""INSERT INTO hook_failures (hook, event_seq, attempts, last_error) VALUES (?,?,1,?)
                ON CONFLICT (hook, event_seq) DO UPDATE SET attempts=attempts+1, last_error=excluded.last_error""",
                       (hook.name, event.seq, error[:500]))
            attempts = db.execute("SELECT attempts FROM hook_failures WHERE hook=? AND event_seq=?",
                                  (hook.name, event.seq)).fetchone()[0]
            if attempts < self.max_attempts:
                db.execute("UPDATE hook_failures SET next_attempt_at=? WHERE hook=? AND event_seq=?",
                           (self.ledger.clock.now() + self.backoff(attempts), hook.name, event.seq))
                return False
            db.execute("UPDATE hook_failures SET gave_up=1 WHERE hook=? AND event_seq=?", (hook.name, event.seq))
            if hook.kind == "action":
                pause = (hook.on_failure or self.action_failure) == "pause"
                self.ledger.anomaly(db, "hook_failed", f"{hook.name}:{event.seq}", {"error": error[:500]}, pause=pause)
                db.execute("INSERT OR IGNORE INTO hook_outputs VALUES (?,?,?,?)",
                           (hook.name, event.seq, canonical({"failed": error[:500]}), self.ledger.clock.now()))
            self._set_cursor(hook, event.seq, db)
        return True

    def _call(self, function, event):
        """Task hooks run on a worker thread when a timeout is configured (design 5.9 item 6); an overrun
        is a failure. The thread cannot be killed: a late return is discarded."""
        view = self.view()
        if self.timeout is None:
            return function(event, view)
        box: dict = {}

        def work():
            try:
                box["value"] = function(event, view)
            except BaseException as exc:  # noqa: BLE001 - re-raised on the control thread
                box["error"] = exc

        worker = threading.Thread(target=work, name="ldm-hook", daemon=True)
        worker.start()
        worker.join(self.timeout)
        if worker.is_alive():
            raise TimeoutError(f"hook exceeded {self.timeout}s")
        if "error" in box:
            raise box["error"]
        return box.get("value")

    def _effect(self, hook, event) -> bool:
        try:
            self._call(hook.effect, event)
        except Exception as exc:  # noqa: BLE001 - observers never break the campaign
            self._failure(hook, event, exc)
            return False
        self.store.check(f"hook.{hook.name}.after_effect")
        self._set_cursor(hook, event.seq)
        return True

    def _action(self, hook, event) -> bool:
        if self.store.value("SELECT 1 FROM hook_outputs WHERE hook=? AND event_seq=?", (hook.name, event.seq)):
            self._set_cursor(hook, event.seq)  # recorded earlier: replay the record, do not run again
            return True
        try:
            raw = list(self._call(hook.decide, event) or ())
            requests = [ActionRequest(f"{hook.name}:{event.seq}:{i}", hook.name, str(item["action"]),
                                      {k: v for k, v in dict(item).items() if k != "action"})
                        for i, item in enumerate(raw)]
            canonical([dict(r.params) for r in requests])
        except Exception as exc:  # noqa: BLE001
            self._failure(hook, event, exc)
            return False
        now = self.ledger.clock.now()
        with self.store.tx(f"hook.{hook.name}") as db:
            decided = []
            for request in requests:
                verdict = self.adjudicator.adjudicate(db, self.ledger, hook, event, request)
                decided.append({"request_id": request.request_id, "action": request.action,
                                "params": dict(request.params), "accepted": verdict.accepted, "reason": verdict.reason})
            db.execute("INSERT INTO hook_outputs VALUES (?,?,?,?)",
                       (hook.name, event.seq, canonical({"requests": decided}), now))
            self._set_cursor(hook, event.seq, db)
        return True


def hook_outputs(store: Store, hook: str | None = None) -> list[dict]:
    sql, args = "SELECT * FROM hook_outputs", ()
    if hook is not None:
        sql, args = sql + " WHERE hook=?", (hook,)
    return [{"hook": r["hook"], "event_seq": r["event_seq"], **json.loads(r["output_json"])}
            for r in store.rows(sql + " ORDER BY event_seq, hook", args)]


__all__ = ["ActionAdjudicator", "ActionHook", "ActionRequest", "CoreAdjudicator", "EffectHook", "Event",
           "HookDispatcher", "TASK_ACTIONS", "Verdict", "WakeHook", "hook_outputs", "matches", "task_hooks"]
