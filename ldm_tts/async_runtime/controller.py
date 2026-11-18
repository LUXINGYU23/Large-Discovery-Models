"""The asynchronous controller (design 5.1): a reconciliation loop over the transaction store.

``step()`` is deterministic: it reads only the persistent state and ``clock.now()`` and does, in
order, change feed -> anti-entropy (when due) -> selection -> cancels -> PUTs -> stop check ->
hook dispatch. Tests drive it with a virtual clock; production calls ``run()``, which sleeps on a
waker that events, a long-polling I/O thread and due times wake up.
"""
from __future__ import annotations

import hashlib
import json
import logging
import threading
from pathlib import Path

from .clock import RealClock
from .codec import ExperimentRequest, plugin_identity
from .config import CampaignConfig, RuntimeOptions, freeze_or_verify
from .es_client import ESClient, ESError
from .events import HookDispatcher, WakeHook
from .ledger import Ledger
from .lock import RunLock
from .reconcile import FEED_CURSOR, Reconciler
from .selection import DedupOnlySelector, FixedProposer, GreedyPacing
from .store import Store, get_cursor
from .wire import TERMINAL_SQL, canonical

log = logging.getLogger(__name__)
WAKE_ON_OUTBOX = ("experiment.planned", "experiment.cancel_requested", "experiment.resubmitted")
WAKE_ON_PACING = ("slot.available", "observation.ingested", "experiment.finished", "campaign.resumed",
                  "generation.*", "reservoir.*")


def decision_seed(campaign_seed: int, decision_id: int) -> int:
    material = canonical([campaign_seed, decision_id]).encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") >> 1  # fits SQLite INTEGER


class Waker:
    def __init__(self):
        self._condition = threading.Condition()
        self._flag = False

    def notify(self, *_):
        with self._condition:
            self._flag = True
            self._condition.notify_all()

    def wait(self, timeout: float | None) -> bool:
        with self._condition:
            if not self._flag:
                self._condition.wait(timeout)
            woke, self._flag = self._flag, False
            return woke


class Controller:
    def __init__(self, path, config: CampaignConfig, *, codec, client: ESClient, clock=None, proposer=None,
                 selector=None, pacing=None, hooks=(), adjudicator=None, objectives=None,
                 options: RuntimeOptions | None = None, faults=None, collect_only: bool = False, builtin_hooks=()):
        self.path, self.config = Path(path), config
        self.clock = clock or RealClock()
        self.options = options or RuntimeOptions()
        self.codec, self.collect_only = codec, collect_only
        self.proposer = proposer or FixedProposer(())
        self.selector = selector or DedupOnlySelector()
        self.pacing = pacing or GreedyPacing()
        self.hooks = list(hooks)
        self.waker = Waker()
        self._run_lock = RunLock(self.path.with_name(self.path.name + ".lock")).acquire()
        try:
            self.store = Store(self.path, faults=faults)
            try:
                ignore = ("plugins.proposer", "plugins.selector", "plugins.pacing", "plugins.selector_config",
                          "plugins.research", "plugins.reservoir", "plugins.history") if collect_only else ()
                freeze_or_verify(self.store, self.frozen_config(), self.clock.now(), ignore=ignore)
            except BaseException:
                self.store.close()
                raise
        except BaseException:
            self._run_lock.release()
            raise
        self.ledger = Ledger(self.store, config, codec, clock=self.clock, objectives=objectives,
                             collect_only=collect_only)
        self.reconciler = Reconciler(self.ledger, client, self.options)
        # ``builtin_hooks`` run first: history publication precedes pacing (design 5.9 "次序").
        builtin = [*builtin_hooks, WakeHook("outbox", WAKE_ON_OUTBOX, self.waker.notify),
                   WakeHook("pacing", WAKE_ON_PACING, self.waker.notify)]
        self.dispatcher = HookDispatcher(self.store, self.ledger, builtin + self.hooks, view=self.view,
                                         adjudicator=adjudicator, max_attempts=self.options.max_hook_attempts,
                                         action_failure=self.options.action_hook_failure,
                                         timeout=self.options.hook_timeout_seconds, backoff=self.options.backoff)
        self._closed = False
        self._startup_pending = True
        self._exhausted = False

    # ---- configuration ---------------------------------------------------------------------------

    def frozen_config(self) -> dict:
        action_hooks = [f"{h.name}@{h.version}" for h in self.hooks if getattr(h, "kind", "") == "action"]
        return {"campaign": self.config.to_json(),
                "plugins": {"codec": plugin_identity(self.codec, "codec"),
                            "proposer": plugin_identity(self.proposer, "proposer"),
                            "selector": plugin_identity(self.selector, "selector"),
                            "pacing": plugin_identity(self.pacing, "pacing"),
                            "action_hooks": action_hooks}}

    @property
    def client(self) -> ESClient:
        return self.reconciler.client

    @client.setter
    def client(self, value: ESClient) -> None:  # the ES address is deployment configuration
        self.reconciler.client = value

    # ---- the loop ----------------------------------------------------------------------------------

    def step(self) -> bool:
        """One deterministic pass; returns True if anything changed (call again until False)."""
        self._exhausted = False
        progressed = False
        if self._startup_pending:
            self._startup()
        progressed |= self.reconciler.pull_changes()
        progressed |= self.reconciler.anti_entropy(force=self._startup_pending)
        self._startup_pending = False
        progressed |= self._select()
        progressed |= self.reconciler.send_cancels()
        progressed |= self.reconciler.send_puts()
        if not progressed:  # never decide exhaustion on state this step's cancels/PUT answers already changed
            progressed |= self._check_stop()
        progressed |= self.dispatcher.dispatch()
        return progressed

    def _startup(self) -> None:
        """Restart (design 5.6): every PLANNED intent is resent at once (idempotent)."""
        now = self.clock.now()
        with self.store.tx("startup") as db:
            db.execute("UPDATE experiments SET next_attempt_at=? WHERE submit_state='PLANNED' AND next_attempt_at > ?",
                       (now, now))

    def _select(self) -> bool:
        made = 0
        for _ in range(self.options.max_intents_per_step):
            if self.ledger.dispatch_blockers():
                break
            view = self.ledger.view()
            decision = self.pacing.decide(view)
            if decision.action == "stop":
                return self.ledger.stop(decision.reason or "pacing") or made > 0
            if decision.action != "dispatch":
                break
            decision_id = self.ledger.next_decision_id()
            seed = decision_seed(self.config.seed, decision_id)
            candidates = list(self.proposer.propose(view))
            selection = self.selector.select(view, candidates, seed)
            if selection is None:
                self._exhausted = True
                break
            candidate = selection.candidate
            if candidate.canonical_key in view.dedup_keys:
                with self.store.tx("selection") as db:
                    self.ledger.anomaly(db, "selector_violation", candidate.canonical_key,
                                        {"selector": plugin_identity(self.selector, "selector")}, pause=True)
                break
            try:
                request = self.codec.encode(candidate)
                if not isinstance(request, ExperimentRequest):
                    raise TypeError(f"codec returned {type(request).__name__}, not ExperimentRequest")
                canonical(candidate.to_dict())
            except Exception as exc:  # noqa: BLE001 - a codec defect pauses dispatch, it does not crash the loop
                with self.store.tx("selection") as db:
                    self.ledger.anomaly(db, "encode_failed", candidate.canonical_key,
                                        {"error": f"{type(exc).__name__}: {exc}"}, pause=True)
                break
            snapshot = {"proposer": plugin_identity(self.proposer, "proposer"),
                        "selector": plugin_identity(self.selector, "selector"),
                        "pacing": plugin_identity(self.pacing, "pacing"),
                        "probability": selection.probability, "record": dict(selection.record),
                        "candidate_count": len(candidates), "budget": view.budget.to_dict(),
                        "outstanding": view.outstanding, "unstarted": view.unstarted,
                        "pending_keys": sorted(view.pending_keys), "head_ingest_seq": view.head_ingest_seq}
            if self.ledger.plan_intent(decision_id, candidate, request, seed, snapshot) is None:
                break
            made += 1
        return made > 0

    def _check_stop(self) -> bool:
        """Minimal stop rule (controller-pacing refines it): nothing outstanding and either no budget
        or no eligible proposal left."""
        if self.collect_only or self.ledger.campaign_row()["status"] != "active" or self.ledger.paused_reasons():
            return False
        if self.ledger.counts()[1] > 0:
            return False
        if self.ledger.budget().free < 1:
            return self.ledger.stop("budget_exhausted")
        if self._exhausted:
            return self.ledger.stop("proposals_exhausted")
        return False

    @property
    def finished(self) -> bool:
        """Nothing left to do: no outstanding experiment and no further decisions will be made."""
        if self.ledger.counts()[1] > 0:
            return False
        return self.collect_only or self.ledger.campaign_row()["status"] != "active"

    def next_due(self) -> float | None:
        """Earliest absolute time at which a step would do time-driven work (retries, anti-entropy)."""
        times = []
        put = self.store.value("""SELECT MIN(next_attempt_at) FROM experiments
            WHERE submit_state='PLANNED' AND cancel_requested_at IS NULL AND written_off_at IS NULL""")
        if put is not None:
            backoff = self.ledger.flag("es_backoff_until")
            times.append(max(put, backoff) if backoff is not None else put)
        cancel = self.store.value(f"""SELECT MIN(COALESCE(cancel_next_attempt_at, 0)) FROM experiments
            WHERE cancel_requested_at IS NOT NULL AND cancel_acked_at IS NULL
            AND (es_state IS NULL OR es_state NOT IN {TERMINAL_SQL})""")
        if cancel is not None:
            times.append(cancel)
        times.append(self.ledger.flag("anti_entropy_due", 0.0))
        hook_retry = self.store.value("""SELECT MIN(f.next_attempt_at) FROM hook_failures f
            LEFT JOIN hook_cursors c ON c.hook = f.hook WHERE f.gave_up=0 AND f.event_seq > COALESCE(c.last_seq, 0)""")
        if hook_retry is not None:
            times.append(hook_retry)
        return min(times) if times else None

    def run(self, stop: threading.Event | None = None, *, until_finished: bool = True) -> None:
        """Production loop: step until idle, then sleep until woken, due, or ``poll_interval``."""
        stop = stop or threading.Event()
        poll_stop = threading.Event()
        poller = None
        if self.options.long_poll_seconds > 0:
            poller = threading.Thread(target=self._long_poll, args=(poll_stop,), name="ldm-es-longpoll", daemon=True)
            poller.start()
        try:
            while not stop.is_set():
                try:
                    if self.step():
                        continue
                except ESError:  # pragma: no cover - reconciler already classifies; keep the loop alive
                    log.exception("controller step failed")
                if until_finished and self.finished:
                    return
                due, now = self.next_due(), self.clock.now()
                timeout = self.options.poll_interval if due is None else min(self.options.poll_interval,
                                                                              max(0.0, due - now))
                self.waker.wait(timeout)
        finally:
            poll_stop.set()
            if poller is not None:
                poller.join(timeout=1.0)

    def _feed_cursor(self) -> str:
        with self.store.lock:
            return get_cursor(self.store.db, FEED_CURSOR)

    def _long_poll(self, stop: threading.Event) -> None:
        """I/O thread: only detects that the feed moved and wakes the control thread (no ingestion here)."""
        while not stop.is_set():
            try:
                cursor = self._feed_cursor()
                page = self.client.changes(self.config.campaign_id, cursor, limit=1,
                                           wait_seconds=self.options.long_poll_seconds)
                if page["items"]:
                    self.waker.notify()
                    for _ in range(max(1, int(self.options.poll_interval / 0.05))):
                        if stop.wait(0.05) or self._feed_cursor() != cursor:
                            break
            except ESError:
                stop.wait(self.options.poll_interval)
            except Exception:  # noqa: BLE001 - e.g. the store closed during shutdown (review item 6)
                return

    # ---- operator actions and reads --------------------------------------------------------------

    # Every operator entry point wakes ``run()`` after its commit (controller-core review item 2).

    def _woken(self, result):
        self.waker.notify()
        return result

    def request_cancel(self, experiment_id: str, reason: str = "operator", *, force: bool = False) -> bool:
        return self._woken(self.ledger.request_cancel(experiment_id, reason, force=force))

    def write_off(self, experiment_id: str, operator: str, reason: str) -> bool:
        return self._woken(self.ledger.write_off(experiment_id, operator, reason))

    def pause(self, reason: str) -> None:
        with self.store.tx("pause") as db:
            self.ledger.pause(db, reason)
        self.waker.notify()

    def resume(self, operator: str) -> bool:
        return self._woken(self.ledger.resume(operator))

    def resolve_anomaly(self, anomaly_id: int, operator: str) -> bool:
        return self._woken(self.ledger.resolve_anomaly(anomaly_id, operator))

    def retry_rejected(self, experiment_id: str) -> bool:
        return self._woken(self.ledger.retry_rejected(experiment_id))

    def close_campaign(self, reason: str = "closed") -> bool:
        return self._woken(self.ledger.close_campaign(reason))

    def view(self):
        return self.ledger.view()

    def budget(self):
        return self.ledger.budget()

    def observations(self, after_seq: int = 0, limit: int | None = None):
        return self.ledger.observations(after_seq, limit)

    def anomalies(self, open_only: bool = True) -> list[dict]:
        sql = "SELECT * FROM anomalies" + (" WHERE resolved_at IS NULL" if open_only else "") + " ORDER BY id"
        return [{**dict(r), "payload": json.loads(r["payload_json"])} for r in self.store.rows(sql)]

    def events(self, after_seq: int = 0, type_prefix: str | None = None) -> list[dict]:
        rows = self.store.rows("SELECT * FROM events WHERE seq > ? ORDER BY seq", (after_seq,))
        return [{"seq": r["seq"], "type": r["type"], "subject_id": r["subject_id"],
                 "payload": json.loads(r["payload_json"]), "created_at": r["created_at"]}
                for r in rows if type_prefix is None or r["type"].startswith(type_prefix)]

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.store.close()
        finally:
            self._run_lock.release()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
