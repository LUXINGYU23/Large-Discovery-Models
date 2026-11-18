"""ES core: idempotent intake, cancellation, state-based change feed, pool scheduling and recovery.

The service is driven by ``tick()``: poll running attempts, forward cancellations and
time limits, then fill free pool capacity. Deterministic simulations call ``tick`` /
``run_until_idle`` / ``advance`` explicitly on a ``VirtualClock``; a deployed service runs a
background ticker (``start_background``).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
from collections.abc import Mapping
from pathlib import Path

from .clock import RealClock, VirtualClock
from .executors import (FAILURE_CLASSES, Artifact, AttemptContext, AttemptStatus, DispatchUncertain,
                        executor_problems)
from .store import Store

log = logging.getLogger(__name__)

ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._~-]{0,199}")
NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
LIMIT_KEYS = frozenset({"max_runtime_seconds"})
CANCEL_RETRY_SECONDS = 5.0
_RUNNING_SQL = """SELECT e.experiment_id, e.campaign_id, e.kind, e.kind_version, e.spec, e.spec_digest, e.limits,
  e.pool, e.units, e.started_at, e.cancel_requested_at, e.condition, a.idx, a.state AS attempt_state,
  a.executor_ref, a.started_at AS attempt_started_at, a.cancel_attempted_at, a.cancel_acknowledged_at, a.timeout_at
  FROM experiments e JOIN attempts a ON a.experiment_id = e.experiment_id
  WHERE e.state = 'RUNNING' AND a.idx = (SELECT MAX(idx) FROM attempts WHERE experiment_id = e.experiment_id)
  ORDER BY e.seq"""


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def digest(value) -> str:
    return "sha256:" + hashlib.sha256(canonical(value).encode()).hexdigest()


PROBLEM_BLANK = "about:blank"
PROBLEM_EXPERIMENT_NOT_FOUND = "urn:ldm-es:problem:experiment-not-found"
PROBLEM_KIND_NOT_FOUND = "urn:ldm-es:problem:kind-not-found"
PROBLEM_ARTIFACT_NOT_FOUND = "urn:ldm-es:problem:artifact-not-found"
PROBLEM_ROUTE_NOT_FOUND = "urn:ldm-es:problem:route-not-found"


class Problem(Exception):
    """An HTTP problem (RFC 9457 shape); raised by service methods, rendered by the server.
    ``problem_type`` is the stable machine-readable identifier of the problem (``type``)."""

    def __init__(self, status, title, detail=None, headers=None, *, problem_type=PROBLEM_BLANK, **extra):
        super().__init__(f"{status} {title}")
        self.status, self.title, self.detail, self.problem_type = status, title, detail, problem_type
        self.headers, self.extra = dict(headers or {}), extra

    def to_json(self):
        body = {"type": self.problem_type, "title": self.title, "status": self.status}
        if self.detail:
            body["detail"] = self.detail
        return {**body, **self.extra}


class ExperimentService:
    def __init__(self, database, executors=(), *, pools=None, clock=None, artifact_dir=None,
                 max_queued_per_campaign: int | None = 1000, max_running_per_campaign: int | None = None,
                 max_infra_retries: int = 1, retry_after_seconds: int = 5, max_wait_seconds: float = 60.0):
        self.clock = clock or RealClock()
        self.pools = dict(pools or {"default": 1})
        for name, capacity in self.pools.items():
            if not isinstance(name, str) or not name or type(capacity) is not int or capacity < 1:
                raise ValueError("pools map non-empty names to positive integer capacities")
        self.executors = {}
        for executor in executors:
            problems = executor_problems(executor)
            if problems:
                raise ValueError(f"executor {getattr(executor, 'kind', executor)!r}: " + "; ".join(problems))
            caps = executor.capabilities
            if executor.kind in self.executors:
                raise ValueError(f"duplicate executor for kind {executor.kind!r}")
            if caps.pool not in self.pools:
                raise ValueError(f"executor {executor.kind!r} uses undeclared pool {caps.pool!r}")
            if caps.units > self.pools[caps.pool]:
                raise ValueError(f"executor {executor.kind!r} needs more units than pool {caps.pool!r} has")
            self.executors[executor.kind] = executor
        self.max_queued, self.max_running = max_queued_per_campaign, max_running_per_campaign
        self.max_infra_retries, self.retry_after, self.max_wait = max_infra_retries, retry_after_seconds, max_wait_seconds
        self.store = Store(database)
        self.artifact_dir = Path(artifact_dir) if artifact_dir else self.store.path.with_name(self.store.path.name + ".artifacts")
        self._tick_lock = threading.RLock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread = None
        self._closed = False
        self._rr: dict[str, str] = {}
        try:
            self._recover()
        except BaseException:
            self.close()
            raise

    # ---- HTTP-facing operations -------------------------------------------------------------

    def submit(self, experiment_id, body):
        """``PUT /v1/experiments/{id}``: returns ``(201|200, resource)`` or raises ``Problem``."""
        self._check_id(experiment_id)
        if not isinstance(body, dict):
            raise Problem(400, "Request body must be a JSON object")
        for name in ("campaign_id", "kind", "kind_version"):
            if not isinstance(body.get(name), str) or not body[name]:
                raise Problem(400, f"{name} must be a non-empty string")
        campaign, kind, version = body["campaign_id"], body["kind"], body["kind_version"]
        spec, limits, client = body.get("spec"), body.get("limits") or {}, body.get("client")
        if not isinstance(spec, dict) or not isinstance(limits, dict) or not isinstance(client, (dict, type(None))):
            raise Problem(400, "spec and limits must be JSON objects; client an object or null")
        try:
            spec_digest, limits_text, client_text = digest(spec), canonical(limits), canonical(client)
        except (TypeError, ValueError) as exc:
            raise Problem(400, "Request is not representable as canonical JSON", str(exc)) from None
        if body.get("spec_digest") not in (None, spec_digest):
            raise Problem(422, "spec_digest does not match spec", f"computed {spec_digest}")
        with self.store.tx() as db:
            row = db.execute("SELECT * FROM experiments WHERE experiment_id=?", (experiment_id,)).fetchone()
            if row is not None:
                if row["tombstone"]:
                    return 200, self.store.resource(row)
                same = (row["campaign_id"], row["kind"], row["kind_version"], row["spec_digest"], row["limits"]) \
                    == (campaign, kind, version, spec_digest, limits_text)
                if not same:
                    raise Problem(409, "experiment_id already exists with a different request",
                                  f"existing spec_digest {row['spec_digest']}")
                return 200, self.store.resource(row)
            executor = self.executors.get(kind)
            if executor is None:
                raise Problem(422, f"Unknown experiment kind {kind!r}")
            if executor.kind_version != version:
                raise Problem(422, f"kind {kind!r} is served at version {executor.kind_version!r}")
            problems = self._limit_problems(limits) + self._spec_problems(executor, spec)
            if problems:
                raise Problem(422, "Experiment spec is invalid", problems=problems)
            queued = db.execute("SELECT count(*) FROM experiments WHERE campaign_id=? AND state='QUEUED'",
                                (campaign,)).fetchone()[0]
            if self.max_queued is not None and queued >= self.max_queued:
                raise Problem(429, "Campaign backlog is full", headers={"Retry-After": str(self.retry_after)})
            revision, now = self.store.bump(), self.clock.now()
            db.execute("""INSERT INTO experiments (experiment_id, seq, campaign_id, kind, kind_version, spec, spec_digest,
                limits, client, pool, units, state, revision, accepted_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,'QUEUED',?,?)""",
                       (experiment_id, revision, campaign, kind, version, canonical(spec), spec_digest, limits_text,
                        client_text, executor.capabilities.pool, executor.capabilities.units, revision, now))
            created = self.store.resource(db.execute("SELECT * FROM experiments WHERE experiment_id=?",
                                                     (experiment_id,)).fetchone())
        self._wake.set()
        return 201, created

    def get(self, experiment_id):
        self._check_id(experiment_id)
        with self.store.lock:
            row = self.store.db.execute("SELECT * FROM experiments WHERE experiment_id=?", (experiment_id,)).fetchone()
            if row is None:
                raise Problem(404, "Experiment not found", problem_type=PROBLEM_EXPERIMENT_NOT_FOUND)
            return self.store.resource(row)

    def cancel(self, experiment_id, body=None):
        """``POST …:cancel``: 200 (tombstone, queued, terminal) or 202 (running), with the resource."""
        self._check_id(experiment_id)
        campaign = (body or {}).get("campaign_id") if isinstance(body, dict) else None
        if campaign is not None and (not isinstance(campaign, str) or not campaign):
            raise Problem(400, "campaign_id must be a non-empty string")
        with self.store.tx() as db:
            row = db.execute("SELECT * FROM experiments WHERE experiment_id=?", (experiment_id,)).fetchone()
            now, status = self.clock.now(), 200
            if row is None:
                revision = self.store.bump()
                db.execute("""INSERT INTO experiments (experiment_id, seq, tombstone, campaign_id, state, revision,
                    finished_at, cancel_requested_at) VALUES (?,?,1,?,'CANCELED',?,?,?)""",
                           (experiment_id, revision, campaign, revision, now, now))
            elif row["state"] == "QUEUED":
                db.execute("""UPDATE experiments SET state='CANCELED', cancel_requested_at=?, finished_at=?, revision=?
                    WHERE experiment_id=?""", (now, now, self.store.bump(), experiment_id))
            elif row["state"] == "RUNNING":
                status = 202
                if row["cancel_requested_at"] is None:
                    db.execute("UPDATE experiments SET cancel_requested_at=?, revision=? WHERE experiment_id=?",
                               (now, self.store.bump(), experiment_id))
            result = self.store.resource(db.execute("SELECT * FROM experiments WHERE experiment_id=?",
                                                    (experiment_id,)).fetchone())
        self._wake.set()
        return status, result

    def changes(self, campaign_id, changed_after="", limit=100, wait_seconds=0.0):
        """State-based change feed: current state of experiments with ``revision > changed_after``."""
        if not isinstance(campaign_id, str) or not campaign_id:
            raise Problem(400, "campaign_id is required")
        if changed_after in (None, ""):
            after = 0
        elif isinstance(changed_after, str) and changed_after.isdigit():
            after = int(changed_after)
        else:
            raise Problem(400, "changed_after must be a cursor returned by this service")
        limit = max(1, min(int(limit), 1000))
        deadline = time.monotonic() + max(0.0, min(float(wait_seconds), self.max_wait))
        while True:
            if self.store.closed:
                raise Problem(503, "Service is shutting down")
            seen = self.store.head
            with self.store.lock:
                rows = self.store.db.execute("""SELECT * FROM experiments WHERE campaign_id=? AND revision>?
                    ORDER BY revision LIMIT ?""", (campaign_id, after, limit + 1)).fetchall()
                items = [self.store.resource(row) for row in rows[:limit]]
            remaining = deadline - time.monotonic()
            if items or remaining <= 0:
                break
            self.store.wait_change(seen, remaining)
        cursor = items[-1]["revision"] if items else f"{after:016d}"
        return {"items": items, "cursor": cursor, "more": len(rows) > limit}

    def kind_info(self, kind):
        executor = self.executors.get(kind)
        if executor is None:
            raise Problem(404, "Unknown experiment kind", problem_type=PROBLEM_KIND_NOT_FOUND)
        caps = executor.capabilities
        return {"kind": kind, "kind_version": executor.kind_version, "capabilities": caps.to_json(),
                "pool": {"name": caps.pool, "capacity": self.pools[caps.pool]},
                "schema": getattr(executor, "schema", None)}

    def artifact(self, experiment_id, name):
        """Return ``(media_type, path, sha256)`` of a stored artifact."""
        self._check_id(experiment_id)
        row = self.store.row("SELECT * FROM artifacts WHERE experiment_id=? AND name=?", (experiment_id, name))
        if row is None:
            raise Problem(404, "Artifact not found", problem_type=PROBLEM_ARTIFACT_NOT_FOUND)
        return row["media_type"], self.artifact_dir / row["path"], row["sha256"]

    # ---- scheduling loop ---------------------------------------------------------------------

    def tick(self) -> int:
        """One scheduling pass; returns the number of state changes it made."""
        with self._tick_lock:
            changed = sum(self._step(r) for r in self.store.rows(_RUNNING_SQL))
            for pool in sorted(self.pools):
                while (candidate := self._next_queued(pool)) is not None:
                    self._dispatch(candidate)
                    changed += 1
            return changed

    def run_until_idle(self, max_ticks: int = 10000) -> int:
        total = 0
        for _ in range(max_ticks):
            changed = self.tick()
            total += changed
            if not changed:
                return total
        raise RuntimeError("experiment service did not become idle")

    def next_event_time(self):
        """Earliest future time at which an executor or a time limit will change something."""
        now, times = self.clock.now(), []
        for executor in self.executors.values():
            probe = getattr(executor, "next_event_time", None)
            if callable(probe) and (t := probe()) is not None:
                times.append(t)
        for r in self.store.rows(_RUNNING_SQL):
            if r["attempt_state"] != "active":
                continue
            limit = json.loads(r["limits"]).get("max_runtime_seconds")
            if limit is not None and r["timeout_at"] is None:
                times.append(r["attempt_started_at"] + limit)
            executor = self._executor_for(r)
            if executor is not None and self._stop_requested(r):
                due = self._cancel_due_at(executor, r, now)
                if due is not None:
                    times.append(due)
        future = [t for t in times if t > now]
        return min(future) if future else None

    def advance(self, seconds=None) -> bool:
        """Virtual clock only: settle, move to the next event (or by ``seconds``), settle again."""
        if not isinstance(self.clock, VirtualClock):
            raise TypeError("advance() needs a VirtualClock")
        self.run_until_idle()
        if seconds is not None:
            self.clock.advance(seconds)
        else:
            when = self.next_event_time()
            if when is None:
                return False
            self.clock.advance_to(when)
        self.run_until_idle()
        return True

    def run_simulation(self, max_events: int = 1_000_000) -> int:
        """Advance event by event until nothing further can happen; returns the number of steps."""
        for steps in range(max_events):
            if not self.advance():
                return steps
        raise RuntimeError("simulation did not settle")

    def start_background(self, interval: float = 1.0):
        if self._thread is not None:
            return
        self._stop.clear()

        def loop():
            while not self._stop.is_set():
                try:
                    self.tick()
                except Exception:  # noqa: BLE001 - keep serving; the error is logged
                    log.exception("experiment service tick failed")
                self._wake.wait(interval)
                self._wake.clear()

        self._thread = threading.Thread(target=loop, name="es-scheduler", daemon=True)
        self._thread.start()

    def close(self):
        """Stop the scheduler, call the optional ``close()`` of every loaded executor (errors are
        logged, never raised), then close the store. Idempotent."""
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
            self._thread = None
        with self._tick_lock:
            if self._closed:
                return
            self._closed = True
            self._close_executors()
            self.store.close()

    def _close_executors(self):
        seen = set()
        for kind, executor in self.executors.items():
            if id(executor) in seen:  # one object may serve several kinds
                continue
            seen.add(id(executor))
            closer = getattr(executor, "close", None)
            if callable(closer):
                try:
                    closer()
                except Exception:  # noqa: BLE001 - closing must not stop the remaining executors or the store
                    log.exception("closing the executor for kind %r failed", kind)

    # ---- operator actions (not part of the LDM contract) -------------------------------------

    def abandon(self, experiment_id, reason):
        """Record failure after an operator verifies the attempt has ended or never started."""
        with self._tick_lock:
            r = self._running_row(experiment_id)
            self._finish(r, "FAILED", failure={"class": "infrastructure", "message": f"abandoned by operator: {reason}"})
            return self.get(experiment_id)

    def adopt(self, experiment_id, executor_ref):
        """Attach an operator-located attempt to a RUNNING experiment whose dispatch is unconfirmed."""
        with self._tick_lock:
            r = self._running_row(experiment_id)
            if r["attempt_state"] != "dispatching":
                raise Problem(409, "the current attempt is not awaiting dispatch confirmation")
            self._activate(r, executor_ref)
            return self.get(experiment_id)

    # ---- internals ---------------------------------------------------------------------------

    def _check_id(self, experiment_id):
        if not isinstance(experiment_id, str) or not ID_PATTERN.fullmatch(experiment_id):
            raise Problem(400, "experiment_id must match [A-Za-z0-9][A-Za-z0-9._~-]{0,199}")

    @staticmethod
    def _limit_problems(limits):
        problems = [f"unknown limit {key!r}" for key in sorted(set(limits) - LIMIT_KEYS)]
        value = limits.get("max_runtime_seconds")
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0):
            problems.append("max_runtime_seconds must be a positive number")
        return problems

    @staticmethod
    def _spec_problems(executor, spec):
        try:
            problems = list(executor.validate(spec))
        except Exception as exc:  # noqa: BLE001 - a crashing validator rejects the spec
            return [f"validator raised {type(exc).__name__}: {exc}"]
        return [p if isinstance(p, (str, dict)) else str(p) for p in problems]

    def _executor_for(self, r):
        executor = self.executors.get(r["kind"])
        return executor if executor is not None and executor.kind_version == r["kind_version"] else None

    def _running_row(self, experiment_id):
        r = self.store.row(_RUNNING_SQL.replace("ORDER BY e.seq", "AND e.experiment_id = ?"), (experiment_id,))
        if r is None:
            raise Problem(409, "experiment is not RUNNING")
        return r

    def _context(self, r):
        return AttemptContext(r["experiment_id"], r["idx"], r["campaign_id"], r["kind"], r["kind_version"],
                              json.loads(r["spec"]), r["spec_digest"], json.loads(r["limits"]))

    def _recover(self):
        """Startup: the queue needs nothing (it is the table); running attempts follow 4.5."""
        for r in self.store.rows(_RUNNING_SQL):
            executor = self._executor_for(r)
            if executor is None:
                continue
            if executor.capabilities.in_process:
                self._infra(executor, r, "attempt lost: the service process restarted")
            elif r["attempt_state"] == "dispatching":
                self._uncertain(executor, r)

    def _step(self, r):
        executor = self._executor_for(r)
        if executor is None:
            return self._condition(r, "executor_unavailable")
        if r["attempt_state"] == "dispatching":
            return self._start(executor, r) if executor.capabilities.dispatch == "idempotent" else 0
        ref, now = r["executor_ref"], self.clock.now()
        limit = json.loads(r["limits"]).get("max_runtime_seconds")
        if limit is not None and r["timeout_at"] is None and now >= r["attempt_started_at"] + limit:
            deadline = r["attempt_started_at"] + limit
            self._mark(r, "timeout_at", deadline)
            r["timeout_at"] = deadline
        try:
            status = executor.poll(ref)
            if not isinstance(status, AttemptStatus):
                raise TypeError(f"poll returned {type(status).__name__}")
        except Exception as exc:  # noqa: BLE001 - the ES does not know the attempt's state
            status = AttemptStatus.unreachable(f"poll failed: {type(exc).__name__}: {exc}")
        if not status.terminal:
            if self._stop_requested(r):
                return self._advance_cancel(executor, r, status, now)
            return self._condition(r, "unreachable" if status.state == "unreachable" else None)
        if status.state == "succeeded":
            return self._finish(r, "SUCCEEDED", status=status)
        if status.state == "failed":
            failure_class = status.failure_class if status.failure_class in FAILURE_CLASSES else "execution_error"
            if failure_class == "infrastructure":
                return self._infra(executor, r, status.message, status)
            return self._finish(r, "FAILED", failure={"class": failure_class, "message": status.message}, status=status)
        timed_out, canceled = r["timeout_at"], r["cancel_requested_at"]
        if timed_out is not None and (canceled is None or timed_out <= canceled):
            return self._finish(r, "FAILED", failure={"class": "timeout", "message": "max_runtime_seconds exceeded"},
                                status=status)
        if canceled is not None:
            return self._finish(r, "CANCELED", status=status)
        return self._infra(executor, r, "executor stopped the attempt without a request", status)

    @staticmethod
    def _stop_requested(r):
        return r["cancel_requested_at"] is not None or r["timeout_at"] is not None

    @staticmethod
    def _cancel_due_at(executor, r, now):
        if r["cancel_acknowledged_at"] is not None:
            return None
        if r["cancel_attempted_at"] is None:
            return now
        if executor.capabilities.cancel_idempotent:
            return r["cancel_attempted_at"] + CANCEL_RETRY_SECONDS
        return None

    def _advance_cancel(self, executor, r, status, now):
        due = self._cancel_due_at(executor, r, now)
        if due is not None and now >= due:
            # Persist the intent first. A crash between the request and its acknowledgement
            # leaves delivery uncertain, so only a reference-idempotent adapter may resend.
            changed = self._mark_cancel_attempt(r, now)
            try:
                executor.cancel(r["executor_ref"])
            except Exception:  # noqa: BLE001 - delivery is uncertain
                log.exception("executor cancel unconfirmed for %s", r["executor_ref"])
                return changed
            self._mark(r, "cancel_acknowledged_at", self.clock.now())
        elif r["cancel_acknowledged_at"] is None:
            return self._condition(r, "cancel_unconfirmed")
        condition = "unreachable" if status.state == "unreachable" else "cancel_pending"
        return self._condition(r, condition)

    def _mark_cancel_attempt(self, r, now):
        with self.store.tx() as db:
            db.execute("UPDATE attempts SET cancel_attempted_at=? WHERE experiment_id=? AND idx=?",
                       (now, r["experiment_id"], r["idx"]))
            if r["condition"] == "cancel_unconfirmed":
                return 0
            db.execute("UPDATE experiments SET condition='cancel_unconfirmed', revision=? WHERE experiment_id=?",
                       (self.store.bump(), r["experiment_id"]))
        r["condition"] = "cancel_unconfirmed"
        return 1

    def _next_queued(self, pool):
        capacity = self.pools[pool]
        with self.store.lock:
            used = self.store.db.execute("SELECT COALESCE(SUM(units), 0) FROM experiments WHERE state='RUNNING' AND pool=?",
                                         (pool,)).fetchone()[0]
            if used >= capacity:
                return None
            queued = [dict(r) for r in self.store.db.execute(
                "SELECT * FROM experiments WHERE state='QUEUED' AND pool=? ORDER BY campaign_id, seq", (pool,))]
            running = dict(self.store.db.execute(
                "SELECT campaign_id, count(*) FROM experiments WHERE state='RUNNING' GROUP BY campaign_id").fetchall())
        heads = {}
        for row in queued:  # FIFO within a campaign; rows of kinds no longer served stay queued and are skipped
            if row["campaign_id"] not in heads and self._executor_for(row) is not None:
                heads[row["campaign_id"]] = row
        if self.max_running is not None:
            heads = {c: row for c, row in heads.items() if running.get(c, 0) < self.max_running}
        if not heads:
            return None
        order = sorted(heads)
        last = self._rr.get(pool)
        chosen = heads[next((c for c in order if last is not None and c > last), order[0])]
        return chosen if chosen["units"] <= capacity - used else None  # no overtaking: the head waits for room

    def _dispatch(self, row):
        """Persist the attempt before calling ``start`` (design 4.5)."""
        now = self.clock.now()
        with self.store.tx() as db:
            if db.execute("SELECT state FROM experiments WHERE experiment_id=?",
                          (row["experiment_id"],)).fetchone()[0] != "QUEUED":
                return
            index = db.execute("SELECT COALESCE(MAX(idx), 0) + 1 FROM attempts WHERE experiment_id=?",
                               (row["experiment_id"],)).fetchone()[0]
            db.execute("INSERT INTO attempts (experiment_id, idx, state, started_at) VALUES (?,?,'dispatching',?)",
                       (row["experiment_id"], index, now))
            db.execute("""UPDATE experiments SET state='RUNNING', started_at=COALESCE(started_at, ?), revision=?
                WHERE experiment_id=?""", (now, self.store.bump(), row["experiment_id"]))
        self._rr[row["pool"]] = row["campaign_id"]
        r = self._running_row(row["experiment_id"])
        self._start(self._executor_for(r), r)

    def _start(self, executor, r):
        try:
            ref = executor.start(self._context(r))
        except DispatchUncertain:
            return self._uncertain(executor, r)
        except Exception as exc:  # noqa: BLE001 - a definite start failure is an infrastructure failure
            return self._infra(executor, r, f"start failed: {type(exc).__name__}: {exc}")
        if not isinstance(ref, str) or not ref:
            return self._infra(executor, r, "executor returned an invalid reference")
        self._activate(r, ref)
        return 1

    def _uncertain(self, executor, r):
        """Dispatch outcome unknown: lookup tries ``find``; idempotent re-starts on the next tick;
        otherwise wait for an operator. Never re-dispatch blindly."""
        if executor.capabilities.dispatch == "lookup":
            try:
                ref = executor.find(r["experiment_id"], r["idx"])
            except Exception:  # noqa: BLE001
                ref = None
            if isinstance(ref, str) and ref:
                self._activate(r, ref)
                return 1
        return self._condition(r, "dispatch_unconfirmed")

    def _activate(self, r, ref):
        with self.store.tx() as db:
            db.execute("UPDATE attempts SET state='active', executor_ref=? WHERE experiment_id=? AND idx=?",
                       (ref, r["experiment_id"], r["idx"]))
            db.execute("UPDATE experiments SET condition=NULL, revision=? WHERE experiment_id=? AND state='RUNNING'",
                       (self.store.bump(), r["experiment_id"]))

    def _condition(self, r, condition):
        if r["condition"] == condition:
            return 0
        with self.store.tx() as db:
            db.execute("UPDATE experiments SET condition=?, revision=? WHERE experiment_id=? AND state='RUNNING'",
                       (condition, self.store.bump(), r["experiment_id"]))
        r["condition"] = condition
        return 1

    def _mark(self, r, column, value):
        with self.store.tx() as db:
            db.execute(f"UPDATE attempts SET {column}=? WHERE experiment_id=? AND idx=?", (value, r["experiment_id"], r["idx"]))

    def _infra(self, executor, r, message, status=None):
        """Retry a confirmed infrastructure failure only while no stop has been requested."""
        if not self._stop_requested(r) and executor.capabilities.retry_safe and r["idx"] <= self.max_infra_retries:
            retry = self._reserve_retry(r, message)
            if retry is not None:
                self._start(executor, retry)
                return 1
        return self._finish(r, "FAILED", failure={"class": "infrastructure", "message": message}, status=status)

    def _reserve_retry(self, r, message):
        now = self.clock.now()
        outcome = canonical({"state": "failed", "failure_class": "infrastructure", "message": message})
        with self.store.tx() as db:
            # HTTP can record cancellation while the scheduler is blocked in executor.poll().
            latest = db.execute("SELECT cancel_requested_at FROM experiments WHERE experiment_id=?",
                                (r["experiment_id"],)).fetchone()
            if latest[0] is not None:
                return None
            db.execute("UPDATE attempts SET state='ended', ended_at=?, outcome=? WHERE experiment_id=? AND idx=?",
                       (now, outcome, r["experiment_id"], r["idx"]))
            db.execute("INSERT INTO attempts (experiment_id, idx, state, started_at) VALUES (?,?,'dispatching',?)",
                       (r["experiment_id"], r["idx"] + 1, now))
            db.execute("UPDATE experiments SET condition=NULL, revision=? WHERE experiment_id=?",
                       (self.store.bump(), r["experiment_id"]))
        return dict(r, idx=r["idx"] + 1, attempt_state="dispatching", executor_ref=None, attempt_started_at=now,
                    cancel_attempted_at=None, cancel_acknowledged_at=None, timeout_at=None, condition=None)

    def _finish(self, r, state, *, failure=None, status=None):
        now, result, result_digest, artifacts = self.clock.now(), None, None, []
        if state == "SUCCEEDED":
            try:
                result = dict(status.result or {})
                result_digest = digest(result)
                artifacts = self._write_artifacts(r["experiment_id"], status.artifacts)
            except (TypeError, ValueError, OSError) as exc:
                state, result, result_digest, artifacts = "FAILED", None, None, []
                failure = {"class": "execution_error", "message": f"invalid executor output: {exc}"}
        runtime = max(0.0, now - r["started_at"])
        usage = {"pool": r["pool"], "units": r["units"], "runtime_seconds": runtime, "unit_seconds": runtime * r["units"]}
        if status is not None and status.usage:
            try:
                canonical(dict(status.usage))
                usage["reported"] = dict(status.usage)
            except (TypeError, ValueError):
                usage["reported_invalid"] = True
        outcome = {"state": status.state if status else "abandoned"}
        if failure:
            outcome.update(failure_class=failure["class"], message=failure.get("message"))
        with self.store.tx() as db:
            current = db.execute("SELECT state FROM experiments WHERE experiment_id=?", (r["experiment_id"],)).fetchone()
            if current[0] != "RUNNING":
                return 0
            db.execute("UPDATE attempts SET state='ended', ended_at=?, outcome=? WHERE experiment_id=? AND idx=?",
                       (now, canonical(outcome), r["experiment_id"], r["idx"]))
            for a in artifacts:
                db.execute("INSERT INTO artifacts VALUES (?,?,?,?,?,?)",
                           (r["experiment_id"], a["name"], a["media_type"], a["size"], a["sha256"], a["path"]))
            db.execute("""UPDATE experiments SET state=?, finished_at=?, condition=NULL, failure=?, result=?,
                result_digest=?, usage=?, revision=? WHERE experiment_id=?""",
                       (state, now, canonical(failure) if failure else None,
                        canonical(result) if result is not None else None, result_digest, canonical(usage),
                        self.store.bump(), r["experiment_id"]))
        return 1

    def _write_artifacts(self, experiment_id, artifacts):
        stored, names = [], set()
        folder = hashlib.sha256(experiment_id.encode()).hexdigest()[:32]
        for item in artifacts or ():
            item = Artifact(**item) if isinstance(item, Mapping) else item
            if not isinstance(item, Artifact) or not NAME_PATTERN.fullmatch(item.name or "") or item.name in names:
                raise ValueError(f"invalid or duplicate artifact {getattr(item, 'name', item)!r}")
            if (item.data is None) == (item.path is None):
                raise ValueError(f"artifact {item.name!r} needs exactly one of data or path")
            data = bytes(item.data) if item.data is not None else Path(item.path).read_bytes()
            target = self.artifact_dir / folder / item.name
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_name(target.name + ".partial")
            with open(partial, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(partial, target)
            names.add(item.name)
            stored.append({"name": item.name, "media_type": str(item.media_type), "size": len(data),
                           "sha256": hashlib.sha256(data).hexdigest(), "path": f"{folder}/{item.name}"})
        return stored
