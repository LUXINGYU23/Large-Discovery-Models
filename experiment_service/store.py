"""Single-writer SQLite store. The experiments table is also the queue (design 3.3, 3.9)."""
from __future__ import annotations

import fcntl
import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

TERMINAL = ("SUCCEEDED", "FAILED", "CANCELED")
_TERMINAL_SQL = "('SUCCEEDED','FAILED','CANCELED')"
SCHEMA_VERSION = 2

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value INTEGER NOT NULL);
INSERT OR IGNORE INTO meta VALUES ('schema_version', {SCHEMA_VERSION}), ('revision', 0);
CREATE TABLE IF NOT EXISTS experiments (
  experiment_id TEXT PRIMARY KEY, seq INTEGER NOT NULL UNIQUE, tombstone INTEGER NOT NULL DEFAULT 0,
  campaign_id TEXT, kind TEXT, kind_version TEXT, spec TEXT, spec_digest TEXT, limits TEXT, client TEXT,
  pool TEXT, units INTEGER, state TEXT NOT NULL, revision INTEGER NOT NULL,
  accepted_at REAL, started_at REAL, finished_at REAL, cancel_requested_at REAL,
  condition TEXT, failure TEXT, result TEXT, result_digest TEXT, usage TEXT);
CREATE INDEX IF NOT EXISTS experiments_changes ON experiments(campaign_id, revision);
CREATE INDEX IF NOT EXISTS experiments_queue ON experiments(state, pool, campaign_id, seq);
CREATE TABLE IF NOT EXISTS attempts (
  experiment_id TEXT NOT NULL REFERENCES experiments(experiment_id), idx INTEGER NOT NULL,
  state TEXT NOT NULL, executor_ref TEXT, started_at REAL NOT NULL, ended_at REAL,
  cancel_attempted_at REAL, cancel_acknowledged_at REAL, timeout_at REAL,
  outcome TEXT, PRIMARY KEY (experiment_id, idx));
CREATE TABLE IF NOT EXISTS artifacts (
  experiment_id TEXT NOT NULL REFERENCES experiments(experiment_id), name TEXT NOT NULL,
  media_type TEXT NOT NULL, size INTEGER NOT NULL, sha256 TEXT NOT NULL, path TEXT NOT NULL,
  PRIMARY KEY (experiment_id, name));
CREATE TRIGGER IF NOT EXISTS experiments_terminal_immutable BEFORE UPDATE ON experiments
  WHEN OLD.state IN {_TERMINAL_SQL}
  BEGIN SELECT RAISE(ABORT, 'terminal experiment is immutable'); END;
CREATE TRIGGER IF NOT EXISTS attempts_terminal_immutable BEFORE UPDATE ON attempts
  WHEN (SELECT state FROM experiments WHERE experiment_id = OLD.experiment_id) IN {_TERMINAL_SQL}
  BEGIN SELECT RAISE(ABORT, 'attempts of a terminal experiment are immutable'); END;
CREATE TRIGGER IF NOT EXISTS attempts_terminal_insert BEFORE INSERT ON attempts
  WHEN (SELECT state FROM experiments WHERE experiment_id = NEW.experiment_id) IN {_TERMINAL_SQL}
  BEGIN SELECT RAISE(ABORT, 'attempts of a terminal experiment are immutable'); END;
CREATE TRIGGER IF NOT EXISTS artifacts_fixed_insert BEFORE INSERT ON artifacts
  WHEN (SELECT state FROM experiments WHERE experiment_id = NEW.experiment_id) IS NOT 'RUNNING'
  BEGIN SELECT RAISE(ABORT, 'artifacts are fixed at the terminal transition'); END;
CREATE TRIGGER IF NOT EXISTS artifacts_immutable BEFORE UPDATE ON artifacts
  BEGIN SELECT RAISE(ABORT, 'artifacts are immutable'); END;
"""


def iso(t):
    return None if t is None else datetime.fromtimestamp(t, timezone.utc).isoformat(timespec="microseconds")


def revision_text(revision: int) -> str:
    return f"{revision:016d}"


def _load(text):
    return None if text is None else json.loads(text)


class ServiceAlreadyRunning(RuntimeError):
    """Another experiment service owns the database."""


class Store:
    """One connection guarded by one lock: writes are serialized, so revisions are
    assigned and committed in the same order and readers never see a gap close later."""

    def __init__(self, path):
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.changed = threading.Condition()
        self.closed = False
        lock_path = self.path.with_name(self.path.name + ".lock")
        self._instance_lock = lock_path.open("a")
        try:
            try:
                fcntl.flock(self._instance_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ServiceAlreadyRunning(f"experiment service already running for {self.path}") from exc
            self.db = sqlite3.connect(str(self.path), check_same_thread=False, isolation_level=None, timeout=30)
            try:
                self._initialize()
            except BaseException:
                self.db.close()
                raise
        except BaseException:
            self._instance_lock.close()
            raise
        self._bumped = None

    def _initialize(self):
        self.db.row_factory = sqlite3.Row
        if self.db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='meta'").fetchone():
            row = self.db.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
            if row is None or row[0] != SCHEMA_VERSION:
                raise RuntimeError(f"unsupported experiment service schema version; expected {SCHEMA_VERSION}")
        with self.lock:
            self.db.execute("PRAGMA journal_mode=WAL")
            self.db.execute("PRAGMA synchronous=FULL")
            self.db.execute("PRAGMA foreign_keys=ON")
            self.db.executescript(SCHEMA)
            self.head = self.db.execute("SELECT value FROM meta WHERE key='revision'").fetchone()[0]

    @contextmanager
    def tx(self):
        with self.lock:
            self.db.execute("BEGIN IMMEDIATE")
            self._bumped = None
            try:
                yield self.db
                self.db.execute("COMMIT")
            except BaseException:
                self.db.execute("ROLLBACK")
                raise
            bumped, self._bumped = self._bumped, None
        if bumped is not None:
            with self.changed:
                self.head = bumped
                self.changed.notify_all()

    def bump(self) -> int:
        """Allocate the next revision inside the current transaction."""
        self.db.execute("UPDATE meta SET value = value + 1 WHERE key='revision'")
        self._bumped = self.db.execute("SELECT value FROM meta WHERE key='revision'").fetchone()[0]
        return self._bumped

    def wait_change(self, seen_head: int, timeout: float) -> None:
        with self.changed:
            self.changed.wait_for(lambda: self.head != seen_head or self.closed, timeout)

    def rows(self, sql, args=()):
        with self.lock:
            return [dict(row) for row in self.db.execute(sql, args).fetchall()]

    def row(self, sql, args=()):
        found = self.rows(sql, args)
        return found[0] if found else None

    def resource(self, row) -> dict:
        """Build the public resource; call while holding ``lock`` for a consistent snapshot."""
        eid = row["experiment_id"]
        attempts = [{"index": a["idx"], "executor_ref": a["executor_ref"], "started_at": iso(a["started_at"]),
                     "ended_at": iso(a["ended_at"]), "outcome": _load(a["outcome"])}
                    for a in self.db.execute("SELECT * FROM attempts WHERE experiment_id=? ORDER BY idx", (eid,))]
        artifacts = [{"name": a["name"], "media_type": a["media_type"], "size": a["size"], "sha256": a["sha256"],
                      "href": f"/v1/experiments/{quote(eid, safe='')}/artifacts/{quote(a['name'], safe='')}"}
                     for a in self.db.execute("SELECT * FROM artifacts WHERE experiment_id=? ORDER BY name", (eid,))]
        return {
            "experiment_id": eid, "campaign_id": row["campaign_id"], "kind": row["kind"],
            "kind_version": row["kind_version"], "spec": _load(row["spec"]), "spec_digest": row["spec_digest"],
            "limits": _load(row["limits"]), "client": _load(row["client"]), "state": row["state"],
            "revision": revision_text(row["revision"]), "tombstone": bool(row["tombstone"]),
            "accepted_at": iso(row["accepted_at"]), "started_at": iso(row["started_at"]),
            "finished_at": iso(row["finished_at"]), "cancel_requested_at": iso(row["cancel_requested_at"]),
            "condition": row["condition"], "attempts": attempts, "failure": _load(row["failure"]),
            "result": _load(row["result"]), "result_digest": row["result_digest"], "artifacts": artifacts,
            "usage": _load(row["usage"]) or {},
        }

    def close(self):
        with self.changed:
            self.closed = True
            self.changed.notify_all()
        with self.lock:
            self.db.close()
            self._instance_lock.close()
