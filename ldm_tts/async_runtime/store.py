"""Controller transaction store (design 5.3): one local SQLite file, WAL, synchronous commits.

Every state transition is one ``tx(name)``; ``name.before_commit`` / ``name.after_commit`` are
crash points. The events table, observations, decisions and hook outputs are append-only
(enforced by triggers). HTTP never runs inside a transaction.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path

from .wire import canonical

SCHEMA_VERSION = "2"

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS campaign (
  singleton INTEGER PRIMARY KEY CHECK (singleton = 1), campaign_id TEXT NOT NULL, config_json TEXT NOT NULL,
  config_digest TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active','stopped','closed')),
  stop_reason TEXT, created_at REAL NOT NULL, closed_at REAL);
CREATE TABLE IF NOT EXISTS decisions (
  decision_id INTEGER PRIMARY KEY, experiment_id TEXT NOT NULL UNIQUE, seed INTEGER NOT NULL,
  snapshot_json TEXT NOT NULL, created_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS experiments (
  experiment_id TEXT PRIMARY KEY, decision_id INTEGER NOT NULL UNIQUE REFERENCES decisions(decision_id),
  canonical_key TEXT NOT NULL, candidate_json TEXT NOT NULL, kind TEXT NOT NULL, kind_version TEXT NOT NULL,
  request_body TEXT NOT NULL, spec_digest TEXT NOT NULL,
  submit_state TEXT NOT NULL CHECK (submit_state IN ('PLANNED','ACKED','REJECTED')),
  reject_status INTEGER,
  attempts INTEGER NOT NULL DEFAULT 0, next_attempt_at REAL NOT NULL, last_error TEXT,
  cancel_requested_at REAL, cancel_reason TEXT, cancel_acked_at REAL,
  cancel_attempts INTEGER NOT NULL DEFAULT 0, cancel_next_attempt_at REAL,
  es_state TEXT, es_revision INTEGER, es_tombstone INTEGER, es_accepted_at TEXT, es_started_at TEXT,
  es_finished_at TEXT, es_cancel_requested_at TEXT, es_condition TEXT, es_failure_class TEXT,
  es_result_digest TEXT, es_resource_json TEXT, observed_at REAL,
  pending INTEGER NOT NULL CHECK (pending IN (0, 1)),
  budget_state TEXT NOT NULL CHECK (budget_state IN ('HOLD','CONSUMED','REFUNDED')),
  disposition TEXT CHECK (disposition IN ('ingested','invalid','no_result')),
  written_off_at REAL, written_off_by TEXT, written_off_reason TEXT, created_at REAL NOT NULL);
CREATE UNIQUE INDEX IF NOT EXISTS experiments_pending_key ON experiments(canonical_key) WHERE pending = 1;
CREATE INDEX IF NOT EXISTS experiments_key ON experiments(canonical_key);
CREATE TABLE IF NOT EXISTS observations (
  experiment_id TEXT PRIMARY KEY REFERENCES experiments(experiment_id), ingest_seq INTEGER NOT NULL UNIQUE,
  decision_id INTEGER NOT NULL, canonical_key TEXT NOT NULL, status TEXT NOT NULL, observation_json TEXT NOT NULL,
  result_digest TEXT, flags_json TEXT NOT NULL, ingested_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS events (
  seq INTEGER PRIMARY KEY AUTOINCREMENT, type TEXT NOT NULL, subject_id TEXT, payload_json TEXT NOT NULL,
  created_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS hook_cursors (hook TEXT PRIMARY KEY, last_seq INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS hook_outputs (
  hook TEXT NOT NULL, event_seq INTEGER NOT NULL, output_json TEXT NOT NULL, created_at REAL NOT NULL,
  PRIMARY KEY (hook, event_seq));
CREATE TABLE IF NOT EXISTS hook_failures (
  hook TEXT NOT NULL, event_seq INTEGER NOT NULL, attempts INTEGER NOT NULL, last_error TEXT NOT NULL,
  gave_up INTEGER NOT NULL DEFAULT 0, next_attempt_at REAL, PRIMARY KEY (hook, event_seq));
CREATE TABLE IF NOT EXISTS cursors (name TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS anomalies (
  id INTEGER PRIMARY KEY AUTOINCREMENT, type TEXT NOT NULL, subject_id TEXT NOT NULL DEFAULT '',
  payload_json TEXT NOT NULL, created_at REAL NOT NULL, resolved_at REAL, resolved_by TEXT);
CREATE UNIQUE INDEX IF NOT EXISTS anomalies_open ON anomalies(type, subject_id) WHERE resolved_at IS NULL;
CREATE TABLE IF NOT EXISTS flags (name TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS occurrences (
  id INTEGER PRIMARY KEY AUTOINCREMENT, canonical_key TEXT NOT NULL, candidate_json TEXT NOT NULL,
  lineage TEXT NOT NULL, generation INTEGER NOT NULL, watermark INTEGER NOT NULL, created_at REAL NOT NULL,
  removed_at REAL, removed_reason TEXT, removed_results INTEGER);
CREATE INDEX IF NOT EXISTS occurrences_live ON occurrences(canonical_key) WHERE removed_at IS NULL;
CREATE TABLE IF NOT EXISTS generations (
  generation INTEGER PRIMARY KEY,
  state TEXT NOT NULL CHECK (state IN ('running','ready','rejected','blocked','abandoned')),
  trigger TEXT NOT NULL, input_path TEXT NOT NULL, input_digest TEXT NOT NULL, base_ingest_seq INTEGER NOT NULL,
  base_event_seq INTEGER NOT NULL, final_watermark INTEGER, started_at REAL NOT NULL, deadline_at REAL NOT NULL,
  finished_at REAL, proposal_count INTEGER, admitted INTEGER, decision_head INTEGER, outcome_json TEXT, error TEXT);
CREATE TABLE IF NOT EXISTS plans (
  plan_id INTEGER PRIMARY KEY, generation INTEGER, count INTEGER NOT NULL, keys_json TEXT NOT NULL,
  probabilities_json TEXT NOT NULL, record_json TEXT NOT NULL, inputs_json TEXT NOT NULL,
  consumed INTEGER NOT NULL DEFAULT 0, closed_reason TEXT, created_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS agent_messages (
  id INTEGER PRIMARY KEY AUTOINCREMENT, request_id TEXT NOT NULL UNIQUE, source TEXT NOT NULL,
  target_generation INTEGER NOT NULL, text TEXT NOT NULL, payload_json TEXT NOT NULL, created_at REAL NOT NULL,
  delivered_at REAL, delivery_json TEXT);
CREATE TRIGGER IF NOT EXISTS events_append_only_u BEFORE UPDATE ON events
  BEGIN SELECT RAISE(ABORT, 'events are append-only'); END;
CREATE TRIGGER IF NOT EXISTS events_append_only_d BEFORE DELETE ON events
  BEGIN SELECT RAISE(ABORT, 'events are append-only'); END;
CREATE TRIGGER IF NOT EXISTS observations_immutable_u BEFORE UPDATE ON observations
  BEGIN SELECT RAISE(ABORT, 'observations are immutable'); END;
CREATE TRIGGER IF NOT EXISTS observations_immutable_d BEFORE DELETE ON observations
  BEGIN SELECT RAISE(ABORT, 'observations are immutable'); END;
CREATE TRIGGER IF NOT EXISTS decisions_immutable_u BEFORE UPDATE ON decisions
  BEGIN SELECT RAISE(ABORT, 'decisions are immutable'); END;
CREATE TRIGGER IF NOT EXISTS decisions_immutable_d BEFORE DELETE ON decisions
  BEGIN SELECT RAISE(ABORT, 'decisions are immutable'); END;
CREATE TRIGGER IF NOT EXISTS hook_outputs_immutable_u BEFORE UPDATE ON hook_outputs
  BEGIN SELECT RAISE(ABORT, 'hook outputs are immutable'); END;
"""


class Store:
    def __init__(self, path, *, faults=None, busy_timeout: float = 30.0):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.faults = faults
        self.db = sqlite3.connect(str(self.path), isolation_level=None, check_same_thread=False, timeout=busy_timeout)
        self.db.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        with self.lock:
            self.db.execute("PRAGMA journal_mode=WAL")
            self.db.execute("PRAGMA synchronous=FULL")
            self.db.execute("PRAGMA foreign_keys=ON")
            self.db.execute(f"PRAGMA busy_timeout={int(busy_timeout * 1000)}")
            self.db.executescript(SCHEMA)
            self.db.execute("INSERT OR IGNORE INTO meta VALUES ('schema_version', ?)", (SCHEMA_VERSION,))
            if self.value("SELECT value FROM meta WHERE key='schema_version'") != SCHEMA_VERSION:
                raise RuntimeError("unsupported controller store schema version")

    def check(self, point: str) -> None:
        if self.faults is not None:
            self.faults.check(point)

    @contextmanager
    def tx(self, name: str):
        with self.lock:
            self.db.execute("BEGIN IMMEDIATE")
            try:
                yield self.db
                self.check(f"{name}.before_commit")
            except BaseException:
                self.db.execute("ROLLBACK")
                raise
            self.db.execute("COMMIT")
        self.check(f"{name}.after_commit")

    def rows(self, sql, args=()):
        with self.lock:
            return self.db.execute(sql, args).fetchall()

    def row(self, sql, args=()):
        with self.lock:
            return self.db.execute(sql, args).fetchone()

    def value(self, sql, args=(), default=None):
        found = self.row(sql, args)
        return default if found is None or found[0] is None else found[0]

    def close(self):
        with self.lock:
            self.db.close()


# ---- helpers used inside a transaction -----------------------------------------------------------

def append_event(db, now: float, type_: str, subject_id: str | None = None, payload=None) -> int:
    cursor = db.execute("INSERT INTO events (type, subject_id, payload_json, created_at) VALUES (?,?,?,?)",
                        (type_, subject_id, canonical(payload or {}), now))
    return cursor.lastrowid


def record_anomaly(db, now: float, type_: str, subject_id: str = "", payload=None) -> bool:
    """Record an open anomaly once per ``(type, subject)``; appends ``anomaly.<type>`` when new."""
    cursor = db.execute("""INSERT INTO anomalies (type, subject_id, payload_json, created_at) VALUES (?,?,?,?)
        ON CONFLICT (type, subject_id) WHERE resolved_at IS NULL DO NOTHING""",
                        (type_, subject_id or "", canonical(payload or {}), now))
    if cursor.rowcount:
        append_event(db, now, f"anomaly.{type_}", subject_id or None, payload or {})
        return True
    return False


def resolve_anomalies(db, now: float, type_: str, subject_id: str = "", by: str = "auto") -> int:
    return db.execute("UPDATE anomalies SET resolved_at=?, resolved_by=? WHERE type=? AND subject_id=? AND resolved_at IS NULL",
                      (now, by, type_, subject_id or "")).rowcount


def get_flag(db, name: str, default=None):
    row = db.execute("SELECT value FROM flags WHERE name=?", (name,)).fetchone()
    return default if row is None else json.loads(row[0])


def set_flag(db, name: str, value) -> None:
    if value is None:
        db.execute("DELETE FROM flags WHERE name=?", (name,))
    else:
        db.execute("INSERT INTO flags VALUES (?,?) ON CONFLICT (name) DO UPDATE SET value=excluded.value",
                   (name, canonical(value)))


def get_cursor(db, name: str, default: str = "") -> str:
    row = db.execute("SELECT value FROM cursors WHERE name=?", (name,)).fetchone()
    return default if row is None else row[0]


def set_cursor(db, name: str, value: str) -> None:
    db.execute("INSERT INTO cursors VALUES (?,?) ON CONFLICT (name) DO UPDATE SET value=excluded.value", (name, value))
