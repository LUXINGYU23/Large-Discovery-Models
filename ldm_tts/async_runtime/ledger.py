"""Transactional state of the controller: intents, the ES mirror, ingestion, budget and dedup.

Three projections per experiment (design 5.2), never conflated:

* submit confirmation ``submit_state``: PLANNED -> ACKED (or REJECTED on a definite 4xx);
* ES mirror ``es_*``: overwritten only by an ES resource with a larger revision;
* result disposition ``disposition`` (ingested / invalid / no_result) plus the write-off columns.

Algorithmic ``pending`` and ``budget_state`` are explicit columns updated in the same transaction
as the mirror (I3 is a partial unique index; budget bounds are derived by SUM over the column).
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

from ldm_tts.contracts import Candidate, EvaluationResult, Observation

from .codec import ExperimentRequest, expected_status
from .config import CampaignConfig
from .store import (Store, append_event, get_flag, record_anomaly, resolve_anomalies, set_flag)
from .wire import STATES, TERMINAL, TERMINAL_SQL, canonical, decision_label, digest, experiment_id_for, revision_number


@dataclass(frozen=True)
class Budget:
    limit: int
    held: int
    consumed: int
    refunded: int
    started: int

    @property
    def free(self) -> int:
        return self.limit - self.held - self.consumed

    @property
    def lower(self) -> int:
        """Experiments the ES confirmed as started (design 2.2)."""
        return self.started

    @property
    def upper(self) -> int:
        return self.held + self.consumed

    def to_dict(self) -> dict:
        return {**asdict(self), "free": self.free, "lower": self.lower, "upper": self.upper}


@dataclass(frozen=True)
class PendingItem:
    experiment_id: str
    decision_id: int
    canonical_key: str
    candidate: Candidate
    state: str  # PLANNED | REJECTED | QUEUED | RUNNING
    cancel_requested: bool


@dataclass(frozen=True)
class StateView:
    """Read-only snapshot handed to proposers, selectors, pacing and task hooks."""

    now: float
    campaign_id: str
    status: str
    stop_reason: str | None
    budget: Budget
    pending: tuple[PendingItem, ...]
    unstarted: int
    outstanding: int
    dedup_keys: frozenset[str]
    observation_count: int
    head_ingest_seq: int
    paused: tuple[str, ...]
    backoff_until: float | None
    collect_only: bool
    slot_available: bool
    config: Mapping[str, Any] = field(default_factory=dict)

    @property
    def pending_keys(self) -> frozenset[str]:
        return frozenset(item.canonical_key for item in self.pending)


def candidate_from_json(text: str) -> Candidate:
    return Candidate(**json.loads(text))


def observation_from_json(text: str) -> Observation:
    data = json.loads(text)
    return Observation(candidate=Candidate(**data["candidate"]), evaluation=EvaluationResult(**data["evaluation"]),
                       surrogate=data.get("surrogate"), round_idx=data.get("round_idx"),
                       metadata=data.get("metadata") or {}, experiment_id=data.get("experiment_id", ""),
                       decision_id=data.get("decision_id", ""), lineage=data.get("lineage") or {})


def dedup_keys(db, max_attempts_per_key: int) -> frozenset[str]:
    """The controller's dedup set (``Ledger.dedup_keys``) over a store connection; shared with the invariants."""
    rows = db.execute("""SELECT canonical_key FROM experiments GROUP BY canonical_key HAVING
        MAX(pending) = 1 OR MAX(written_off_at IS NOT NULL) = 1 OR MAX(es_state = 'SUCCEEDED') = 1
        OR MAX(es_state = 'FAILED' AND es_failure_class = 'infeasible') = 1
        OR SUM(es_state = 'SUCCEEDED' OR (es_state = 'FAILED'
               AND COALESCE(es_failure_class, '') != 'infrastructure')) >= ?""", (max_attempts_per_key,)).fetchall()
    return frozenset(row[0] for row in rows)


class Ledger:
    def __init__(self, store: Store, config: CampaignConfig, codec, *, clock, objectives=None, collect_only=False):
        self.store, self.config, self.codec, self.clock = store, config, codec, clock
        self.objectives, self.collect_only = objectives, collect_only
        self.campaign_id = config.campaign_id
        # Same-transaction observers (controller-pacing: reservoir ageing/invalidation, plans, breaker).
        # Optional methods: on_intent(db, candidate, decision_id, snapshot, now),
        # on_ingested(db, canonical_key, ingest_seq, now), on_settled(db, experiment_id, state, now).
        self.listeners: list = []

    def _notify(self, method: str, db, *args) -> None:
        for listener in self.listeners:
            function = getattr(listener, method, None)
            if function is not None:
                function(db, *args)

    # ---- derived quantities (callable inside or outside a transaction) --------------------------

    def _db(self, db):
        return db if db is not None else self.store.db

    def budget(self, db=None) -> Budget:
        with self.store.lock:
            rows = self._db(db).execute("""SELECT budget_state, es_started_at IS NOT NULL AS started, COUNT(*) AS n
                FROM experiments GROUP BY 1, 2""").fetchall()
        counts = {"HOLD": 0, "CONSUMED": 0, "REFUNDED": 0}
        started = 0
        for row in rows:
            counts[row["budget_state"]] += row["n"]
            if row["started"] and row["budget_state"] != "REFUNDED":
                started += row["n"]
        return Budget(self.config.budget, counts["HOLD"], counts["CONSUMED"], counts["REFUNDED"], started)

    def counts(self, db=None) -> tuple[int, int]:
        """``(N, O)``: unstarted and outstanding (algorithmically pending) experiments."""
        with self.store.lock:
            row = self._db(db).execute("""SELECT COUNT(*), COALESCE(SUM(es_started_at IS NULL), 0)
                FROM experiments WHERE pending=1""").fetchone()
        return row[1], row[0]

    def has_slot(self, db=None) -> bool:
        unstarted, outstanding = self.counts(db)
        return outstanding < self.config.max_outstanding and unstarted < self.config.max_unstarted

    def dedup_keys(self, db=None) -> frozenset[str]:
        """Pending, written off, succeeded, infeasible, or tried ``max_attempts_per_key`` times (5.2). A FAILED
        attempt of class ``infrastructure`` is not a try of the candidate (coordinator decision): the ES already
        retries infrastructure faults and a resource fault must not block a key for good."""
        with self.store.lock:
            return dedup_keys(self._db(db), self.config.max_attempts_per_key)

    def campaign_row(self, db=None):
        with self.store.lock:
            return self._db(db).execute("SELECT status, stop_reason FROM campaign WHERE singleton=1").fetchone()

    def flag(self, name, default=None, db=None):
        with self.store.lock:
            return get_flag(self._db(db), name, default)

    def paused_reasons(self, db=None) -> tuple[str, ...]:
        return tuple(item["reason"] for item in self.flag("paused", [], db))

    def view(self) -> StateView:
        now = self.clock.now()
        with self.store.lock:
            db = self.store.db
            rows = db.execute("""SELECT experiment_id, decision_id, canonical_key, candidate_json, submit_state, es_state,
                cancel_requested_at FROM experiments WHERE pending=1 ORDER BY decision_id""").fetchall()
            pending = tuple(PendingItem(r["experiment_id"], r["decision_id"], r["canonical_key"],
                                        candidate_from_json(r["candidate_json"]),
                                        r["es_state"] if r["submit_state"] == "ACKED" else r["submit_state"],
                                        r["cancel_requested_at"] is not None) for r in rows)
            unstarted, outstanding = self.counts(db)
            observations, head = db.execute("SELECT COUNT(*), COALESCE(MAX(ingest_seq), 0) FROM observations").fetchone()
            campaign = self.campaign_row(db)
            backoff = get_flag(db, "es_backoff_until")
            return StateView(now=now, campaign_id=self.campaign_id, status=campaign["status"],
                             stop_reason=campaign["stop_reason"], budget=self.budget(db), pending=pending,
                             unstarted=unstarted, outstanding=outstanding, dedup_keys=self.dedup_keys(db),
                             observation_count=observations, head_ingest_seq=head, paused=self.paused_reasons(db),
                             backoff_until=backoff if backoff is not None and backoff > now else None,
                             collect_only=self.collect_only,
                             slot_available=outstanding < self.config.max_outstanding
                             and unstarted < self.config.max_unstarted, config=self.config.to_json())

    def dispatch_blockers(self, db=None, key: str | None = None) -> list[str]:
        """Core guards no pacing policy can override (design 2.3, 5.4 D2-D5)."""
        db = self._db(db)
        blockers = []
        campaign = self.campaign_row(db)
        if campaign["status"] != "active":
            blockers.append(f"campaign_{campaign['status']}")
        if self.collect_only:
            blockers.append("collect_only")
        if self.paused_reasons(db):
            blockers.append("paused")
        if self.budget(db).free < 1:
            blockers.append("budget")
        unstarted, outstanding = self.counts(db)
        if unstarted >= self.config.max_unstarted:
            blockers.append("short_queue")
        if outstanding >= self.config.max_outstanding:
            blockers.append("max_outstanding")
        if key is not None and key in self.dedup_keys(db):
            blockers.append("duplicate_key")
        return blockers

    def next_decision_id(self, db=None) -> int:
        with self.store.lock:
            return self._db(db).execute("SELECT COALESCE(MAX(decision_id), 0) + 1 FROM decisions").fetchone()[0]

    # ---- intent (design 4.1 W1/W2) --------------------------------------------------------------

    def plan_intent(self, decision_id: int, candidate: Candidate, request: ExperimentRequest, seed: int,
                    snapshot: Mapping[str, Any]) -> str | None:
        """One transaction: decision record + experiment intent + budget hold + ``experiment.planned``."""
        now = self.clock.now()
        experiment_id = experiment_id_for(self.campaign_id, decision_id)
        body = canonical({"campaign_id": self.campaign_id, "kind": request.kind, "kind_version": request.kind_version,
                          "spec": request.spec, "limits": request.limits, "spec_digest": digest(request.spec),
                          "client": {"decision_id": decision_label(decision_id), "candidate_key": candidate.canonical_key}})
        candidate_json = canonical(candidate.to_dict())
        with self.store.tx("intent") as db:
            if self.next_decision_id(db) != decision_id or self.dispatch_blockers(db, candidate.canonical_key):
                return None
            slot = self.has_slot(db)
            db.execute("INSERT INTO decisions VALUES (?,?,?,?,?)",
                       (decision_id, experiment_id, seed, canonical(dict(snapshot)), now))
            db.execute("""INSERT INTO experiments (experiment_id, decision_id, canonical_key, candidate_json, kind,
                kind_version, request_body, spec_digest, submit_state, next_attempt_at, pending, budget_state, created_at)
                VALUES (?,?,?,?,?,?,?,?,'PLANNED',?,1,'HOLD',?)""",
                       (experiment_id, decision_id, candidate.canonical_key, candidate_json, request.kind,
                        request.kind_version, body, digest(request.spec), now, now))
            append_event(db, now, "experiment.planned", experiment_id,
                         {"decision_id": decision_id, "canonical_key": candidate.canonical_key, "kind": request.kind})
            self._notify("on_intent", db, candidate, decision_id, snapshot, now)
            if self.budget(db).free == 0:
                append_event(db, now, "budget.exhausted", None, self.budget(db).to_dict())
            self._slot_event(db, slot, now)
        return experiment_id

    # ---- applying ES resources (PUT/cancel responses, change feed, anti-entropy) ----------------

    def apply_resource(self, db, resource: Mapping[str, Any], source: str) -> bool:
        """Revision rule (design 5.2, I6): only a larger revision overwrites the mirror; a terminal
        mirror settles budget/pending and ingests the result in this same transaction."""
        now = self.clock.now()
        experiment_id = resource.get("experiment_id")
        row = db.execute("SELECT * FROM experiments WHERE experiment_id=?", (experiment_id,)).fetchone()
        if row is None:
            return self.anomaly(db, "unknown_experiment", str(experiment_id), {"source": source, "resource": resource},
                                pause=True)
        try:
            revision, state = revision_number(resource.get("revision")), resource.get("state")
            if state not in STATES:
                raise ValueError(f"unknown state {state!r}")
        except ValueError as exc:
            return self.anomaly(db, "malformed_resource", experiment_id, {"source": source, "error": str(exc)},
                                pause=True)
        tombstone = bool(resource.get("tombstone"))
        if resource.get("campaign_id") not in (None, self.campaign_id) or \
                (not tombstone and resource.get("spec_digest") != row["spec_digest"]):
            return self.anomaly(db, "spec_mismatch", experiment_id,
                                {"source": source, "spec_digest": resource.get("spec_digest"),
                                 "campaign_id": resource.get("campaign_id")}, pause=True)
        changed = False
        if row["submit_state"] != "ACKED":
            db.execute("UPDATE experiments SET submit_state='ACKED', last_error=NULL WHERE experiment_id=?",
                       (experiment_id,))
            append_event(db, now, "experiment.accepted", experiment_id,
                         {"source": source, "state": state, "revision": revision, "tombstone": tombstone})
            changed = True
        local = row["es_revision"]
        if local is not None and revision < local:
            return changed  # stale (a delayed response or an older page): ignore
        if local is not None and revision == local:
            if canonical(dict(resource)) != row["es_resource_json"]:
                self.anomaly(db, "revision_conflict", experiment_id, {"source": source, "revision": revision})
            return changed
        if row["es_state"] in TERMINAL:
            if state != row["es_state"] or resource.get("result_digest") != row["es_result_digest"]:
                self.anomaly(db, "terminal_changed", experiment_id,
                             {"source": source, "kept": row["es_state"], "incoming": state})
            return changed
        slot = self.has_slot(db)
        failure_class = (resource.get("failure") or {}).get("class")
        db.execute("""UPDATE experiments SET es_state=?, es_revision=?, es_tombstone=?, es_accepted_at=?, es_started_at=?,
            es_finished_at=?, es_cancel_requested_at=?, es_condition=?, es_failure_class=?, es_result_digest=?,
            es_resource_json=?, observed_at=? WHERE experiment_id=?""",
                   (state, revision, int(tombstone), resource.get("accepted_at"), resource.get("started_at"),
                    resource.get("finished_at"), resource.get("cancel_requested_at"), resource.get("condition"),
                    failure_class, resource.get("result_digest"), canonical(dict(resource)), now, experiment_id))
        started_now = row["es_started_at"] is None and bool(resource.get("started_at"))
        if started_now:
            append_event(db, now, "experiment.started", experiment_id,
                         {"state": state, "started_at": resource.get("started_at")})
        if state in TERMINAL:
            append_event(db, now, "experiment.finished", experiment_id,
                         {"state": state, "failure_class": failure_class, "started": bool(resource.get("started_at"))})
            self._settle(db, row, resource, now)
            self._notify("on_settled", db, experiment_id, state, now)
        elif not started_now:
            append_event(db, now, "experiment.updated", experiment_id,
                         {"state": state, "condition": resource.get("condition"),
                          "cancel_requested_at": resource.get("cancel_requested_at"), "revision": revision})
        self._slot_event(db, slot, now)
        return True

    def _settle(self, db, row, resource, now):
        experiment_id, state = row["experiment_id"], resource["state"]
        if row["written_off_at"] is None:
            # refund only with ES evidence that nothing started (design 2.2)
            budget_state = "CONSUMED" if resource.get("started_at") else "REFUNDED"
            db.execute("UPDATE experiments SET pending=0, budget_state=? WHERE experiment_id=? AND budget_state='HOLD'",
                       (budget_state, experiment_id))
        flags = []
        if row["cancel_requested_at"] is not None or resource.get("cancel_requested_at"):
            flags.append("after_cancel_request")
        if row["written_off_at"] is not None:
            flags.append("after_write_off")
        if self.campaign_row(db)["status"] == "closed":
            flags.append("after_close")
        if state == "CANCELED":
            db.execute("UPDATE experiments SET disposition='no_result' WHERE experiment_id=?", (experiment_id,))
            if row["cancel_requested_at"] is None:
                self.anomaly(db, "unexpected_cancel", experiment_id, {"started": bool(resource.get("started_at"))})
            return
        candidate = candidate_from_json(row["candidate_json"])
        try:
            if state == "SUCCEEDED" and digest(resource.get("result")) != resource.get("result_digest"):
                raise ValueError("result does not match result_digest")
            evaluation = self.codec.decode(candidate, resource)
            if not isinstance(evaluation, EvaluationResult):
                raise TypeError(f"codec returned {type(evaluation).__name__}, not EvaluationResult")
            if evaluation.candidate_id != candidate.candidate_id:
                raise ValueError("codec result names a different candidate")
            if evaluation.status != expected_status(resource):
                raise ValueError(f"codec status {evaluation.status!r} contradicts ES state {state}")
            if self.objectives is not None:
                self.objectives.validate_result(evaluation)
        except Exception as exc:  # noqa: BLE001 - a task-side defect makes the result invalid, not the campaign
            db.execute("UPDATE experiments SET disposition='invalid' WHERE experiment_id=?", (experiment_id,))
            self.anomaly(db, "invalid_result", experiment_id, {"error": f"{type(exc).__name__}: {exc}"})
            return
        existing = db.execute("SELECT result_digest FROM observations WHERE experiment_id=?", (experiment_id,)).fetchone()
        if existing is not None:
            if existing[0] != resource.get("result_digest"):
                self.anomaly(db, "result_changed", experiment_id, {"kept": existing[0]})
            return
        seq = db.execute("SELECT COALESCE(MAX(ingest_seq), 0) + 1 FROM observations").fetchone()[0]
        observation = Observation(
            candidate=candidate, evaluation=evaluation, experiment_id=experiment_id,
            decision_id=decision_label(row["decision_id"]),
            metadata={"ingest_seq": seq, "flags": flags,
                      "es": {key: resource.get(key) for key in ("revision", "started_at", "finished_at", "usage")}})
        db.execute("INSERT INTO observations VALUES (?,?,?,?,?,?,?,?,?)",
                   (experiment_id, seq, row["decision_id"], row["canonical_key"], evaluation.status,
                    canonical(observation.to_dict()), resource.get("result_digest"), canonical(flags), now))
        db.execute("UPDATE experiments SET disposition='ingested' WHERE experiment_id=?", (experiment_id,))
        append_event(db, now, "observation.ingested", experiment_id,
                     {"ingest_seq": seq, "status": evaluation.status, "canonical_key": row["canonical_key"],
                      "flags": flags})
        self._notify("on_ingested", db, row["canonical_key"], seq, now)

    def _slot_event(self, db, before: bool, now: float) -> None:
        if not before and self.has_slot(db):
            append_event(db, now, "slot.available", None, {})

    # ---- outbox bookkeeping -----------------------------------------------------------------------

    def record_put_retry(self, experiment_id: str, error: str, delay: float, *, backoff_until: float | None = None):
        now = self.clock.now()
        with self.store.tx("retry") as db:
            db.execute("""UPDATE experiments SET attempts=attempts+1, next_attempt_at=?, last_error=?
                WHERE experiment_id=? AND submit_state='PLANNED'""", (now + delay, error[:500], experiment_id))
            if backoff_until is not None:
                set_flag(db, "es_backoff_until", max(backoff_until, get_flag(db, "es_backoff_until", 0.0)))

    def record_put_rejected(self, experiment_id: str, status, problem, error: str):
        now = self.clock.now()
        with self.store.tx("reject") as db:
            changed = db.execute("""UPDATE experiments SET submit_state='REJECTED', attempts=attempts+1, last_error=?,
                reject_status=? WHERE experiment_id=? AND submit_state='PLANNED'""",
                                 (error[:500], status, experiment_id)).rowcount
            if changed:
                append_event(db, now, "experiment.rejected", experiment_id, {"status": status})
                self.anomaly(db, "put_rejected", experiment_id, {"status": status, "problem": problem}, pause=True)

    def retry_rejected(self, experiment_id: str) -> bool:
        """Operator: after fixing the cause of a 4xx, send the stored request again."""
        now = self.clock.now()
        with self.store.tx("retry_rejected") as db:
            changed = db.execute("""UPDATE experiments SET submit_state='PLANNED', next_attempt_at=?
                WHERE experiment_id=? AND submit_state='REJECTED' AND written_off_at IS NULL""",
                                 (now, experiment_id)).rowcount
            if changed:
                resolve_anomalies(db, now, "put_rejected", experiment_id, by="operator")
                append_event(db, now, "experiment.resubmitted", experiment_id, {})
        return bool(changed)

    def request_cancel(self, experiment_id: str, reason: str, *, force: bool = False) -> bool:
        """A 409-rejected intent names an id the ES holds for a *different* experiment: cancelling it would cancel
        someone else's work, so it is refused unless ``force`` (write it off instead)."""
        now = self.clock.now()
        row = self.store.row("SELECT submit_state, reject_status FROM experiments WHERE experiment_id=?", (experiment_id,))
        if row is not None and row["submit_state"] == "REJECTED" and row["reject_status"] == 409 and not force:
            raise ValueError(f"{experiment_id} was rejected with 409: the ES holds another experiment under this id; "
                             "write it off instead (or pass force=True)")
        with self.store.tx("cancel_request") as db:
            changed = db.execute(f"""UPDATE experiments SET cancel_requested_at=?, cancel_reason=?, cancel_next_attempt_at=?
                WHERE experiment_id=? AND cancel_requested_at IS NULL AND pending=1
                AND (es_state IS NULL OR es_state NOT IN {TERMINAL_SQL})""",
                                 (now, reason, now, experiment_id)).rowcount
            if changed:
                append_event(db, now, "experiment.cancel_requested", experiment_id, {"reason": reason})
        return bool(changed)

    def record_cancel_retry(self, experiment_id: str, error: str, delay: float):
        now = self.clock.now()
        with self.store.tx("retry") as db:
            db.execute("""UPDATE experiments SET cancel_attempts=cancel_attempts+1, cancel_next_attempt_at=?, last_error=?
                WHERE experiment_id=?""", (now + delay, error[:500], experiment_id))

    def record_cancel_response(self, status: int, resource: Mapping[str, Any]):
        """200 or 202: the request is recorded by the ES. The body is a full resource and goes through
        the revision rule; budget only ever settles on a terminal state, never on the 202 itself."""
        now = self.clock.now()
        with self.store.tx("cancel_ack") as db:
            db.execute("UPDATE experiments SET cancel_acked_at=?, cancel_attempts=cancel_attempts+1 WHERE experiment_id=?",
                       (now, resource.get("experiment_id")))
            self.apply_resource(db, resource, f"cancel_{status}")

    def write_off(self, experiment_id: str, operator: str, reason: str) -> bool:
        """Operator: the only way out of algorithmic pending without a terminal state (design 2.2, 4.3).
        Budget stays consumed and the key stays in the dedup set; a later result is still ingested."""
        if not operator or not reason:
            raise ValueError("a write-off needs an operator identity and a reason")
        row = self.store.row("SELECT submit_state, es_state FROM experiments WHERE experiment_id=?", (experiment_id,))
        if row is not None and row["es_state"] is None and row["submit_state"] == "PLANNED":
            # review item 3: the outbox would still deliver it; a cancel leaves a tombstone and refunds instead
            raise ValueError(f"{experiment_id} was never acknowledged by the ES: use request_cancel, not write_off")
        now = self.clock.now()
        with self.store.tx("write_off") as db:
            slot = self.has_slot(db)
            changed = db.execute(f"""UPDATE experiments SET pending=0, budget_state='CONSUMED', written_off_at=?,
                written_off_by=?, written_off_reason=? WHERE experiment_id=? AND pending=1 AND written_off_at IS NULL
                AND (es_state IS NULL OR es_state NOT IN {TERMINAL_SQL})""",
                                 (now, operator, reason, experiment_id)).rowcount
            if changed:
                append_event(db, now, "experiment.written_off", experiment_id, {"operator": operator, "reason": reason})
                self._slot_event(db, slot, now)
        return bool(changed)

    # ---- campaign lifecycle, anomalies, reachability ----------------------------------------------

    def anomaly(self, db, type_: str, subject: str, payload=None, *, pause: bool = False) -> bool:
        """Record (once while open) and alert; a pausing anomaly pauses dispatch when first seen,
        so an operator's ``resume`` is not undone by the next anti-entropy pass."""
        now = self.clock.now()
        new = record_anomaly(db, now, type_, subject, payload)
        if new and pause:
            self.pause(db, f"anomaly.{type_}:{subject}")
        return new

    def resolve_anomaly(self, anomaly_id: int, operator: str) -> bool:
        now = self.clock.now()
        with self.store.tx("resolve") as db:
            row = db.execute("SELECT type, subject_id FROM anomalies WHERE id=? AND resolved_at IS NULL",
                             (anomaly_id,)).fetchone()
            if row is None:
                return False
            resolve_anomalies(db, now, row["type"], row["subject_id"], by=operator)
            append_event(db, now, "anomaly.resolved", row["subject_id"] or None,
                         {"id": anomaly_id, "type": row["type"], "operator": operator})
        return True

    def pause(self, db, reason: str) -> None:
        reasons = get_flag(db, "paused", [])
        if all(item["reason"] != reason for item in reasons):
            now = self.clock.now()
            set_flag(db, "paused", reasons + [{"reason": reason, "at": now}])
            append_event(db, now, "campaign.paused", None, {"reason": reason})

    def resume(self, operator: str) -> bool:
        now = self.clock.now()
        with self.store.tx("resume") as db:
            reasons = get_flag(db, "paused", [])
            if not reasons:
                return False
            set_flag(db, "paused", None)
            append_event(db, now, "campaign.resumed", None, {"operator": operator, "cleared": reasons})
        return True

    def stop(self, reason: str) -> bool:
        now = self.clock.now()
        with self.store.tx("stop") as db:
            changed = db.execute("UPDATE campaign SET status='stopped', stop_reason=? WHERE singleton=1 AND status='active'",
                                 (reason,)).rowcount
            if changed:
                append_event(db, now, "campaign.stopped", None, {"reason": reason})
        return bool(changed)

    def close_campaign(self, reason: str = "closed") -> bool:
        now = self.clock.now()
        with self.store.tx("close") as db:
            changed = db.execute("""UPDATE campaign SET status='closed', closed_at=?, stop_reason=COALESCE(stop_reason, ?)
                WHERE singleton=1 AND status != 'closed'""", (now, reason)).rowcount
            if changed:
                append_event(db, now, "campaign.closed", None, {"reason": reason})
        return bool(changed)

    def note_unreachable(self, error: str, alert_after: float) -> None:
        now = self.clock.now()
        since = self.flag("es_unreachable_since")
        if since is None:
            with self.store.tx("reachability") as db:
                set_flag(db, "es_unreachable_since", now)
        elif now - since >= alert_after and not self.flag("es_unreachable_alerted"):
            with self.store.tx("reachability") as db:
                set_flag(db, "es_unreachable_alerted", True)
                record_anomaly(db, now, "es_unreachable", "", {"since": since, "error": error[:500]})

    def note_reachable(self) -> None:
        if self.flag("es_unreachable_since") is None:
            return
        now = self.clock.now()
        with self.store.tx("reachability") as db:
            set_flag(db, "es_unreachable_since", None)
            set_flag(db, "es_unreachable_alerted", None)
            resolve_anomalies(db, now, "es_unreachable", "")

    # ---- reading observations ----------------------------------------------------------------------

    def observations(self, after_seq: int = 0, limit: int | None = None) -> list[Observation]:
        """Booked observations in ingest order (``metadata['ingest_seq']`` is monotonic)."""
        sql = "SELECT observation_json FROM observations WHERE ingest_seq > ? ORDER BY ingest_seq"
        args: tuple = (after_seq,)
        if limit is not None:
            sql, args = sql + " LIMIT ?", (after_seq, int(limit))
        return [observation_from_json(row[0]) for row in self.store.rows(sql, args)]
