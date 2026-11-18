"""Talking to the ES: outbox (PUT/cancel), change-feed ingestion and anti-entropy (design 3.5, 4.1, 5.1).

Rhythm everywhere: read rows (no transaction) -> HTTP -> one transaction applying the answer.
Each function returns True when it changed controller state.
"""
from __future__ import annotations

import logging

from .config import RuntimeOptions
from .es_client import ESBackpressure, ESClient, ESRejected, ESUnavailable
from .ledger import Ledger
from .store import get_cursor, set_cursor, set_flag
from .wire import TERMINAL_SQL, revision_number

log = logging.getLogger(__name__)
FEED_CURSOR = "changes"


class Reconciler:
    def __init__(self, ledger: Ledger, client: ESClient, options: RuntimeOptions):
        self.ledger, self.client, self.options = ledger, client, options
        self.store, self.clock = ledger.store, ledger.clock

    # ---- outbox -----------------------------------------------------------------------------------

    def send_puts(self) -> bool:
        now = self.clock.now()
        backoff = self.ledger.flag("es_backoff_until")
        if backoff is not None and backoff > now:
            return False
        rows = self.store.rows("""SELECT experiment_id, request_body, attempts FROM experiments
            WHERE submit_state='PLANNED' AND cancel_requested_at IS NULL AND written_off_at IS NULL AND next_attempt_at <= ?
            ORDER BY decision_id""", (now,))
        changed = False
        for row in rows:
            experiment_id = row["experiment_id"]
            try:
                status, resource = self.client.put_experiment(experiment_id, row["request_body"].encode("utf-8"))
            except ESBackpressure as exc:
                until = self.clock.now() + exc.retry_after
                self.ledger.record_put_retry(experiment_id, str(exc), exc.retry_after, backoff_until=until)
                return True
            except ESUnavailable as exc:
                self.ledger.record_put_retry(experiment_id, str(exc), self.options.backoff(row["attempts"] + 1))
                self.ledger.note_unreachable(str(exc), self.options.unreachable_alert_seconds)
                return True  # the ES is likely down: leave the remaining rows for their next attempt
            except ESRejected as exc:
                self.ledger.record_put_rejected(experiment_id, exc.status, exc.problem, str(exc))
                changed = True
                continue
            self.store.check("put.after_send")
            self.ledger.note_reachable()
            with self.store.tx("ack") as db:
                self.ledger.apply_resource(db, resource, f"put_{status}")
            changed = True
        return changed

    def send_cancels(self) -> bool:
        now = self.clock.now()
        rows = self.store.rows(f"""SELECT experiment_id, cancel_attempts FROM experiments
            WHERE cancel_requested_at IS NOT NULL AND cancel_acked_at IS NULL
            AND (es_state IS NULL OR es_state NOT IN {TERMINAL_SQL}) AND COALESCE(cancel_next_attempt_at, 0) <= ?
            ORDER BY decision_id""", (now,))
        changed = False
        for row in rows:
            experiment_id = row["experiment_id"]
            try:
                status, resource = self.client.cancel_experiment(experiment_id, self.ledger.campaign_id)
            except ESUnavailable as exc:
                delay = exc.retry_after if isinstance(exc, ESBackpressure) else self.options.backoff(row["cancel_attempts"] + 1)
                self.ledger.record_cancel_retry(experiment_id, str(exc), delay)
                self.ledger.note_unreachable(str(exc), self.options.unreachable_alert_seconds)
                return True
            except ESRejected as exc:
                self.ledger.record_cancel_retry(experiment_id, str(exc), self.options.retry_max_seconds)
                with self.store.tx("reject") as db:
                    self.ledger.anomaly(db, "cancel_rejected", experiment_id, {"status": exc.status, "problem": exc.problem})
                changed = True
                continue
            self.store.check("cancel.after_send")
            self.ledger.note_reachable()
            self.ledger.record_cancel_response(status, resource)
            changed = True
        return changed

    # ---- change feed (design 3.5): cursor advances in the same transaction as the ingestion ----------

    def pull_changes(self) -> bool:
        changed = False
        for _ in range(self.options.max_pages_per_step):
            with self.store.lock:
                cursor = get_cursor(self.store.db, FEED_CURSOR)
            try:
                page = self.client.changes(self.ledger.campaign_id, cursor, limit=self.options.page_limit)
            except ESUnavailable as exc:
                self.ledger.note_unreachable(str(exc), self.options.unreachable_alert_seconds)
                return changed
            except ESRejected as exc:
                log.error("change feed rejected: %s", exc)
                return changed
            self.ledger.note_reachable()
            items, new_cursor = page["items"], page["cursor"]
            if not items:  # an empty page only echoes the cursor: nothing to record
                return changed
            with self.store.tx("ingest") as db:
                for item in items:
                    changed = self.ledger.apply_resource(db, item, "feed") or changed
                try:
                    regressed = cursor != "" and revision_number(new_cursor) < revision_number(cursor)
                except ValueError:
                    regressed = True
                if regressed:
                    self.ledger.anomaly(db, "cursor_regressed", "", {"old": cursor, "new": new_cursor}, pause=True)
                else:
                    set_cursor(db, FEED_CURSOR, new_cursor)
            if regressed or not page.get("more"):
                return changed
        return changed

    # ---- anti-entropy (design 3.5, 4.3): full listing + GET of locally live ids missing from it ------

    def anti_entropy(self, force: bool = False) -> bool:
        now = self.clock.now()
        due = self.ledger.flag("anti_entropy_due", 0.0)
        if not force and now < due:
            return False
        changed, seen, cursor = False, set(), ""
        try:
            while True:
                page = self.client.changes(self.ledger.campaign_id, cursor, limit=self.options.page_limit)
                if page["items"]:
                    with self.store.tx("anti_entropy") as db:
                        for item in page["items"]:
                            seen.add(item["experiment_id"])
                            changed = self.ledger.apply_resource(db, item, "anti_entropy") or changed
                if not page.get("more") or not page["items"]:
                    break
                cursor = page["cursor"]
            live = self.store.rows(f"""SELECT experiment_id FROM experiments WHERE submit_state='ACKED'
                AND (es_state IS NULL OR es_state NOT IN {TERMINAL_SQL}) ORDER BY decision_id""")
            for row in live:
                if row["experiment_id"] in seen:
                    continue
                resource = self.client.get_experiment(row["experiment_id"])
                with self.store.tx("anti_entropy") as db:
                    if resource is None:
                        changed = self.ledger.anomaly(db, "lost_experiment", row["experiment_id"], {}, pause=True) or changed
                    else:
                        changed = self.ledger.apply_resource(db, resource, "anti_entropy") or changed
        except ESUnavailable as exc:
            self.ledger.note_unreachable(str(exc), self.options.unreachable_alert_seconds)
            self._reschedule(min(self.options.anti_entropy_interval, self.options.retry_max_seconds))
            return changed
        except ESRejected as exc:
            log.error("anti-entropy rejected: %s", exc)
        self.ledger.note_reachable()
        self._reschedule(self.options.anti_entropy_interval)
        return changed

    def _reschedule(self, delay: float) -> None:
        with self.store.tx("schedule") as db:
            set_flag(db, "anti_entropy_due", self.clock.now() + delay)
