"""Shared experiment history wired into the paced controller (design 5.5, 5.8, 5.9).

``HistoryBridge`` owns the agent-visible history of one campaign (``<dir>/log.jsonl``, the host visibility
record ``<dir>/visibility.jsonl``) and connects it to the controller:

* the built-in **publish-and-notify hook** (ordered before the pacing hook): controller events -> idempotent
  ``HistoryPublisher.sync()`` -> ``publish_history_head`` to every research client that is running a turn ->
  pending task messages pushed to the running turn;
* a **lane extension**: the frozen round input gets ``watermarks.base_history_seq`` (the log head after publishing
  every event up to ``base_event_seq``), the research sessions' previous cursors and visibility watermarks, the
  abandoned generations whose session state is unknown (``unsettled``) and the task messages not yet delivered;
  the start / end / abandon transactions keep ``agent_messages`` delivery marks (at least once);
* ``ReservoirProposalOutcomes``: the ``proposals`` view of the history tool, from the occurrence records.

Watermark units. Pacing counts booked results (``ingest_seq``, one per observation). The history log numbers
``result`` and ``selected`` entries together (``seq``) and also records terminal outcomes without an observation
(``canceled``, ``result_invalid``). The mapping is a pure function of the immutable log prefix::

    ingest_watermark(log, h) = #{e in log, e.seq <= h, e.kind == result, e.status in OBSERVATION_STATUSES}

Observations are logged in booking order, so this is the largest ``ingest_seq`` booked at or before entry ``h``.
"""
from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ldm_tts.harness.history import RESULT, ControllerHistorySource, HistoryLog, HistoryProjection, HistoryPublisher
from ldm_tts.harness.history_host import HistoryService
from ldm_tts.harness.protocol import HarnessHistoryConfig, canonical_sha256

from .events import EffectHook
from .ledger import candidate_from_json
from .proposals import results_booked
from .reservoir import Occurrence, occurrence_weight
from .wire import canonical

OBSERVATION_STATUSES = frozenset({"succeeded", "invalid", "timed_out", "failed"})
PUBLISH_EVENTS = ("experiment.planned", "observation.ingested", "experiment.finished", "anomaly.invalid_result",
                  "agent.message", "generation.started")
# Occurrence removal reason -> proposal status of the history tool (``PROPOSAL_STATUSES``).
_REMOVED = {"renewed": "retired", "replaced": "retired", "evicted": "retired", "capacity": "retired",
            "withdrawn": "withdrawn", "blocked": "not_admitted"}


def ingest_watermark(log: HistoryLog, history_seq: int) -> int:
    """Booked results (pacing units) visible at history ``seq``; canceled, result_invalid and selected entries
    do not count."""
    return sum(1 for e in log.entries(0, history_seq) if e.kind == RESULT and e.status in OBSERVATION_STATUSES)


def history_seq_at_event(log: HistoryLog, event_seq: int) -> int:
    """Largest log ``seq`` published from controller events up to ``event_seq`` (the log is in event order)."""
    seq = 0
    for entry in log.entries():
        if int(entry.source.rsplit(":", 1)[1]) > event_seq:
            break
        seq = entry.seq
    return seq


class ReservoirProposalOutcomes:
    """``ProposalOutcomes`` from the occurrence records: one row per immutable occurrence id. Status at response
    time: live -> ``in_reservoir`` (weight, results since the proposal's watermark); selected -> ``pending`` while
    in flight, ``measured`` once measured (``retired`` with reason ``selected_canceled`` if it ended unmeasured);
    other removals per ``_REMOVED`` with the removal reason."""

    def __init__(self, bridge: "HistoryBridge"):
        self.bridge = bridge

    def rows(self) -> list[dict[str, Any]]:
        controller = self.bridge.controller
        if controller is None:
            return []
        store, maintenance = controller.store, controller.maintenance
        with store.lock:
            db = store.db
            occurrences = db.execute("SELECT * FROM occurrences ORDER BY id").fetchall()
            booked = results_booked(db)
            pending = {r[0] for r in db.execute("SELECT canonical_key FROM experiments WHERE pending=1")}
            measured = {r[0] for r in db.execute("""SELECT canonical_key FROM experiments
                WHERE disposition IN ('ingested','invalid')""")}
        half_life = getattr(maintenance, "half_life", None)
        out = []
        for r in occurrences:
            candidate = candidate_from_json(r["candidate_json"])
            row = {"proposal_id": r["id"], "lineage": r["lineage"], "key": r["canonical_key"],
                   "candidate_id": candidate.candidate_id, "generation": r["generation"],
                   "fields": self.bridge.projection.project({"candidate": candidate.to_dict()}, candidate_only=True)}
            reason = r["removed_reason"]
            if r["removed_at"] is None:
                occurrence = Occurrence(candidate, r["lineage"], r["generation"], r["watermark"])
                row.update(status="in_reservoir", weight=round(occurrence_weight(occurrence, booked, half_life), 6),
                           results_since=booked - r["watermark"])
            elif reason in ("selected", "measured"):
                key = r["canonical_key"]
                row["status"] = "pending" if key in pending else "measured" if key in measured else "retired"
                if row["status"] == "retired":
                    row["reason"] = "selected_canceled"
            else:
                row.update(status=_REMOVED.get(reason, "retired"), reason=reason,
                           results_since=(r["removed_results"] or 0) - r["watermark"])
            out.append(row)
        return out


class HistoryBridge:
    """One per campaign: log, host service, publish hook and lane extension. Construct it, pass it to
    ``PacedController(history=...)`` (which calls ``attach``), and register research clients with ``add_client``."""

    name = "history"
    version = "1"

    def __init__(self, directory: Path | str, projection, *, config: HarnessHistoryConfig | None = None,
                 lineage_of: Callable[[str], str] = lambda profile_id: profile_id,
                 clock: Callable[[], float] = time.time):
        self.directory = Path(directory)
        self.projection = HistoryProjection.from_declaration(projection)
        self.log = HistoryLog(self.directory / "log.jsonl")
        self.service = HistoryService(self.log, self.projection, config=config,
                                      record_path=self.directory / "visibility.jsonl",
                                      proposals=ReservoirProposalOutcomes(self), state_of=self.state_of,
                                      lineage_of=lineage_of, clock=clock)
        self.controller = None
        self.publisher: HistoryPublisher | None = None
        self.clients: list = []

    def identity(self) -> str:
        """Frozen with the campaign: a different projection would rewrite the published log."""
        return f"{self.name}@{self.version}:{canonical_sha256(self.projection.to_dict())[:16]}"

    def attach(self, controller) -> None:
        self.controller = controller
        self.publisher = HistoryPublisher(self.log, ControllerHistorySource.from_controller(controller, self.projection))

    def add_client(self, client):
        """A ``HarnessClient`` (or anything with ``publish_history_head`` / ``publish_task_message``)."""
        self.clients.append(client)
        return client

    def hook(self) -> EffectHook:
        return EffectHook(self.name, PUBLISH_EVENTS, self._effect, builtin=True)

    # ---- the publish-and-notify hook (control thread) -------------------------------------------------------

    def sync(self) -> int:
        if self.publisher is None:
            raise RuntimeError("history bridge is not attached to a controller")
        return self.publisher.sync()

    def _effect(self, event, view) -> None:
        """Idempotent per event: the sync resumes from the log's last source token, a head hint is only a hint,
        and the sidecar ignores a task message it has already queued or delivered."""
        head = self.sync()
        for client in self.clients:
            client.publish_history_head(head)
        self.push_messages()

    def push_messages(self) -> int:
        """Undelivered task messages for the running generation go to every client running a turn."""
        lane = getattr(self.controller, "lane", None)
        row = None if lane is None else lane.latest()
        if row is None or row["state"] != "running":
            return 0
        messages = self.controller.store.rows("""SELECT request_id, source, text FROM agent_messages
            WHERE delivered_at IS NULL AND target_generation <= ? ORDER BY id""", (row["generation"],))
        sent = 0
        for message in messages:
            for client in self.clients:
                sent += bool(client.publish_task_message(message["request_id"], message["text"],
                                                         source=message["source"]))
        return sent

    def state_of(self, ref: str) -> str:
        """Coarse state of a pending experiment for the history tool (read on the client thread)."""
        found = self.controller.store.row("SELECT es_state FROM experiments WHERE experiment_id=?", (ref,)) \
            if self.controller is not None else None
        return "running" if found is not None and found["es_state"] == "RUNNING" else "queued"

    # ---- lane extension (control thread, inside the lane's lock / transactions) ------------------------------

    def extend_round_input(self, db, generation: int, round_input: dict) -> None:
        self.sync()
        watermarks = round_input["watermarks"]
        base_seq = history_seq_at_event(self.log, watermarks["base_event_seq"])
        if ingest_watermark(self.log, base_seq) != watermarks["base_ingest_seq"]:
            raise RuntimeError(f"history log {self.log.path} disagrees with the controller at event "
                               f"{watermarks['base_event_seq']} (another campaign's log?)")
        watermarks["base_history_seq"] = base_seq
        round_input["sessions"] = self.sessions(db)
        unsettled = self.unsettled(db)
        if unsettled:
            round_input["unsettled"] = unsettled
        round_input["messages"] = [
            {"id": r["request_id"], "source": r["source"], "text": r["text"]}
            for r in db.execute("""SELECT request_id, source, text FROM agent_messages
                WHERE delivered_at IS NULL AND target_generation <= ? ORDER BY id""", (generation,)).fetchall()]

    @staticmethod
    def sessions(db) -> dict[str, dict]:
        """Each research session's state after its latest committed turn (``metadata.sessions`` of ready or
        rejected generations): the next turn continues its cursor chain and counts its header from there."""
        sessions: dict[str, dict] = {}
        for row in db.execute("""SELECT generation, outcome_json FROM generations WHERE state IN ('ready','rejected')
                ORDER BY generation""").fetchall():
            metadata = json.loads(row["outcome_json"] or "{}").get("metadata") or {}
            for profile, state in (metadata.get("sessions") or {}).items():
                sessions[profile] = {"generation": row["generation"], "turn_id": state.get("turn_id"),
                                     "history_to_seq": int(state["history_to_seq"]),
                                     "visible_seq": int(state["visible_seq"])}
        return dict(sorted(sessions.items()))

    @staticmethod
    def unsettled(db) -> list[int]:
        """Abandoned generations after the newest committed one: whether their turns committed on the sidecar is
        unknown to the controller, so the research source settles the session cursors before the next turn."""
        return [r[0] for r in db.execute("""SELECT generation FROM generations WHERE state='abandoned' AND generation >
            (SELECT COALESCE(MAX(generation), 0) FROM generations WHERE state IN ('ready','rejected'))
            ORDER BY generation""").fetchall()]

    def on_start(self, db, generation: int, round_input: dict, now: float) -> None:
        """Messages frozen into the turn input count as delivered (undone if the generation is abandoned)."""
        mark = canonical({"via": "turn_input", "generation": generation})
        for message in round_input.get("messages", ()):
            db.execute("UPDATE agent_messages SET delivered_at=?, delivery_json=? WHERE request_id=? "
                       "AND delivered_at IS NULL", (now, mark, message["id"]))

    def on_finish(self, db, generation: int, outcome, now: float) -> None:
        """Messages that every session of the turn received between steps (``metadata.task_messages``)."""
        mark = canonical({"via": "steer", "generation": generation})
        for request_id in (outcome.metadata or {}).get("task_messages") or ():
            db.execute("UPDATE agent_messages SET delivered_at=?, delivery_json=? WHERE request_id=? "
                       "AND delivered_at IS NULL", (now, mark, str(request_id)))

    def on_abandon(self, db, generation: int, now: float) -> None:
        mark = canonical({"via": "turn_input", "generation": generation})
        db.execute("UPDATE agent_messages SET delivered_at=NULL, delivery_json=NULL WHERE delivery_json=?", (mark,))


__all__ = ["HistoryBridge", "OBSERVATION_STATUSES", "PUBLISH_EVENTS", "ReservoirProposalOutcomes",
           "history_seq_at_event", "ingest_watermark"]
