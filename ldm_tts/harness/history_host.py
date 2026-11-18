"""Host side of the sidecar experiment-history component: query responder, submission CAS, visibility record.

The sidecar never reads the log itself; it asks the host (``history_query_requested``) and the host
answers from the shared ``HistoryLog``. Every answer carries the log head and, when the session's
visibility watermark is behind, a bounded notice of the gap; the host appends a visibility record line
before the answer is sent. At submission time the host compares the session's ``visible_seq`` with the
head: behind means "look first, then submit" without calling the task validator (design 5.8).
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ldm_tts.harness.history import HistoryLog, HistoryProjection
from ldm_tts.harness.history_query import HistoryQueryEngine, HistoryQueryError, ProposalOutcomes, QueryLimits
from ldm_tts.harness.protocol import HarnessError, HarnessHistoryConfig, canonical_sha256

OPERATIONS = ("query", "notify", "head")


def history_config(projection: HistoryProjection, **options: Any) -> HarnessHistoryConfig:
    """Initialization-frame history configuration from a task projection."""
    objective = projection.objective
    return HarnessHistoryConfig(fields=tuple(item.to_dict() for item in projection.fields),
                                objective=None if objective is None else objective.name, **options)


@dataclass(frozen=True)
class HistoryAdmission:
    stale: bool
    head_seq: int
    history_seq: int | None = None
    notice: Mapping[str, Any] | None = None
    forced: bool = False


class HistoryService:
    """One per harness pool. Thread-safe: queries arrive on the client thread while the controller appends."""

    def __init__(self, log: HistoryLog, projection: HistoryProjection | Mapping[str, Any], *,
                 config: HarnessHistoryConfig | None = None, record_path: Path | str | None = None,
                 proposals: ProposalOutcomes | None = None, state_of: Callable[[str], str] | None = None,
                 lineage_of: Callable[[str], str] = lambda profile_id: profile_id,
                 clock: Callable[[], float] = time.time):
        self.log = log
        self.projection = HistoryProjection.from_declaration(projection)
        self.config = config or history_config(self.projection)
        self.engine = HistoryQueryEngine(
            log, self.projection, proposals=proposals, state_of=state_of, tool_name=self.config.tool_name,
            limits=QueryLimits(self.config.page_size_default, self.config.page_size_max,
                               self.config.response_max_bytes, self.config.inline_limit))
        self.record_path = None if record_path is None else Path(record_path)
        self.lineage_of, self.clock = lineage_of, clock
        self._lock = threading.Lock()

    # ---- lane inputs (part B) --------------------------------------------------------------------------------

    def baseline(self) -> int:
        """Round baseline ``base_seq`` = current head; becomes ``HarnessTurn.history_to_seq``."""
        return self.log.head

    def turn_window(self, from_seq: int, base_seq: int | None = None) -> tuple[int, int, str]:
        """``(history_from_seq, history_to_seq, history_digest)`` for a ``HarnessTurn`` that continues the
        session cursor chain at ``from_seq`` (the previous turn's ``history_to_seq``)."""
        base_seq = self.log.head if base_seq is None else base_seq
        return from_seq, base_seq, self.log.digest(from_seq, base_seq)

    # ---- sidecar requests -------------------------------------------------------------------------------------

    def handle(self, profile_id: str, turn_id: str, operation: str, query: Mapping[str, Any] | None,
               visible_seq: int) -> dict[str, Any]:
        """Answer one ``history_query_requested`` frame: ``{"headSeq", "result", "notice", "error"}``."""
        if operation not in OPERATIONS:
            raise HarnessError(f"unknown history operation {operation!r}")
        head = self._check_visible(visible_seq)
        result: dict[str, Any] | None = None
        error = None
        # Agent-controlled input (queries, cursors) and task callbacks must never take the turn down:
        # every engine failure becomes an error answer the agent can read.
        if operation == "query":
            try:
                result = self.engine.query(query or {}, lineage=self.lineage_of(profile_id))
                head = result["head"]
            except HistoryQueryError as exc:
                error = {"code": exc.code, "message": str(exc)}
            except Exception as exc:  # noqa: BLE001
                result, error = None, {"code": "internal_error",
                                       "message": f"The history query failed on the host ({type(exc).__name__})."}
        notice = None
        if operation != "head" and visible_seq < head:
            try:
                notice = self.engine.notice(visible_seq, head)
            except Exception as exc:  # noqa: BLE001 - no notice: the watermark simply does not advance
                error = error or {"code": "internal_error",
                                  "message": f"The history notice failed on the host ({type(exc).__name__})."}
        # ``served_to`` is what the host answered, not what the agent saw; delivery is the sidecar's to confirm.
        self.record(profile_id, turn_id, operation, visible=visible_seq, head=head,
                    view=None if result is None else result["view"],
                    watermark=None if result is None else result["watermark"],
                    served_to=None if notice is None else notice["to_seq"],
                    notice_sha256=None if notice is None else canonical_sha256(notice["text"]),
                    error=None if error is None else error["code"])
        return {"headSeq": head, "result": result, "notice": notice, "error": error}

    def admit(self, profile_id: str, turn_id: str, attempt_index: int, visible_seq: int,
              cas_retries: int) -> HistoryAdmission:
        """Compare-and-swap before validation. Stale answers do not reach the validator and are not
        counted as attempts; after ``max_cas_retries`` consecutive stale answers validate at the head."""
        head = self._check_visible(visible_seq)
        if visible_seq < head and cas_retries < self.config.max_cas_retries:
            notice = self.engine.notice(visible_seq, head)
            self.record(profile_id, turn_id, "cas_stale", attempt=attempt_index, visible=visible_seq, head=head,
                        cas_retries=cas_retries, served_to=head, notice_sha256=canonical_sha256(notice["text"]))
            return HistoryAdmission(True, head, notice=notice)
        forced = visible_seq < head
        self.record(profile_id, turn_id, "validate", attempt=attempt_index, visible=visible_seq, head=head,
                    history_seq=head, forced=forced, cas_retries=cas_retries)
        return HistoryAdmission(False, head, history_seq=head, forced=forced)

    def turn_end(self, profile_id: str, turn_id: str, visible_seq: int | None, *, replayed: bool = False,
                 task_messages: tuple[str, ...] = ()) -> None:
        self.record(profile_id, turn_id, "turn_end", visible=visible_seq, replayed=replayed,
                    task_messages=list(task_messages) or None)

    def turn_sent(self, turn) -> None:
        """Record the exact ``HarnessTurn`` about to be sent (so a later generation can replay it verbatim)."""
        self.record(turn.profile_id, turn.turn_id, "turn_sent", turn=turn.to_dict())

    def sent_turn(self, profile_id: str, turn_id: str) -> dict[str, Any] | None:
        return next((item["turn"] for item in reversed(self.records(profile_id, turn_id))
                     if item["kind"] == "turn_sent"), None)

    def terminal_decision(self, profile_id: str, turn_id: str) -> str | None:
        """``accept`` / ``reject_turn`` if the host answered a terminal validation for the turn: only then can the
        sidecar have committed it (a committed turn always ends with such an answer)."""
        return next((item["decision"] for item in reversed(self.records(profile_id, turn_id))
                     if item["kind"] == "decision"), None)

    def accepted_history_seq(self, profile_id: str, turn_id: str) -> int | None:
        """Watermark recorded for the turn's last validated attempt (the accepted one when the turn accepted)."""
        return next((item["history_seq"] for item in reversed(self.records(profile_id, turn_id))
                     if item["kind"] == "validate"), None)

    def _check_visible(self, visible_seq: int) -> int:
        head = self.log.head
        if isinstance(visible_seq, bool) or not isinstance(visible_seq, int) or not 0 <= visible_seq <= head:
            raise HarnessError(f"sidecar visibility watermark {visible_seq!r} is outside the history (head {head})")
        return head

    # ---- visibility record ------------------------------------------------------------------------------------

    def record(self, profile_id: str, turn_id: str, kind: str, **fields: Any) -> None:
        if self.record_path is None:
            return
        line = json.dumps({"at": self.clock(), "kind": kind, "profile_id": profile_id, "turn_id": turn_id,
                           **{k: v for k, v in fields.items() if v is not None}}, sort_keys=True,
                          separators=(",", ":"))
        with self._lock:
            self.record_path.parent.mkdir(parents=True, exist_ok=True)
            with self.record_path.open("a", encoding="utf-8") as stream:
                stream.write(line + "\n")
                stream.flush()
                os.fsync(stream.fileno())

    def records(self, profile_id: str | None = None, turn_id: str | None = None) -> list[dict[str, Any]]:
        if self.record_path is None or not self.record_path.exists():
            return []
        out = []
        for line in self.record_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                if profile_id in (None, item["profile_id"]) and turn_id in (None, item["turn_id"]):
                    out.append(item)
        return out

    def last_visible(self, profile_id: str) -> int:
        """Largest watermark the sidecar reported as seen by this profile (its requests and turn results);
        notices the host merely served (``served_to``) do not count — the sidecar may have crashed first."""
        return max((item["visible"] for item in self.records(profile_id) if isinstance(item.get("visible"), int)),
                   default=0)


__all__ = ["HistoryAdmission", "HistoryService", "OPERATIONS", "history_config"]
