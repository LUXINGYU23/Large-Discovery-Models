"""Append-only experiment history log, watermark views and task history projections (design 5.8).

The log is the agent-visible history: one monotonic ``seq`` shared by two entry kinds, ``result``
(an experiment reached a terminal state) and ``selected`` (a candidate was chosen and became pending).
Its canonical order is booking order, not submission order, so "seq > w" is exactly "learned after w".
Every view at a watermark ``w`` (measured keys, pending set, query pages) is a pure function of the
prefix ``seq <= w``, which never changes once published.
"""

from __future__ import annotations

import json
import math
import os
import threading
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from ldm_tts.harness.protocol import HarnessSubmissionError, canonical_sha256

RESULT, SELECTED = "result", "selected"
# Result statuses: the four ``EvaluationResult`` values plus terminal outcomes that produce no observation.
MEASURED_STATUSES = frozenset({"succeeded", "invalid", "timed_out", "failed", "result_invalid"})
RESULT_STATUSES = MEASURED_STATUSES | {"canceled"}
FIELD_TYPES = frozenset({"integer", "number", "string", "boolean", "json"})
# Error codes of history-relative submission checks (design 5.8 "提交工具").
INVALID, ALREADY_MEASURED, ALREADY_PENDING, DUPLICATE_IN_SUBMISSION, COUNT_OUT_OF_RANGE = (
    "invalid", "already_measured", "already_pending", "duplicate_in_submission", "count_out_of_range")


# ---- projection contract ------------------------------------------------------------------------------------

@dataclass(frozen=True)
class HistoryField:
    name: str
    type: str
    source: str
    filterable: bool = False
    sortable: bool = False
    direction: str | None = None
    minimum: float | None = None
    maximum: float | None = None
    values: tuple[Any, ...] = ()
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name.isidentifier() or self.name in _RESERVED:
            raise ValueError(f"history field name is invalid or reserved: {self.name!r}")
        if self.type not in FIELD_TYPES:
            raise ValueError(f"history field {self.name} has unsupported type {self.type!r}")
        if self.type == "json" and (self.filterable or self.sortable):
            raise ValueError(f"history field {self.name} of type json cannot be filtered or sorted")
        if self.direction not in (None, "maximize", "minimize"):
            raise ValueError(f"history field {self.name} has invalid direction")
        if not self.source or any(not part for part in self.source.replace("[", ".[").split(".")):
            raise ValueError(f"history field {self.name} has an invalid source path")

    def read(self, record: Mapping[str, Any]) -> Any:
        value: Any = record
        for part in self.source.replace("[", ".[").split("."):
            if part.startswith("["):
                index = int(part[1:-1])
                value = value[index] if isinstance(value, list) and -len(value) <= index < len(value) else None
            else:
                value = value.get(part) if isinstance(value, Mapping) else None
            if value is None:
                return None
        return _coerce(self, value)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"name": self.name, "type": self.type, "filterable": self.filterable,
                               "sortable": self.sortable}
        for key in ("direction", "minimum", "maximum", "description"):
            if getattr(self, key) not in (None, ""):
                out[key] = getattr(self, key)
        if self.values:
            out["values"] = list(self.values)
        return out


_RESERVED = frozenset({"seq", "kind", "status", "key", "candidate_id", "generation", "lineage", "state"})


def _coerce(spec: HistoryField, value: Any) -> Any:
    if spec.type == "integer":
        return value if isinstance(value, int) and not isinstance(value, bool) else None
    if spec.type == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            return None
        return float(value)
    if spec.type == "string":
        return value if isinstance(value, str) else None
    if spec.type == "boolean":
        return value if isinstance(value, bool) else None
    return value


@dataclass(frozen=True)
class HistoryProjection:
    """Task declaration: record fields, which are filterable/sortable, the objective and visible artifacts."""

    fields: tuple[HistoryField, ...]
    artifacts: tuple[str, ...] = ()
    version: int = 1

    def __post_init__(self) -> None:
        names = [item.name for item in self.fields]
        if len(set(names)) != len(names):
            raise ValueError("history projection field names must be unique")
        if sum(1 for item in self.fields if item.direction) > 1:
            raise ValueError("history projection declares at most one objective field (with a direction)")

    @classmethod
    def from_declaration(cls, value: "HistoryProjection | Mapping[str, Any]") -> "HistoryProjection":
        """Accept the manifest ``history_projection`` dict (numeric-async section 4 format)."""
        if isinstance(value, HistoryProjection):
            return value
        unknown = set(value) - {"version", "fields", "artifacts"}
        if unknown or value.get("version", 1) != 1:
            raise ValueError(f"unsupported history projection declaration: {sorted(unknown) or value.get('version')}")
        fields = []
        for raw in value.get("fields", ()):
            extra = set(raw) - {"name", "type", "source", "filterable", "sortable", "direction", "minimum",
                                "maximum", "values", "description"}
            if extra:
                raise ValueError(f"history field has unknown keys: {sorted(extra)}")
            fields.append(HistoryField(**{**raw, "values": tuple(raw.get("values", ()))}))
        return cls(tuple(fields), tuple(value.get("artifacts", ())), 1)

    @property
    def objective(self) -> HistoryField | None:
        return next((item for item in self.fields if item.direction), None)

    def field(self, name: str) -> HistoryField | None:
        return next((item for item in self.fields if item.name == name), None)

    def project(self, record: Mapping[str, Any], *, candidate_only: bool = False) -> dict[str, Any]:
        return {item.name: item.read(record) for item in self.fields
                if not candidate_only or item.source.startswith("candidate.")}

    def to_dict(self) -> dict[str, Any]:
        return {"version": self.version, "fields": [item.to_dict() for item in self.fields],
                "artifacts": list(self.artifacts)}


# ---- log entries ----------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class HistoryRecord:
    """A log entry before it has a ``seq``. ``ref`` pairs a selection with its result; it is internal."""

    kind: str
    ref: str
    key: str
    candidate_id: str
    status: str
    fields: Mapping[str, Any] = field(default_factory=dict)
    generation: int | None = None
    lineage: Mapping[str, Any] = field(default_factory=dict)
    source: str = ""

    def __post_init__(self) -> None:
        if self.kind not in (RESULT, SELECTED):
            raise ValueError(f"history entry kind must be result or selected, not {self.kind!r}")
        if self.kind == SELECTED and self.status != "selected":
            raise ValueError("selected history entries have status 'selected'")
        if self.kind == RESULT and self.status not in RESULT_STATUSES:
            raise ValueError(f"unknown history result status {self.status!r}")
        if not self.ref or not self.key:
            raise ValueError("history entries require ref and key")
        object.__setattr__(self, "fields", dict(self.fields))
        object.__setattr__(self, "lineage", dict(self.lineage))


@dataclass(frozen=True)
class HistoryEntry(HistoryRecord):
    seq: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {"seq": self.seq, "kind": self.kind, "ref": self.ref, "key": self.key,
                "candidate_id": self.candidate_id, "status": self.status, "fields": dict(self.fields),
                "generation": self.generation, "lineage": dict(self.lineage), "source": self.source}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "HistoryEntry":
        return cls(**dict(value))

    def record(self) -> HistoryRecord:
        return HistoryRecord(self.kind, self.ref, self.key, self.candidate_id, self.status, self.fields,
                             self.generation, self.lineage, self.source)


@dataclass(frozen=True)
class HistoryView:
    """Everything known at watermark ``w``: reproducible from the immutable log prefix."""

    watermark: int
    entries: tuple[HistoryEntry, ...]

    @property
    def results(self) -> tuple[HistoryEntry, ...]:
        return tuple(e for e in self.entries if e.kind == RESULT)

    @property
    def pending(self) -> tuple[HistoryEntry, ...]:
        finished = {e.ref for e in self.entries if e.kind == RESULT}
        return tuple(e for e in self.entries if e.kind == SELECTED and e.ref not in finished)

    @property
    def measured_keys(self) -> frozenset[str]:
        return frozenset(e.key for e in self.results if e.status in MEASURED_STATUSES)

    @property
    def pending_keys(self) -> frozenset[str]:
        return frozenset(e.key for e in self.pending)


def _write(fd: int, data: memoryview) -> int:
    return os.write(fd, data)  # a seam for failure-injection tests


class HistoryLog:
    """Append-only JSONL log with an in-memory index. Readers trust only complete lines."""

    def __init__(self, path: Path | str | None = None):
        self.path = None if path is None else Path(path)
        self._entries: list[HistoryEntry] = []
        self._sources: dict[str, int] = {}
        self._lock = threading.RLock()
        self._listeners: list[Callable[[int], None]] = []
        if self.path is not None and self.path.exists():
            with self.path.open("rb") as stream:
                body = stream.read()
            complete = body[:body.rfind(b"\n") + 1]
            if len(complete) != len(body):
                with self.path.open("r+b") as stream:  # drop a torn final line (crash mid-append)
                    stream.truncate(len(complete))
            for line in complete.splitlines():
                self._index(HistoryEntry.from_dict(json.loads(line)))

    def _index(self, entry: HistoryEntry) -> None:
        if entry.seq != len(self._entries) + 1:
            raise ValueError(f"history log seq gap: expected {len(self._entries) + 1}, found {entry.seq}")
        self._entries.append(entry)
        if entry.source:
            self._sources[entry.source] = entry.seq

    @property
    def head(self) -> int:
        with self._lock:
            return len(self._entries)

    def subscribe(self, listener: Callable[[int], None]) -> None:
        """Called with the new head after every append that advanced it (the B-part push path)."""
        self._listeners.append(listener)

    def append(self, records: Iterable[HistoryRecord]) -> int:
        """Append in order; a record whose ``source`` is already logged must be identical (idempotent
        at-least-once delivery) and is skipped. Returns the new head.

        All or nothing: the batch is validated and made durable before any entry becomes visible to
        readers, listeners or later appends; a failed write is truncated back off the file."""
        with self._lock:
            before, batch, sources = len(self._entries), [], {}
            for record in records:
                if record.source and (record.source in self._sources or record.source in sources):
                    known = (self._entries[self._sources[record.source] - 1] if record.source in self._sources
                             else batch[sources[record.source]])
                    if known.record() != record:
                        raise ValueError(f"history source {record.source!r} was already published differently")
                    continue
                if record.source:
                    sources[record.source] = len(batch)
                batch.append(HistoryEntry(**record.__dict__, seq=before + len(batch) + 1))
            if batch and self.path is not None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                body = "".join(json.dumps(e.to_dict(), sort_keys=True, separators=(",", ":")) + "\n"
                               for e in batch).encode("utf-8")
                fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
                try:
                    size = os.fstat(fd).st_size
                    try:
                        view = memoryview(body)
                        while view:
                            view = view[_write(fd, view):]
                        os.fsync(fd)
                    except BaseException:
                        os.ftruncate(fd, size)
                        raise
                finally:
                    os.close(fd)
            for entry in batch:
                self._index(entry)
            head = len(self._entries)
        if head > before:
            for listener in list(self._listeners):
                listener(head)
        return head

    def entries(self, after: int = 0, upto: int | None = None) -> tuple[HistoryEntry, ...]:
        with self._lock:
            upto = len(self._entries) if upto is None else upto
            if not 0 <= after <= upto <= len(self._entries):
                raise ValueError(f"history range ({after}, {upto}] is outside the log (head {len(self._entries)})")
            return tuple(self._entries[after:upto])

    def view(self, watermark: int | None = None) -> HistoryView:
        with self._lock:
            watermark = len(self._entries) if watermark is None else watermark
            return HistoryView(watermark, self.entries(0, watermark))

    def last_source(self) -> str | None:
        with self._lock:
            return next((e.source for e in reversed(self._entries) if e.source), None)

    def digest(self, after: int, upto: int) -> str:
        """Digest of the slice ``(after, upto]``; with ``HarnessTurn`` it binds ``history_digest``."""
        return canonical_sha256([entry.to_dict() for entry in self.entries(after, upto)])


# ---- data source and publisher --------------------------------------------------------------------------------

class HistorySource(Protocol):
    """Where log records come from. ``read(after)`` returns records newer than a source cursor, in order;
    each record's ``source`` is its cursor token (``"<name>:<int>"``)."""

    def read(self, after: int) -> Sequence[HistoryRecord]: ...


class HistoryPublisher:
    """Copies new source records into the log. Idempotent; the B-part publish hook calls ``sync``."""

    def __init__(self, log: HistoryLog, source: HistorySource):
        self.log, self.source = log, source

    def cursor(self) -> int:
        token = self.log.last_source()
        return 0 if token is None else int(token.rsplit(":", 1)[1])

    def sync(self) -> int:
        return self.log.append(self.source.read(self.cursor()))


class ControllerHistorySource:
    """Read-only adapter over the async controller's event table.

    ``experiment.planned`` gives a ``selected`` entry; ``observation.ingested`` a result with the
    projected observation; a ``CANCELED`` finish and ``anomaly.invalid_result`` give results without an
    observation (``canceled`` / ``result_invalid``) so that the pending set derived from the log empties.
    """

    def __init__(self, projection: HistoryProjection, *, events: Callable[[int], Sequence[Mapping[str, Any]]],
                 observation: Callable[[int], Any], candidate: Callable[[str], Any], name: str = "event"):
        self.projection, self.name = projection, name
        self._events, self._observation, self._candidate = events, observation, candidate

    @classmethod
    def from_controller(cls, controller, projection: HistoryProjection) -> "ControllerHistorySource":
        from ldm_tts.async_runtime.ledger import candidate_from_json

        def observation(ingest_seq: int):
            found = controller.observations(ingest_seq - 1, 1)
            if not found or found[0].metadata.get("ingest_seq") != ingest_seq:
                raise LookupError(f"observation {ingest_seq} is not booked")
            return found[0]

        def candidate(experiment_id: str):
            row = controller.store.row("SELECT candidate_json FROM experiments WHERE experiment_id=?",
                                       (experiment_id,))
            if row is None:
                raise LookupError(f"unknown experiment {experiment_id}")
            return candidate_from_json(row[0])

        return cls(projection, events=lambda after: controller.events(after), observation=observation,
                   candidate=candidate)

    def read(self, after: int) -> list[HistoryRecord]:
        records = []
        for event in self._events(after):
            kind, ref, payload = event["type"], event.get("subject_id"), event.get("payload") or {}
            token = f"{self.name}:{event['seq']}"
            if kind == "experiment.planned":
                records.append(self._from_candidate(SELECTED, "selected", ref, token))
            elif kind == "observation.ingested":
                obs = self._observation(int(payload["ingest_seq"]))
                lineage = dict(obs.lineage)
                # The async ledger leaves round_idx unset; the proposing generation travels in candidate metadata.
                generation = obs.round_idx if obs.round_idx is not None else (obs.candidate.metadata or {}).get("generation")
                records.append(HistoryRecord(RESULT, ref, obs.canonical_key, obs.candidate_id, obs.evaluation.status,
                                             self.projection.project(obs.to_dict()), generation, lineage, token))
            elif kind == "experiment.finished" and payload.get("state") == "CANCELED":
                records.append(self._from_candidate(RESULT, "canceled", ref, token))
            elif kind == "anomaly.invalid_result" and ref:
                records.append(self._from_candidate(RESULT, "result_invalid", ref, token))
        return records

    def _from_candidate(self, kind: str, status: str, ref: str, token: str) -> HistoryRecord:
        candidate = self._candidate(ref)
        metadata = getattr(candidate, "metadata", {}) or {}
        return HistoryRecord(kind, ref, candidate.canonical_key, candidate.candidate_id, status,
                             self.projection.project({"candidate": candidate.to_dict()}, candidate_only=True),
                             metadata.get("generation"), dict(metadata.get("lineage") or {}), token)


# ---- history-relative submission checks -------------------------------------------------------------------------

def collision_errors(view: HistoryView, keys: Sequence[str | None], *, pointer: str = "/candidates",
                     count: tuple[int, int] | None = None) -> tuple[HarnessSubmissionError, ...]:
    """Pure check of submitted canonical keys against the history at the recorded watermark.

    ``None`` marks a candidate the task already found invalid. Codes: ``invalid``, ``already_measured``,
    ``already_pending``, ``duplicate_in_submission``, ``count_out_of_range``."""
    errors = []
    if count is not None and not count[0] <= len(keys) <= count[1]:
        errors.append(HarnessSubmissionError(pointer, COUNT_OUT_OF_RANGE,
                                             f"Expected {count[0]} to {count[1]} candidates; found {len(keys)}."))
    measured, pending, seen = view.measured_keys, view.pending_keys, set()
    for index, key in enumerate(keys):
        path = f"{pointer}/{index}"
        if key is None:
            errors.append(HarnessSubmissionError(path, INVALID, "The candidate is not legal for this task."))
        elif key in measured:
            errors.append(HarnessSubmissionError(path, ALREADY_MEASURED, "The candidate has already been measured.",
                                                 f"Visible in history up to seq {view.watermark}."))
        elif key in pending:
            errors.append(HarnessSubmissionError(path, ALREADY_PENDING, "The candidate is already being measured.",
                                                 "Use the pending view of the history tool."))
        elif key in seen:
            errors.append(HarnessSubmissionError(path, DUPLICATE_IN_SUBMISSION, "The candidate appears twice."))
        if key is not None:
            seen.add(key)
    return tuple(errors)


__all__ = [
    "ALREADY_MEASURED", "ALREADY_PENDING", "COUNT_OUT_OF_RANGE", "ControllerHistorySource", "DUPLICATE_IN_SUBMISSION",
    "HistoryEntry", "HistoryField", "HistoryLog", "HistoryProjection", "HistoryPublisher", "HistoryRecord",
    "HistorySource", "HistoryView", "INVALID", "MEASURED_STATUSES", "RESULT", "RESULT_STATUSES", "SELECTED",
    "collision_errors",
]
