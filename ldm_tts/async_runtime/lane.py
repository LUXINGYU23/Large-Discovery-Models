"""The research lane (design 5.5): one generation at a time, frozen inputs on disk, outcome by receipt.

State machine (the newest ``generations`` row; no row or an ended one means idle):

    idle --start--> running --outcome committed--> ready     (proposals merged into the reservoir)
                            --outcome rejected---> rejected  (reservoir unchanged)
                            --timeout / error----> blocked --operator retry--> running (same input)
                                                           --operator abandon-> abandoned

The round input is written to ``<db>.research/gNNNNNN.input.json`` before the start transaction records
its digest; recovery resubmits the same bytes. The job writes ``gNNNNNN.outcome.json`` before the control
thread commits it, so a crash after the round finished does not rerun research. Research sources follow
the ``ReservoirExpander`` shape: frozen input (+ the history up to the round's base watermark) -> proposals.
Executors: ``InlineExecutor`` (tests: control thread, virtual-clock duration, deterministic) and
``ThreadExecutor`` (production: a worker thread that wakes the controller when done).
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from ldm_tts.contracts import Candidate, CandidateRejection

from .codec import plugin_identity
from .pacing import LaneStatus
from .proposals import Proposal, results_booked
from .store import append_event, get_flag, record_anomaly, resolve_anomalies, set_flag
from .wire import canonical, digest


@dataclass(frozen=True)
class ResearchOutcome:
    status: str = "committed"  # committed | rejected (an explicit research-barrier rejection receipt)
    proposals: tuple[Proposal, ...] = ()
    withdrawals: tuple[tuple[str, str], ...] = ()  # (lineage, canonical_key)
    reason: str = ""
    final_watermark: int | None = None  # minimum final visibility watermark over the round's sessions
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.status not in ("committed", "rejected"):
            raise ValueError(f"unknown research outcome status {self.status!r}")
        object.__setattr__(self, "proposals", tuple(self.proposals))
        object.__setattr__(self, "withdrawals", tuple((str(a), str(b)) for a, b in self.withdrawals))

    def to_json(self) -> dict:
        return {"status": self.status, "proposals": [p.to_json() for p in self.proposals],
                "withdrawals": [list(w) for w in self.withdrawals], "reason": self.reason,
                "final_watermark": self.final_watermark, "metadata": dict(self.metadata)}

    @classmethod
    def from_json(cls, data) -> "ResearchOutcome":
        return cls(data["status"], tuple(Proposal.from_json(p) for p in data["proposals"]),
                   tuple(tuple(w) for w in data["withdrawals"]), data.get("reason", ""),
                   data.get("final_watermark"), data.get("metadata") or {})


@runtime_checkable
class ResearchSource(Protocol):
    name: str
    version: str

    def research(self, round_input: Mapping[str, Any], history: Sequence) -> ResearchOutcome: ...


def _rank(seed, generation, key) -> str:
    return hashlib.sha256(canonical([seed, generation, key]).encode()).hexdigest()


class DeterministicProposer:
    """A reproducible stand-in for a research panel: generation g proposes ``per_generation`` pool candidates in a
    seeded order, skipping those measured or pending at the round's base watermark, assigned to lineages
    round-robin. ``lineages`` > 1 with ``overlap`` makes several lineages propose the same candidate."""

    version = "1"

    def __init__(self, pool: Sequence[Candidate], *, per_generation: int, lineages: Sequence[str] = ("agent-0",),
                 seed: int = 0, skip_known: bool = True, overlap: int = 0, name: str = "deterministic"):
        self.pool, self.per_generation, self.lineages = tuple(pool), per_generation, tuple(lineages)
        self.seed, self.skip_known, self.overlap, self.name = seed, skip_known, overlap, name

    def research(self, round_input, history):
        known = {o.canonical_key for o in history} | set(round_input.get("pending_keys", ()))
        generation = round_input["generation"]
        ranked = sorted(self.pool, key=lambda c: _rank(self.seed, generation, c.canonical_key))
        chosen = [c for c in ranked if not self.skip_known or c.canonical_key not in known][:self.per_generation]
        proposals = [Proposal(c, self.lineages[i % len(self.lineages)]) for i, c in enumerate(chosen)]
        proposals += [Proposal(c, self.lineages[(i + 1) % len(self.lineages)]) for i, c in enumerate(chosen[:self.overlap])]
        return ResearchOutcome(proposals=tuple(proposals), metadata={"pool_left": len(ranked) - len(known)})


class ScriptedSource:
    """Test source: ``script(round_input, history)`` returns a ``ResearchOutcome`` or a list of ``Proposal``."""

    version = "1"

    def __init__(self, script: Callable, name: str = "scripted"):
        self.script, self.name = script, name

    def research(self, round_input, history):
        result = self.script(round_input, history)
        return result if isinstance(result, ResearchOutcome) else ResearchOutcome(proposals=tuple(result))


class ExpanderSource:
    """Adapter for an existing ``ReservoirExpander`` plus the task's ``CandidateDomainAdapter``: the round input
    becomes an ``ExpansionRequest`` (history = observations up to the base watermark) and every raw proposal is
    admitted by the task; the lineage is the raw proposal's ``source``. Rejections are counted in metadata."""

    def __init__(self, expander, adapter, *, name: str = "expander", version: str = "1"):
        self.expander, self.adapter, self.name, self.version = expander, adapter, name, version

    def research(self, round_input, history):
        from ldm_tts.engine.expansion import ExpansionRequest
        result = self.expander.expand(ExpansionRequest(
            round_idx=round_input["generation"], reservoir_size=round_input["reservoir_size"],
            observations=tuple(history), context={"async_round": dict(round_input)}))
        proposals, rejected = [], {}
        for raw in result.proposals:
            admitted = self.adapter.admit(raw)
            if isinstance(admitted, CandidateRejection):
                rejected[admitted.reason] = rejected.get(admitted.reason, 0) + 1
                continue
            proposals.append(Proposal(admitted, raw.source or "panel"))
        return ResearchOutcome(proposals=tuple(proposals), metadata={"task_rejections": rejected,
                                                                     "expansion": dict(result.metadata)})


# ---- executors ------------------------------------------------------------------------------------------

class InlineExecutor:
    """Deterministic executor: the job runs on the control thread once the (virtual) clock passes
    ``started_at + duration(round_input)``."""

    def __init__(self, clock, duration: float | Callable[[Mapping], float] = 0.0):
        self.clock = clock
        self.duration = duration if callable(duration) else (lambda _input, value=float(duration): value)
        self.jobs: dict[int, tuple[Callable, float]] = {}
        self.done: dict[int, Any] = {}

    def submit(self, generation, job, round_input, started_at):
        """The job keeps its schedule across a restart (a durable job): its effective start is ``started_at``."""
        self.jobs[generation] = (job, started_at + float(self.duration(round_input)))
        return started_at

    def preload(self, generation, outcome):
        self.done[generation] = outcome

    def discard_failure(self, generation) -> None:
        if isinstance(self.done.get(generation), BaseException):
            del self.done[generation]

    def has(self, generation) -> bool:
        return generation in self.jobs or generation in self.done

    def poll(self, generation):
        if generation in self.done:
            return self.done.pop(generation)
        if generation not in self.jobs or self.clock.now() < self.jobs[generation][1]:
            return None
        job, _ = self.jobs.pop(generation)
        try:
            return job()
        except Exception as exc:  # noqa: BLE001 - an unknown outcome blocks the lane
            return exc

    def next_due(self):
        return min((due for _, due in self.jobs.values()), default=None)


class ThreadExecutor:
    """Production executor: one worker thread per generation (the lane runs at most one); ``on_done`` wakes the
    control thread. A research round cannot be cancelled (5.5 rule 4): a thread that outlived its generation (timed
    out, then abandoned) keeps the executor ``busy`` and the lane starts no new generation until it ends, so two
    rounds never share a research client. A job submitted after a restart starts now (``submit`` returns None)."""

    def __init__(self, on_done: Callable[[], None] | None = None):
        self.on_done = on_done
        self.running: set[int] = set()
        self.done: dict[int, Any] = {}
        self._lock = threading.Lock()

    def submit(self, generation, job, round_input, started_at):
        def work():
            result = None
            try:
                result = job()
            except BaseException as exc:  # noqa: BLE001 - recorded (the lane blocks); a non-Exception propagates
                result = exc
                if not isinstance(exc, Exception):
                    raise
            finally:  # never leave the executor busy for ever
                with self._lock:
                    self.running.discard(generation)
                    self.done[generation] = result
                if self.on_done is not None:
                    self.on_done()

        with self._lock:
            self.running.add(generation)
        threading.Thread(target=work, name=f"ldm-research-g{generation}", daemon=True).start()
        return None

    def preload(self, generation, outcome):
        with self._lock:
            self.done[generation] = outcome

    def discard_failure(self, generation) -> None:
        """Operator retry: an exception left by a timed-out run is stale; a late outcome is kept and collected."""
        with self._lock:
            if isinstance(self.done.get(generation), BaseException):
                del self.done[generation]

    def busy(self) -> bool:
        with self._lock:
            return bool(self.running)

    def has(self, generation) -> bool:
        with self._lock:
            return generation in self.running or generation in self.done

    def poll(self, generation):
        with self._lock:
            return self.done.pop(generation, None)

    def next_due(self):
        return None


# ---- files ----------------------------------------------------------------------------------------------

def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


# ---- the lane -------------------------------------------------------------------------------------------

class ResearchLane:
    """``extension`` (optional, e.g. the history bridge) adds fields to the frozen round input and takes part in the
    start / end / abandon transactions: ``extend_round_input(db, generation, round_input)``,
    ``on_start(db, generation, round_input, now)``, ``on_finish(db, generation, outcome, now)``,
    ``on_abandon(db, generation, now)``."""

    def __init__(self, *, store, ledger, keeper, source, executor, config, directory: Path, clock, extension=None):
        self.store, self.ledger, self.keeper, self.source = store, ledger, keeper, source
        self.executor, self.config, self.directory, self.clock = executor, config, Path(directory), clock
        # Optional readiness step (policy-prior): ``prepare(generation, outcome)`` outside the transaction, then
        # ``commit(db, generation, prepared, now)`` inside ``generation_end`` before ``generation.ready``.
        self.readiness = None
        self.extension = extension

    def input_path(self, generation: int) -> Path:
        return self.directory / f"g{generation:06d}.input.json"

    def outcome_path(self, generation: int) -> Path:
        return self.directory / f"g{generation:06d}.outcome.json"

    # ---- reads ----------------------------------------------------------------------------------------

    def latest(self, db=None):
        db = db or self.store.db
        with self.store.lock:
            return db.execute("SELECT * FROM generations ORDER BY generation DESC LIMIT 1").fetchone()

    def status(self, db=None) -> LaneStatus:
        row = self.latest(db)
        if row is None or row["state"] not in ("running", "blocked"):
            return LaneStatus()
        return LaneStatus(row["state"], row["generation"], row["started_at"], row["deadline_at"], row["trigger"])

    def generations(self) -> list[dict]:
        return [dict(r) for r in self.store.rows("SELECT * FROM generations ORDER BY generation")]

    # ---- start ----------------------------------------------------------------------------------------

    def round_input(self, db, generation: int, trigger: str, note: str) -> dict:
        previous = db.execute("""SELECT generation, state, admitted, proposal_count, outcome_json, error FROM generations
            WHERE state IN ('ready','rejected','abandoned') ORDER BY generation DESC LIMIT 1""").fetchone()
        feedback = None
        if previous is not None:
            outcome = json.loads(previous["outcome_json"] or "{}")
            feedback = {"generation": previous["generation"], "state": previous["state"],
                        "admitted": previous["admitted"], "proposals": previous["proposal_count"],
                        "reason": outcome.get("reason") or previous["error"] or "",
                        "removed": outcome.get("merge", {}).get("removed", {})}
        pending = sorted(r[0] for r in db.execute("SELECT canonical_key FROM experiments WHERE pending=1").fetchall())
        data = {"campaign_id": self.ledger.campaign_id, "generation": generation, "trigger": trigger, "note": note,
                "watermarks": {"base_ingest_seq": results_booked(db),
                               "base_event_seq": db.execute("SELECT COALESCE(MAX(seq),0) FROM events").fetchone()[0]},
                "reservoir_size": self.config.reservoir_size, "budget": self.ledger.budget(db).to_dict(),
                "pending_keys": pending, "previous": feedback, "source": plugin_identity(self.source, "research"),
                "seed": self.ledger.config.seed}
        extras = getattr(self.source, "round_input_extras", None)  # e.g. the policy turn's frozen inputs
        if extras is not None:
            data.update(extras(db, generation))
        if self.extension is not None:
            self.extension.extend_round_input(db, generation, data)
        return data

    def start(self, trigger: str, note: str = "") -> int | None:
        busy = getattr(self.executor, "busy", None)
        if busy is not None and busy():
            return None  # a round of an abandoned generation is still running on the research client
        now = self.clock.now()
        with self.store.lock:
            db = self.store.db
            generation = db.execute("SELECT COALESCE(MAX(generation),0)+1 FROM generations").fetchone()[0]
            round_input = self.round_input(db, generation, trigger, note)
        path = self.input_path(generation)
        _write_atomic(path, canonical(round_input))  # the frozen input exists before the lane records it
        deadline = now + self.config.research_timeout_seconds
        with self.store.tx("generation_start") as db:
            if self.status(db).state != "idle" or \
                    db.execute("SELECT COALESCE(MAX(generation),0)+1 FROM generations").fetchone()[0] != generation:
                return None
            db.execute("""INSERT INTO generations (generation, state, trigger, input_path, input_digest, base_ingest_seq,
                base_event_seq, started_at, deadline_at) VALUES (?,?,?,?,?,?,?,?,?)""",
                       (generation, "running", trigger, path.name, digest(round_input),
                        round_input["watermarks"]["base_ingest_seq"], round_input["watermarks"]["base_event_seq"],
                        now, deadline))
            request = get_flag(db, "research_requested")
            if request is not None and trigger == f"hook:{request['request_id']}":
                set_flag(db, "research_requested", None)
            if self.extension is not None:
                self.extension.on_start(db, generation, round_input, now)
            append_event(db, now, "generation.started", str(generation),
                         {"trigger": trigger, "base_ingest_seq": round_input["watermarks"]["base_ingest_seq"]})
        self._submit(generation, round_input, now)
        return generation

    def _submit(self, generation: int, round_input: dict, started_at: float) -> None:
        history = tuple(self.ledger.observations(0, round_input["watermarks"]["base_ingest_seq"]))
        expected = digest(round_input)
        source, out = self.source, self.outcome_path(generation)

        def job():
            outcome = source.research(round_input, history)
            if not isinstance(outcome, ResearchOutcome):
                raise TypeError(f"research source returned {type(outcome).__name__}, not ResearchOutcome")
            _write_atomic(out, canonical({"input_digest": expected, "outcome": outcome.to_json()}))
            return outcome

        started = self.executor.submit(generation, job, round_input, started_at)
        return self.clock.now() if started is None else started

    def recover(self) -> None:
        """Restart (5.6): a running generation is resubmitted with the same input, or its saved outcome reused."""
        row = self.latest()
        if row is None or row["state"] != "running" or self.executor.has(row["generation"]):
            return
        generation = row["generation"]
        try:
            text = self.input_path(generation).read_text(encoding="utf-8")
            round_input = json.loads(text)
            if digest(round_input) != row["input_digest"]:
                raise ValueError("round input differs from the recorded digest")
        except (OSError, ValueError) as exc:
            self.block(generation, f"input: {exc}")
            return
        saved = self.outcome_path(generation)
        if saved.exists():
            data = json.loads(saved.read_text(encoding="utf-8"))
            if data.get("input_digest") == row["input_digest"]:
                self.executor.preload(generation, ResearchOutcome.from_json(data["outcome"]))
                return
        started = self._submit(generation, round_input, row["started_at"])
        deadline = started + self.config.research_timeout_seconds
        if deadline > row["deadline_at"]:  # a job restarted from scratch gets the full wall-clock limit again
            with self.store.tx("lane") as db:
                db.execute("UPDATE generations SET deadline_at=? WHERE generation=? AND state='running'",
                           (deadline, generation))

    # ---- progress -------------------------------------------------------------------------------------

    def poll(self) -> bool:
        row = self.latest()
        if row is None or row["state"] != "running":
            return False
        generation = row["generation"]
        result = self.executor.poll(generation)
        if result is None:
            if self.clock.now() >= row["deadline_at"]:
                self.block(generation, "timeout")
                return True
            return False
        if isinstance(result, BaseException):
            self.block(generation, f"{type(result).__name__}: {result}")
            return True
        prepared = None
        if self.readiness is not None and result.status == "committed":
            try:
                prepared = self.readiness.prepare(generation, result)
            except Exception as exc:  # noqa: BLE001 - a readiness defect blocks the lane (operator retry reuses the outcome)
                self.block(generation, f"readiness: {type(exc).__name__}: {exc}")
                return True
        self.finish(generation, result, prepared)
        return True

    def finish(self, generation: int, outcome: ResearchOutcome, prepared=None) -> bool:
        now = self.clock.now()
        with self.store.tx("generation_end") as db:
            row = db.execute("SELECT * FROM generations WHERE generation=?", (generation,)).fetchone()
            if row is None or row["state"] != "running":
                return False
            booked = results_booked(db)
            # Never below the round baseline: the sessions saw at least that much, and a smaller value would keep
            # delta >= 1 for ever (rule R clause d) and let research restart without anything new.
            final = row["base_ingest_seq"] if outcome.final_watermark is None else \
                max(row["base_ingest_seq"], min(int(outcome.final_watermark), booked))
            record = outcome.to_json()
            if outcome.status == "rejected":
                record["eligible"] = self.eligible(db)
                record["merge"] = {"admitted": 0, "proposals": len(outcome.proposals), "removed": {}}
                head = db.execute("SELECT COALESCE(MAX(decision_id),0) FROM decisions").fetchone()[0]
                db.execute("""UPDATE generations SET state='rejected', finished_at=?, final_watermark=?, proposal_count=?,
                    admitted=0, decision_head=?, outcome_json=? WHERE generation=?""",
                           (now, final, len(outcome.proposals), head, canonical(record), generation))
                append_event(db, now, "generation.rejected", str(generation), {"reason": outcome.reason})
            else:
                withdrawn = sum(self.keeper.withdraw(db, lineage, [key], now) for lineage, key in outcome.withdrawals)
                merged = self.keeper.merge(db, outcome.proposals, generation=generation,
                                           base_watermark=row["base_ingest_seq"], now=now)
                record["merge"] = {**merged, "withdrawn": withdrawn}
                record["eligible"] = self.eligible(db)
                head = db.execute("SELECT COALESCE(MAX(decision_id),0) FROM decisions").fetchone()[0]
                db.execute("""UPDATE generations SET state='ready', finished_at=?, final_watermark=?, proposal_count=?,
                    admitted=?, decision_head=?, outcome_json=? WHERE generation=?""",
                           (now, final, merged["proposals"], merged["admitted"], head, canonical(record), generation))
                if prepared is not None:  # ready = panel committed + policy turn + policy application (5.5 rule 5)
                    self.readiness.commit(db, generation, prepared, now)
                append_event(db, now, "generation.ready", str(generation),
                             {"admitted": merged["admitted"], "proposals": merged["proposals"]})
            if self.extension is not None:
                self.extension.on_finish(db, generation, outcome, now)
            self.check_stagnation(db, now)
        return True

    def block(self, generation: int, error: str) -> None:
        now = self.clock.now()
        with self.store.tx("lane") as db:
            changed = db.execute("UPDATE generations SET state='blocked', error=? WHERE generation=? AND state='running'",
                                 (error[:500], generation)).rowcount
            if changed:
                append_event(db, now, "generation.blocked", str(generation), {"error": error[:500]})
                record_anomaly(db, now, "lane_blocked", str(generation), {"error": error[:500]})

    def resolve(self, action: str, operator: str) -> bool:
        """Operator reconciliation of a blocked generation: ``retry`` (same input) or ``abandon``."""
        if action not in ("retry", "abandon"):
            raise ValueError("lane resolution is 'retry' or 'abandon'")
        now = self.clock.now()
        row = self.latest()
        if row is None or row["state"] != "blocked":
            return False
        generation = row["generation"]
        with self.store.tx("lane") as db:
            if action == "retry":
                changed = db.execute("""UPDATE generations SET state='running', started_at=?, deadline_at=?, error=NULL
                    WHERE generation=? AND state='blocked'""",
                                     (now, now + self.config.research_timeout_seconds, generation)).rowcount
                event = "generation.started"
            else:
                changed = db.execute("""UPDATE generations SET state='abandoned', finished_at=?, admitted=0,
                    final_watermark=base_ingest_seq, decision_head=(SELECT COALESCE(MAX(decision_id),0) FROM decisions),
                    outcome_json=? WHERE generation=? AND state='blocked'""",
                                     (now, canonical({"eligible": self.eligible(db)}), generation)).rowcount
                event = "generation.abandoned"
            if not changed:
                return False
            resolve_anomalies(db, now, "lane_blocked", str(generation), by=operator)
            append_event(db, now, event, str(generation), {"operator": operator, "resolution": action})
            if action == "abandon":
                if self.extension is not None:
                    self.extension.on_abandon(db, generation, now)
                self.check_stagnation(db, now)
        if action == "retry":
            discard = getattr(self.executor, "discard_failure", None)
            if discard is not None:
                discard(generation)
            self.recover()
        return True

    def eligible(self, db) -> int:
        """E: live reservoir candidates outside the dedup set (recorded when a generation ends)."""
        return len(self.keeper.view(db, self.ledger.dedup_keys(db)).candidates)

    def check_stagnation(self, db, now: float) -> None:
        """``max_empty_reservoir_rounds`` consecutive ended generations that left the reservoir without an eligible
        candidate -> drain and stop with ``empty_reservoir_limit`` (the synchronous "empty reservoir" round)."""
        empty = empty_rounds(db)
        if empty >= self.config.max_empty_reservoir_rounds and get_flag(db, "draining") is None:
            set_flag(db, "draining", {"reason": "empty_reservoir_limit", "mode": "drain", "at": now})
            append_event(db, now, "campaign.draining", None, {"reason": "empty_reservoir_limit", "empty_rounds": empty})

    def next_due(self) -> float | None:
        times = [t for t in (self.executor.next_due(),) if t is not None]
        row = self.latest()
        if row is not None and row["state"] == "running":
            times.append(row["deadline_at"])
        return min(times) if times else None


def empty_rounds(db) -> int:
    """Consecutive most recent ended generations with E = 0 at their end (coordinator decision, pacing review
    item 8): a generation that admitted nothing while eligible candidates remained is not an empty round and ends
    the streak. Rows without the recorded E (older stores) fall back to "admitted nothing"."""
    count = 0
    for row in db.execute("""SELECT admitted, outcome_json FROM generations WHERE state IN ('ready','rejected','abandoned')
            ORDER BY generation DESC""").fetchall():
        eligible = json.loads(row["outcome_json"] or "{}").get("eligible")
        if (row["admitted"] if eligible is None else eligible):
            break
        count += 1
    return count


__all__ = ["DeterministicProposer", "ExpanderSource", "InlineExecutor", "ResearchLane", "ResearchOutcome",
           "ResearchSource", "ScriptedSource", "ThreadExecutor", "empty_rounds"]
