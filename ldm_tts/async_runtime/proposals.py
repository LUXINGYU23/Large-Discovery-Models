"""The resident reservoir in the controller store (design 6.7, I12).

Occurrences live in the ``occurrences`` table; a removed occurrence keeps its row with a reason
(``renewed``, ``replaced``, ``withdrawn``, ``evicted``, ``capacity``, ``selected``, ``measured``,
``blocked``), which is the proposal-destination record that history tools page over. The weights,
q0 and fresh stock are never stored: they are recomputed from live occurrences and the booked-result
count by the pure maintenance object (``ResidentDecayReservoir`` or ``GenerationReservoir``).

``ReservoirKeeper`` is a ledger listener: selecting a candidate invalidates its occurrences in the
intent transaction, and booking a result ages every occurrence and evicts those under the weight
threshold in the ingest transaction. ``reservoir.stale`` / ``reservoir.exhausted`` are appended when
the reservoir becomes stale or runs out of eligible candidates.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from ldm_tts.contracts import Candidate

from .ledger import candidate_from_json
from .reservoir import GenerationReservoir, Occurrence, ReservoirView, occurrence_weight
from .store import append_event, get_flag, set_flag
from .wire import canonical


@dataclass(frozen=True)
class Proposal:
    """One proposal from a research round: a task-admitted candidate, the lineage that proposed it and,
    optionally, the visibility watermark (booked results) it was proposed at; default: the round's base."""

    candidate: Candidate
    lineage: str
    watermark: int | None = None

    def to_json(self) -> dict:
        return {"candidate": self.candidate.to_dict(), "lineage": self.lineage, "watermark": self.watermark}

    @classmethod
    def from_json(cls, data) -> "Proposal":
        return cls(Candidate(**data["candidate"]), str(data["lineage"]), data.get("watermark"))


def results_booked(db) -> int:
    return db.execute("SELECT COALESCE(MAX(ingest_seq), 0) FROM observations").fetchone()[0]


class ReservoirKeeper:
    def __init__(self, maintenance, ledger):
        self.maintenance, self.ledger = maintenance, ledger

    # ---- reading ------------------------------------------------------------------------------------

    def live(self, db) -> list[tuple[int, Occurrence]]:
        rows = db.execute("""SELECT id, candidate_json, lineage, generation, watermark FROM occurrences
            WHERE removed_at IS NULL ORDER BY id""").fetchall()
        return [(r["id"], Occurrence(candidate_from_json(r["candidate_json"]), r["lineage"], r["generation"],
                                     r["watermark"])) for r in rows]

    def view(self, db, excluded: Iterable[str] | None = None) -> ReservoirView:
        excluded = self.ledger.dedup_keys(db) if excluded is None else excluded
        return self.maintenance.view([o for _, o in self.live(db)], results_booked=results_booked(db),
                                     excluded_keys=excluded)

    def stale(self, view: ReservoirView) -> bool:
        return self.maintenance.is_stale(view)

    # ---- writing (all inside the caller's transaction) -----------------------------------------------

    def _remove(self, db, ids: Iterable[int], reason: str, now: float, booked: int) -> int:
        count = 0
        for row_id in ids:
            count += db.execute("""UPDATE occurrences SET removed_at=?, removed_reason=?, removed_results=?
                WHERE id=? AND removed_at IS NULL""", (now, reason, booked, row_id)).rowcount
        return count

    def _insert(self, db, occurrence: Occurrence, now: float) -> int:
        return db.execute("""INSERT INTO occurrences (canonical_key, candidate_json, lineage, generation, watermark,
            created_at) VALUES (?,?,?,?,?,?)""", (occurrence.key, canonical(occurrence.candidate.to_dict()),
                                                  occurrence.lineage, occurrence.generation, occurrence.watermark,
                                                  now)).lastrowid

    def merge(self, db, proposals: Sequence[Proposal], *, generation: int, base_watermark: int, now: float) -> dict:
        """New proposals join the reservoir (renewal for the same lineage; generation replacement clears
        the previous generation). Keys in the dedup set are blocked: admission is the authoritative dedup."""
        booked = results_booked(db)
        blocked = self.ledger.dedup_keys(db)
        live = self.live(db)
        replacing = isinstance(self.maintenance, GenerationReservoir)
        counts: dict[str, int] = {}
        if replacing:  # a new generation replaces the old one even when it admits nothing
            stale_ids = [row_id for row_id, o in live if o.generation < generation]
            if stale_ids:
                counts["replaced"] = self._remove(db, stale_ids, "replaced", now, booked)
            live = [(row_id, o) for row_id, o in live if o.generation >= generation]
        new = []
        for item in proposals:
            watermark = base_watermark if item.watermark is None else int(item.watermark)
            new.append(Occurrence(item.candidate, item.lineage, generation, min(watermark, booked)))
        state = self.maintenance.add([o for _, o in live], new, results_booked=booked, blocked_keys=blocked)
        kept = {id(o) for o in state}
        renewed = {(o.lineage, o.key) for o in new if id(o) in kept}
        for row_id, occurrence in live:
            if id(occurrence) in kept:
                continue
            if replacing:
                reason = "replaced"
            elif (occurrence.lineage, occurrence.key) in renewed:
                reason = "renewed"
            elif occurrence_weight(occurrence, booked, self.maintenance.half_life) < self.maintenance.min_weight:
                reason = "evicted"
            else:
                reason = "capacity"
            self._remove(db, [row_id], reason, now, booked)
            counts[reason] = counts.get(reason, 0) + 1
        admitted = 0
        for index, occurrence in enumerate(new):
            row_id = self._insert(db, occurrence, now)
            if id(occurrence) in kept:
                admitted += 1
                continue
            later = any((o.lineage, o.key) == (occurrence.lineage, occurrence.key) and id(o) in kept
                        for o in new[index + 1:])
            reason = "blocked" if occurrence.key in blocked else "renewed" if later and not replacing else "capacity"
            self._remove(db, [row_id], reason, now, booked)
            counts[f"new_{reason}"] = counts.get(f"new_{reason}", 0) + 1
        self.refresh(db, now)
        return {"admitted": admitted, "proposals": len(new), "removed": counts}

    def withdraw(self, db, lineage: str, keys: Iterable[str], now: float) -> int:
        keys = set(keys)
        ids = [row_id for row_id, o in self.live(db) if o.lineage == lineage and o.key in keys]
        removed = self._remove(db, ids, "withdrawn", now, results_booked(db))
        if removed:
            append_event(db, now, "reservoir.withdrawn", lineage, {"keys": sorted(keys), "removed": removed})
            self.refresh(db, now)
        return removed

    def invalidate(self, db, key: str, reason: str, now: float) -> int:
        ids = [r[0] for r in db.execute("SELECT id FROM occurrences WHERE canonical_key=? AND removed_at IS NULL",
                                        (key,)).fetchall()]
        return self._remove(db, ids, reason, now, results_booked(db))

    def evict(self, db, now: float) -> int:
        booked = results_booked(db)
        live = self.live(db)
        keep = {id(o) for o in self.maintenance.evict([o for _, o in live], results_booked=booked)}
        return self._remove(db, [row_id for row_id, o in live if id(o) not in keep], "evicted", now, booked)

    def refresh(self, db, now: float) -> None:
        """Edge-triggered ``reservoir.stale`` / ``reservoir.exhausted`` (5.9 event table)."""
        view = self.view(db)
        for flag, value, event in (("reservoir_stale", self.stale(view), "reservoir.stale"),
                                   ("reservoir_exhausted", not view.candidates, "reservoir.exhausted")):
            current = get_flag(db, flag)  # None: never evaluated (first start) -> a true value is an edge
            if current is None or bool(current) != value:
                set_flag(db, flag, value)
                if value:
                    append_event(db, now, event, None, {"fresh_stock": view.fresh_stock,
                                                        "eligible": len(view.candidates)})

    # ---- ledger listener ------------------------------------------------------------------------------

    def on_intent(self, db, candidate, decision_id, snapshot, now):
        self.invalidate(db, candidate.canonical_key, "selected", now)
        self.refresh(db, now)

    def on_ingested(self, db, key, ingest_seq, now):
        self.invalidate(db, key, "measured", now)
        self.evict(db, now)
        self.refresh(db, now)


def destinations(store, lineage: str | None = None, after_id: int = 0, limit: int = 100) -> list[dict]:
    """Proposal-destination rows keyed by the immutable occurrence id (keyset pagination for history-tools)."""
    sql, args = "SELECT * FROM occurrences WHERE id > ?", [after_id]
    if lineage is not None:
        sql, args = sql + " AND lineage = ?", args + [lineage]
    rows = store.rows(sql + " ORDER BY id LIMIT ?", (*args, limit))
    return [{"id": r["id"], "canonical_key": r["canonical_key"], "lineage": r["lineage"], "generation": r["generation"],
             "watermark": r["watermark"], "created_at": r["created_at"],
             "status": "live" if r["removed_at"] is None else r["removed_reason"],
             "removed_results": r["removed_results"]} for r in rows]


__all__ = ["Proposal", "ReservoirKeeper", "destinations", "results_booked"]
