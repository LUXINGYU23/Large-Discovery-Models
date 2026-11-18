"""Stateless history queries at a fixed watermark: views, filters, total orders and keyset cursors (design 5.8).

Each query's first page fixes ``w = head`` and binds it into an opaque cursor together with a digest of
the query. Later pages only look at ``seq <= w`` of the immutable log, so a paginated enumeration is
the same frozen total order however many results are booked meanwhile; ``unseen = head - w`` tells the
agent to fetch the rest with ``since_seq = w``. Old cursors keep working; a cursor used with a different
query is an error.
"""

from __future__ import annotations

import base64
import functools
import hashlib
import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from ldm_tts.harness.history import RESULT_STATUSES, HistoryEntry, HistoryLog, HistoryProjection, HistoryView
from ldm_tts.harness.protocol import canonical_sha256

VIEWS = ("summary", "observations", "pending", "proposals")
PROPOSAL_STATUSES = ("not_admitted", "in_reservoir", "withdrawn", "retired", "pending", "measured")
_SHARED_PROPOSAL_STATUSES = frozenset({"pending", "measured"})
_FILTER_KEYS = {"status", "since_seq", "until_seq", "generation", "keys", "fields", "mine"}


class HistoryQueryError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class QueryLimits:
    page_size_default: int = 20
    page_size_max: int = 100
    response_max_bytes: int = 32_768
    inline_limit: int = 3

    def __post_init__(self) -> None:
        if not 1 <= self.page_size_default <= self.page_size_max or self.response_max_bytes < 1024 \
                or self.inline_limit < 0:
            raise ValueError("invalid history query limits")


class ProposalOutcomes(Protocol):
    """Proposal-fate table (filled by the research lane in part B). Rows are keyed by an immutable,
    totally ordered ``proposal_id`` and carry ``lineage``, ``status`` (``PROPOSAL_STATUSES``) and
    optional ``key``, ``generation``, ``weight``, ``results_since``, ``reason``, ``fields``."""

    def rows(self) -> Sequence[Mapping[str, Any]]: ...


def _encode_cursor(payload: Mapping[str, Any]) -> str:
    body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    check = hashlib.sha256(body).hexdigest()[:16]
    return base64.urlsafe_b64encode(body).decode().rstrip("=") + "." + check


def _decode_cursor(cursor: str) -> dict[str, Any]:
    try:
        text, check = cursor.rsplit(".", 1)
        body = base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))
        if hashlib.sha256(body).hexdigest()[:16] != check:
            raise ValueError
        payload = json.loads(body)
        # The checksum only detects corruption; an agent can re-encode any payload, so validate the shape.
        if not isinstance(payload, dict) or set(payload) != {"v", "w", "q", "k"} or payload["v"] != 1 \
                or not _is_int(payload["w"]) or not isinstance(payload["q"], str) or not isinstance(payload["k"], list):
            raise ValueError
        return payload
    except (ValueError, TypeError, UnicodeDecodeError, json.JSONDecodeError):
        raise _bad_cursor() from None


def _bad_cursor(detail: str = "malformed") -> HistoryQueryError:
    return HistoryQueryError("invalid_cursor", f"The cursor is {detail}; start the query again without it.")


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _natural_key(value: str) -> tuple[Any, ...]:
    """``"g1-10"`` after ``"g1-9"``: alternating text and integer parts (text at even positions)."""
    parts = re.split(r"(\d+)", value)
    return tuple(int(part) if index % 2 else part for index, part in enumerate(parts))


def _valid_natural_key(key: Sequence[Any]) -> bool:
    return len(key) % 2 == 1 and all(
        (_is_int(part) and part >= 0) if index % 2 else isinstance(part, str) for index, part in enumerate(key))


def best_result(view_entries: Sequence[HistoryEntry], projection: HistoryProjection) -> HistoryEntry | None:
    objective = projection.objective
    if objective is None:
        return None
    sign = 1 if objective.direction == "maximize" else -1
    scored = [e for e in view_entries if e.kind == "result" and e.status == "succeeded"
              and e.fields.get(objective.name) is not None]
    return max(scored, key=lambda e: (sign * e.fields[objective.name], -e.seq), default=None)


class HistoryQueryEngine:
    def __init__(self, log: HistoryLog, projection: HistoryProjection, *, limits: QueryLimits = QueryLimits(),
                 proposals: ProposalOutcomes | None = None, state_of: Callable[[str], str] | None = None,
                 tool_name: str = "experiment_history"):
        self.log, self.projection, self.limits = log, projection, limits
        self.proposals, self.state_of, self.tool_name = proposals, state_of, tool_name

    # ---- rows ------------------------------------------------------------------------------------------------

    def row(self, entry: HistoryEntry, *, pending: bool = False) -> dict[str, Any]:
        out: dict[str, Any] = {"seq": entry.seq, "status": entry.status, "key": entry.key,
                               "candidate_id": entry.candidate_id}
        if entry.generation is not None:
            out["generation"] = entry.generation
        if pending:
            out.update(status="pending", measured=False,
                       state=self.state_of(entry.ref) if self.state_of is not None else "pending")
        out.update({name: value for name, value in entry.fields.items() if value is not None})
        return out

    # ---- query -----------------------------------------------------------------------------------------------

    def query(self, request: Mapping[str, Any], *, lineage: str | None = None) -> dict[str, Any]:
        if not isinstance(request, Mapping):
            raise HistoryQueryError("invalid_query", "The query must be an object.")
        unknown = set(request) - {"view", "filter", "sort", "page_size", "cursor"}
        if unknown:
            raise HistoryQueryError("invalid_query", f"Unknown query keys: {sorted(unknown)}.")
        view = request.get("view", "summary")
        if view not in VIEWS:
            raise HistoryQueryError("invalid_query", f"view must be one of {list(VIEWS)}.")
        flt = self._filter(request.get("filter") or {}, view)
        sort = self._sort(request.get("sort"), view)
        page_size = request.get("page_size", self.limits.page_size_default)
        if isinstance(page_size, bool) or not isinstance(page_size, int) or not 1 <= page_size <= self.limits.page_size_max:
            raise HistoryQueryError("invalid_query", f"page_size must be an integer in [1, {self.limits.page_size_max}].")
        identity = canonical_sha256({"view": view, "filter": flt, "sort": sort, "lineage": lineage})[:16]
        head = self.log.head
        cursor = request.get("cursor")
        after: list[Any] | None = None
        watermark = head
        if cursor is not None:
            if not isinstance(cursor, str):
                raise HistoryQueryError("invalid_cursor", "cursor must be the next_cursor string of a previous page.")
            payload = _decode_cursor(cursor)
            if payload.get("q") != identity:
                raise HistoryQueryError("cursor_query_mismatch",
                                        "This cursor belongs to a different query; repeat that query or drop the cursor.")
            watermark, after = payload["w"], payload["k"]
            if not 0 <= watermark <= head:
                raise _bad_cursor("outside the history")
            if not self._valid_after(after, view, sort):
                raise _bad_cursor("malformed")
        snapshot = self.log.view(watermark)
        base = {"view": view, "watermark": watermark, "head": head, "unseen": head - watermark}
        if view == "summary":
            return {**base, **self._summary(snapshot, flt)}
        if view == "pending":
            rows = [self.row(e, pending=True) for e in reversed(snapshot.pending) if self._match(e, flt)]
            return {**base, "total": len(rows), "rows": rows, "next_cursor": None}
        if view == "observations":
            items = [(self._sort_key(e, sort), self.row(e)) for e in snapshot.results if self._match(e, flt)]
            compare = self._comparator(sort)
        else:
            items = [(_natural_key(str(r["proposal_id"])), r) for r in self._proposal_rows(flt, lineage)]
            compare = lambda a, b: (a > b) - (a < b)  # noqa: E731
        ordered = sorted(items, key=functools.cmp_to_key(lambda a, b: compare(a[0], b[0])))
        if after is not None:
            ordered = [item for item in ordered if compare(item[0], tuple(after)) > 0]
        rows, size = [], 0
        for item in ordered[:page_size]:
            size += len(json.dumps(item[1], sort_keys=True, separators=(",", ":"))) + 1
            if rows and size > self.limits.response_max_bytes:
                break
            rows.append(item)
        total = len(items)
        more = len(rows) < len(ordered)
        next_cursor = _encode_cursor({"v": 1, "w": watermark, "q": identity, "k": list(rows[-1][0])}) if more else None
        return {**base, "total": total, "rows": [item[1] for item in rows], "next_cursor": next_cursor}

    def _valid_after(self, after: list[Any], view: str, sort: Mapping[str, str]) -> bool:
        """The cursor's sort key must have the arity and element types of this query's sort key."""
        if view in ("summary", "pending"):
            return False  # these views never issue cursors
        if view == "proposals":
            return _valid_natural_key(after)
        if sort["by"] == "seq":
            return len(after) == 1 and _is_int(after[0])
        if len(after) != 2 or not _is_int(after[1]):
            return False
        value, kind = after[0], self.projection.field(sort["by"]).type
        if value is None:
            return True
        if kind == "integer":
            return _is_int(value)
        if kind == "number":
            return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
        return isinstance(value, {"string": str, "boolean": bool}[kind])

    def _filter(self, raw: Any, view: str) -> dict[str, Any]:
        if not isinstance(raw, Mapping) or set(raw) - _FILTER_KEYS:
            raise HistoryQueryError("invalid_filter", f"filter must be an object with keys from {sorted(_FILTER_KEYS)}.")
        out: dict[str, Any] = {}
        if "status" in raw:
            allowed = PROPOSAL_STATUSES if view == "proposals" else tuple(sorted(RESULT_STATUSES))
            status = raw["status"]
            if not isinstance(status, list) or not status or any(s not in allowed for s in status):
                raise HistoryQueryError("invalid_filter", f"filter.status must be a non-empty list from {list(allowed)}.")
            out["status"] = sorted(set(status))
        for name in ("since_seq", "until_seq"):
            if name in raw:
                value = raw[name]
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise HistoryQueryError("invalid_filter", f"filter.{name} must be a non-negative integer.")
                out[name] = value
        if "generation" in raw:
            out["generation"] = self._range(raw["generation"], "filter.generation", int)
        if "keys" in raw:
            keys = raw["keys"]
            if not isinstance(keys, list) or not keys or len(keys) > 200 or any(not isinstance(k, str) for k in keys):
                raise HistoryQueryError("invalid_filter", "filter.keys must be a list of 1 to 200 candidate keys.")
            out["keys"] = sorted(set(keys))
        if "fields" in raw:
            fields = raw["fields"]
            if not isinstance(fields, Mapping):
                raise HistoryQueryError("invalid_filter", "filter.fields must map field names to conditions.")
            out["fields"] = {}
            for name in sorted(fields):
                spec = self.projection.field(name)
                if spec is None or not spec.filterable:
                    names = [f.name for f in self.projection.fields if f.filterable]
                    raise HistoryQueryError("invalid_filter", f"{name!r} is not filterable; filterable fields: {names}.")
                out["fields"][name] = self._condition(fields[name], name)
        if "mine" in raw:
            if view != "proposals" or not isinstance(raw["mine"], bool):
                raise HistoryQueryError("invalid_filter", "filter.mine is a boolean of the proposals view.")
            out["mine"] = raw["mine"]
        return out

    @staticmethod
    def _range(value: Any, name: str, kind: type) -> dict[str, Any]:
        if not isinstance(value, Mapping) or not value or set(value) - {"min", "max"} or any(
                isinstance(v, bool) or not isinstance(v, (kind, float) if kind is float else kind) for v in value.values()):
            raise HistoryQueryError("invalid_filter", f"{name} must be an object with min and/or max.")
        return dict(sorted(value.items()))

    def _condition(self, value: Any, name: str) -> dict[str, Any]:
        if not isinstance(value, Mapping) or len(value) == 0 or set(value) - {"eq", "in", "min", "max"} \
                or ("eq" in value or "in" in value) and len(value) != 1:
            raise HistoryQueryError("invalid_filter",
                                    f"filter.fields.{name} must be one of {{eq}}, {{in: [...]}} or {{min, max}}.")
        if "in" in value and (not isinstance(value["in"], list) or not value["in"] or len(value["in"]) > 200):
            raise HistoryQueryError("invalid_filter", f"filter.fields.{name}.in must be a list of 1 to 200 values.")
        if any(isinstance(value.get(k), (dict, list)) for k in ("eq", "min", "max")):
            raise HistoryQueryError("invalid_filter", f"filter.fields.{name} compares scalar values only.")
        return dict(sorted(value.items()))

    def _sort(self, raw: Any, view: str) -> dict[str, str]:
        if raw is None:
            return {"by": "proposal_id", "order": "asc"} if view == "proposals" else {"by": "seq", "order": "desc"}
        if not isinstance(raw, Mapping) or set(raw) - {"by", "order"}:
            raise HistoryQueryError("invalid_sort", 'sort must be {"by": ..., "order": "asc"|"desc"}.')
        by = raw.get("by", "seq")
        objective = self.projection.objective
        if view == "proposals":
            if by != "proposal_id":
                raise HistoryQueryError("invalid_sort", "The proposals view is ordered by proposal_id only.")
            default = "asc"
        elif by == "seq":
            default = "desc"
        elif by == "objective":
            if objective is None:
                raise HistoryQueryError("invalid_sort", "This task declares no objective field.")
            by, default = objective.name, "desc" if objective.direction == "maximize" else "asc"
        else:
            spec = self.projection.field(by)
            if spec is None or not spec.sortable:
                names = ["seq", "objective"] + [f.name for f in self.projection.fields if f.sortable]
                raise HistoryQueryError("invalid_sort", f"{by!r} is not sortable; sortable keys: {names}.")
            default = "desc" if spec is objective and objective.direction == "maximize" else "asc"
        order = raw.get("order", default)
        if order not in ("asc", "desc"):
            raise HistoryQueryError("invalid_sort", 'sort.order must be "asc" or "desc".')
        return {"by": by, "order": order}

    # ---- matching and ordering -------------------------------------------------------------------------------

    def _match(self, entry: HistoryEntry, flt: Mapping[str, Any]) -> bool:
        if "status" in flt and entry.status not in flt["status"]:
            return False
        if entry.seq <= flt.get("since_seq", -1) or entry.seq > flt.get("until_seq", entry.seq):
            return False
        if "generation" in flt and not _in_range(entry.generation, flt["generation"]):
            return False
        if "keys" in flt and entry.key not in flt["keys"]:
            return False
        return all(_condition_holds(entry.fields.get(name), cond) for name, cond in flt.get("fields", {}).items())

    @staticmethod
    def _sort_key(entry: HistoryEntry, sort: Mapping[str, str]) -> tuple[Any, ...]:
        return (entry.seq,) if sort["by"] == "seq" else (entry.fields.get(sort["by"]), entry.seq)

    @staticmethod
    def _comparator(sort: Mapping[str, str]) -> Callable[[Sequence[Any], Sequence[Any]], int]:
        sign = -1 if sort["order"] == "desc" else 1

        def compare(a: Sequence[Any], b: Sequence[Any]) -> int:
            if len(a) == 2:
                (av, aseq), (bv, bseq) = a, b
                if (av is None) != (bv is None):
                    return 1 if av is None else -1  # missing values last in either order
                if av is not None and av != bv:
                    return sign * ((av > bv) - (av < bv))
            else:
                aseq, bseq = a[0], b[0]
            return sign * ((aseq > bseq) - (aseq < bseq))  # seq is always the final order

        return compare

    def _proposal_rows(self, flt: Mapping[str, Any], lineage: str | None) -> list[dict[str, Any]]:
        if self.proposals is None:
            return []
        out, keys = [], set()
        for row in self.proposals.rows():
            pid = row.get("proposal_id") if isinstance(row, Mapping) else None
            if not (isinstance(pid, str) and pid or _is_int(pid) and pid >= 0) \
                    or row.get("status") not in PROPOSAL_STATUSES or not isinstance(row.get("lineage"), (str, type(None))):
                raise HistoryQueryError("invalid_proposals", "The proposal table has a malformed row; this is a host defect.")
            key = _natural_key(str(pid))
            if key in keys:
                raise HistoryQueryError("invalid_proposals", f"The proposal table repeats proposal_id {pid!r}; this is a host defect.")
            keys.add(key)
            own =lineage is not None and row.get("lineage") == lineage
            if not own and (flt.get("mine") or row.get("status") not in _SHARED_PROPOSAL_STATUSES):
                continue
            if "status" in flt and row.get("status") not in flt["status"]:
                continue
            if "generation" in flt and not _in_range(row.get("generation"), flt["generation"]):
                continue
            if "keys" in flt and row.get("key") not in flt["keys"]:
                continue
            fields = row.get("fields") or {}
            if not all(_condition_holds(fields.get(n), c) for n, c in flt.get("fields", {}).items()):
                continue
            out.append({k: v for k, v in dict(row).items() if v is not None})
        return out

    # ---- summary, notices, turn header ------------------------------------------------------------------------

    def _summary(self, snapshot: HistoryView, flt: Mapping[str, Any]) -> dict[str, Any]:
        results = [e for e in snapshot.results if self._match(e, flt)]
        counts: dict[str, int] = {}
        for entry in results:
            counts[entry.status] = counts.get(entry.status, 0) + 1
        best = best_result(results, self.projection)
        objective = self.projection.objective
        return {"counts": dict(sorted(counts.items())), "results": len(results),
                "pending": sum(1 for e in snapshot.pending if self._match(e, flt)),
                "best": None if best is None else self.row(best),
                "objective": None if objective is None else {"name": objective.name, "direction": objective.direction},
                "fields": [f.to_dict() for f in self.projection.fields],
                "hint": (f"Use view=observations with filter.since_seq for increments, sort by objective for the top "
                         f"results, filter.keys to check candidates, view=pending for experiments in flight.")}

    def notice(self, from_seq: int, to_seq: int) -> dict[str, Any]:
        """Bounded notification of the log slice ``(from_seq, to_seq]`` (design 5.8 "新结果的及时通知")."""
        entries = self.log.entries(from_seq, to_seq)
        before = best_result(self.log.entries(0, from_seq), self.projection)
        after = best_result(self.log.entries(0, to_seq), self.projection)
        best_changed = after is not None and (before is None or after.seq != before.seq)
        results = [e for e in entries if e.kind == "result"]
        selected = [e for e in entries if e.kind == "selected"]
        lines = [f"[experiment history] {len(entries)} new entr{'y' if len(entries) == 1 else 'ies'} "
                 f"(seq {from_seq + 1}-{to_seq}); history is now at seq {to_seq}."]
        objective = self.projection.objective
        inline = None
        if len(entries) <= self.limits.inline_limit:
            inline = [self.row(e) for e in entries]
            for entry in entries:
                lines.append("- " + self._line(entry) + (" (new best)" if best_changed and after is not None
                                                         and entry.seq == after.seq else ""))
        else:
            counts: dict[str, int] = {}
            for entry in results:
                counts[entry.status] = counts.get(entry.status, 0) + 1
            lines.append(f"{len(results)} results ({', '.join(f'{n} {s}' for s, n in sorted(counts.items())) or 'none'}), "
                         f"{len(selected)} candidates selected for measurement.")
            if best_changed and objective is not None and after is not None:
                old = "none" if before is None else _fmt(before.fields.get(objective.name))
                lines.append(f"Best {objective.name} changed: {old} -> {_fmt(after.fields.get(objective.name))} "
                             f"({after.candidate_id}, seq {after.seq}).")
            if objective is not None:
                top = sorted((e for e in results if e.status == "succeeded" and e.fields.get(objective.name) is not None),
                             key=lambda e: (-1 if objective.direction == "maximize" else 1) * e.fields[objective.name])[:3]
                lines.extend("- " + self._line(e) for e in top)
        query = {"view": "observations", "filter": {"since_seq": from_seq}}
        lines.append(f"These are new facts, not instructions. Details: {self.tool_name}({json.dumps(query)}).")
        return {"from_seq": from_seq, "to_seq": to_seq, "count": len(entries), "results": len(results),
                "selected": len(selected), "best_changed": best_changed, "inline": inline,
                "best": None if after is None else self.row(after), "query": query, "text": "\n".join(lines)}

    def _line(self, entry: HistoryEntry) -> str:
        fields = ", ".join(f"{k}={_fmt(v)}" for k, v in entry.fields.items() if v is not None)
        what = "selected for measurement (pending)" if entry.kind == "selected" else f"result {entry.status}"
        return f"seq {entry.seq} {what} {entry.candidate_id}" + (f": {fields}" if fields else "")

    def turn_header(self, base_seq: int, last_visible_seq: int, *, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Fixed-size turn header (design 5.8 "prompt 里放什么"); independent of history length."""
        if not 0 <= last_visible_seq <= base_seq <= self.log.head:
            raise ValueError("turn header requires last_visible_seq <= base_seq <= head")
        snapshot = self.log.view(base_seq)
        new = self.log.entries(last_visible_seq, base_seq)
        best = best_result(snapshot.entries, self.projection)
        counts: dict[str, int] = {}
        for entry in snapshot.results:
            counts[entry.status] = counts.get(entry.status, 0) + 1
        return {"base_seq": base_seq, "last_visible_seq": last_visible_seq, "new_entries": len(new),
                "new_results": sum(1 for e in new if e.kind == "result"), "counts": dict(sorted(counts.items())),
                "measured": len(snapshot.measured_keys), "pending": len(snapshot.pending),
                "best": None if best is None else self.row(best), **dict(extra or {}),
                "instructions": (f"The history is not listed here. Use {self.tool_name}: view=summary first, then "
                                 f"view=observations with filter.since_seq={last_visible_seq} for what is new.")}


def _in_range(value: Any, bounds: Mapping[str, Any]) -> bool:
    return value is not None and bounds.get("min", value) <= value <= bounds.get("max", value)


def _same(a: Any, b: Any) -> bool:
    return isinstance(a, bool) == isinstance(b, bool) and a == b


def _condition_holds(value: Any, cond: Mapping[str, Any]) -> bool:
    if "eq" in cond:
        return _same(value, cond["eq"])
    if "in" in cond:
        return any(_same(value, item) for item in cond["in"])
    if value is None or isinstance(value, bool):
        return False
    try:
        return _in_range(value, cond)
    except TypeError:
        return False


def _fmt(value: Any) -> str:
    return f"{value:.6g}" if isinstance(value, float) else json.dumps(value) if not isinstance(value, str) else value


__all__ = ["HistoryQueryEngine", "HistoryQueryError", "PROPOSAL_STATUSES", "ProposalOutcomes", "QueryLimits",
           "VIEWS", "best_result"]
