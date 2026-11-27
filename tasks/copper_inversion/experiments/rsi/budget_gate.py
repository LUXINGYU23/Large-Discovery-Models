"""Framework-neutral evaluation budget gate over RiskOnlyService.

Enforces the fairness contract shared by external search-framework baselines:
1. Validate Python syntax + ``solve(problem, budget, seed)`` entrypoint, strip
   comments/docstrings, and enforce the 32,000-byte normalized-code limit.
2. Deduplicate by stripped-source SHA-256 (cache hits and concurrent in-flight
   duplicates consume zero evaluation slots).
3. Count evaluations (calls to ``RiskClient.evaluate_source``), including service
   interface rejections and all-failed solves; unparseable or oversized proposals
   rejected before service submission do not consume evaluation slots.
4. Refuse submissions once ``slots`` (default 321) novel evaluations are reserved
   or completed, touch a ``gate-exhausted`` marker file, fire ``on_exhausted``,
   and never raise on candidate or budget errors.
5. Append strictly aggregate-only records to ``ledger.jsonl`` compatible with
   ``experiments/rsi/freeze.py`` and ``rsi/audit.py``.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Callable

from ldm_tts.engine.run_store import BudgetExceededError
from tasks.copper_inversion.core.candidate import seal, validate_source
from tasks.copper_inversion.core.encoding import MAX_NORMALIZED_BYTES, normalized_code
from tasks.copper_inversion.rsi import ledger as ledgers
from tasks.copper_inversion.rsi.render import stripped_source

DEFAULT_SLOTS = 321


def prepare_submission(source: bytes | bytearray | str) -> tuple[bytes, str]:
    """Validate syntax + solve() signature, strip comments/docstrings, enforce 32 kB limit."""
    if isinstance(source, str):
        raw = source.encode("utf-8")
    elif isinstance(source, (bytes, bytearray)):
        raw = bytes(source)
    else:
        raise ValueError("solver source must be str or bytes")
    validate_source(raw)
    stripped_text = stripped_source(raw.decode("utf-8"))
    stripped_bytes = stripped_text.encode("utf-8")
    sha = validate_source(stripped_bytes)
    norm_size = len(normalized_code(stripped_bytes).encode("utf-8"))
    if norm_size > MAX_NORMALIZED_BYTES:
        raise ValueError(
            f"normalized solver is {norm_size} bytes (limit {MAX_NORMALIZED_BYTES} bytes)"
        )
    return stripped_bytes, sha


def read_submissions(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


class BudgetGate:
    """Thread- and process-safe 321-evaluation budget gate over RiskOnlyService."""

    def __init__(
        self,
        run_dir: str | Path,
        evaluator,
        *,
        slots: int = DEFAULT_SLOTS,
        case_feedback: str = "none",
        submissions_filename: str = "gate-submissions.jsonl",
        on_exhausted: Callable[[], None] | None = None,
    ):
        if case_feedback != "none":
            raise ValueError("external framework arms require aggregate-only feedback (case_feedback='none')")
        if type(slots) is not int or slots < 1:
            raise ValueError("slots must be a positive integer")
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.evaluator = evaluator
        self.slots = slots
        self.case_feedback = case_feedback
        self.ledger_path = self.run_dir / "ledger.jsonl"
        self.submissions_path = self.run_dir / submissions_filename
        self.lock_path = self.run_dir / ".gate-eval.lock"
        self.exhausted_path = self.run_dir / "gate-exhausted"
        self._on_exhausted = on_exhausted
        self._thread_lock = threading.Lock()
        self._in_flight: dict[str, threading.Event] = {}
        self._attempt_counter = len(read_submissions(self.submissions_path))
        self.exhausted = len(ledgers.read(self.ledger_path)) >= self.slots
        if self.exhausted:
            self.exhausted_path.touch(exist_ok=True)

    def set_on_exhausted(self, callback: Callable[[], None] | None) -> None:
        with self._thread_lock:
            self._on_exhausted = callback
            already = self.exhausted
        if already and callback is not None:
            callback()

    @property
    def used_slots(self) -> int:
        return len(ledgers.read(self.ledger_path))

    def _mark_exhausted_unlocked(self) -> Callable[[], None] | None:
        self.exhausted = True
        self.exhausted_path.touch(exist_ok=True)
        return self._on_exhausted

    def _next_attempt_unlocked(self) -> int:
        self._attempt_counter += 1
        return self._attempt_counter

    def submit(self, source: bytes | bytearray | str) -> tuple[float | None, str]:
        """Submit candidate source; return ``(risk | None, status)`` and never raise on candidate defects."""
        try:
            raw_bytes = source.encode("utf-8") if isinstance(source, str) else bytes(source)
        except Exception:
            raw_bytes = b""
        raw_sha = hashlib.sha256(raw_bytes).hexdigest()

        try:
            stripped_bytes, sha = prepare_submission(raw_bytes)
        except (ValueError, SyntaxError, UnicodeError) as exc:
            reason = "oversized" if "bytes" in str(exc) else "invalid_source"
            with self._thread_lock:
                with self.lock_path.open("a+", encoding="utf-8") as lock_stream:
                    fcntl.flock(lock_stream, fcntl.LOCK_EX)
                    attempt = self._next_attempt_unlocked()
                    _append_jsonl(self.submissions_path, {
                        "attempt": attempt,
                        "raw_sha256": raw_sha,
                        "status": reason,
                        "counted": False,
                        "service_slot": False,
                        "risk": None,
                    })
            return None, reason

        while True:
            wait_event = None
            fire_callback = None
            refused_exhausted = False
            with self._thread_lock:
                with self.lock_path.open("a+", encoding="utf-8") as lock_stream:
                    fcntl.flock(lock_stream, fcntl.LOCK_EX)
                    existing = ledgers.read(self.ledger_path)
                    by_sha = {row["program_id"]: row for row in existing}
                    if sha in by_sha:
                        prev = by_sha[sha]
                        attempt = self._next_attempt_unlocked()
                        is_valid = prev["status"] == "valid"
                        dup_status = "duplicate" if is_valid else f"duplicate_{prev['status']}"
                        _append_jsonl(self.submissions_path, {
                            "attempt": attempt,
                            "raw_sha256": raw_sha,
                            "program_id": sha,
                            "status": dup_status,
                            "counted": False,
                            "service_slot": False,
                            "risk": float(prev["risk"]) if is_valid else None,
                        })
                        return (float(prev["risk"]) if is_valid else None), dup_status

                    if sha in self._in_flight:
                        wait_event = self._in_flight[sha]
                    elif len(existing) + len(self._in_flight) >= self.slots:
                        refused_exhausted = True
                        attempt = self._next_attempt_unlocked()
                        fire_callback = self._mark_exhausted_unlocked()
                        _append_jsonl(self.submissions_path, {
                            "attempt": attempt,
                            "raw_sha256": raw_sha,
                            "program_id": sha,
                            "status": "budget_exhausted",
                            "counted": False,
                            "service_slot": False,
                            "risk": None,
                        })
                    else:
                        done_event = threading.Event()
                        self._in_flight[sha] = done_event
                        sealed = seal(self.run_dir, stripped_bytes)
                        source_file = self.run_dir / sealed["source_path"]
                        if len(existing) + len(self._in_flight) >= self.slots:
                            fire_callback = self._mark_exhausted_unlocked()
                        break

            if wait_event is not None:
                wait_event.wait()
                continue
            if refused_exhausted:
                if fire_callback is not None:
                    fire_callback()
                return None, "budget_exhausted"

        if fire_callback is not None:
            fire_callback()

        post_callback = None
        try:
            try:
                feedback = self.evaluator.evaluate_source(source_file)
            except BudgetExceededError:
                with self._thread_lock:
                    with self.lock_path.open("a+", encoding="utf-8") as lock_stream:
                        fcntl.flock(lock_stream, fcntl.LOCK_EX)
                        attempt = self._next_attempt_unlocked()
                        post_callback = self._mark_exhausted_unlocked()
                        _append_jsonl(self.submissions_path, {
                            "attempt": attempt,
                            "raw_sha256": raw_sha,
                            "program_id": sha,
                            "status": "budget_exhausted",
                            "counted": False,
                            "service_slot": False,
                            "risk": None,
                        })
                if post_callback is not None:
                    post_callback()
                return None, "budget_exhausted"
            except Exception:
                with self._thread_lock:
                    with self.lock_path.open("a+", encoding="utf-8") as lock_stream:
                        fcntl.flock(lock_stream, fcntl.LOCK_EX)
                        attempt = self._next_attempt_unlocked()
                        _append_jsonl(self.submissions_path, {
                            "attempt": attempt,
                            "raw_sha256": raw_sha,
                            "program_id": sha,
                            "status": "evaluator_error",
                            "counted": False,
                            "service_slot": False,
                            "risk": None,
                        })
                return None, "evaluator_error"

            if feedback.get("risk_by_case"):
                raise ValueError("service returned per-case risks under aggregate-only mode")

            with self._thread_lock:
                with self.lock_path.open("a+", encoding="utf-8") as lock_stream:
                    fcntl.flock(lock_stream, fcntl.LOCK_EX)
                    existing = ledgers.read(self.ledger_path)
                    seq = len(existing)
                    record = {
                        "seq": seq,
                        "program_id": sha,
                        "epoch": 0,
                        "round": 0 if seq == 0 else seq,
                        "branch": -1 if seq == 0 else 0,
                        "status": feedback["status"],
                        "risk": float(feedback["risk"]),
                        "risk_by_case": {},
                        "legal_rate": float(feedback["legal_rate"]),
                        "failed_solves": int(feedback["failed_solves"]),
                        "seconds_per_solve": float(feedback["seconds_per_solve"]),
                        "message": feedback["message"],
                        "selection": "initial" if seq == 0 else "acquisition",
                    }
                    ledgers.append(self.ledger_path, record)
                    attempt = self._next_attempt_unlocked()
                    _append_jsonl(self.submissions_path, {
                        "attempt": attempt,
                        "seq": seq,
                        "raw_sha256": raw_sha,
                        "program_id": sha,
                        "status": record["status"],
                        "counted": True,
                        "service_slot": True,
                        "risk": record["risk"],
                    })
                    if seq + 1 >= self.slots:
                        post_callback = self._mark_exhausted_unlocked()
        finally:
            with self._thread_lock:
                ev = self._in_flight.pop(sha, None)
                if ev is not None:
                    ev.set()

        if post_callback is not None:
            post_callback()
        if record["status"] == "valid":
            return record["risk"], "valid"
        return None, record["status"]

    def summary(self) -> dict:
        records = ledgers.read(self.ledger_path)
        subs = read_submissions(self.submissions_path)
        counts: dict[str, int] = {}
        for row in subs:
            st = str(row.get("status") or "unknown")
            counts[st] = counts.get(st, 0) + 1
        return {
            "slots_limit": self.slots,
            "used_slots": len(records),
            "exhausted": bool(self.exhausted or len(records) >= self.slots),
            "total_submissions": len(subs),
            "counted_evaluations": sum(1 for r in subs if r.get("counted", False)),
            "valid_evaluations": sum(1 for r in records if r["status"] == "valid"),
            "rejected_evaluations": sum(1 for r in records if r["status"] == "rejected"),
            "failed_evaluations": sum(1 for r in records if r["status"] == "failed"),
            "duplicate_submissions": sum(v for k, v in counts.items() if k.startswith("duplicate")),
            "invalid_source_submissions": counts.get("invalid_source", 0),
            "oversized_submissions": counts.get("oversized", 0),
            "budget_exhausted_submissions": counts.get("budget_exhausted", 0),
            "status_counts": counts,
        }
