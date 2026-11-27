"""Shared process-safe 321-evaluation budget gate for external search-framework baselines.

Enforces the same candidate hygiene and RiskOnlyService contract across OpenEvolve,
ShinkaEvolve, LLM4AD, and other external search frameworks:
1. Validate Python syntax + ``solve(problem, budget, seed)`` entrypoint, strip comments
   and docstrings, and enforce the 32,000-byte normalized code limit.
2. Replay cached scores for duplicate stripped-code SHA-256 digests without charging a slot.
3. Refuse malformed/oversized proposals locally without consuming one of the 321 service
   evaluations (failed proposals still count toward the outer LLM-call safety cap).
4. Evaluate novel valid sources concurrently against ``RiskOnlyService`` (up to 16 workers
   per run), charging 1 slot per service evaluation (valid, failed, or interface-rejected),
   stopping cleanly at exact ``slots`` (321) evaluations and writing ``gate.exhausted``.
"""

import fcntl
import hashlib
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

from ldm_tts.engine.run_store import BudgetExceededError
from tasks.copper_inversion.core.candidate import seal, validate_source
from tasks.copper_inversion.core.encoding import MAX_NORMALIZED_BYTES, normalized_code
from tasks.copper_inversion.rsi import ledger as ledgers
from tasks.copper_inversion.rsi.case_isolation import check_source_case_ids
from tasks.copper_inversion.rsi.render import stripped_source

DEFAULT_SLOTS = 321


def prepare_submission(source_bytes, *, forbidden_case_tokens=()):
    """Validate Python syntax + solve() entrypoint, strip comments/docstrings, and enforce 32 kB limit."""
    if not isinstance(source_bytes, (bytes, bytearray)):
        raise ValueError("solver source must be bytes")
    raw = bytes(source_bytes)
    validate_source(raw)
    stripped_text = stripped_source(raw.decode("utf-8"))
    stripped_bytes = stripped_text.encode("utf-8")
    sha = validate_source(stripped_bytes)
    if forbidden_case_tokens:
        check_source_case_ids(stripped_bytes, forbidden_case_tokens)
    norm_size = len(normalized_code(stripped_bytes).encode("utf-8"))
    if norm_size > MAX_NORMALIZED_BYTES:
        raise ValueError(
            f"normalized solver is {norm_size} bytes (limit {MAX_NORMALIZED_BYTES} bytes)"
        )
    return stripped_bytes, sha


def read_submissions(path):
    path = Path(path)
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_submission(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _probe_active_inflight(inflight_dir):
    """Return set of SHA-256s currently held by live worker processes; reap stale files."""
    active = set()
    if not inflight_dir.is_dir():
        return active
    for entry in sorted(inflight_dir.glob("*.active")):
        sha = entry.stem
        try:
            fd = os.open(entry, os.O_RDWR)
        except FileNotFoundError:
            continue
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                active.add(sha)
            else:
                try:
                    entry.unlink()
                except FileNotFoundError:
                    pass
                fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)
    return active


class BudgetGate:
    """Process-safe, concurrency-safe 321-evaluation gate over RiskOnlyService."""

    def __init__(
        self,
        run_dir,
        evaluator,
        *,
        slots=DEFAULT_SLOTS,
        case_feedback="none",
        submissions_filename="gate-submissions.jsonl",
    ):
        if case_feedback != "none":
            raise ValueError("external framework gate requires aggregate-only feedback (case_feedback='none')")
        if type(slots) is not int or slots < 1:
            raise ValueError("slots must be a positive integer")
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.evaluator = evaluator
        self.slots = slots
        self.case_feedback = case_feedback
        self.ledger_path = self.run_dir / "ledger.jsonl"
        self.submissions_path = self.run_dir / submissions_filename
        self.lock_path = self.run_dir / ".gate.lock"
        self.inflight_dir = self.run_dir / ".gate-inflight"
        self.inflight_dir.mkdir(parents=True, exist_ok=True)
        self.exhausted_path = self.run_dir / "gate.exhausted"

    @property
    def exhausted(self):
        if self.exhausted_path.is_file():
            return True
        if len(ledgers.read(self.ledger_path)) >= self.slots:
            self._mark_exhausted()
            return True
        return False

    def used_slots(self):
        return len(ledgers.read(self.ledger_path))

    def _mark_exhausted(self):
        if not self.exhausted_path.is_file():
            self.exhausted_path.write_text(f"{self.slots}\n", encoding="utf-8")

    @contextmanager
    def _meta_lock(self):
        with self.lock_path.open("a+", encoding="utf-8") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)

    def submit(self, source, *, raise_on_leak=True):
        """Submit candidate bytes/str/Path; return ``(risk, status)`` and never raise on bad code."""
        if isinstance(source, Path):
            source_bytes = source.read_bytes()
        elif isinstance(source, str):
            source_bytes = source.encode("utf-8")
        elif isinstance(source, (bytes, bytearray)):
            source_bytes = bytes(source)
        else:
            return 1.0, "invalid_source"

        raw_sha = hashlib.sha256(source_bytes).hexdigest()
        try:
            stripped_bytes, sha = prepare_submission(source_bytes)
        except (ValueError, SyntaxError, UnicodeError) as exc:
            reason = "oversized" if "bytes" in str(exc) else "invalid_source"
            with self._meta_lock():
                existing = ledgers.read(self.ledger_path)
                submissions = read_submissions(self.submissions_path)
                if len(existing) >= self.slots:
                    self._mark_exhausted()
                    append_submission(self.submissions_path, {
                        "attempt": len(submissions) + 1, "raw_sha256": raw_sha,
                        "status": "budget_exhausted", "counted": False, "service_slot": False, "risk": 1.0,
                    })
                    return 1.0, "budget_exhausted"
                append_submission(self.submissions_path, {
                    "attempt": len(submissions) + 1, "raw_sha256": raw_sha,
                    "status": reason, "counted": False, "service_slot": False, "risk": 1.0,
                })
            return 1.0, reason

        inflight_fd = None
        inflight_file = self.inflight_dir / f"{sha}.active"
        source_file = None

        while True:
            wait_on_same_sha = False
            wait_on_capacity = False
            with self._meta_lock():
                existing = ledgers.read(self.ledger_path)
                submissions = read_submissions(self.submissions_path)
                by_sha = {row["program_id"]: row for row in existing}
                if sha in by_sha:
                    prev = by_sha[sha]
                    append_submission(self.submissions_path, {
                        "attempt": len(submissions) + 1, "raw_sha256": raw_sha, "program_id": sha,
                        "status": "duplicate", "counted": False, "service_slot": False,
                        "risk": float(prev["risk"]),
                    })
                    return float(prev["risk"]), "duplicate"

                if len(existing) >= self.slots:
                    self._mark_exhausted()
                    append_submission(self.submissions_path, {
                        "attempt": len(submissions) + 1, "raw_sha256": raw_sha, "program_id": sha,
                        "status": "budget_exhausted", "counted": False, "service_slot": False, "risk": 1.0,
                    })
                    return 1.0, "budget_exhausted"

                active = _probe_active_inflight(self.inflight_dir)
                if sha in active:
                    wait_on_same_sha = True
                elif len(existing) + len(active) >= self.slots:
                    wait_on_capacity = True
                else:
                    inflight_fd = os.open(inflight_file, os.O_CREAT | os.O_RDWR, 0o666)
                    fcntl.flock(inflight_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    sealed = seal(self.run_dir, stripped_bytes)
                    source_file = self.run_dir / sealed["source_path"]
                    break

            if wait_on_same_sha:
                try:
                    wfd = os.open(inflight_file, os.O_RDONLY)
                except FileNotFoundError:
                    continue
                try:
                    fcntl.flock(wfd, fcntl.LOCK_SH)
                    fcntl.flock(wfd, fcntl.LOCK_UN)
                finally:
                    os.close(wfd)
            elif wait_on_capacity:
                time.sleep(0.05)

        try:
            try:
                feedback = self.evaluator.evaluate_source(source_file)
            except BudgetExceededError:
                with self._meta_lock():
                    self._mark_exhausted()
                    submissions = read_submissions(self.submissions_path)
                    append_submission(self.submissions_path, {
                        "attempt": len(submissions) + 1, "raw_sha256": raw_sha, "program_id": sha,
                        "status": "budget_exhausted", "counted": False, "service_slot": False, "risk": 1.0,
                    })
                return 1.0, "budget_exhausted"
            except Exception as exc:
                with self._meta_lock():
                    submissions = read_submissions(self.submissions_path)
                    append_submission(self.submissions_path, {
                        "attempt": len(submissions) + 1, "raw_sha256": raw_sha, "program_id": sha,
                        "status": "evaluator_error", "error": str(exc)[:200],
                        "counted": False, "service_slot": False, "risk": 1.0,
                    })
                return 1.0, "evaluator_error"

            if feedback.get("risk_by_case"):
                if raise_on_leak:
                    raise ValueError("service returned per-case risks under aggregate-only mode")
                return 1.0, "leak_refused"

            with self._meta_lock():
                existing = ledgers.read(self.ledger_path)
                submissions = read_submissions(self.submissions_path)
                by_sha = {row["program_id"]: row for row in existing}
                if sha in by_sha:
                    prev = by_sha[sha]
                    return float(prev["risk"]), "duplicate"
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
                append_submission(self.submissions_path, {
                    "attempt": len(submissions) + 1, "seq": seq, "raw_sha256": raw_sha,
                    "program_id": sha, "status": record["status"], "counted": True,
                    "service_slot": True, "risk": record["risk"],
                })
                if seq + 1 >= self.slots:
                    self._mark_exhausted()
                return record["risk"], record["status"]
        finally:
            if inflight_fd is not None:
                try:
                    inflight_file.unlink()
                except FileNotFoundError:
                    pass
                try:
                    fcntl.flock(inflight_fd, fcntl.LOCK_UN)
                finally:
                    os.close(inflight_fd)


def submit(source, *, gate):
    """Framework-neutral helper forwarding ``source`` to ``gate.submit``."""
    return gate.submit(source)
