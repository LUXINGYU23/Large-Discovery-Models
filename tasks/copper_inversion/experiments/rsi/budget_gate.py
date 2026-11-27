"""Framework-neutral evaluation budget gate for external search baselines.

Enforces the exact fairness contract shared with LDM4Geo (contracts §2.1,
frameworks-common.md):
  1. Validate Python syntax + ``solve(problem, budget, seed)`` entrypoint,
     strip comments/docstrings (including framework markers like
     ``# EVOLVE-BLOCK-START``), and enforce the 32,000-byte normalized code
     limit (``prepare_submission``).
  2. Deduplicate by stripped-source SHA-256 so re-evaluating an identical
     program costs zero evaluation slots.
  3. Submit novel valid-syntax candidates to ``RiskOnlyService`` via
     ``RiskClient.evaluate_source``, counting one slot per service evaluation
     (valid, failed, and interface-rejected candidates included) and writing
     contiguous ``ledger.jsonl`` records compatible with ``freeze.py``.
  4. Pre-evaluation proposal failures (syntax errors, missing ``solve``,
     oversized code) are logged in ``submissions.jsonl`` without calling the
     service or consuming an evaluation slot.
  5. Refuse beyond ``slots`` (default 321), set ``gate.exhausted = True``, and
     touch ``gate-exhausted`` atomically without calling the service.
  6. Never raise on candidate errors (only raise on aggregate-only contract
     violations from the service).
"""

import fcntl
import hashlib
import json
import os
from pathlib import Path

from ldm_tts.engine.run_store import BudgetExceededError, atomic_json_write
from tasks.copper_inversion.core.candidate import seal, validate_source
from tasks.copper_inversion.core.encoding import MAX_NORMALIZED_BYTES, normalized_code
from tasks.copper_inversion.rsi import ledger as ledgers
from tasks.copper_inversion.rsi.render import stripped_source

DEFAULT_SLOTS = 321
EXHAUSTED_FILENAME = "gate-exhausted"
SUBMISSIONS_FILENAME = "submissions.jsonl"


def prepare_submission(source_bytes):
    """Validate Python syntax + solve() entrypoint, strip comments/docstrings, and enforce 32 kB limit."""
    if isinstance(source_bytes, str):
        raw = source_bytes.encode("utf-8")
    elif isinstance(source_bytes, (bytes, bytearray)):
        raw = bytes(source_bytes)
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


def score_from_risk(risk, *, accepted=True):
    """Map aggregate confirmation risk in [0, 1] to a maximized score in [0, 1]."""
    if not accepted or risk is None:
        return 0.0
    value = float(risk)
    return float(max(0.0, min(1.0, 1.0 - value)))


def read_submissions(path):
    path = Path(path)
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _append_submission(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        )
        stream.flush()
        os.fsync(stream.fileno())


def _coerce_source_bytes(source):
    if isinstance(source, Path):
        return source.read_bytes()
    if isinstance(source, (bytes, bytearray)):
        return bytes(source)
    if isinstance(source, str):
        if "\n" not in source and len(source) < 1024:
            candidate_path = Path(source)
            if candidate_path.is_file():
                return candidate_path.read_bytes()
        return source.encode("utf-8")
    raise ValueError("source must be bytes, str, or Path")


class BudgetGate:
    """Process-safe 321-evaluation gate shared across external search frameworks."""

    def __init__(
        self,
        run_dir,
        evaluator,
        *,
        slots=DEFAULT_SLOTS,
        case_feedback="none",
        submissions_filename=SUBMISSIONS_FILENAME,
    ):
        if case_feedback != "none":
            raise ValueError(
                "external search arm requires aggregate-only feedback (case_feedback='none')"
            )
        if type(slots) is not int or slots < 1:
            raise ValueError("slots must be a positive integer")
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.evaluator = evaluator
        self.slots = slots
        self.case_feedback = case_feedback
        self.ledger_path = self.run_dir / "ledger.jsonl"
        self.submissions_path = self.run_dir / submissions_filename
        self.exhausted_path = self.run_dir / EXHAUSTED_FILENAME
        self.lock_path = self.run_dir / ".budget-gate.lock"
        self.exhausted = self.exhausted_path.is_file() or (
            len(ledgers.read(self.ledger_path)) >= self.slots
        )
        if self.exhausted and not self.exhausted_path.is_file():
            self._mark_exhausted(len(ledgers.read(self.ledger_path)))

    def _mark_exhausted(self, used_slots):
        self.exhausted = True
        atomic_json_write(
            self.exhausted_path,
            {"exhausted": True, "used_slots": int(used_slots), "slots": int(self.slots)},
        )

    @property
    def used_slots(self):
        return len(ledgers.read(self.ledger_path))

    def submit(self, source):
        """Submit candidate source or file path; return ``(risk | None, status)`` and never raise on candidate error."""
        detail = self.submit_detailed(source)
        return (detail["risk"] if detail["accepted"] else None), detail["status"]

    def submit_detailed(self, source):
        """Submit candidate source or file path and return full aggregate outcome dict."""
        try:
            source_bytes = _coerce_source_bytes(source)
        except (OSError, UnicodeError, ValueError) as exc:
            return {
                "risk": None,
                "combined_score": 0.0,
                "accepted": False,
                "status": "invalid_source",
                "counted": False,
                "used_slots": self.used_slots,
                "exhausted": self.exhausted,
                "program_id": None,
                "error": str(exc)[:200],
            }
        with self.lock_path.open("a+", encoding="utf-8") as lock_stream:
            fcntl.flock(lock_stream, fcntl.LOCK_EX)
            return self._submit_locked(source_bytes)

    def _submit_locked(self, source_bytes):
        existing = ledgers.read(self.ledger_path)
        used = len(existing)
        if used >= self.slots and not self.exhausted:
            self._mark_exhausted(used)
        submissions = read_submissions(self.submissions_path)
        by_sha = {row["program_id"]: row for row in existing}
        raw_sha = hashlib.sha256(source_bytes).hexdigest()

        try:
            stripped_bytes, sha = prepare_submission(source_bytes)
        except (ValueError, SyntaxError, UnicodeError) as exc:
            reason = "oversized" if "bytes" in str(exc) else "invalid_source"
            _append_submission(
                self.submissions_path,
                {
                    "attempt": len(submissions) + 1,
                    "raw_sha256": raw_sha,
                    "status": reason,
                    "counted": False,
                    "service_slot": False,
                    "risk": 1.0,
                    "error": str(exc)[:200],
                },
            )
            return {
                "risk": None,
                "combined_score": 0.0,
                "accepted": False,
                "status": reason,
                "counted": False,
                "used_slots": used,
                "exhausted": self.exhausted,
                "program_id": None,
            }

        if sha in by_sha:
            prev = by_sha[sha]
            prev_risk = float(prev["risk"])
            accepted = prev["status"] == "valid"
            _append_submission(
                self.submissions_path,
                {
                    "attempt": len(submissions) + 1,
                    "raw_sha256": raw_sha,
                    "program_id": sha,
                    "status": "duplicate",
                    "counted": False,
                    "service_slot": False,
                    "risk": prev_risk,
                },
            )
            return {
                "risk": prev_risk if accepted else None,
                "combined_score": score_from_risk(prev_risk, accepted=accepted),
                "accepted": accepted,
                "status": "duplicate",
                "counted": False,
                "used_slots": used,
                "exhausted": self.exhausted,
                "program_id": sha,
            }

        if used >= self.slots:
            self._mark_exhausted(used)
            _append_submission(
                self.submissions_path,
                {
                    "attempt": len(submissions) + 1,
                    "raw_sha256": raw_sha,
                    "program_id": sha,
                    "status": "budget_exhausted",
                    "counted": False,
                    "service_slot": False,
                    "risk": 1.0,
                },
            )
            return {
                "risk": None,
                "combined_score": 0.0,
                "accepted": False,
                "status": "budget_exhausted",
                "counted": False,
                "used_slots": used,
                "exhausted": True,
                "program_id": sha,
            }

        sealed = seal(self.run_dir, stripped_bytes)
        source_file = self.run_dir / sealed["source_path"]
        try:
            feedback = self.evaluator.evaluate_source(source_file)
        except BudgetExceededError:
            self._mark_exhausted(used)
            _append_submission(
                self.submissions_path,
                {
                    "attempt": len(submissions) + 1,
                    "raw_sha256": raw_sha,
                    "program_id": sha,
                    "status": "budget_exhausted",
                    "counted": False,
                    "service_slot": False,
                    "risk": 1.0,
                },
            )
            return {
                "risk": None,
                "combined_score": 0.0,
                "accepted": False,
                "status": "budget_exhausted",
                "counted": False,
                "used_slots": used,
                "exhausted": True,
                "program_id": sha,
            }

        if feedback.get("risk_by_case"):
            raise ValueError("service returned per-case risks under aggregate-only mode")

        seq = used
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
        used = seq + 1
        if used >= self.slots:
            self._mark_exhausted(used)
        _append_submission(
            self.submissions_path,
            {
                "attempt": len(submissions) + 1,
                "seq": seq,
                "raw_sha256": raw_sha,
                "program_id": sha,
                "status": record["status"],
                "counted": True,
                "service_slot": True,
                "risk": record["risk"],
            },
        )
        accepted = record["status"] == "valid"
        return {
            "risk": record["risk"] if accepted else None,
            "combined_score": score_from_risk(record["risk"], accepted=accepted),
            "accepted": accepted,
            "status": record["status"],
            "counted": True,
            "used_slots": used,
            "exhausted": self.exhausted,
            "program_id": sha,
        }


def submit(source, *, gate):
    """Module-level helper delegating ``submit(source) -> (risk | None, status)`` to ``gate``."""
    return gate.submit(source)
