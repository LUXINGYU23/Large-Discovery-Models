"""Strict client for the risk-only service; never accepts hidden metrics."""

import json
import re
from pathlib import Path

from ldm_tts.contracts.evaluation import EvaluationPaused
from ldm_tts.engine.run_store import BudgetExceededError, atomic_json_write
from ..core.candidate import validate_source
from ..v2.service_client import ServiceEvaluator
from . import exposure
from .exposure import FIELDS, VERSION, validate_feedback


INFO = frozenset({"status", "contract_digest", "public_digest", "seconds", "slots", "used_slots",
                  "exposure_version", "objective", "tier", "import_ledger_prefix_sha256"})
_ANY = object()
# Service refusals that are not results; anything else must be exactly allowlisted feedback.
REFUSALS = frozenset({"paused", "infrastructure_error", "contract_mismatch", "rejected_request",
                      "invalid_source", "training_sealed"})


def checked_feedback(response, sha):
    """Allowlisted feedback for sha; any other key set is an infrastructure pause."""
    if not isinstance(response, dict):
        raise EvaluationPaused("risk-only evaluator response is not an object")
    if set(response) - FIELDS and set(response) != {"status"} and not exposure.UNSAFE_DISABLE_WHITELIST:
        # Never a result and never saved: extra keys mean the whitelist failed.
        raise EvaluationPaused("risk-only evaluator returned non-allowlisted fields; infrastructure pause")
    if set(response) == {"status"}:
        raise EvaluationPaused("risk-only evaluation did not complete: "
                               + (response["status"] if response["status"] in REFUSALS else "unknown"))
    try:
        result = validate_feedback(response)
        if result["source_sha256"] != sha:
            raise ValueError("feedback source differs")
    except (ValueError, KeyError, TypeError) as exc:
        raise EvaluationPaused("risk-only evaluation unavailable or unsafe") from exc
    return result


class RiskClient(ServiceEvaluator):
    def verify_contract(self, public_digest, seconds, slots, *, objective=None, tier=None,
                        import_ledger_prefix_sha256=_ANY):
        """import_ledger_prefix_sha256: expected sha256 hex of the imported ledger-prefix
        file bytes, or None to require a service that was not started by import."""
        info = self.request({"operation": "info"})
        if (set(info) != INFO
                or info["status"] != "ready" or info["contract_digest"] != self.contract_digest
                or info["public_digest"] != public_digest or info["seconds"] != seconds
                or info["slots"] != slots or info["exposure_version"] != VERSION
                or info["objective"] not in ("confirmation", "oracle_iou")
                or (objective is not None and info["objective"] != objective)
                or (tier is not None and info["tier"] != tier)
                or not (info["import_ledger_prefix_sha256"] is None
                        or re.fullmatch(r"[0-9a-f]{64}", str(info["import_ledger_prefix_sha256"])))
                or (import_ledger_prefix_sha256 is not _ANY
                    and info["import_ledger_prefix_sha256"] != import_ledger_prefix_sha256)
                or type(info["used_slots"]) is not int or not 0 <= info["used_slots"] <= slots):
            raise EvaluationPaused("risk-only evaluator identity or budget differs")
        return info

    def evaluate_source(self, source):
        body = Path(source).read_bytes()
        sha = validate_source(body)
        path = self.root / "feedback" / (sha + ".json")
        if path.exists():
            try:
                saved = json.loads(path.read_text())
                if set(saved) != {"contract_digest", "feedback"} or saved["contract_digest"] != self.contract_digest:
                    raise ValueError("feedback contract differs")
                return checked_feedback(saved["feedback"], sha)
            except (ValueError, KeyError, TypeError, EvaluationPaused) as exc:
                raise EvaluationPaused("saved risk-only feedback is invalid") from exc
        response = self.request({"operation": "evaluate", "source": body.decode("utf-8"),
                                 "contract_digest": self.contract_digest})
        if response == {"status": "budget_exhausted"}:
            raise BudgetExceededError("risk-only feedback budget exhausted")
        result = checked_feedback(response, sha)
        atomic_json_write(path, {"contract_digest": self.contract_digest, "feedback": result})
        return result

    def seal_training(self):
        reply = self.request({"operation": "seal_training", "contract_digest": self.contract_digest})
        if (set(reply) != {"status", "contract_digest", "source_sha256s", "used_slots"}
                or reply["status"] != "training_sealed" or reply["contract_digest"] != self.contract_digest):
            raise EvaluationPaused("private training seal unavailable")
        return reply
