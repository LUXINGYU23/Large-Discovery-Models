"""Strict client for the risk-only service; never accepts hidden metrics."""

import json
from pathlib import Path

from ldm_tts.contracts.evaluation import EvaluationPaused
from ldm_tts.engine.run_store import BudgetExceededError, atomic_json_write
from ..core.candidate import validate_source
from ..v2.service_client import ServiceEvaluator
from .exposure import VERSION, validate_feedback


class RiskClient(ServiceEvaluator):
    def verify_contract(self, public_digest, seconds, slots):
        info = self.request({"operation": "info"})
        if (set(info) != {"status", "contract_digest", "public_digest", "seconds", "slots",
                          "used_slots", "exposure_version"}
                or info["status"] != "ready" or info["contract_digest"] != self.contract_digest
                or info["public_digest"] != public_digest or info["seconds"] != seconds
                or info["slots"] != slots or info["exposure_version"] != VERSION
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
                result = validate_feedback(saved["feedback"])
                if result["source_sha256"] != sha:
                    raise ValueError("feedback source differs")
                return result
            except (ValueError, KeyError, TypeError) as exc:
                raise EvaluationPaused("saved risk-only feedback is invalid") from exc
        response = self.request({"operation": "evaluate", "source": body.decode("utf-8"),
                                 "contract_digest": self.contract_digest})
        if response.get("status") == "budget_exhausted":
            raise BudgetExceededError("risk-only feedback budget exhausted")
        try:
            result = validate_feedback(response)
            if result["source_sha256"] != sha:
                raise ValueError("feedback source differs")
        except (ValueError, KeyError, TypeError) as exc:
            raise EvaluationPaused("risk-only evaluation unavailable or unsafe") from exc
        atomic_json_write(path, {"contract_digest": self.contract_digest, "feedback": result})
        return result

    def seal_training(self):
        reply = self.request({"operation": "seal_training", "contract_digest": self.contract_digest})
        if (set(reply) != {"status", "contract_digest", "source_sha256s", "used_slots"}
                or reply["status"] != "training_sealed" or reply["contract_digest"] != self.contract_digest):
            raise EvaluationPaused("private training seal unavailable")
        return reply
