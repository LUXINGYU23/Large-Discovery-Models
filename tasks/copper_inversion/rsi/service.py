"""Risk-only Unix-socket service for the evidence-only controller.

The old scientific evaluator runs inside this trusted process. Its full metrics
and per-case receipts remain private; only the allowlisted projection is sent.
"""

import argparse
import json
import logging
import math
import os
import secrets
import socketserver
from pathlib import Path
from types import SimpleNamespace

from ldm_tts.engine.run_store import atomic_json_write
from ..core.execution import digest
from ..v2.service import EvaluatorService, Handler
from ..v2.training_receipts import eligible
from .exposure import VERSION, risk_by_case, valid_case_id, validate_feedback
from .integrity import shared_identity


class RiskOnlyService(EvaluatorService):
    def __init__(self, configuration):
        super().__init__(configuration)
        path = self.root / "rsi-private-audit.json"
        cases = self.evaluator.manifest["cases"]
        parents = sorted({row["parent_id"] for row in cases})
        case_ids = sorted(row["case_id"] for row in cases)
        if len(parents) != len(cases) or any(not valid_case_id(value) for value in case_ids):
            raise ValueError("risk-only service requires one public case per geology parent")
        if path.exists():
            audit = json.loads(path.read_text())
            if (set(audit) != {"version", "sentinel", "private_parent_ids", "case_ids"}
                    or audit["version"] != VERSION or audit["private_parent_ids"] != parents
                    or audit["case_ids"] != case_ids):
                raise ValueError("private feedback audit identity changed")
        else:
            audit = {"version": VERSION, "sentinel": secrets.token_hex(16),
                     "private_parent_ids": parents, "case_ids": case_ids}
            atomic_json_write(path, audit)
        self.audit = audit
        self.shared_implementation = shared_identity()
        self.public_contract = digest({"service": digest(self.identity), "version": VERSION,
                                       "private_audit_sha256": digest(audit),
                                       "shared_implementation_sha256": self.shared_implementation})

    def seal_training(self):
        """Freeze the three lowest-risk eligible sources with at least one legal solve."""
        path = self.root / "training-sealed.json"
        if path.exists():
            record = json.loads(path.read_text())
            if record["contract_digest"] != digest(self.identity):
                raise ValueError("sealed contract differs")
        else:
            records = {sha: row for sha, row in eligible(self.root, self.manifest_path).items()
                       if row["metrics"]["legal_rate"] > 0}
            if not records:
                raise ValueError("no valid training candidate can be frozen")
            ordered = sorted(records, key=lambda sha: (records[sha]["metrics"]["loss_total"],
                             -records[sha]["metrics"]["legal_rate"],
                             records[sha]["metrics"]["mean_solver_wall_seconds"], sha))[:3]
            record = {"contract_digest": digest(self.identity),
                      "used_slots": self.ledger.counters.get("algorithm_feedback_slots", 0),
                      "source_sha256s": ordered, "selected_receipts": {sha: records[sha] for sha in ordered}}
            atomic_json_write(path, record)
        return {"status": "training_sealed", "contract_digest": record["contract_digest"],
                "source_sha256s": record["source_sha256s"], "used_slots": record["used_slots"]}

    def dispatch(self, request):
        if request == {"operation": "info"}:
            return dict(status="ready", contract_digest=self.public_contract,
                        public_digest=self.public_digest, seconds=self.seconds,
                        slots=self.slot_limit,
                        used_slots=self.ledger.counters.get("algorithm_feedback_slots", 0),
                        exposure_version=VERSION)
        if not isinstance(request, dict) or request.get("contract_digest") != self.public_contract:
            return {"status": "contract_mismatch"}
        if shared_identity() != self.shared_implementation:
            return {"status": "paused"}
        operation = request.get("operation")
        if operation not in {"evaluate", "seal_training"}:
            return {"status": "rejected_request"}
        underlying = {**request, "contract_digest": digest(self.identity)}
        response = super().dispatch(underlying)
        if operation == "seal_training":
            if response.get("status") == "training_sealed":
                return {**response, "contract_digest": self.public_contract}
            return response
        if response.get("status") not in {"succeeded", "invalid_interface"}:
            return response
        try:
            sha = response["source_sha256"]
            metrics = response["metrics"]
            failures = metrics["failed_solves"]
            if (type(failures) not in (int, float) or not math.isfinite(failures)
                    or failures < 0 or int(failures) != failures):
                raise ValueError("invalid private failure count")
            if response["status"] == "invalid_interface":
                case_risks = {name: 1.0 for name in self.audit["case_ids"]}
                status, message = "rejected", "interface rejected"
            else:
                key = self.evaluator.evaluation_attempt_usage_key(SimpleNamespace(canonical_key=sha))
                rows = json.loads((self.root / "evaluations" / key / "private-case-results.json").read_text())
                expected = {(case["case_id"], seed, case["parent_id"])
                            for case in self.evaluator.manifest["cases"]
                            for seed in self.evaluator.manifest["seeds"]}
                actual = [(row["case_id"], row["seed"], row["parent_id"]) for row in rows]
                if len(actual) != len(expected) or set(actual) != expected:
                    raise ValueError("private receipt does not cover the planned cases")
                case_risks = risk_by_case(rows, self.audit["case_ids"])
                status = "failed" if metrics["legal_rate"] == 0 else "valid"
                message = "all planned solves failed" if status == "failed" else ""
            feedback = validate_feedback(dict(
                source_sha256=sha, status=status, risk=metrics["loss_total"],
                risk_by_case=case_risks, legal_rate=metrics["legal_rate"],
                failed_solves=int(failures),
                seconds_per_solve=metrics["mean_solver_wall_seconds"], message=message))
            # The sentinel exists only in a private receipt. It must never be
            # present in any response, branch view, or provider request.
            private_path = self.root / "rsi-feedback" / (sha + ".json")
            private = {"feedback": feedback, "sentinel": self.audit["sentinel"]}
            if private_path.exists() and json.loads(private_path.read_text()) != private:
                raise ValueError("private feedback receipt changed")
            atomic_json_write(private_path, private)
            return feedback
        except Exception:
            logging.exception("private risk-only projection failed")
            return {"status": "infrastructure_error"}


def main():
    parser = argparse.ArgumentParser(description="Isolated risk-only copper evaluator")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    configuration = json.loads(args.config.read_text())
    backend = RiskOnlyService(configuration)
    with socketserver.UnixStreamServer(configuration["socket"], Handler) as server:
        os.chmod(configuration["socket"], 0o600)
        server.backend = backend
        print(json.dumps({"status": "listening", "pid": os.getpid(),
                          "contract_digest": backend.public_contract}), flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()
