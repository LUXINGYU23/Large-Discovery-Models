"""Risk-only Unix-socket service for the evidence-only controller.

The old scientific evaluator runs inside this trusted process. Its full metrics
and per-case receipts remain private; only the allowlisted projection is sent.
"""

import argparse
import hashlib
import json
import logging
import math
import os
import secrets
import shutil
import socketserver
from pathlib import Path
from types import SimpleNamespace

from ldm_tts.engine.run_store import atomic_json_write
from ..core.candidate import seal, validate_source
from ..core.execution import digest
from ..v2.service import EvaluatorService, Handler
from ..v2.training_receipts import eligible
from .case_isolation import (LEGACY_CASE_ISOLATION, check_source_case_ids, forbidden_case_tokens,
                             opaque_case_id, resolve_case_isolation)
from .exposure import FIELDS, VERSION, project, risk_by_case, valid_case_id, validate_case_feedback
from .integrity import shared_identity

IMPORT_KEYS = frozenset({"source_root", "ledger_prefix", "ledger_prefix_sha256"})
# Per-run paths and the import itself may differ between a source service and its continuation.
_RUN_LOCAL = frozenset({"root", "socket", "caller_uid", "import"})


def _comparable(identity):
    return {**identity, "configuration": {k: v for k, v in identity["configuration"].items()
                                          if k not in _RUN_LOCAL}}


def _link_tree(source, destination):
    """Copy immutable receipts, sharing inodes when the filesystem allows it."""
    def link(src, dst):
        if Path(dst).exists():
            if Path(src).read_bytes() != Path(dst).read_bytes():
                raise ValueError("imported receipt differs from an existing copy")
            return dst
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
        return dst
    shutil.copytree(source, destination, copy_function=link, dirs_exist_ok=True, symlinks=False)


class RiskOnlyService(EvaluatorService):
    optional_keys = EvaluatorService.optional_keys | {"import", "case_isolation"}
    objective = "confirmation"
    tier = None
    case_isolation = LEGACY_CASE_ISOLATION
    _forbidden_case_tokens = frozenset()
    # SHA-256 hex of the exact ledger-prefix file bytes this service imported
    # (the launcher's copy of --import-ledger, pinned in import.ledger_prefix_sha256);
    # None when the service was not started by import.
    import_ledger_prefix_sha256 = None

    def __init__(self, configuration):
        if isinstance(configuration, dict) and "purpose" in configuration:
            raise ValueError("discovery feedback cannot use a measurement-only suite")
        raw_iso = configuration.get("case_isolation") if isinstance(configuration, dict) else None
        isolation = resolve_case_isolation(raw_iso)
        if isinstance(configuration, dict) and "case_isolation" in configuration:
            for flag in ("anonymize_cases", "canonical_origin", "randomize_orientation"):
                if flag in configuration and configuration[flag] != isolation[flag]:
                    raise ValueError(f"{flag} differs from case_isolation.{flag}")
            stored_iso = dict(isolation)
            if isinstance(raw_iso, dict) and "randomize_orientation" not in raw_iso and not isolation["randomize_orientation"]:
                stored_iso.pop("randomize_orientation")
            extra = {"randomize_orientation": isolation["randomize_orientation"]} if (
                isolation["randomize_orientation"] or "randomize_orientation" in configuration) else {}
            configuration = {**configuration, "case_isolation": stored_iso,
                             "anonymize_cases": isolation["anonymize_cases"],
                             "canonical_origin": isolation["canonical_origin"], **extra}
        super().__init__(configuration)
        self.case_isolation = isolation
        path = self.root / "rsi-private-audit.json"
        cases = self.evaluator.manifest["cases"]
        parents = sorted({row["parent_id"] for row in cases})
        case_ids = sorted(row["case_id"] for row in cases)
        if len(parents) != len(cases) or any(not valid_case_id(value) for value in case_ids):
            raise ValueError("risk-only service requires one public case per geology parent")
        self._forbidden_case_tokens = (forbidden_case_tokens(case_ids, len(case_ids))
                                       if isolation["refuse_case_ids"] else frozenset())
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
        if "import" in configuration:
            self.import_prefix(configuration["import"])
            self.import_ledger_prefix_sha256 = configuration["import"]["ledger_prefix_sha256"]
        self.public_contract = digest({"service": digest(self.identity), "version": VERSION,
                                       "private_audit_sha256": digest(audit),
                                       "shared_implementation_sha256": self.shared_implementation})

    def import_prefix(self, spec):
        """Continuation hook: charge a source run's ledger prefix to this new service.

        Copies the prefix's sources and whole-suite receipts from the source
        service directory, charges one slot per prefix record under the same
        usage keys, and re-projects each record to prove it equals the public
        prefix. Idempotent, so an interrupted import is completed on restart.
        """
        if not isinstance(spec, dict) or set(spec) != IMPORT_KEYS:
            raise ValueError("invalid continuation import")
        body = Path(spec["ledger_prefix"]).read_bytes()
        if hashlib.sha256(body).hexdigest() != spec["ledger_prefix_sha256"]:
            raise ValueError("imported ledger prefix differs from its pinned digest")
        from .ledger import validate
        rows = [validate(json.loads(line)) for line in body.splitlines() if line.strip()]
        if not rows or [r["seq"] for r in rows] != list(range(len(rows))):
            raise ValueError("imported ledger prefix is empty or not contiguous")
        if len({r["program_id"] for r in rows}) != len(rows):
            raise ValueError("imported ledger prefix repeats a program")
        source = Path(spec["source_root"]).resolve()
        if source == self.root.resolve():
            raise ValueError("continuation requires a new service directory")
        if _comparable(json.loads((source / "identity.json").read_text())) != _comparable(self.identity):
            raise ValueError("source service measured a different scientific contract")
        source_audit = json.loads((source / "rsi-private-audit.json").read_text())
        if (source_audit["private_parent_ids"] != self.audit["private_parent_ids"]
                or source_audit["case_ids"] != self.audit["case_ids"]):
            raise ValueError("source service evaluated different cases")
        usage = json.loads((source / "budget.json").read_text()).get("metadata", {}).get("cumulative_usage", {})
        header = {"source_contract_digest": digest(json.loads((source / "identity.json").read_text())),
                  "ledger_prefix_sha256": spec["ledger_prefix_sha256"], "records": len(rows)}
        receipt_path = self.root / "import-receipt.json"
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            if {k: receipt.get(k) for k in header} != header:
                raise ValueError("continuation import receipt differs")
            return receipt
        if self.ledger.counters.get("algorithm_feedback_slots", 0) > len(rows):
            raise ValueError("service was charged before its continuation import completed")
        for row in rows:
            sha = row["program_id"]
            key = "copper:v2:scientific-source:" + sha
            charged = usage.get(key, {})
            if charged.get("algorithm_feedback_slots") != 1:
                raise ValueError("prefix record was not charged by the source service")
            rejected = charged.get("interface_rejected_sources") == 1
            if rejected != (row["status"] == "rejected"):
                raise ValueError("prefix rejection differs from the source service")
            payload = seal(self.root, (source / "candidates" / sha / "solver.py").read_bytes())
            if payload["sha256"] != sha:
                raise ValueError("source candidate changed")
            if rejected:
                response = {"status": "invalid_interface", "source_sha256": sha,
                            "metrics": dict(self.interface_failure_metrics)}
            else:
                evaluation = self.evaluator.evaluation_attempt_usage_key(SimpleNamespace(canonical_key=sha))
                _link_tree(source / "evaluations" / evaluation, self.root / "evaluations" / evaluation)
                result = json.loads((self.root / "evaluations" / evaluation / "result.json").read_text())
                if result.get("status") != "succeeded" or result.get("candidate_id") != sha:
                    raise ValueError("source evaluation receipt is incomplete")
                response = {"status": "succeeded", "source_sha256": sha, "metrics": result["metrics"]}
            self.ledger.consume_many({"algorithm_feedback_slots": 1,
                                      **({"interface_rejected_sources": 1} if rejected else {})},
                                     usage_key=key)
            feedback = self._project(response)
            if feedback != {"source_sha256": sha, **{k: row[k] for k in FIELDS - {"source_sha256"}}}:
                raise ValueError("public prefix differs from the private receipts")
        if self.ledger.counters.get("algorithm_feedback_slots", 0) != len(rows):
            raise ValueError("continuation slot count differs from the prefix")
        if any(r["status"] != "rejected" for r in rows):
            measured = eligible(self.root, self.manifest_path)
            if not {r["program_id"] for r in rows if r["status"] != "rejected"} <= set(measured):
                raise ValueError("imported receipts are not eligible training evidence")
        receipt = {**header, "source_root": str(source), "used_slots": len(rows),
                   "program_ids": [r["program_id"] for r in rows]}
        atomic_json_write(receipt_path, receipt)
        return receipt

    def seal_training(self):
        """Freeze the lowest-risk eligible sources with at least one legal solve (ranked by full judge when resampled)."""
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
            count = getattr(self, "finalists", 3)
            ordered = sorted(records, key=lambda sha: (
                             records[sha]["metrics"].get("full_loss_total", records[sha]["metrics"]["loss_total"]),
                             -records[sha]["metrics"]["legal_rate"],
                             records[sha]["metrics"]["mean_solver_wall_seconds"], sha))[:count]
            record = {"contract_digest": digest(self.identity),
                      "used_slots": self.ledger.counters.get("algorithm_feedback_slots", 0),
                      "source_sha256s": ordered, "selected_receipts": {sha: records[sha] for sha in ordered}}
            atomic_json_write(path, record)
        return {"status": "training_sealed", "contract_digest": record["contract_digest"],
                "source_sha256s": record["source_sha256s"], "used_slots": record["used_slots"]}

    def _project(self, response):
        """Full private record, sentinel included, then the whitelist projection."""
        sha = response["source_sha256"]
        metrics = dict(response["metrics"])
        # contracts §3.2: the sentinel always enters the metric stream, so the
        # whitelist below is the only barrier keeping it from the controller.
        metrics["sentinel"] = self.audit["sentinel"]
        failures = metrics["failed_solves"]
        if (type(failures) not in (int, float) or not math.isfinite(failures)
                or failures < 0 or int(failures) != failures):
            raise ValueError("invalid private failure count")
        if response["status"] == "invalid_interface":
            private_case_risks = {name: 1.0 for name in self.audit["case_ids"]}
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
            private_case_risks = risk_by_case(rows, self.audit["case_ids"], self.objective)
            status = "failed" if metrics["legal_rate"] == 0 else "valid"
            message = "all planned solves failed" if status == "failed" else ""
        mode = self.case_isolation["case_feedback"]
        if mode == "none":
            case_risks = {}
        elif mode == "opaque":
            eval_key = self.evaluator.evaluation_attempt_usage_key(SimpleNamespace(canonical_key=sha))
            case_risks = {opaque_case_id(eval_key, name): val for name, val in sorted(private_case_risks.items())}
        else:
            case_risks = private_case_risks
        record = {**metrics, "source_sha256": sha, "status": status, "risk": metrics["loss_total"],
                  "risk_by_case": case_risks, "legal_rate": metrics["legal_rate"],
                  "failed_solves": int(failures), "seconds_per_solve": metrics["mean_solver_wall_seconds"],
                  "message": message}
        feedback = validate_case_feedback(project(record), mode)
        private_path = self.root / "rsi-feedback" / (sha + ".json")
        private = {"feedback": feedback, "private_record": {**record, "risk_by_case": private_case_risks}}
        if private_path.exists() and json.loads(private_path.read_text()) != private:
            raise ValueError("private feedback receipt changed")
        atomic_json_write(private_path, private)
        return feedback

    def dispatch(self, request):
        if request == {"operation": "info"}:
            return dict(status="ready", contract_digest=self.public_contract,
                        public_digest=self.public_digest, seconds=self.seconds,
                        slots=self.slot_limit,
                        used_slots=self.ledger.counters.get("algorithm_feedback_slots", 0),
                        exposure_version=VERSION, objective=self.objective,
                        tier=None if self.tier is None else self.tier["name"],
                        import_ledger_prefix_sha256=self.import_ledger_prefix_sha256,
                        case_isolation=dict(self.case_isolation))
        if not isinstance(request, dict) or request.get("contract_digest") != self.public_contract:
            return {"status": "contract_mismatch"}
        if shared_identity() != self.shared_implementation:
            return {"status": "paused"}
        operation = request.get("operation")
        if operation not in {"evaluate", "seal_training"}:
            return {"status": "rejected_request"}
        if operation == "evaluate" and self._forbidden_case_tokens:
            if set(request) != {"operation", "source", "contract_digest"} or not isinstance(request["source"], str):
                return {"status": "rejected_request"}
            if (self.root / "training-sealed.json").exists():
                return {"status": "training_sealed"}
            try:
                body = request["source"].encode()
                validate_source(body)
            except (ValueError, SyntaxError, UnicodeError):
                return {"status": "invalid_source"}
            try:
                check_source_case_ids(body, self._forbidden_case_tokens)
            except ValueError:
                from ldm_tts.engine.run_store import BudgetExceededError
                try:
                    payload = seal(self.root, body)
                    self.slots.reserve(payload["sha256"], interface_rejected=True)
                    return self._project({"status": "invalid_interface", "source_sha256": payload["sha256"],
                                          "metrics": dict(self.interface_failure_metrics)})
                except BudgetExceededError:
                    return {"status": "budget_exhausted"}
                except Exception:
                    logging.exception("private static case-id rejection failed")
                    return {"status": "infrastructure_error"}
        underlying = {**request, "contract_digest": digest(self.identity)}
        response = super().dispatch(underlying)
        if operation == "seal_training":
            if response.get("status") == "training_sealed":
                return {**response, "contract_digest": self.public_contract}
            return response
        if response.get("status") not in {"succeeded", "invalid_interface"}:
            return response
        try:
            return self._project(response)
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
