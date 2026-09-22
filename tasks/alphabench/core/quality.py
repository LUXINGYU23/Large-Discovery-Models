"""Frozen post-search quality audit; its results are never proposal feedback."""

import json
from ldm_tts.contracts import Candidate, RawProposal
from ldm_tts.engine.run_store import BudgetExceededError
from .candidate import FactorDomain
from .protocol import digest
from .receipts import Receipts


def audit_quality(protocol, runtime, gateway):
    domain = FactorDomain(protocol.backend, protocol.grammar_depth)
    occurrences = []
    for path in sorted((runtime.run_dir / "generation").glob("*.json")):
        generation = json.loads(path.read_text(encoding="utf-8"))
        occurrences.extend({**row, "generation": path.stem} for row in generation["occurrences"])
    Receipts(runtime.run_dir / "private/stages").accept("quality_set_frozen", occurrences)
    original, checked = {}, {}
    for occurrence in occurrences:
        if "check_receipt" not in occurrence:
            continue
        identity = occurrence["check_receipt"]
        record = gateway.receipts.load(identity)
        if record is None or record["state"] != "completed":
            raise ValueError("a recorded generation check has no completed receipt")
        request, response = record["request"], record["response"]
        candidate = domain.admit(RawProposal(occurrence["payload"], "quality_audit"))
        if (not isinstance(candidate, Candidate) or record["request_digest"] != digest(request)
            or record["response_digest"] != digest(response) or request["request_id"] != digest(identity)
            or response["request_id"] != request["request_id"] or request["protocol"] != protocol.to_dict()
            or request["operation"] != "check" or not request["fast"]
            or (request["start"], request["end"]) != protocol.interval("check")
            or request["expression"] != candidate.payload["expression"]):
            raise ValueError("quality check receipt differs from the frozen evaluation contract")
        original[digest(identity)] = response
        checked.setdefault(candidate.candidate_id, response)
    results, admitted = [], set()
    for occurrence in occurrences:
        candidate = domain.admit(RawProposal(occurrence["payload"], "quality_audit"))
        row = {"generation": occurrence["generation"], "attempt": occurrence["attempt"],
               "index": occurrence["index"], "original_status": occurrence["status"], "static_valid": isinstance(candidate, Candidate)}
        if isinstance(candidate, Candidate):
            if occurrence["status"] == "accepted":
                admitted.add(candidate.candidate_id)
            if candidate.candidate_id not in checked:
                try:
                    checked[candidate.candidate_id] = gateway.evaluate(candidate, phase="quality", position=candidate.candidate_id)
                except BudgetExceededError as exc:
                    row["reason"] = str(exc)
            raw = original[digest(occurrence["check_receipt"])] if "check_receipt" in occurrence else checked.get(candidate.candidate_id)
            row.update(candidate_id=candidate.candidate_id, backend_valid=raw["success"] if raw else None,
                       paper_valid=(raw["success"] and raw.get("nan_ratio", 1) <= .01 and raw.get("elapsed_seconds", 31) <= 30) if raw else None)
        else:
            row.update(backend_valid=False, paper_valid=False, reason=candidate.message)
        results.append(row)
    total = len(occurrences)
    accepted = sum(row["status"] == "accepted" for row in occurrences)
    audited = sum(row["backend_valid"] is not None for row in results)
    complete = audited == total
    return {"domain": "all raw occurrences from search model responses; excludes initialization",
            "raw_occurrences": total, "audited": audited, "coverage": audited / total if total else None,
            "complete": complete, "unavailable_reason": "no_occurrences" if not total else None if complete else "quality_budget_exhausted",
            "static_success_rate": sum(row["static_valid"] for row in results) / total if total else None,
            protocol.backend + "_dynamic_success_rate": sum(row["backend_valid"] for row in results) / total if total and complete else None,
            "paper_dynamic_success_rate": sum(row["paper_valid"] for row in results) / total if total and complete else None,
            "accepted_occurrences": accepted, "unique_checked": len(checked),
            "unique_valid": sum(raw["success"] for raw in checked.values()) if complete else None,
            "unique_admitted": len(admitted), "rows": results}
