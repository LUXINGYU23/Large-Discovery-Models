"""Frozen post-search quality audit; its results are never proposal feedback."""

import json
from ldm_tts.contracts import Candidate, RawProposal
from .candidate import FactorDomain
from .receipts import Receipts


def audit_quality(protocol, runtime, gateway):
    domain = FactorDomain(protocol.backend, protocol.grammar_depth)
    occurrences = []
    for path in sorted((runtime.run_dir / "generation").glob("*.json")):
        generation = json.loads(path.read_text(encoding="utf-8"))
        occurrences.extend({**row, "generation": path.stem} for row in generation["occurrences"])
    Receipts(runtime.run_dir / "private/stages").accept("quality_set_frozen", occurrences)
    results, checked = [], {}
    for index, occurrence in enumerate(occurrences):
        candidate = domain.admit(RawProposal(occurrence["payload"], "quality_audit"))
        row = {"generation": occurrence["generation"], "attempt": occurrence["attempt"],
               "index": occurrence["index"], "original_status": occurrence["status"], "static_valid": isinstance(candidate, Candidate)}
        if isinstance(candidate, Candidate):
            if candidate.candidate_id not in checked:
                checked[candidate.candidate_id] = gateway.evaluate(candidate, phase="quality", position=candidate.candidate_id)
            raw = checked[candidate.candidate_id]
            row.update(candidate_id=candidate.candidate_id, backend_valid=raw["success"],
                       paper_valid=raw["success"] and raw.get("nan_ratio", 1) <= .01 and raw.get("elapsed_seconds", 31) <= 30)
        else:
            row.update(backend_valid=False, paper_valid=False, reason=candidate.message)
        results.append(row)
    total = len(occurrences)
    accepted = sum(row["status"] == "accepted" for row in occurrences)
    return {"domain": "all raw occurrences from search model responses; excludes initialization",
            "raw_occurrences": total, "audited": len(results), "coverage": len(results) / total if total else None,
            "static_success_rate": sum(row["static_valid"] for row in results) / total if total else None,
            "backend_success_rate": sum(row["backend_valid"] for row in results) / total if total else None,
            "paper_success_rate": sum(row["paper_valid"] for row in results) / total if total else None,
            "accepted_occurrences": accepted, "unique_checked": len(checked),
            "unique_valid": sum(raw["success"] for raw in checked.values()), "rows": results}
