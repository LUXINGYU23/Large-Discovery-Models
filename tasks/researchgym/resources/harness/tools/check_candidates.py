"""Guest-side candidate checker worker; the host re-validates every submission.

Reads {"catalog": {"global_rules": ...}, "case": {"entry", "forbidden_import_prefixes"},
"candidates": [...], "measured_keys": [...]} on stdin and writes indexed errors
using the same program rules as task admission.
"""

import json
import sys

from program_rules import canonical_key, check_program

FIELDS = {"program", "change_summary", "rationale"}
OPTIONAL = {"comparison_candidate_ids"}


def main():
    request = json.load(sys.stdin)
    rules, case = request["catalog"]["global_rules"], request["case"]
    measured = set(request.get("measured_keys", ()))
    seen = {}
    rows = []
    for index, row in enumerate(request["candidates"]):
        errors = []
        if not isinstance(row, dict) or not FIELDS <= set(row) or set(row) - FIELDS - OPTIONAL:
            errors.append({"code": "invalid_candidate_fields", "message": "use program, change_summary, rationale "
                           "and optional comparison_candidate_ids", "hint": "Remove other fields."})
            rows.append({"index": index, "canonical_key": None, "errors": errors})
            continue
        for field in ("change_summary", "rationale"):
            if not isinstance(row[field], str) or not row[field].strip():
                errors.append({"code": f"invalid_{field}", "message": f"{field} must be a nonempty string",
                               "hint": "Write a short research note."})
        errors += check_program(row["program"], entry=case["entry"], rules=rules,
                                forbidden_prefixes=case["forbidden_import_prefixes"])
        key = None
        if not errors:
            key = canonical_key(row["program"])
            if key in measured:
                errors.append({"code": "already_measured", "message": "this program was already evaluated",
                               "hint": "Query get_measured_history for its result and submit a different method."})
            elif key in seen:
                errors.append({"code": "duplicate_in_submission",
                               "message": f"same canonical program as candidates[{seen[key]}]",
                               "hint": "Each entry of one submission must be a distinct method."})
            else:
                seen[key] = index
        rows.append({"index": index, "canonical_key": key, "errors": errors})
    json.dump({"candidates": rows, "count": len(rows), "valid": all(not r["errors"] for r in rows)}, sys.stdout)


if __name__ == "__main__":
    main()
