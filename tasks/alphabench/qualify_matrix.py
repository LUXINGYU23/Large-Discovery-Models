"""Build the complete T3 functional qualification inventory from audited inputs."""

import argparse
import json
from pathlib import Path

from .core.protocol import BACKENDS, METHODS, digest


ROOT = Path(__file__).resolve().parents[2]
DATA_EVIDENCE = "tasks/alphabench/resources/evidence/data_preparation.json"
TASK_EVIDENCE = "tasks/alphabench/resources/qualification_evidence.json"
OUTPUT = "tasks/alphabench/qualification_records/completeness_matrix.json"


def build_matrix(data, task):
    policy = data.get("comparison_data_policy", "qualified_only")
    if policy not in {"qualified_only", "partial_comparison"}:
        raise ValueError("unknown comparison data policy")
    cells, market_limitations = [], {}
    for backend, markets in BACKENDS.items():
        for market in markets:
            section = "cn" if market.startswith("csi") else "us"
            audit = data[section][market]
            if audit["qualification"] not in {"blocked", "qualified"}:
                raise ValueError("market audit has no recognized qualification: " + market)
            blocked = audit["qualification"] != "qualified"
            issues = sorted({issue["code"] for issue in audit["issues"]}) if blocked else []
            market_limitations[f"{backend}/{market}"] = issues
            for method in METHODS:
                cells.append({
                    "id": f"{backend}/{market}/{method}",
                    "backend": backend, "market": market, "method": method,
                    "scientific_profile": "ldm_matched_v1", "contract_profile": None,
                    "dialects": ["qlib", "assay"] if backend == "assay" else ["qlib"],
                    "initialization": None, "filter_profile": None, "budget": None,
                    "stage": "not_started", "status": "blocked",
                    "source_audit_qualification": audit["qualification"],
                    "capabilities": {"data": "unverified", "operators": "unverified",
                                     "method": "unverified", "portfolio": "unverified", "recovery": "unverified"},
                    "blockers": (["unqualified_market_data"] if blocked and policy == "qualified_only" else []) +
                                ["frozen_backend_manifest_missing", "formal_profile_unfrozen", "real_run_missing"],
                    "data_evidence": {"file": DATA_EVIDENCE, "section": f"{section}.{market}"},
                    "run_id": None, "compact_record": None, "raw_hashes": {},
                })
    if len(cells) != 72 or len({cell["id"] for cell in cells}) != 72:
        raise ValueError("the full T3 functional matrix must contain 72 distinct cells")
    counts = {status: sum(cell["status"] == status for cell in cells)
              for status in ("passed", "failed", "blocked")}
    return {"schema_version": 1, "task_id": "alphabench", "kind": "functional_72",
            "comparison_data_policy": policy,
            "market_data_limitations": market_limitations,
            "data_evidence": {"file": DATA_EVIDENCE, "digest": digest(data)},
            "task_evidence": {"file": TASK_EVIDENCE, "digest": digest(task), "stage": task["stage"]},
            "required": len(cells), "counts": counts, "complete_t3": False, "cells": cells}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write", action="store_true")
    action.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    data = json.loads((ROOT / DATA_EVIDENCE).read_text(encoding="utf-8"))
    task = json.loads((ROOT / TASK_EVIDENCE).read_text(encoding="utf-8"))
    target = ROOT / OUTPUT
    rendered = json.dumps(build_matrix(data, task), indent=2, sort_keys=True) + "\n"
    if args.write:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(rendered, encoding="utf-8")
    elif not target.exists() or target.read_text(encoding="utf-8") != rendered:
        raise SystemExit("T3 completeness matrix differs from current evidence")
    print(f"{OUTPUT}: 72 required; task stage {task['stage']}; complete_t3=false")


if __name__ == "__main__":
    main()
