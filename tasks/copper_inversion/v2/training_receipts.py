"""Audit candidate eligibility against completed, whole-training-suite receipts."""
import json
import math
from pathlib import Path
from ..core.candidate import validate_source
from ..core.execution import digest
from .suite import load_manifest,file_hash,aggregate


def eligible(run,manifest):
    root=Path(run).resolve();path=Path(manifest).resolve();suite=load_manifest(path)
    if suite["split"]!="train":raise ValueError("candidate eligibility requires training suite")
    expected={(c["case_id"],seed,c["parent_id"]) for c in suite["cases"] for seed in suite["seeds"]}
    records={}
    for result_path in sorted((root/"evaluations").glob("*/result.json")):
        result=json.loads(result_path.read_text())
        if result.get("status")!="succeeded":continue
        if result.get("metadata",{}).get("suite_digest")!=digest(suite):
            raise ValueError("training evaluation refers to a different suite")
        source_id=result["candidate_id"]
        if len(source_id)!=64 or any(c not in "0123456789abcdef" for c in source_id):
            raise ValueError("invalid training source identity")
        source=root/"candidates"/source_id/"solver.py"
        if validate_source(source.read_bytes())!=source_id:raise ValueError("training source changed")
        rows_path=result_path.parent/"private-case-results.json"
        rows=json.loads(rows_path.read_text())
        identities=[(r["case_id"],r["seed"],r["parent_id"]) for r in rows]
        if len(identities)!=len(expected) or set(identities)!=expected:
            raise ValueError("training evaluation is missing or duplicating planned cases")
        for row in rows:
            receipt=result_path.parent/row["case_id"]/str(row["seed"])/"receipt.json"
            if json.loads(receipt.read_text())!=row:raise ValueError("training child receipt differs")
        metrics=aggregate(rows)
        if set(metrics)!=set(result["metrics"]) or any(not math.isclose(value,result["metrics"][key],rel_tol=1e-12,abs_tol=1e-12) for key,value in metrics.items()):
            raise ValueError("training aggregate differs from child receipts")
        if source_id in records:raise ValueError("ambiguous duplicate training evaluations")
        records[source_id]=dict(source_path=str(source),metrics=metrics,
            result_path=str(result_path),result_sha256=file_hash(result_path),
            cases_path=str(rows_path),cases_sha256=file_hash(rows_path))
    if not records:raise ValueError("no completed whole-training-suite candidates")
    return records
