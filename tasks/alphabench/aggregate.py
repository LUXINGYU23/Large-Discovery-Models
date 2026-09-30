"""Aggregate pre-registered CoT/ToT run outcomes without executing a model or oracle."""

import argparse
import hashlib
import json
from pathlib import Path

from ldm_tts.engine.run_store import atomic_json_write
from .core.protocol import digest
from .core.reporting import aggregate_fraction_success


def aggregate(roster_path, invalidations_path=None):
    roster_path = Path(roster_path)
    manifest = json.loads(roster_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("method") not in {"alphabench_cot", "alphabench_tot"}:
        raise ValueError("expected a version-one CoT/ToT pre-registered roster")
    invalidations_path = Path(invalidations_path) if invalidations_path else None
    invalidations = json.loads(invalidations_path.read_text(encoding="utf-8")) if invalidations_path else {}
    if not isinstance(invalidations, dict):
        raise ValueError("invalidation decisions must map run IDs to evidence files")
    evidence = {}
    for run_id, path in invalidations.items():
        evidence_path = Path(path)
        evidence_path = evidence_path if evidence_path.is_absolute() else invalidations_path.parent / evidence_path
        raw = evidence_path.read_bytes()
        if not raw:
            raise ValueError("infrastructure invalidation evidence is empty")
        evidence[run_id] = {"path": str(evidence_path.resolve()),
                            "sha256": hashlib.sha256(raw).hexdigest()}
    reports, hashes, common_protocol = {}, {}, None
    for item in manifest["runs"]:
        directory = Path(item["run_dir"])
        directory = directory if directory.is_absolute() else roster_path.parent / directory
        path = directory / "result.json"
        if path.exists():
            status_path = directory / "status.json"
            if not status_path.exists() or json.loads(status_path.read_text(encoding="utf-8"))["status"] != "completed":
                raise ValueError("a result exists before the run reached completed status")
            raw = path.read_bytes()
            report = json.loads(raw)
            if digest(report["protocol"]) != report["protocol_digest"]:
                raise ValueError("run report protocol digest mismatch")
            if (report["method"], report["protocol"]["backend"], report["protocol"]["market"],
                    report["protocol"]["profile"]) != (manifest["method"], manifest["backend"],
                    manifest["market"], manifest["profile"]):
                raise ValueError("run report differs from the pre-registered cohort")
            comparable = {key: value for key, value in report["protocol"].items() if key != "random_seed"}
            if common_protocol is not None and comparable != common_protocol:
                raise ValueError("completed cohort runs differ beyond the random seed")
            common_protocol = comparable
            reports[item["run_id"]] = report
            hashes[item["run_id"]] = hashlib.sha256(raw).hexdigest()
    result = aggregate_fraction_success(manifest["runs"], reports,
        invalid_evidence={run_id: item["sha256"] for run_id, item in evidence.items()},
        invalid_rule=manifest["invalid_rule"])
    qualified = bool(result["included_runs"]) and all(
        row["infrastructure_invalid_evidence"] is not None or
        reports.get(row["run_id"], {}).get("complete_t3") is True for row in result["runs"])
    result.update(cohort={key: manifest[key] for key in ("method", "backend", "market", "profile")},
                  cohort_protocol_digest=digest(common_protocol) if common_protocol else None,
                  aggregation_qualified=qualified,
                  qualification_reason=None if qualified else "included_run_not_t3_qualified",
                  roster_sha256=hashlib.sha256(roster_path.read_bytes()).hexdigest(),
                  result_sha256=hashes, invalidation_evidence=evidence)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roster", required=True, type=Path)
    parser.add_argument("--invalidations", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    atomic_json_write(args.output, aggregate(args.roster, args.invalidations))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
