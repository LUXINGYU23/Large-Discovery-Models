"""Read-only archive checks: brief reconstruction, sentinel scan, slot reconciliation.

    python -m tasks.copper_inversion.experiments.rsi.analysis.checks {briefs,sentinel,slots,all} \
        --archive ARCHIVE [--private PRIVATE] [--run EXP/RUN_ID]

PRIVATE holds the evaluator side as <exp>/<run_id>/ (rsi-private-audit.json,
budget.json, rsi-feedback/). Secrets are never printed; a failing check exits 1.
"""

import argparse
import json
import re
import sys
from pathlib import Path

from ....rsi.audit import reconcile_budget, verify_briefs
from .rundir import discover

FEEDBACK = ("status", "risk", "risk_by_case", "legal_rate", "failed_solves", "seconds_per_solve", "message")
IOU = re.compile(rb"iou", re.IGNORECASE)


def check_briefs(run):
    return verify_briefs(run.path)


def _markers(private):
    audit = json.loads((private / "rsi-private-audit.json").read_text())
    markers = [audit["sentinel"].encode()] + [p.encode() for p in audit.get("private_parent_ids", [])]
    if any(not m for m in markers):
        raise ValueError("empty private marker")
    return markers


def check_sentinel(run, private):
    """Every public file of the run, byte level. The private feedback receipts, which
    carry the sentinel by construction, are the positive control."""
    markers = _markers(private)
    receipts = sorted((private / "rsi-feedback").glob("*.json"))
    control = sum(any(m in p.read_bytes() for m in markers[:1]) for p in receipts)
    if receipts and control != len(receipts):
        raise ValueError("sentinel positive control failed on private receipts")
    leaks, iou_files, scanned = [], [], 0
    for path in sorted(run.path.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        body = path.read_bytes()
        scanned += 1
        if any(m in body for m in markers):
            leaks.append(str(path.relative_to(run.path)))
        if IOU.search(body):
            iou_files.append(str(path.relative_to(run.path)))
    result = {"files_scanned": scanned, "positive_control_receipts": control, "leaks": len(leaks),
              "leaked_files": leaks, "iou_mentions": iou_files}
    if leaks:
        raise LeakError(result)
    return result


class LeakError(ValueError):
    pass


def check_slots(run, private):
    """Public ledger against the private budget and the private feedback receipts."""
    result = reconcile_budget(run.path, private)
    mismatched = []
    for record in run.ledger:
        path = private / "rsi-feedback" / f"{record['program_id']}.json"
        if not path.is_file():
            mismatched.append(record["program_id"][:12])
            continue
        feedback = json.loads(path.read_text())["feedback"]
        if any(feedback[k] != record[k] for k in FEEDBACK):
            mismatched.append(record["program_id"][:12])
    receipts = {p.stem for p in (private / "rsi-feedback").glob("*.json")}
    extra = receipts - {r["program_id"] for r in run.ledger}
    if mismatched or extra:
        raise ValueError(f"public and private records differ: {len(mismatched)} mismatched, {len(extra)} unrecorded")
    return {**result, "feedback_receipts_matched": len(run.ledger)}


def run_checks(mode, archive, private=None, only=None):
    report, failed = {}, False
    for run in discover(archive):
        key = f"{run.experiment}/{run.run_id}"
        if only and key != only:
            continue
        entry = {}
        for name in (["briefs", "sentinel", "slots"] if mode == "all" else [mode]):
            try:
                if name == "briefs":
                    entry[name] = check_briefs(run)
                else:
                    if private is None:
                        raise ValueError("--private is required for this check")
                    side = Path(private) / run.experiment / run.run_id
                    entry[name] = check_sentinel(run, side) if name == "sentinel" else check_slots(run, side)
            except LeakError as exc:
                entry[name], failed = {"status": "failed", **exc.args[0]}, True
            except (ValueError, OSError, KeyError) as exc:
                entry[name], failed = {"status": "failed", "error": str(exc)}, True
        report[key] = entry
    return report, failed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("briefs", "sentinel", "slots", "all"))
    parser.add_argument("--archive", required=True)
    parser.add_argument("--private")
    parser.add_argument("--run")
    args = parser.parse_args(argv)
    report, failed = run_checks(args.mode, args.archive, args.private, args.run)
    print(json.dumps({"failed": failed, "runs": report}, sort_keys=True, indent=1))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
