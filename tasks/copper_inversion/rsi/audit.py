"""Read-only reconstruction, budget reconciliation and blind-channel checks."""

import argparse
import hashlib
import json
from pathlib import Path

from .controller import Config, make_message
from .ledger import read


def verify_briefs(run):
    run = Path(run)
    manifest = json.loads((run / "manifest.json").read_text())
    config = Config(**manifest["config"])
    task = (run / "task_brief.txt").read_text()
    if hashlib.sha256(task.encode()).hexdigest() != manifest["task_brief_sha256"]:
        raise ValueError("task brief differs from the frozen run")
    ledger = read(run / "ledger.jsonl")
    intervention = None
    if manifest.get("intervention"):
        item = manifest["intervention"]
        note = (run / "intervention/note.txt").read_text()
        if hashlib.sha256(note.encode()).hexdigest() != item["note_sha256"]:
            raise ValueError("intervention note differs from the frozen run")
        intervention = {"round": item["round"], "target_tag": item["target_tag"], "note": note}
    checked = 0
    for path in sorted((run / "harness").glob("epoch_*/branch_*/briefs/round_*.json")):
        branch = int(path.parent.parent.name.removeprefix("branch_"))
        round_index = int(path.stem.removeprefix("round_"))
        prefix = [row for row in ledger if row["round"] < round_index]
        previous = []
        if round_index > 1:
            folder = run / "proposals" / f"round_{round_index-1:02d}"
            for receipt in sorted(folder.glob("branch_*.json")):
                previous.extend(json.loads(receipt.read_text())["proposals"])
        message, _ = make_message(config, run, prefix, branch, round_index, task, previous, intervention)
        expected = json.dumps(message, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
        if path.read_text() != expected:
            raise ValueError("branch brief differs from its ledger prefix: " + str(path))
        checked += 1
    return {"status": "passed", "briefs_checked": checked}


def reconcile_budget(run, private):
    run, private = Path(run), Path(private)
    records = read(run / "ledger.jsonl")
    ledger = json.loads((private / "budget.json").read_text())
    prefix = "copper:v2:scientific-source:"
    private_ids = {key[len(prefix):] for key, value in ledger.get("metadata", {}).get("cumulative_usage", {}).items()
                   if key.startswith(prefix) and value.get("algorithm_feedback_slots") == 1}
    public_ids = {record["program_id"] for record in records}
    if private_ids != public_ids or ledger["counters"].get("algorithm_feedback_slots", 0) != len(public_ids):
        raise ValueError("private and public scientific-source budgets differ")
    return {"status": "passed", "charged_sources": len(public_ids)}


def scan_blind(run, private, *, require_requests=True, private_manifest=None, key_file=None):
    """Search only internally; never print the sentinel or private parent IDs."""
    run, private = Path(run), Path(private)
    audit = json.loads((private / "rsi-private-audit.json").read_text())
    markers = [audit["sentinel"].encode()] + [value.encode() for value in audit["private_parent_ids"]]
    if private_manifest is not None:
        manifest = json.loads(Path(private_manifest).read_text())
        markers += [row["sha256"]["truth.npy"].encode() for row in manifest["cases"]]
    if key_file is not None:
        markers.append(Path(key_file).read_bytes().strip())
    if any(not marker for marker in markers):
        raise ValueError("blind-audit marker is empty")
    requests = sorted((run / "harness").rglob("*.request.bin"))
    if require_requests and not requests:
        raise ValueError("blind audit requires at least one preserved provider request")
    paths = list(requests)
    paths.extend((run / "harness").rglob("briefs/*.json"))
    paths.extend((run / "harness").rglob("public/evidence/history.txt"))
    paths.extend((run / "harness").rglob("public/evidence/sources/*.py"))
    paths.append(run / "ledger.jsonl")
    leaks = sum(any(marker in path.read_bytes() for marker in markers) or b"mean_iou" in path.read_bytes().lower()
                for path in paths if path.is_file())
    # Verify that the same byte-level predicate detects the private canary.
    if not any(marker in b"positive-control:" + audit["sentinel"].encode() for marker in markers):
        raise ValueError("blind-audit positive control failed")
    if leaks:
        raise ValueError(f"blind-channel audit found {leaks} contaminated files")
    return {"status": "passed", "provider_requests_checked": len(requests),
            "files_checked": len(paths), "sentinel_positive_control": "passed", "leaks": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("briefs", "budget", "blind"))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--private", type=Path)
    parser.add_argument("--private-manifest", type=Path)
    parser.add_argument("--key-file", type=Path)
    args = parser.parse_args()
    if args.mode == "briefs":
        result = verify_briefs(args.run)
    else:
        if args.private is None:
            parser.error("--private is required for this audit")
        result = (reconcile_budget(args.run, args.private) if args.mode == "budget" else
                  scan_blind(args.run, args.private, private_manifest=args.private_manifest, key_file=args.key_file))
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
