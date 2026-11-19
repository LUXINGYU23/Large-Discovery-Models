"""Read-only reconstruction, budget reconciliation and blind-channel checks."""

import argparse
from fractions import Fraction
import hashlib
import json
import re
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


IOU = re.compile(rb"iou", re.IGNORECASE)


def _private_markers(private, private_manifest):
    """Sentinel, private parent IDs and every truth-side file hash (AT-05)."""
    audit = json.loads((private / "rsi-private-audit.json").read_text())
    markers = {"sentinel": [audit["sentinel"].encode()],
               "private_parent_id": [value.encode() for value in audit["private_parent_ids"]],
               "truth_hash": []}
    if private_manifest is None and (private / "identity.json").is_file():
        private_manifest = json.loads((private / "identity.json").read_text())["configuration"]["manifest"]
    if private_manifest is not None:
        manifest = json.loads(Path(private_manifest).read_text())
        markers["truth_hash"] = [value.encode() for row in manifest["cases"]
                                 for name, value in sorted(row["sha256"].items())
                                 if name == "truth.npy" or name.startswith("confirmation/")]
    markers["private_value"] = _private_values(private)
    return markers


def _significant(text):
    digits = text.lower().split("e")[0].lstrip("-").replace(".", "").lstrip("0")
    return len(digits)


def _simple_ratio(value, max_denominator=1000):
    """A float that is exactly p/q with a small q (e.g. 7/24 = 0.2916666666666667)."""
    return float(Fraction(value).limit_denominator(max_denominator)) == value


def _private_values(private, min_digits=8):
    """Every hidden IoU the service recorded (per program and per case), written as
    JSON writes a float. Short values and exact small-denominator ratios are left out:
    they collide with ordinary numbers (the M1 pilot under rsi-r1-a4dbd1606448faca hit a
    hidden per-case IoU of 7/24 in an agent's own count of ore samples, 35 of 120)."""
    values = set()
    for path in sorted((private / "evaluations").glob("*/result.json")):
        values.add(json.loads(path.read_text()).get("metrics", {}).get("mean_iou"))
        cases = path.parent / "private-case-results.json"
        if cases.is_file():
            values.update(row.get("metrics", {}).get("iou") for row in json.loads(cases.read_text()))
    texts = {json.dumps(v) for v in values if type(v) is float and not _simple_ratio(v)}
    return sorted(t.encode() for t in texts if _significant(t) >= min_digits)


def _allowed_phrases(texts):
    """Task-description lines that mention IoU, raw and as JSON string content."""
    phrases = set()
    for text in texts:
        for line in text.splitlines():
            line = line.strip()
            if IOU.search(line.encode()):
                phrases.add(line.encode())
                phrases.add(json.dumps(line)[1:-1].encode())
                phrases.add(json.dumps(line, ensure_ascii=False)[1:-1].encode())
    return sorted(phrases, key=len, reverse=True)


def _classify(body, start, phrases):
    for phrase in phrases:
        index = body.find(phrase, max(0, start + 3 - len(phrase)))
        while index != -1 and index <= start:
            if start + 3 <= index + len(phrase):
                return "task_description"
            index = body.find(phrase, index + 1)
    left = start
    while left > 0 and chr(body[left - 1]).isalpha():
        left -= 1
    following = body[start + 3:start + 4]
    # Random identifiers such as provider tool-call IDs ("call_00_UFFVU367c0srmiOUaqeE4451"): the
    # alphanumeric token around the match is long and holds a digit.
    first, last = start, start + 3
    while first > 0 and chr(body[first - 1]).isalnum():
        first -= 1
    while last < len(body) and chr(body[last]).isalnum():
        last += 1
    token = body[first:last]
    if len(token) >= 16 and any(chr(b).isdigit() for b in token):
        return "identifier"
    # English words such as "previous" or "Various": lowercase "iou" inside a longer word.
    if body[start:start + 3] == b"iou" and following.isalpha() and following.islower() and start > left:
        return "english_word"
    return "unexplained"


def scan_blind(run, private, *, require_requests=True, private_manifest=None, key_file=None,
               allowed_text=(), expect_leak=False):
    """Search the whole public run directory; never print a private marker.

    Markers (sentinel, private parent IDs, truth-side file hashes, every hidden
    IoU value at full precision, the key) are searched in every file; any hit
    fails the scan. Case-insensitive "iou" is searched in text files and every
    hit is listed with a classification. Task-description wording and ordinary
    English words are explained; the rest (an agent writing "IoU" in its notes)
    does not fail the scan but must be signed off before unsealing:
    `review_sha256` identifies that list. With expect_leak (the whitelist-off
    control run) the audit passes only if the sentinel is found.
    """
    run, private = Path(run), Path(private)
    markers = _private_markers(private, private_manifest)
    if key_file is not None:
        markers["api_key"] = [Path(key_file).read_bytes().strip()]
    if any(not marker for values in markers.values() for marker in values):
        raise ValueError("blind-audit marker is empty")
    requests = sorted(path for path in (run / "harness").rglob("*.request.bin") if path.is_file())
    if require_requests and not requests:
        raise ValueError("blind audit requires at least one preserved provider request")
    texts = list(allowed_text)
    if (run / "task_brief.txt").is_file():
        texts.append((run / "task_brief.txt").read_text())
    phrases = _allowed_phrases(texts)
    paths = sorted(path for path in run.rglob("*") if path.is_file() and not path.is_symlink())
    hits = {name: 0 for name in markers}
    contaminated, iou_hits, binary = set(), [], 0
    for path in paths:
        body = path.read_bytes()
        marked = False
        for name, values in markers.items():
            found = sum(body.count(marker) for marker in values)
            if found:
                hits[name] += found
                contaminated.add(path)
                marked = True
        if b"\x00" in body[:8192]:
            binary += 1
            continue
        for match in IOU.finditer(body):
            kind = _classify(body, match.start(), phrases)
            # Never echo bytes from a file that holds a private marker.
            context = ("<withheld: file contains a private marker>" if marked else
                       body[max(0, match.start() - 24):match.end() + 24].decode("utf-8", "replace"))
            iou_hits.append({"path": str(path.relative_to(run)), "offset": match.start(),
                             "kind": kind, "context": context})
    review = [{"path": h["path"], "offset": h["offset"]} for h in iou_hits if h["kind"] == "unexplained"]
    result = {"status": "review_required" if review else "passed", "provider_requests_checked": len(requests),
              "files_checked": len(paths), "binary_files_marker_only": binary,
              "marker_hits": hits, "iou_hits": iou_hits,
              "iou_hits_by_kind": {kind: sum(h["kind"] == kind for h in iou_hits)
                                   for kind in ("task_description", "english_word", "identifier", "unexplained")},
              "leaks": len(contaminated), "review_required": len(review),
              "review_sha256": hashlib.sha256(json.dumps(review, sort_keys=True).encode()).hexdigest() if review else None}
    if expect_leak:
        if not hits["sentinel"]:
            raise ValueError("whitelist-off control run did not expose the sentinel")
        return {**result, "status": "control_detected"}
    if contaminated:
        raise ValueError(f"blind-channel audit found {len(contaminated)} contaminated files "
                         f"(markers: {', '.join(k for k, v in hits.items() if v)})")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("briefs", "budget", "blind"))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--private", type=Path)
    parser.add_argument("--private-manifest", type=Path)
    parser.add_argument("--key-file", type=Path)
    parser.add_argument("--allowed-text", type=Path, nargs="*", default=(),
                        help="task-description files whose IoU wording is explained, not a leak")
    parser.add_argument("--expect-leak", action="store_true", help="whitelist-off control run")
    parser.add_argument("--hits-out", type=Path, help="write every listed iou hit here")
    args = parser.parse_args()
    if args.mode == "briefs":
        result = verify_briefs(args.run)
    else:
        if args.private is None:
            parser.error("--private is required for this audit")
        result = (reconcile_budget(args.run, args.private) if args.mode == "budget" else
                  scan_blind(args.run, args.private, private_manifest=args.private_manifest, key_file=args.key_file,
                             allowed_text=[p.read_text() for p in args.allowed_text], expect_leak=args.expect_leak))
        if args.mode == "blind" and args.hits_out is not None:
            args.hits_out.write_text(json.dumps(result["iou_hits"], indent=2, sort_keys=True) + "\n")
        if args.mode == "blind":
            result = {**result, "iou_hits": len(result["iou_hits"])}
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
