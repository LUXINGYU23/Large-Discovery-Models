"""Audit one synthetic real-model smoke without printing private marker values."""

import argparse
import json
from pathlib import Path

from tasks.copper_inversion.rsi.usage import collect


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "private", "private-manifest", "key-file"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    run, private = args.run, args.private
    audit = json.loads((private / "rsi-private-audit.json").read_text())
    manifest = json.loads(args.private_manifest.read_text())
    markers = [audit["sentinel"].encode(), args.key_file.read_bytes().strip()]
    markers += [key.encode() for key in audit["private_parent_ids"]]
    markers += [row["sha256"]["truth.npy"].encode() for row in manifest["cases"]]
    requests = sorted((run / "harness").rglob("*.request.bin"))
    if not requests or any(not marker for marker in markers):
        raise ValueError("missing requests or audit marker")
    leaks, models = 0, set()
    for path in requests:
        body = path.read_bytes()
        leaks += int(any(marker in body for marker in markers) or b'mean_iou' in body.lower())
        payload = json.loads(body)
        models.add((payload.get("model"), (payload.get("reasoning") or {}).get("effort")))
    if leaks or models != {("deepseek-flash", "low")}:
        raise ValueError("provider identity or blind-channel audit failed")
    ledger = [json.loads(line) for line in (run / "ledger.jsonl").read_text().splitlines()]
    private_budget = json.loads((private / "budget.json").read_text())
    prefix = "copper:v2:scientific-source:"
    private_ids = {key[len(prefix):] for key, value in private_budget["metadata"]["cumulative_usage"].items()
                   if key.startswith(prefix) and value.get("algorithm_feedback_slots") == 1}
    if private_ids != {r["program_id"] for r in ledger}:
        raise ValueError("scientific budget does not reconcile")
    total = 0
    for turn in (run / "harness").rglob("turns/round_001"):
        usage = collect(turn)
        if usage["responses_missing_usage"]:
            raise ValueError("provider usage is incomplete")
        total += usage["input_tokens"]
    reported = json.loads((run / "result.json").read_text())["input_tokens"]
    if total != reported:
        raise ValueError("provider usage does not reproduce the run total")
    selection = json.loads((run / "selection/round_01.json").read_text())["selected"]
    if {item["branch"] for item in selection if item["selection"] == "reserved"} != set(range(4)):
        raise ValueError("four branch reservations were not satisfied")
    print(json.dumps({"scope": "synthetic_fixture_only", "status": "passed", "provider_requests": len(requests),
                      "model": "deepseek-flash", "thinking": "low", "marker_leaks": leaks,
                      "charged_sources": len(private_ids), "input_tokens": total,
                      "reserved_branches": 4}, sort_keys=True))


if __name__ == "__main__":
    main()
