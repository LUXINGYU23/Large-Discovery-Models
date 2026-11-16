"""Audit one real-model run (AT-16, AT-17) without printing private marker values.

The expected provider identity comes from the run manifest (``manifest.model``, which
the workflow copies from protocol.json), never from constants here.
"""

import argparse
import json
from pathlib import Path

from tasks.copper_inversion.rsi.usage import verify


def expected_provider(manifest, effort=None):
    """(model, reasoning effort) every request must carry; compaction uses the run's level too."""
    model = manifest["model"]
    if effort is None:
        effort = None if model["thinking"] == "off" else model["thinking"]
    return model["name"], effort


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "private", "private-manifest", "key-file"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--effort", help="override the reasoning effort expected for the manifest's thinking level")
    args = parser.parse_args(argv)
    run, private = args.run, args.private
    run_manifest = json.loads((run / "manifest.json").read_text())
    expected = expected_provider(run_manifest, args.effort)
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
    if leaks or models != {expected}:
        raise ValueError("provider identity or blind-channel audit failed")
    ledger = [json.loads(line) for line in (run / "ledger.jsonl").read_text().splitlines()]
    private_budget = json.loads((private / "budget.json").read_text())
    prefix = "copper:v2:scientific-source:"
    private_ids = {key[len(prefix):] for key, value in private_budget["metadata"]["cumulative_usage"].items()
                   if key.startswith(prefix) and value.get("algorithm_feedback_slots") == 1}
    if private_ids != {r["program_id"] for r in ledger}:
        raise ValueError("scientific budget does not reconcile")
    result = json.loads((run / "result.json").read_text())
    if result["status"] != "completed" or result["charged_records"] != len(ledger):
        raise ValueError("run did not complete with one record per charged slot")
    usage = verify(run)  # AT-17: usage files and result.json against an independent recount
    if usage["status"] != "passed" or usage["unrecorded_rounds"]:
        raise ValueError("provider usage does not reproduce: " + json.dumps(usage, sort_keys=True))
    branches = run_manifest["config"]["policy"]["branches"]
    for path in sorted((run / "selection").glob("round_*.json")):
        selected = json.loads(path.read_text())["selected"]
        if {item["branch"] for item in selected if item["selection"] == "reserved"} != set(range(branches)):
            raise ValueError("branch reservations were not satisfied: " + path.name)
    print(json.dumps({"scope": "synthetic_fixture_only", "status": "passed", "provider_requests": len(requests),
                      "model": expected[0], "thinking": run_manifest["model"]["thinking"], "effort": expected[1],
                      "marker_leaks": leaks, "charged_sources": len(private_ids),
                      "input_tokens": usage["input_tokens"],
                      "responses_missing_usage": usage["responses_missing_usage"],
                      "usage_rounds": usage["rounds"], "reserved_branches": branches}, sort_keys=True))


if __name__ == "__main__":
    main()
