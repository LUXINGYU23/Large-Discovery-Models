"""Read-only audit of one completed PoC HTTP campaign (run on H20)."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sqlite3


def audit(directory: Path) -> dict:
    config = json.loads((directory / "config.json").read_text())
    expected = config["iterations"]
    problems = []

    def require(condition, message):
        if not condition:
            problems.append(message)

    with sqlite3.connect(f"file:{directory / 'controller.sqlite3'}?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        experiments = list(db.execute("SELECT * FROM experiments ORDER BY decision_id"))
        observations = [json.loads(r[0]) for r in db.execute(
            "SELECT observation_json FROM observations ORDER BY ingest_seq")]
        decisions = [json.loads(r[0]) for r in db.execute(
            "SELECT snapshot_json FROM decisions ORDER BY decision_id")]
        unresolved = db.execute("SELECT COUNT(*) FROM anomalies WHERE resolved_at IS NULL").fetchone()[0]
    with sqlite3.connect(f"file:{directory / 'experiment-service.sqlite3'}?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        attempts = list(db.execute("SELECT * FROM attempts ORDER BY started_at"))
        remote = list(db.execute("SELECT * FROM experiments ORDER BY seq"))
    require(len(experiments) == len(observations) == len(attempts) == expected,
            "Expected one decision, attempt and observation per budget unit")
    require(len({r["canonical_key"] for r in experiments}) == expected, "Duplicate selected candidate")
    require({r["experiment_id"] for r in experiments} == {r["experiment_id"] for r in remote},
            "Controller and HTTP experiment service IDs differ")
    require(all(r["es_state"] == "SUCCEEDED" and r["budget_state"] == "CONSUMED"
                and r["pending"] == 0 for r in experiments), "Unsettled or unsuccessful controller records")
    require(all(r["state"] == "SUCCEEDED" for r in remote), "Unsuccessful ES records")
    require(unresolved == 0, "Unresolved controller anomalies")
    for observation in observations:
        sequence = observation["candidate"]["payload"]["sequence"]
        require(isinstance(sequence, str) and len(sequence) == 11, "Not a PoC sequence candidate")
        if config["oracle"] == "mock":
            digest = hashlib.sha256(f"mock-v1/{config['antigen']}/{sequence}".encode()).digest()
            energy = -int.from_bytes(digest[:4], "big") / 2**32
            require(observation["evaluation"]["metrics"] == {"mock_energy": energy},
                    "Oracle output differs from original sequence-hash contract")
    receipts = [json.loads(p.read_text()) for p in sorted((directory / "design_jobs").glob("job-*/receipt.json"))]
    require(bool(receipts), "Missing actual task generation receipts")
    require(all(r["status"] == "succeeded" for r in receipts), "Incomplete generation receipt")
    require(all(r["mock_generator"] == config["mock"] for r in receipts), "Generator mode differs from config")
    if not config["mock"]:
        for path in (directory / "design_jobs").glob("job-*/receipt.json"):
            require((path.parent / "output/designs.json").is_file(), "Missing model design artifact")
    visibility = directory / "history/visibility.jsonl"
    history_events = [json.loads(line) for line in visibility.read_text().splitlines()] if visibility.exists() else []
    kinds = {kind: sum(r["kind"] == kind for r in history_events) for kind in {r["kind"] for r in history_events}}
    if config["method"] in ("harness", "ldm_harness"):
        require(kinds.get("query", 0) > 0, "Pi did not query history through the host tool")
        require(any(r["kind"] == "decision" and r["decision"] == "accept" for r in history_events),
                "No accepted history-aware submission")
    edges = [(r["started_at"], 1) for r in attempts] + [(r["ended_at"], -1) for r in attempts if r["ended_at"]]
    active = peak = 0
    for _, change in sorted(edges):
        active += change
        peak = max(peak, active)
    require(peak <= config["concurrency"], "ES exceeded configured concurrency")
    return {"run_dir": str(directory), "method": config["method"], "observations": len(observations),
            "attempts": len(attempts), "peak_concurrency": peak, "design_jobs": len(receipts),
            "history_events": kinds, "decisions": decisions, "problems": problems}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    report = audit(args.run_dir.resolve())
    print(json.dumps(report, indent=2))
    raise SystemExit(bool(report["problems"]))
