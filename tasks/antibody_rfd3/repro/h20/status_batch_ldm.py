#!/usr/bin/env python3
"""Read batch progress without touching a live campaign or exposing provider content."""
import argparse
import json
from pathlib import Path
import sqlite3


def status(root):
    launch = json.loads((root / "launch.json").read_text())
    result = {"run_dir": str(root), "source_commit": launch["source_commit"],
              "launcher_status": launch["status"], "started_at": launch["started_at"]}
    campaign = root / "campaign"
    database = campaign / "controller.sqlite3"
    if not database.is_file():
        return result
    connection = sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        result["research"] = [dict(row) for row in connection.execute(
            "SELECT generation,state,proposal_count,admitted,error FROM generations ORDER BY generation")]
        result["plans"] = [dict(row) for row in connection.execute(
            "SELECT plan_id,generation,count,consumed,closed_reason FROM plans ORDER BY plan_id")]
        observations = [json.loads(row[0]) for row in connection.execute(
            "SELECT observation_json FROM observations ORDER BY ingest_seq")]
        successful = [row for row in observations if row["evaluation"]["status"] == "succeeded"]
        result["evaluations"] = {"settled": len(observations), "succeeded": len(successful),
                                 "failed": len(observations) - len(successful)}
        result["best_energy"] = min((row["evaluation"]["metrics"]["absolut_energy"]
                                     for row in successful), default=None)
        result["anomalies"] = [dict(row) for row in connection.execute(
            "SELECT type,subject_id,payload_json FROM anomalies WHERE resolved_at IS NULL")]
    finally:
        connection.close()
    jobs = []
    for receipt in sorted((campaign / "design_jobs").glob("job-*/receipt.json")):
        record = json.loads(receipt.read_text())
        jobs.append({"job": receipt.parent.name, "round": record["round"],
                     "role": record.get("profile_id"), "status": record["status"], "count": record["count"]})
    result["generation_jobs"] = jobs
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    args = parser.parse_args()
    print(json.dumps(status(args.run_root.resolve()), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
