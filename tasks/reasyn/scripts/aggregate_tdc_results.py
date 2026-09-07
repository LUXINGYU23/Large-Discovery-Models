#!/usr/bin/env python3
"""Aggregate only a complete real released-oracle three-seed result grid."""

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from tasks.reasyn.core.metrics import ORACLES


def aggregate(paths):
    found = {}
    for path in paths:
        data = json.loads(Path(path).read_text())
        identity = json.loads(
            (Path(path).parent / "scientific_identity.json").read_text()
        )
        if (
            data.get("mock")
            or data.get("benchmark") != "tdc"
            or not data.get("official_budget")
            or data.get("completed_oracle_calls") != 10000
            or data.get("metrics", {}).get("auc_top10") is None
        ):
            raise ValueError(f"{path}: not a completed real 10000-call TDC run")
        key = (data["oracle"], identity["seed"])
        if key in found:
            raise ValueError(f"duplicate oracle/seed {key}")
        found[key] = data["metrics"]["auc_top10"]
    expected = {(o, s) for o in ORACLES for s in (0, 1, 2)}
    if set(found) != expected:
        raise ValueError(
            f"incomplete or unexpected result grid: missing={sorted(expected - set(found))}, extra={sorted(set(found) - expected)}"
        )
    values = {o: [found[o, s] for s in (0, 1, 2)] for o in ORACLES}
    return {
        "task": "reasyn",
        "benchmark": "released_tdc_13",
        "seeds": [0, 1, 2],
        "oracles": {
            o: {"mean": statistics.mean(v), "std": statistics.pstdev(v)}
            for o, v in values.items()
        },
        "mean_auc_top10": statistics.mean(found.values()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("results", type=Path, nargs="+")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = aggregate(args.results)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
