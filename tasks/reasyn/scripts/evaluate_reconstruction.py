#!/usr/bin/env python3
"""Evaluate released projection CSV against the full requested target manifest."""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from tasks.reasyn.core.metrics import reconstruction_metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--targets", type=Path, required=True)
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    targets = [s.strip() for s in args.targets.read_text().splitlines() if s.strip()]
    if targets and targets[0].lower() == "smiles":
        targets = targets[1:]
    with args.predictions.open() as handle:
        rows = list(csv.DictReader(handle))
    result = reconstruction_metrics(targets, rows)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
