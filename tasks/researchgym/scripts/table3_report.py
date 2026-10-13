#!/usr/bin/env python3
"""Create an auditable five-task ResearchGym Table 3 report from run receipts.

The launcher writes one normalized ``result.json`` per task/seed.  This command
does no grading and never fills in missing values: it only validates and
aggregates receipts produced from official task grader summaries.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tasks.researchgym.table3_protocol import ProtocolError, aggregate, load_json_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--result", type=Path, action="append", required=True,
                        help="One normalized completed task/seed result; repeat 15 times.")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        table = aggregate(manifest, load_json_records(args.result))
    except (OSError, json.JSONDecodeError, ProtocolError) as exc:
        raise SystemExit(f"Table 3 report rejected: {exc}") from exc
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing report: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(table, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
