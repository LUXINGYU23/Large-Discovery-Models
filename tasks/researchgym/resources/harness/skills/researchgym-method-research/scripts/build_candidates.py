"""Build candidates.json from program files so source is never hand-escaped."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="candidates.json")
    parser.add_argument("--program", action="append", required=True, help="Python file; repeat per candidate")
    parser.add_argument("--summary", action="append", required=True, help="change summary per candidate")
    parser.add_argument("--rationale", action="append", required=True, help="rationale per candidate")
    parser.add_argument("--compare", action="append", default=[],
                        help="comma-separated measured candidate IDs for the most recent --program")
    args = parser.parse_args()
    if not len(args.program) == len(args.summary) == len(args.rationale):
        parser.error("give one --summary and one --rationale per --program")
    compare = [c.split(",") if c else [] for c in args.compare] + [[]] * (len(args.program) - len(args.compare))
    rows = []
    for path, summary, rationale, ids in zip(args.program, args.summary, args.rationale, compare):
        row = {"program": Path(path).read_text(), "change_summary": summary, "rationale": rationale}
        if ids:
            row["comparison_candidate_ids"] = [i.strip() for i in ids if i.strip()]
        rows.append(row)
    Path(args.out).write_text(json.dumps({"candidates": rows}, indent=1))
    print(f"wrote {len(rows)} candidates to {args.out}")


if __name__ == "__main__":
    main()
