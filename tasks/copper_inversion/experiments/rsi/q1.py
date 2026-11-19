"""Select a reproducible stratified Q1 sample from rehearsal ledgers."""

import argparse
import hashlib
import json
import random
from pathlib import Path

from tasks.copper_inversion.rsi.ledger import read


def select(runs, output, seed):
    """Sample six distinct evaluated programs per risk decile, then twelve repeats.

    Identical program IDs across runs are counted once, using the occurrence in
    the lexicographically first resolved run path. Rejected programs have no
    complete training evaluation and are excluded.
    """
    paths = sorted({Path(run).resolve() for run in runs}, key=str)
    if not paths or type(seed) is not int or seed < 0:
        raise ValueError("run directories and a nonnegative integer seed are required")
    found, hashes = {}, {}
    for run in paths:
        ledger = run / "ledger.jsonl"
        body = ledger.read_bytes()
        hashes[str(run)] = hashlib.sha256(body).hexdigest()
        for row in read(ledger):
            if row["status"] != "rejected":
                found.setdefault(row["program_id"], {"program_id": row["program_id"],
                                                    "source_run": str(run), "risk": row["risk"]})
    ordered = sorted(found.values(), key=lambda r: (r["risk"], r["program_id"]))
    if len(ordered) < 60:
        raise ValueError("Q1 requires at least 60 distinct evaluated programs")
    rng = random.Random(seed)
    selected = []
    for decile in range(10):
        group = ordered[len(ordered) * decile // 10:len(ordered) * (decile + 1) // 10]
        selected.extend({**row, "decile": decile + 1} for row in rng.sample(group, 6))
    selected.sort(key=lambda row: (row["decile"], row["program_id"]))
    repeats = sorted(rng.sample([row["program_id"] for row in selected], 12))
    result = {"schema": "rsi-q1-selection-v1", "seed": seed, "input_run_hashes": hashes,
              "programs": selected, "repeat_program_ids": repeats}
    data = (json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        stream.write(data)
    return hashlib.sha256(data).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    choice = sub.add_parser("select", help="sample evaluated rehearsal programs")
    choice.add_argument("--run", action="append", required=True, dest="runs")
    choice.add_argument("--seed", type=int, required=True)
    choice.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(select(args.runs, args.output, args.seed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
