"""Prepare a reproducible released-data subset; never relabel it a verified paper split."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from tasks.atomworld.core.data import DEFAULT_UPSTREAM, PAPER_ACTIONS, prepare_dataset


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir", type=Path, default=DEFAULT_UPSTREAM / "src/data"
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--actions", default=",".join(PAPER_ACTIONS))
    parser.add_argument(
        "--per-action", type=int, default=1, help="0 selects all released rows"
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    result = prepare_dataset(
        args.source_dir,
        args.out_dir,
        actions=tuple(args.actions.split(",")),
        per_action=args.per_action,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                k: result[k]
                for k in ("sample_count", "dataset_kind", "available_counts")
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
