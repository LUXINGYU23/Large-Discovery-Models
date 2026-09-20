#!/usr/bin/env python3
"""Materialize seed-replicated configs from runner-enforced task profiles."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--benchmark", choices=("reconstruction", "tdc"), required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    contract = json.loads((root / "tasks/reasyn/experiment.json").read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    if args.benchmark == "tdc":
        items = [
            ("tdc_10000", {"oracle": oracle})
            for oracle in contract["evaluation"]["settings"]["released_tdc_oracles"]
        ]
    else:
        items = [
            (name, {})
            for name in contract["profiles"]
            if name.startswith("reconstruction_") and name != "reconstruction_tiny"
        ]
    for profile, extra in items:
        for seed in (0, 1, 2):
            name = f"reasyn_{profile}_{extra.get('oracle', 'suite')}_seed{seed}"
            config = {
                "name": name,
                "task": "reasyn",
                "algorithm": "ldm_tanimoto_gp_ucb"
                if profile.endswith("ldm") or profile.startswith("tdc")
                else "projection_baseline",
                "mode": "real",
                "contract_profile": profile,
                "args": {
                    **contract["profiles"][profile]["locked_args"],
                    **extra,
                    "seed": seed,
                    "out-dir": str((args.output_dir / "runs" / name).resolve()),
                },
            }
            (args.output_dir / (name + ".yaml")).write_text(
                json.dumps(config, indent=2) + "\n"
            )
            count += 1
    print(
        json.dumps(
            {
                "generated_configs": count,
                "output_dir": str(args.output_dir.resolve()),
                "launched": False,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
