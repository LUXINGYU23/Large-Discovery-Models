"""Generate candidate-density open-split sub-releases of `rsi-r2` (holding off sealed splits).

Creates `$R/private/rsi-r2/master-plan`, `$R/private/rsi-r2/datasets/<density>`,
`$R/private/rsi-r2/release/<density>`, and `$R/public/rsi-r2-<density>-train-{s,m,f}`.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from tasks.copper_inversion.experiments.v2.publish_training import publish
from tasks.copper_inversion.v2.public_catalog import catalog_digest
from .plan import load_generation_plan, make_generation_plan, write_generation_plan
from .release import build

DEFAULT_CANDIDATES = (
    ("h8", (8, 8)),
    ("h12", (12, 12)),
    ("h16", (16, 16)),
    ("h24", (24, 24)),
    ("lo", (3, 12)),
    ("mid", (12, 24)),
    ("hi", (24, 48)),
)


def parse_spec(item):
    if "=" not in item:
        for label, bounds in DEFAULT_CANDIDATES:
            if label == item:
                return label, bounds
        raise ValueError(f"unknown candidate density {item!r}")
    label, rhs = item.split("=", 1)
    parts = [int(x) for x in rhs.replace("-", ",").split(",")]
    if len(parts) == 1:
        parts = [parts[0], parts[0]]
    if len(parts) != 2 or parts[0] < 1 or parts[0] > parts[1]:
        raise ValueError(f"invalid density specification {item!r}")
    return label, (parts[0], parts[1])


def run_step(cmd):
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-root", type=Path, required=True, help="e.g. $R/private/rsi-r2")
    parser.add_argument("--public-root", type=Path, required=True, help="e.g. $R/public")
    parser.add_argument("--gempy-python", type=Path, required=True)
    parser.add_argument("--master-seed", type=int, default=2026112401)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--historical", nargs="+", required=True, metavar="LABEL=PATH")
    parser.add_argument("--candidates", nargs="*", help="subset or custom LABEL=LO,HI specs")
    parser.add_argument("--summary-out", type=Path)
    args = parser.parse_args()
    if os.geteuid() != 991:
        raise PermissionError("rsi-r2 candidate generation requires evaluator uid 991")
    if not 1 <= args.workers <= 16:
        raise ValueError("workers must be in [1, 16]")

    private_root = args.private_root.resolve()
    public_root = args.public_root.resolve()
    private_root.mkdir(parents=True, mode=0o700, exist_ok=True)
    private_root.chmod(0o700)

    candidates = [parse_spec(x) for x in args.candidates] if args.candidates else list(DEFAULT_CANDIDATES)
    historical = dict(item.split("=", 1) for item in args.historical)

    master_dir = private_root / "master-plan"
    if not (master_dir / "generation-plan.json").is_file():
        plan = make_generation_plan("rsi-r2", args.master_seed, "resample_until_separated", historical)
        write_generation_plan(master_dir, plan)
    rsi, rsi_sha = load_generation_plan(master_dir)

    datasets_root = private_root / "datasets"
    releases_root = private_root / "release"
    datasets_root.mkdir(mode=0o700, exist_ok=True)
    releases_root.mkdir(mode=0o700, exist_ok=True)

    base_dataset = None
    for existing_dir in sorted(datasets_root.glob("*")):
        if (existing_dir / "id" / "completion.json").is_file() and (existing_dir / "ood" / "completion.json").is_file():
            base_dataset = existing_dir
            break

    summaries = []
    for label, (lo, hi) in candidates:
        t0 = time.monotonic()
        ds = datasets_root / label
        rel = releases_root / label
        if not (rel / "bench-card.json").is_file():
            if not ds.exists():
                ds.mkdir(parents=True, mode=0o700)
                shutil.copy2(master_dir / "generation-plan.json", ds / "generation-plan.json")
                shutil.copy2(master_dir / "generation-plan-receipt.json", ds / "generation-plan-receipt.json")
            if not (ds / "id" / "completion.json").is_file():
                cmd_id = [
                    sys.executable, "-m", "tasks.copper_inversion.experiments.v2.generate_pilot",
                    "--output", str(ds / "id"), "--generator-python", str(args.gempy_python),
                    "--seed", str(rsi["id_seed"]), "--workers", str(args.workers),
                    "--holes", str(lo), str(hi), "--splits", "train", "calibration", "validation",
                ]
                if base_dataset is not None and base_dataset != ds:
                    cmd_id += ["--reuse-parents", str(base_dataset / "id")]
                run_step(cmd_id)
            if not (ds / "ood" / "completion.json").is_file():
                cmd_ood = [
                    sys.executable, "-m", "tasks.copper_inversion.experiments.v2.generate_pilot",
                    "--output", str(ds / "ood"), "--ood-for", str(ds / "id"),
                    "--generator-python", str(args.gempy_python),
                    "--seed", str(rsi["ood_seed"]), "--workers", str(args.workers),
                    "--holes", str(lo), str(hi), "--splits", "calibration_ood",
                ]
                if base_dataset is not None and base_dataset != ds:
                    cmd_ood += ["--reuse-parents", str(base_dataset / "ood")]
                run_step(cmd_ood)
            if base_dataset is None:
                base_dataset = ds
            if not (ds / "population-qa.json").is_file():
                run_step([
                    sys.executable, "-m", "tasks.copper_inversion.experiments.v2.qualify_population",
                    "--root", str(ds), "--output", str(ds / "population-qa.json"),
                ])
            if not (ds / "confirmation-complete.json").is_file():
                run_step([
                    sys.executable, "-m", "tasks.copper_inversion.experiments.v2.generate_confirmation",
                    "--root", str(ds), "--python", str(args.gempy_python), "--workers", str(args.workers),
                ])
            build(ds, rel, open_only=True, sub_release=f"rsi-r2-{label}")

        catalogs = {}
        for tier_suffix, manifest_rel in (("s", "train_s/manifest.json"),
                                          ("m", "train_m/manifest.json"),
                                          ("f", "train/manifest.json")):
            pub_dest = public_root / f"rsi-r2-{label}-train-{tier_suffix}"
            if not (pub_dest / "catalog.json").is_file():
                publish(rel / manifest_rel, pub_dest)
            catalogs[tier_suffix] = dict(path=str(pub_dest), observation_digest=catalog_digest(pub_dest / "catalog.json"))

        card = json.loads((rel / "bench-card.json").read_text())
        elapsed = round(time.monotonic() - t0, 2)
        entry = dict(
            sub_release=card["sub_release"],
            density_label=label,
            public_holes=[lo, hi],
            elapsed_seconds=elapsed,
            release_dir=str(rel),
            bench_card_sha256=json.loads((rel / "development-receipt.json").read_text())["artifacts_sha256"]["bench_card"],
            hole_audit=card["open_split_aggregates"]["hole_audit"],
            by_id_family=card["open_split_aggregates"]["by_id_family"],
            by_split=card["open_split_aggregates"]["by_split"],
            public_catalogs=catalogs,
        )
        summaries.append(entry)
        print(json.dumps({"completed": label, "holes": [lo, hi], "elapsed_seconds": elapsed,
                          "hole_audit": entry["hole_audit"]["status"]}), flush=True)

    report = dict(
        release="rsi-r2",
        generation_plan_sha256=rsi_sha,
        planned_counts=rsi["counts"],
        sealed_splits_held_off=["test_id", "test_ood"],
        candidates=summaries,
    )
    if args.summary_out:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"status": "completed", "sub_releases": [s["sub_release"] for s in summaries]}), flush=True)


if __name__ == "__main__":
    main()
