"""Evaluate Engineer B baselines on calibration splits and log h12 runs to dev-log.jsonl.

Strictly evaluates ONLY the calibration split; never touches validation or sealed splits.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import fcntl
import hashlib
import inspect
import json
import os
import subprocess
import time
from pathlib import Path
import numpy as np

from tasks.copper_inversion.experiments.baselines import fdi_baseline, gempy_reference, inr_reference, offline_baselines, rf_reference
from tasks.copper_inversion.v2 import baselines
from tasks.copper_inversion.v2.data import read_problem
from tasks.copper_inversion.v2.metrics import compare

DENSITIES = ("lo", "h8", "h12", "h16", "mid", "h24", "hi")

CONFIG_LISTS = {
    "fdi_smooth": fdi_baseline.FDI_CONFIGS,
    "fdi_smooth_native": fdi_baseline.FDI_CONFIGS,
    "potential_field": gempy_reference.GEMPY_CONFIGS,
    "potential_field_native": gempy_reference.GEMPY_CONFIGS,
    "inr_mlp": inr_reference.INR_CONFIGS,
    "inr_mlp_native": inr_reference.INR_CONFIGS,
    "rf_xyz": rf_reference.RF_CONFIGS,
    "rf_xyz_native": rf_reference.RF_CONFIGS,
}


def _config_hash(solver_name):
    payload = json.dumps(CONFIG_LISTS[solver_name], sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "bl-b-sync"


def _append_dev_log(dev_log_path, record):
    dev_log_path = Path(dev_log_path)
    dev_log_path.parent.mkdir(parents=True, exist_ok=True)
    with dev_log_path.open("a+") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        fh.seek(0)
        existing = [json.loads(line) for line in fh.read().splitlines() if line.strip()]
        count = sum(
            1 for r in existing
            if r.get("method") == record["method"] and r.get("density") == "h12"
        )
        if count >= 10:
            raise RuntimeError(f"h12 calibration budget (10) exhausted for {record['method']}")
        fh.seek(0, os.SEEK_END)
        fh.write(json.dumps(record, sort_keys=True) + "\n")
        fh.flush()


def _eval_one_case(task):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    pub_dir, truth_path, fam, solver_name, fields_dir, split, seed = task
    solver = offline_baselines.EXTENDED_REFERENCE_SOLVERS[solver_name]
    params = inspect.signature(solver).parameters
    problem = read_problem(pub_dir)
    truth = np.load(truth_path, allow_pickle=False)
    diag = {}
    kwargs = {}
    if "diagnostics" in params:
        kwargs["diagnostics"] = diag
    if "fields_dir" in params and fields_dir is not None:
        kwargs["fields_dir"] = fields_dir
    if "split" in params:
        kwargs["split"] = split
    t0 = time.monotonic()
    field = solver(problem, {"wall_time_seconds": 120.0}, int(seed), **kwargs)
    elapsed = float(diag["wall_seconds"]) if ("diagnostics" in params and "wall_seconds" in diag) else (time.monotonic() - t0)
    stats = compare(problem, field, truth)
    return fam, stats, elapsed, diag


def evaluate_release_parallel(release_dir, split, solver_name, *, fields_dir=None, seed=0, workers=8):
    if split != "calibration":
        raise ValueError("run_bl_b_calibration only evaluates the calibration split")
    manifest_path = baselines._resolve_manifest(release_dir, split)
    manifest = json.loads(manifest_path.read_text())
    root = manifest_path.parent
    families = baselines._family_map(release_dir, split, manifest)
    solver = offline_baselines.EXTENDED_REFERENCE_SOLVERS[solver_name]
    params = inspect.signature(solver).parameters
    accepts_diag = "diagnostics" in params
    accepts_fields = "fields_dir" in params

    tasks = [
        (
            str(root / case["public"]),
            str(root / case["truth"] / "truth.npy"),
            families[case["case_id"]],
            solver_name,
            str(fields_dir) if fields_dir is not None else None,
            split,
            int(seed),
        )
        for case in manifest["cases"]
    ]
    started = time.monotonic()
    if workers <= 1:
        results = [_eval_one_case(t) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=min(int(workers), 16)) as pool:
            results = list(pool.map(_eval_one_case, tasks))
    wall = time.monotonic() - started

    per_family, all_rows, case_times = {}, [], []
    repaired_count, fallback_count = 0, 0
    rule_counts, s_z_counts = {}, {}
    for fam, stats, elapsed, diag in results:
        per_family.setdefault(fam, []).append(stats)
        all_rows.append(stats)
        case_times.append(elapsed)
        if accepts_diag:
            if diag.get("repaired"):
                repaired_count += 1
            if diag.get("fallback"):
                fallback_count += 1
            rule = diag.get("selected_rule")
            if rule is not None:
                rule_counts[str(rule)] = rule_counts.get(str(rule), 0) + 1
            sz = diag.get("selected_s_z")
            if sz is not None:
                s_z_counts[str(sz)] = s_z_counts.get(str(sz), 0) + 1

    def summarize(items):
        ious = [r["iou"] for r in items]
        return {
            "n": len(items),
            "mean_iou": float(np.mean(ious)) if ious else 0.0,
            "std_iou": float(np.std(ious)) if ious else 0.0,
            "min_iou": float(np.min(ious)) if ious else 0.0,
            "mean_surface_f1": float(np.mean([r["surface_f1"] for r in items])) if items else 0.0,
            "mean_rel_vol_err": float(np.mean([r["relative_volume_error"] for r in items])) if items else 0.0,
        }

    by_family = {fam: summarize(per_family[fam]) for fam in sorted(per_family)}
    family_macro_iou = float(np.mean([v["mean_iou"] for v in by_family.values()])) if by_family else 0.0
    n_cases = len(all_rows)
    return {
        "release": str(Path(release_dir)),
        "split": split,
        "solver": solver_name,
        "seed": int(seed),
        "generator_matched": bool(solver_name in ("potential_field", "potential_field_native")),
        "overall": summarize(all_rows),
        "family_macro_iou": family_macro_iou,
        "by_family": by_family,
        "wall_seconds": round(float(sum(case_times)) if accepts_fields else wall, 3),
        "max_case_seconds": round(float(np.max(case_times)), 3) if case_times else 0.0,
        "median_case_seconds": round(float(np.median(case_times)), 3) if case_times else 0.0,
        "repair_rate": (float(repaired_count) / n_cases) if (accepts_diag and n_cases) else None,
        "fallback_count": fallback_count if accepts_diag else None,
        "fallback_rate": (float(fallback_count) / n_cases) if (accepts_diag and n_cases) else None,
        "rule_counts": rule_counts if accepts_diag else None,
        "s_z_counts": s_z_counts if accepts_diag else None,
    }


def _aggregate_seeds(seed_reports):
    if len(seed_reports) == 1:
        rep = dict(seed_reports[0])
        rep["seeds_evaluated"] = [rep["seed"]]
        return rep

    def mean_range(vals):
        arr = np.asarray(vals, dtype=float)
        return {
            "mean": float(np.mean(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "range": float(np.max(arr) - np.min(arr)),
            "std": float(np.std(arr)),
        }

    families = sorted(seed_reports[0]["by_family"].keys())
    by_family = {}
    for fam in families:
        by_family[fam] = {
            "n": seed_reports[0]["by_family"][fam]["n"],
            "iou": mean_range([r["by_family"][fam]["mean_iou"] for r in seed_reports]),
            "surface_f1": mean_range([r["by_family"][fam]["mean_surface_f1"] for r in seed_reports]),
            "rel_vol_err": mean_range([r["by_family"][fam]["mean_rel_vol_err"] for r in seed_reports]),
        }

    return {
        "solver": seed_reports[0]["solver"],
        "generator_matched": seed_reports[0]["generator_matched"],
        "seeds_evaluated": [r["seed"] for r in seed_reports],
        "family_macro_iou": mean_range([r["family_macro_iou"] for r in seed_reports]),
        "overall_iou": mean_range([r["overall"]["mean_iou"] for r in seed_reports]),
        "surface_f1": mean_range([r["overall"]["mean_surface_f1"] for r in seed_reports]),
        "rel_vol_err": mean_range([r["overall"]["mean_rel_vol_err"] for r in seed_reports]),
        "repair_rate": mean_range([r["repair_rate"] for r in seed_reports]),
        "fallback_rate": mean_range([r["fallback_rate"] for r in seed_reports]),
        "median_case_seconds": mean_range([r["median_case_seconds"] for r in seed_reports]),
        "max_case_seconds": float(max(r["max_case_seconds"] for r in seed_reports)),
        "by_family": by_family,
        "per_seed": seed_reports,
    }


def evaluate_method_calibration(release_root, solver_name, densities, *, fields_root=None,
                                seeds=(0,), workers=8, dev_log=None, commit_sha=None, output_path=None):
    release_root = Path(release_root).resolve()
    commit = commit_sha or _git_commit()
    cfg_hash = _config_hash(solver_name)
    out = {}
    if output_path is not None and Path(output_path).is_file():
        out = json.loads(Path(output_path).read_text())
    for dens in densities:
        if dens in out:
            continue
        rel_dir = release_root / dens
        f_dir = (Path(fields_root) / dens) if fields_root is not None else None
        seed_reps = [
            evaluate_release_parallel(rel_dir, "calibration", solver_name, fields_dir=f_dir, seed=int(s), workers=workers)
            for s in seeds
        ]
        summary = _aggregate_seeds(seed_reps)
        out[dens] = summary
        if output_path is not None:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).write_text(json.dumps(out, indent=2, sort_keys=True))
        if dens == "h12" and dev_log is not None:
            if len(seed_reps) == 1:
                macro_iou = seed_reps[0]["family_macro_iou"]
                ov_iou = seed_reps[0]["overall"]["mean_iou"]
                rep_rate = seed_reps[0]["repair_rate"]
            else:
                macro_iou = summary["family_macro_iou"]["mean"]
                ov_iou = summary["overall_iou"]["mean"]
                rep_rate = summary["repair_rate"]["mean"]
            _append_dev_log(dev_log, {
                "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "track": "bl-b",
                "method": solver_name,
                "generator_matched": bool(solver_name in ("potential_field", "potential_field_native")),
                "density": "h12",
                "seeds": [int(s) for s in seeds],
                "commit": commit,
                "config_list_sha256": cfg_hash,
                "family_macro_iou": round(float(macro_iou), 6),
                "overall_mean_iou": round(float(ov_iou), 6),
                "repair_rate": round(float(rep_rate), 6) if rep_rate is not None else None,
            })
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate Engineer B baselines on calibration splits.")
    parser.add_argument("--release-root", type=Path, required=True, help="Root holding {lo,h8,h12,h16,mid,h24,hi}.")
    parser.add_argument("--solver", required=True, choices=tuple(CONFIG_LISTS.keys()))
    parser.add_argument("--densities", default=",".join(DENSITIES))
    parser.add_argument("--fields-root", type=Path, default=None, help="Root holding <density>/calibration/<case_id>.")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--dev-log", type=Path, default=None)
    parser.add_argument("--commit", default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    densities = [d.strip() for d in args.densities.split(",") if d.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    evaluate_method_calibration(
        args.release_root, args.solver, densities,
        fields_root=args.fields_root, seeds=seeds, workers=args.workers,
        dev_log=args.dev_log, commit_sha=args.commit, output_path=args.output,
    )
    print(json.dumps({"solver": args.solver, "densities": densities, "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
