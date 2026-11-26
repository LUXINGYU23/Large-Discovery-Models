"""Stage-B loaders, legalisers, and calibration CLI for offline two-stage baselines (M4, M7, M8-RF).

Kept under ``experiments/baselines`` rather than ``v2`` so candidate-visible
runtime modules and ``PublicTrainingExport`` match the pilot checkout.
"""

import argparse
import inspect
import json
import time
from pathlib import Path
import numpy as np
import skfmm
from ...v2 import baseline_harness as harness
from ...v2 import baselines as v2_baselines
from ...v2.data import read_problem
from ...v2.metrics import compare
from .fdi_baseline import fdi_smooth, fdi_smooth_native

OFFLINE_SOLVER_NAMES = ("potential_field", "inr_mlp", "rf_xyz")
OFFLINE_NATIVE_SOLVER_NAMES = ("potential_field_native", "inr_mlp_native", "rf_xyz_native")


def resolve_stage_a_dir(fields_dir, case_id, *, seed=0, split=None):
    """Locate Stage-A output directory for a (case_id, seed) pair."""
    if fields_dir is None:
        raise ValueError("offline solver requires fields_dir (--fields-dir)")
    base = Path(fields_dir)
    candidates = []
    if split is not None:
        candidates.extend([
            base / split / case_id / f"seed_{seed}",
            base / split / case_id / str(seed),
            base / split / case_id,
        ])
    candidates.extend([
        base / case_id / f"seed_{seed}",
        base / case_id / str(seed),
        base / case_id,
    ])
    for cand in candidates:
        if (cand / "choice.json").is_file():
            return cand
    raise FileNotFoundError(
        f"Stage-A output choice.json not found for case_id={case_id!r}, seed={seed} under {base}"
    )


def _signed_distance_on_feasible(ore_mask, data):
    ore = np.asarray(ore_mask, dtype=bool)
    if ore.shape != data["shape"]:
        raise ValueError(f"precomputed ore.npy shape {ore.shape} != expected {data['shape']}")
    if ore.any() and (~ore).any():
        phi = np.where(ore, 1.0, -1.0)
        dist = np.asarray(skfmm.distance(phi, dx=data["spacing"]), dtype=float)
        return dist[data["feasible"]]
    inward = 1e-4 * np.linalg.norm(data["G"] - data["center"], axis=1)
    if ore.any():
        return np.ones(data["M"], dtype=float) - inward
    return -np.ones(data["M"], dtype=float) - inward


def _load_continuous_on_feasible(field_arr, data):
    arr = np.asarray(field_arr, dtype=float)
    if arr.shape == data["shape"]:
        arr = arr[data["feasible"]]
    else:
        arr = arr.reshape(-1)
    if len(arr) != data["M"] or not np.isfinite(arr).all():
        raise ValueError(f"invalid precomputed field shape {arr.shape} for M={data['M']}")
    return arr


def _solve_offline(method_name, tau0, is_mask, problem, budget=None, seed=0, *,
                   fields_dir=None, split=None, decision_rule=None, diagnostics=None):
    del budget
    t0 = time.monotonic()
    data = harness.extract_public_data(problem)
    if fields_dir is None:
        raise ValueError(
            f"{method_name} is an offline two-stage baseline; supply fields_dir (--fields-dir)"
        )
    case_dir = resolve_stage_a_dir(fields_dir, problem["case_id"], seed=int(seed), split=split)
    choice = json.loads((case_dir / "choice.json").read_text())
    stage_a_wall = float(choice.get("wall_seconds", 0.0))

    if choice.get("no_ore") or not (data["y_b"] > 0.5).any():
        out = harness.no_ore_fallback(problem, data, diagnostics=diagnostics, started=t0)
        if diagnostics is not None:
            diagnostics["wall_seconds"] = stage_a_wall + float(time.monotonic() - t0)
        return out

    force_native = decision_rule == "native"
    block = choice.get("native" if force_native and "native" in choice else "cv", choice)
    if block.get("fallback") or choice.get("fallback"):
        out = harness.stretched_fallback(
            problem, data, diagnostics=diagnostics, started=t0,
            cv_evaluated=int(block.get("cv_evaluated", choice.get("cv_evaluated", 0))),
            deadline_stopped=bool(choice.get("deadline_stopped", False)),
        )
        if diagnostics is not None:
            diagnostics["wall_seconds"] = stage_a_wall + float(time.monotonic() - t0)
        return out

    rule = "native" if force_native else str(block.get("selected_rule", "share"))
    if decision_rule == "share":
        rule = "share"
    use_native_file = force_native and not bool(block.get("same_as_cv", True))
    if is_mask:
        fname = "ore_native.npy" if use_native_file and (case_dir / "ore_native.npy").is_file() else "ore.npy"
        raw = np.load(case_dir / fname, allow_pickle=False)
        f = _signed_distance_on_feasible(raw, data)
    else:
        fname = "field_native.npy" if use_native_file and (case_dir / "field_native.npy").is_file() else "field.npy"
        raw = np.load(case_dir / fname, allow_pickle=False)
        f = _load_continuous_on_feasible(raw, data)

    cfg = block.get("selected_config")
    out = harness.finalize_field(
        problem,
        data,
        f,
        tau0,
        rule,
        selected_config=cfg,
        cv_score=block.get("cv_score"),
        cv_evaluated=int(block.get("cv_evaluated", choice.get("cv_evaluated", 0))),
        deadline_stopped=bool(choice.get("deadline_stopped", False)),
        diagnostics=diagnostics,
        started=t0,
    )
    if diagnostics is not None:
        diagnostics["wall_seconds"] = stage_a_wall + float(time.monotonic() - t0)
        if method_name == "potential_field":
            diagnostics["generator_matched"] = True
    return out


def potential_field(problem, budget=None, seed=0, *, fields_dir=None, split=None,
                    decision_rule=None, diagnostics=None):
    """M4 GemPy potential-field co-kriging Stage-B solver (offline, generator-matched)."""
    return _solve_offline(
        "potential_field", 0.0, True, problem, budget, seed,
        fields_dir=fields_dir, split=split, decision_rule=decision_rule, diagnostics=diagnostics,
    )


def potential_field_native(problem, budget=None, seed=0, *, fields_dir=None, split=None, diagnostics=None):
    """M4 potential_field with decision rule forced to native (generator-matched)."""
    return potential_field(
        problem, budget, seed, fields_dir=fields_dir, split=split,
        decision_rule="native", diagnostics=diagnostics,
    )


def inr_mlp(problem, budget=None, seed=0, *, fields_dir=None, split=None,
            decision_rule=None, diagnostics=None):
    """M7 GeoINR-style Softplus MLP Stage-B solver (offline)."""
    return _solve_offline(
        "inr_mlp", 0.0, False, problem, budget, seed,
        fields_dir=fields_dir, split=split, decision_rule=decision_rule, diagnostics=diagnostics,
    )


def inr_mlp_native(problem, budget=None, seed=0, *, fields_dir=None, split=None, diagnostics=None):
    """M7 inr_mlp with decision rule forced to native."""
    return inr_mlp(
        problem, budget, seed, fields_dir=fields_dir, split=split,
        decision_rule="native", diagnostics=diagnostics,
    )


def rf_xyz(problem, budget=None, seed=0, *, fields_dir=None, split=None,
           decision_rule=None, diagnostics=None):
    """M8 Random-forest coordinate classifier Stage-B solver (offline)."""
    return _solve_offline(
        "rf_xyz", 0.5, False, problem, budget, seed,
        fields_dir=fields_dir, split=split, decision_rule=decision_rule, diagnostics=diagnostics,
    )


def rf_xyz_native(problem, budget=None, seed=0, *, fields_dir=None, split=None, diagnostics=None):
    """M8 rf_xyz with decision rule forced to native."""
    return rf_xyz(
        problem, budget, seed, fields_dir=fields_dir, split=split,
        decision_rule="native", diagnostics=diagnostics,
    )


OFFLINE_REFERENCE_SOLVERS = {
    "potential_field": potential_field,
    "potential_field_native": potential_field_native,
    "inr_mlp": inr_mlp,
    "inr_mlp_native": inr_mlp_native,
    "rf_xyz": rf_xyz,
    "rf_xyz_native": rf_xyz_native,
}

EXTENDED_REFERENCE_SOLVERS = {
    **v2_baselines.REFERENCE_SOLVERS,
    "fdi_smooth": fdi_smooth,
    "fdi_smooth_native": fdi_smooth_native,
    **OFFLINE_REFERENCE_SOLVERS,
}


def materialize_precomputed_reference(solver_name, stage_a_dir, release_dir, output_dir,
                                      *, splits=("train", "validation", "test_id", "test_ood"), tier=None):
    """Materialize Stage-B legal model.npy files on public inputs only and return precomputed_manifest sha256."""
    from ldm_tts.engine.run_store import atomic_json_write
    from ...v2.selection import precomputed_manifest
    if solver_name not in OFFLINE_REFERENCE_SOLVERS:
        raise ValueError(f"unknown offline solver: {solver_name!r}")
    solver = OFFLINE_REFERENCE_SOLVERS[solver_name]
    stage_a_dir = Path(stage_a_dir).resolve()
    release_dir = Path(release_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if tier is not None:
        rec_path = next((release_dir / n for n in ("qualification.json", "development-receipt.json")
                         if (release_dir / n).is_file()), None)
        if rec_path is None:
            raise ValueError(f"release record missing in {release_dir}")
        row = json.loads(rec_path.read_text())["tiers"][tier]
        budget_s = float(row["solver_seconds"])
        rel_manifests = [row["train_manifest"], row["validation_manifest"], *row["sealed_manifests"]]
        plan = [(release_dir / rel, Path(rel).parent.as_posix(), budget_s) for rel in rel_manifests]
    else:
        plan = [(release_dir / split / "manifest.json", split, 120.0) for split in splits]

    for manifest_path, stage_a_sub, budget_s in plan:
        manifest = json.loads(manifest_path.read_text())
        target_split = manifest["split"]
        use_split = stage_a_sub if (stage_a_dir / stage_a_sub).is_dir() else target_split
        seeds = manifest["seeds"]
        for case in manifest["cases"]:
            problem = read_problem(manifest_path.parent / case["public"])
            for seed in seeds:
                diag = {}
                field = solver(
                    problem, {"wall_time_seconds": budget_s}, int(seed),
                    fields_dir=stage_a_dir, split=use_split, diagnostics=diag,
                )
                dst = output_dir / target_split / case["case_id"] / str(seed)
                dst.mkdir(parents=True, exist_ok=True)
                np.save(dst / "model.npy", np.asarray(field, dtype=np.float64), allow_pickle=False)
                atomic_json_write(dst / "meta.json", {"wall_seconds": round(float(diag.get("wall_seconds", 0.0)), 6)})

    _, sha = precomputed_manifest(output_dir)
    return sha


def evaluate_release(release_dir, split, solver_name, *, fields_dir=None, seed=0):
    """Evaluate an extended or offline reference solver on a train or calibration split."""
    if solver_name not in EXTENDED_REFERENCE_SOLVERS:
        raise ValueError(f"unknown solver: {solver_name!r}")
    manifest_path = v2_baselines._resolve_manifest(release_dir, split)
    manifest = json.loads(manifest_path.read_text())
    root = manifest_path.parent
    families = v2_baselines._family_map(release_dir, split, manifest)
    solver = EXTENDED_REFERENCE_SOLVERS[solver_name]
    params = inspect.signature(solver).parameters
    accepts_diag = "diagnostics" in params
    accepts_fields = "fields_dir" in params
    accepts_split = "split" in params
    budget = {"wall_time_seconds": 120.0}
    per_family, all_rows, case_times = {}, [], []
    repaired_count, fallback_count = 0, 0
    rule_counts, s_z_counts = {}, {}
    started = time.monotonic()
    for case in manifest["cases"]:
        problem = read_problem(root / case["public"])
        truth = np.load(root / case["truth"] / "truth.npy", allow_pickle=False)
        diag = {}
        kwargs = {}
        if accepts_diag:
            kwargs["diagnostics"] = diag
        if accepts_fields and fields_dir is not None:
            kwargs["fields_dir"] = fields_dir
        if accepts_split:
            kwargs["split"] = split
        t0 = time.monotonic()
        field = solver(problem, budget, int(seed), **kwargs)
        elapsed = float(diag["wall_seconds"]) if (accepts_diag and "wall_seconds" in diag) else (time.monotonic() - t0)
        case_times.append(elapsed)
        stats = compare(problem, field, truth)
        fam = families[case["case_id"]]
        per_family.setdefault(fam, []).append(stats)
        all_rows.append(stats)
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
    wall = time.monotonic() - started

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
        "median_case_seconds": round(float(np.median(case_times)), 3) if case_times else 0.0,
        "repair_rate": (float(repaired_count) / n_cases) if (accepts_diag and n_cases) else None,
        "fallback_count": fallback_count if accepts_diag else None,
        "fallback_rate": (float(fallback_count) / n_cases) if (accepts_diag and n_cases) else None,
        "rule_counts": rule_counts if accepts_diag else None,
        "s_z_counts": s_z_counts if accepts_diag else None,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate extended/offline reference solvers on train or calibration splits.")
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "train_s", "calibration"), required=True)
    parser.add_argument("--solver", choices=tuple(EXTENDED_REFERENCE_SOLVERS.keys()), required=True)
    parser.add_argument("--fields-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    rep = evaluate_release(args.release, args.split, args.solver, fields_dir=args.fields_dir, seed=args.seed)
    if args.json:
        print(json.dumps(rep, indent=2, sort_keys=True))
    else:
        print(v2_baselines._format_table([rep]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
