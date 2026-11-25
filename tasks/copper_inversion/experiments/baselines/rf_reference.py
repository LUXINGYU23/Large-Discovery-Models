"""M8 Stage-A Random-Forest coordinate classifier runner (rf_xyz, Breiman 2001).

Runs under $R/env/embedding/bin/python using sklearn.ensemble.RandomForestClassifier.
Features: (x, y, s_z * z, x + y, x - y) centred at the domain midpoint.
12 configurations: min_samples_leaf in (10, 3, 1) x max_features in (1, 3) x s_z in (4, 1),
n_estimators = 300, 5 seeds via numpy.random.default_rng(seed).
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import time

import numpy as np

from tasks.copper_inversion.v2 import baseline_harness as harness
from tasks.copper_inversion.v2.data import read_problem


def _build_rf_configs():
    default = {"s_z": 4, "min_samples_leaf": 3, "max_features": 3}
    configs = [default]
    for min_leaf in (10, 3, 1):
        for max_feat in (1, 3):
            for s_z in (4, 1):
                cfg = {"s_z": s_z, "min_samples_leaf": min_leaf, "max_features": max_feat}
                if cfg != default:
                    configs.append(cfg)
    return tuple(configs)


RF_CONFIGS = _build_rf_configs()


def rf_features(points, center, s_z):
    """Build 5D coordinate features (dx, dy, s_z*dz, dx+dy, dx-dy)."""
    u = harness.stretch_points(points, center, s_z)
    if len(u) == 0:
        return np.zeros((0, 5), dtype=float)
    dx, dy, dz_s = u[:, 0], u[:, 1], u[:, 2]
    return np.column_stack([dx, dy, dz_s, dx + dy, dx - dy])


def fit_rf_predictor(cfg, X_b, y_b, w_b, center, *, seed):
    from sklearn.ensemble import RandomForestClassifier
    rng = np.random.default_rng(int(seed))
    rf_seed = int(rng.integers(0, 2 ** 31 - 1))
    F_tr = rf_features(X_b, center, cfg["s_z"])
    y_int = (y_b > 0.5).astype(int)
    if len(F_tr) == 0:
        raise ValueError("empty training samples for RF")
    if y_int.all():
        return lambda pts: np.ones(len(pts), dtype=float)
    if not y_int.any():
        return lambda pts: np.zeros(len(pts), dtype=float)
    clf = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=int(cfg["min_samples_leaf"]),
        max_features=int(cfg["max_features"]),
        n_jobs=1,
        random_state=rf_seed,
    )
    clf.fit(F_tr, y_int, sample_weight=np.asarray(w_b, dtype=float))
    classes = list(clf.classes_)
    ore_idx = classes.index(1)

    def predict(points_xyz):
        F_q = rf_features(points_xyz, center, cfg["s_z"])
        return np.asarray(clf.predict_proba(F_q)[:, ore_idx], dtype=float)

    return predict


def run_case(public_dir, out_dir, *, seed=0, wall_budget_seconds=120.0):
    """Run M8 rf_xyz Stage-A on one (public_dir, seed) and write field.npy + choice.json."""
    t0 = time.monotonic()
    out_dir = Path(out_dir)
    if (out_dir / "choice.json").is_file() and (
        (out_dir / "field.npy").is_file()
        or json.loads((out_dir / "choice.json").read_text()).get("fallback")
        or json.loads((out_dir / "choice.json").read_text()).get("no_ore")
    ):
        return json.loads((out_dir / "choice.json").read_text())

    problem = read_problem(public_dir)
    data = harness.extract_public_data(problem)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not (data["y_b"] > 0.5).any():
        choice = {
            "method": "rf_xyz",
            "case_id": problem["case_id"],
            "seed": int(seed),
            "no_ore": True,
            "fallback": False,
            "wall_seconds": round(float(time.monotonic() - t0), 4),
        }
        (out_dir / "choice.json").write_text(json.dumps(choice, indent=2, sort_keys=True))
        return choice

    folds = harness.make_hole_folds(data, problem)
    deadline = t0 + 0.5 * float(wall_budget_seconds) if wall_budget_seconds > 0 else None

    best_cv_idx, best_cv_rule, best_cv_score = None, "share", -1.0
    best_nat_idx, best_nat_score = None, -1.0
    cv_evaluated = 0
    deadline_stopped = False

    if folds:
        for idx, cfg in enumerate(RF_CONFIGS):
            if deadline is not None and time.monotonic() >= deadline:
                deadline_stopped = True
                break
            counts = {"native": [0, 0, 0], "share": [0, 0, 0]}
            failed = False
            for f_i, fold in enumerate(folds):
                if deadline is not None and time.monotonic() >= deadline:
                    deadline_stopped = True
                    failed = True
                    break
                fold_seed = int(seed) * 1009 + idx * 37 + f_i
                try:
                    pred_fn = fit_rf_predictor(
                        cfg, fold["X_b_tr"], fold["y_b_tr"], fold["w_b_tr"],
                        data["center"], seed=fold_seed,
                    )
                    n_val = len(fold["X_val"])
                    f_both = harness.predict_chunked(pred_fn, np.vstack([fold["X_val"], data["S"]]))
                    f_val, f_S = f_both[:n_val], f_both[n_val:]
                except Exception:
                    failed = True
                    break
                y_true = fold["y_val"]
                for r in ("native", "share"):
                    if r == "native":
                        pred = f_val > 0.5
                    else:
                        q_cut = float(np.quantile(f_S, 1.0 - fold["p_train"]))
                        if np.any(f_S > q_cut) or float(f_S.max()) <= float(f_S.min()):
                            pred = f_val > q_cut
                        else:
                            pred = f_val >= q_cut
                    counts[r][0] += int(np.count_nonzero(pred & y_true))
                    counts[r][1] += int(np.count_nonzero(pred & ~y_true))
                    counts[r][2] += int(np.count_nonzero(~pred & y_true))
            if failed:
                if deadline_stopped:
                    break
                continue
            cv_evaluated += 1
            sc_nat = harness.pooled_jaccard(*counts["native"])
            sc_shr = harness.pooled_jaccard(*counts["share"])
            if sc_nat > best_nat_score + harness.TIE_TOL:
                best_nat_score = sc_nat
                best_nat_idx = idx
            for r, sc in (("native", sc_nat), ("share", sc_shr)):
                if sc > best_cv_score + harness.TIE_TOL:
                    best_cv_score = sc
                    best_cv_idx = idx
                    best_cv_rule = r

    def _fit_ordered(preferred_idx):
        order = (
            list(range(len(RF_CONFIGS)))
            if preferred_idx is None
            else [preferred_idx] + [i for i in range(len(RF_CONFIGS)) if i != preferred_idx]
        )
        for i in order:
            cfg = RF_CONFIGS[i]
            try:
                pred_fn = fit_rf_predictor(
                    cfg, data["X_b"], data["y_b"], data["w_b"],
                    data["center"], seed=int(seed) * 10007 + i,
                )
                f_grid = harness.predict_chunked(pred_fn, data["G"])
                return i, f_grid.astype(np.float32)
            except Exception:
                continue
        return None, None

    fit_cv_idx, field_cv = _fit_ordered(best_cv_idx)
    if field_cv is None:
        choice = {
            "method": "rf_xyz",
            "case_id": problem["case_id"],
            "seed": int(seed),
            "no_ore": False,
            "fallback": True,
            "cv_evaluated": cv_evaluated,
            "deadline_stopped": deadline_stopped,
            "wall_seconds": round(float(time.monotonic() - t0), 4),
        }
        (out_dir / "choice.json").write_text(json.dumps(choice, indent=2, sort_keys=True))
        return choice

    np.save(out_dir / "field.npy", field_cv, allow_pickle=False)

    target_nat_idx = best_nat_idx if best_nat_idx is not None else fit_cv_idx
    same_as_cv = target_nat_idx == fit_cv_idx
    fit_nat_idx = fit_cv_idx
    if not same_as_cv:
        cand_idx, field_nat = _fit_ordered(target_nat_idx)
        if field_nat is not None and cand_idx != fit_cv_idx:
            fit_nat_idx = cand_idx
            np.save(out_dir / "field_native.npy", field_nat, allow_pickle=False)
        else:
            same_as_cv = True

    choice = {
        "method": "rf_xyz",
        "case_id": problem["case_id"],
        "seed": int(seed),
        "no_ore": False,
        "fallback": False,
        "cv_evaluated": cv_evaluated,
        "deadline_stopped": deadline_stopped,
        "cv": {
            "selected_rule": best_cv_rule if best_cv_idx is not None else "share",
            "selected_config": RF_CONFIGS[fit_cv_idx],
            "cv_score": best_cv_score if fit_cv_idx == best_cv_idx and best_cv_idx is not None else None,
            "cv_evaluated": cv_evaluated,
        },
        "native": {
            "selected_rule": "native",
            "selected_config": RF_CONFIGS[fit_nat_idx],
            "cv_score": best_nat_score if fit_nat_idx == best_nat_idx and best_nat_idx is not None else None,
            "cv_evaluated": cv_evaluated,
            "same_as_cv": same_as_cv,
        },
        "wall_seconds": round(float(time.monotonic() - t0), 4),
    }
    (out_dir / "choice.json").write_text(json.dumps(choice, indent=2, sort_keys=True))
    return choice


def _worker(task):
    pub_dir, out_dir, seed, budget_s = task
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    return run_case(pub_dir, out_dir, seed=seed, wall_budget_seconds=budget_s)


def run_split(release_dir, split, output_root, *, seeds=(0, 1, 2, 3, 4), workers=8, wall_budget_seconds=120.0):
    release_dir = Path(release_dir).resolve()
    manifest_path = release_dir / split / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    tasks = [
        (
            str(manifest_path.parent / c["public"]),
            str(Path(output_root) / split / c["case_id"] / str(int(s))),
            int(s), float(wall_budget_seconds),
        )
        for s in seeds
        for c in manifest["cases"]
    ]
    if workers <= 1:
        results = [_worker(t) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=min(int(workers), 16)) as pool:
            results = list(pool.map(_worker, tasks))
    return {
        "release": str(release_dir),
        "split": split,
        "seeds": [int(s) for s in seeds],
        "runs": len(results),
        "fallbacks": sum(1 for r in results if r.get("fallback")),
        "median_seconds": float(np.median([r["wall_seconds"] for r in results])) if results else 0.0,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run M8 Random-Forest Stage-A on public split inputs.")
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument(
        "--split", required=True,
        choices=("calibration", "train", "train_s", "train_m", "validation", "test_id", "test_ood"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", default="0,1,2,3,4", help="Comma-separated integer seeds.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--wall-budget-seconds", type=float, default=120.0)
    args = parser.parse_args(argv)
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    summary = run_split(
        args.release, args.split, args.output, seeds=seeds,
        workers=args.workers, wall_budget_seconds=args.wall_budget_seconds,
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
