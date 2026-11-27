"""Shared fairness harness for published baselines (baselines-lit.md §2).

Provides common public-data extraction, grouped K-fold CV over drill holes,
the shared vertical anisotropy grid, native/share decision rules, 50% wall-time
deadline enforcement, legality/fallback handling, and repair-rate diagnostics.
"""

import time
import numpy as np
from scipy.spatial import cKDTree
from .contract import validate

SZ_GRID = (1, 2, 4, 8)
EVAL_CHUNK_SIZE = 8192
TIE_TOL = 1e-12
SHARE_CLIP = (0.02, 0.65)
REPAIR_FRACTION_THRESHOLD = 0.01


def _base():
    from . import baselines
    return baselines


def stretch_points(points, center, s_z):
    """Center coordinates at the domain center and stretch z by s_z.

    Centering makes reflections across x/y exact bit-for-bit negations while
    leaving all pairwise distances and linear polynomials unchanged.
    """
    pts = np.asarray(points, dtype=float)
    if len(pts) == 0:
        return np.zeros((0, 3), dtype=float)
    sc = np.array([1.0, 1.0, float(s_z)], dtype=float)
    return (pts - center) * sc


def _symmetric_subsample_mask(feasible):
    """Every third cell along each axis, centred symmetrically for D4 equivariance."""
    shape = feasible.shape
    slices = tuple(slice((1 - n) % 3, n, 3) if n >= 3 else slice(None) for n in shape)
    sub = np.zeros(shape, dtype=bool)
    sub[slices] = True
    sub &= feasible
    return sub if sub.any() else feasible


def extract_public_data(problem):
    """Extract common public-only quantities specified in baselines-lit.md §2.2."""
    validate(problem)
    b = _base()
    shape = tuple(int(n) for n in problem["shape"])
    origin = np.asarray(problem["origin"], dtype=float)
    spacing = np.asarray(problem["spacing"], dtype=float)
    E = np.asarray(shape, dtype=float) * spacing
    center = origin + 0.5 * E
    feasible = np.asarray(problem["domain_mask"], dtype=bool) & ~np.asarray(problem["forbidden_mask"], dtype=bool)
    xyz = b._grid_centers(problem)
    feasible_idx = np.argwhere(feasible)
    G = xyz[feasible]
    sub_mask = _symmetric_subsample_mask(feasible)
    S = xyz[sub_mask]

    drill = b._drill_observation(problem)
    d_pts = np.asarray(drill["points"], dtype=float)
    d_lab = np.asarray(drill["labels"], dtype=float)
    d_grp = np.asarray(drill["groups"])
    d_w = float(drill.get("weight", 1.0))
    eps = float(drill.get("label_error", 0.0))

    d_bin = d_lab != 0.5
    d_con = ~d_bin
    hole_ids = np.unique(d_grp) if len(d_grp) else np.zeros(0, dtype=np.int64)
    bin_hole_ids = np.unique(d_grp[d_bin]) if d_bin.any() else np.zeros(0, dtype=np.int64)
    n_h = int(len(hole_ids))
    h_hole = float(np.sqrt(E[0] * E[1] / max(1, n_h)))

    x_parts, y_parts, w_parts, is_drill_parts, grp_parts = [], [], [], [], []
    if d_bin.any():
        x_parts.append(d_pts[d_bin])
        y_parts.append((d_lab[d_bin] > 0.5).astype(float))
        w_parts.append(np.full(int(d_bin.sum()), d_w, dtype=float))
        is_drill_parts.append(np.ones(int(d_bin.sum()), dtype=bool))
        grp_parts.append(d_grp[d_bin].astype(np.int64, copy=False))
    for obs in problem["observations"]:
        if obs is drill:
            continue
        oy = np.asarray(obs["labels"], dtype=float)
        ob = oy != 0.5
        if ob.any():
            n_ob = int(ob.sum())
            x_parts.append(np.asarray(obs["points"], dtype=float)[ob])
            y_parts.append((oy[ob] > 0.5).astype(float))
            w_parts.append(np.full(n_ob, float(obs.get("weight", 1.0)), dtype=float))
            is_drill_parts.append(np.zeros(n_ob, dtype=bool))
            grp_parts.append(np.full(n_ob, -1, dtype=np.int64))

    if x_parts:
        X_b = np.concatenate(x_parts, axis=0)
        y_b = np.concatenate(y_parts, axis=0)
        w_b = np.concatenate(w_parts, axis=0)
        is_drill_b = np.concatenate(is_drill_parts, axis=0)
        grp_b = np.concatenate(grp_parts, axis=0)
    else:
        X_b = np.zeros((0, 3), dtype=float)
        y_b = np.zeros(0, dtype=float)
        w_b = np.zeros(0, dtype=float)
        is_drill_b = np.zeros(0, dtype=bool)
        grp_b = np.zeros(0, dtype=np.int64)

    X_c = d_pts[d_con] if d_con.any() else np.zeros((0, 3), dtype=float)
    w_c = np.full(len(X_c), d_w, dtype=float)
    grp_c = d_grp[d_con].astype(np.int64, copy=False) if d_con.any() else np.zeros(0, dtype=np.int64)

    return {
        "shape": shape,
        "origin": origin,
        "spacing": spacing,
        "E": E,
        "center": center,
        "hz": float(spacing[2]),
        "hbar": float(np.mean(spacing)),
        "feasible": feasible,
        "feasible_idx": feasible_idx,
        "xyz": xyz,
        "G": G,
        "M": int(len(G)),
        "S": S,
        "hole_ids": hole_ids,
        "bin_hole_ids": bin_hole_ids,
        "n_h": n_h,
        "h_hole": h_hole,
        "X_b": X_b,
        "y_b": y_b,
        "w_b": w_b,
        "is_drill_b": is_drill_b,
        "grp_b": grp_b,
        "X_c": X_c,
        "w_c": w_c,
        "grp_c": grp_c,
        "drill_weight": d_w,
        "eps": eps,
        "p_hat": float(b._drill_ore_share(problem)),
    }


def _filtered_problem(problem, keep_hole_ids):
    """Shallow copy of problem keeping only drill samples whose group is in keep_hole_ids."""
    drill = _base()._drill_observation(problem)
    keep = np.isin(np.asarray(drill["groups"]), keep_hole_ids)
    new_obs = []
    for obs in problem["observations"]:
        if obs is drill:
            new_obs.append({
                **obs,
                "points": np.asarray(obs["points"], dtype=float)[keep],
                "labels": np.asarray(obs["labels"], dtype=float)[keep],
                "groups": np.asarray(obs["groups"])[keep],
            })
        else:
            new_obs.append(obs)
    return {**problem, "observations": new_obs}


def make_hole_folds(data, problem):
    """Build grouped K-fold splits over sorted drill hole IDs (K = min(n_h, 6)).

    Non-drill modalities always stay in training; contact points follow their hole.
    Returns an empty list when CV is skipped (n_h < 2 or single class in binary drill labels).
    """
    is_drill_b = data["is_drill_b"]
    if not is_drill_b.any():
        return []
    drill_y = data["y_b"][is_drill_b] > 0.5
    bin_holes = data["bin_hole_ids"]
    if len(bin_holes) < 2 or not drill_y.any() or drill_y.all():
        return []

    hole_ids = data["hole_ids"]
    K = min(len(bin_holes), 6)
    # Assign binary-bearing holes round-robin by sorted rank; attach any contact-only hole similarly.
    hole_to_fold = {int(h): int(r % K) for r, h in enumerate(bin_holes)}
    for r, h in enumerate(hole_ids):
        if int(h) not in hole_to_fold:
            hole_to_fold[int(h)] = int(r % K)

    grp_b = data["grp_b"]
    grp_c = data["grp_c"]
    X_b, y_b, w_b, X_c = data["X_b"], data["y_b"], data["w_b"], data["X_c"]
    folds = []
    for k in range(K):
        val_holes = np.array([h for h in hole_ids if hole_to_fold[int(h)] == k], dtype=hole_ids.dtype)
        tr_holes = np.array([h for h in hole_ids if hole_to_fold[int(h)] != k], dtype=hole_ids.dtype)
        val_mask = is_drill_b & np.isin(grp_b, val_holes)
        if not val_mask.any():
            continue
        tr_mask = ~val_mask
        tr_c_mask = ~np.isin(grp_c, val_holes)
        tr_drill_mask = tr_mask & is_drill_b
        tr_bin_holes = np.unique(grp_b[tr_drill_mask])
        if len(tr_bin_holes):
            p_raw = float(np.mean([
                (y_b[tr_drill_mask & (grp_b == h)] > 0.5).mean() for h in tr_bin_holes
            ]))
        else:
            p_raw = SHARE_CLIP[0]
        p_train = float(np.clip(p_raw, SHARE_CLIP[0], SHARE_CLIP[1]))
        folds.append({
            "fold": k,
            "val_holes": val_holes,
            "tr_holes": tr_holes,
            "X_b_tr": X_b[tr_mask],
            "y_b_tr": y_b[tr_mask],
            "w_b_tr": w_b[tr_mask],
            "X_c_tr": X_c[tr_c_mask],
            "X_val": X_b[val_mask],
            "y_val": y_b[val_mask] > 0.5,
            "p_train": p_train,
            "problem_tr": _filtered_problem(problem, tr_holes),
        })
    return folds


def pooled_jaccard(tp, fp, fn):
    return float(tp) / float(tp + fp + fn) if tp > 0 else 0.0


def predict_chunked(predict, points, chunk_size=EVAL_CHUNK_SIZE):
    """Evaluate predict(points) in fixed chunks, enforcing 1D finite float output."""
    pts = np.asarray(points, dtype=float)
    n = len(pts)
    if n == 0:
        return np.zeros(0, dtype=float)
    if n <= chunk_size:
        out = np.asarray(predict(pts), dtype=float).reshape(-1)
        if len(out) != n or not np.isfinite(out).all():
            raise ValueError("non-finite or malformed predictor output")
        return out
    parts = []
    for start in range(0, n, chunk_size):
        chunk = pts[start:start + chunk_size]
        part = np.asarray(predict(chunk), dtype=float).reshape(-1)
        if len(part) != len(chunk) or not np.isfinite(part).all():
            raise ValueError("non-finite or malformed predictor output")
        parts.append(part)
    return np.concatenate(parts, axis=0)


def evaluate_config_cv(config, fit, folds, S, tau0, rules=("native", "share"), deadline=None):
    """Run grouped K-fold CV for one configuration; returns dict rule -> pooled Jaccard or None."""
    counts = {r: [0, 0, 0] for r in rules}
    need_S = "share" in rules
    for fold in folds:
        if deadline is not None and time.monotonic() >= deadline:
            return None
        try:
            pred_fn = fit(
                config,
                fold["X_b_tr"],
                fold["y_b_tr"],
                fold["w_b_tr"],
                fold["X_c_tr"],
                fold["problem_tr"],
            )
            f_val = predict_chunked(pred_fn, fold["X_val"])
            f_S = predict_chunked(pred_fn, S) if need_S else None
        except Exception:
            return None
        y_true = fold["y_val"]
        for r in rules:
            if r == "native":
                pred = f_val > tau0
            else:
                q = float(np.quantile(f_S, 1.0 - fold["p_train"]))
                if np.any(f_S > q) or float(f_S.max()) <= float(f_S.min()):
                    pred = f_val > q
                else:
                    pred = f_val >= q
            counts[r][0] += int(np.count_nonzero(pred & y_true))
            counts[r][1] += int(np.count_nonzero(pred & ~y_true))
            counts[r][2] += int(np.count_nonzero(~pred & y_true))
    return {r: pooled_jaccard(*counts[r]) for r in rules}


def apply_decision_rule(f, tau0, rule, p_hat, problem, feasible, feasible_idx):
    """Convert real field f on feasible cells G into a 3D boolean mask."""
    shape = feasible.shape
    mask = np.zeros(shape, dtype=bool)
    M = len(f)
    if M == 0:
        return mask
    if rule == "native":
        mask[feasible] = f > tau0
        return mask
    if rule == "share":
        lo, hi = _base()._volume_cell_bounds(problem, feasible)
        if hi <= 0:
            return mask
        lo_eff = min(max(1, lo), hi)
        target = min(max(int(round(p_hat * M)), lo_eff), hi)
        order = np.argsort(-f, kind="mergesort")
        chosen = feasible_idx[order[:target]]
        mask[tuple(chosen.T)] = True
        return mask
    raise ValueError(f"unknown decision rule: {rule!r}")


def no_ore_fallback(problem, data, diagnostics=None, started=None):
    """Common fallback when no binary sample is ore (§2.5)."""
    shape = data["shape"]
    feasible = data["feasible"]
    rank = -np.linalg.norm(data["xyz"] - data["center"], axis=-1)
    raw_mask = np.zeros(shape, dtype=bool)
    field = _base()._enforce_legal(problem, raw_mask, rank)
    final_mask = field > 0
    changed = float(np.count_nonzero((final_mask != raw_mask) & feasible)) / max(1, data["M"])
    if diagnostics is not None:
        diagnostics.update({
            "selected_rule": "none",
            "selected_config": None,
            "selected_s_z": None,
            "cv_score": None,
            "cv_evaluated": 0,
            "deadline_stopped": False,
            "fallback": False,
            "no_ore": True,
            "repaired": bool(changed > REPAIR_FRACTION_THRESHOLD),
            "repair_changed_fraction": changed,
            "native_changed_fraction": changed,
            "wall_seconds": float(time.monotonic() - started) if started is not None else 0.0,
        })
    return field


def finalize_field(problem, data, f, tau0, rule, *, selected_config=None, cv_score=None,
                   cv_evaluated=0, deadline_stopped=False, diagnostics=None, started=None):
    """Apply decision rule and _enforce_legal, recording repair-rate diagnostics."""
    shape = data["shape"]
    feasible = data["feasible"]
    f = np.asarray(f, dtype=float).reshape(-1)
    if len(f) != data["M"] or not np.isfinite(f).all():
        raise ValueError("non-finite field on feasible cells")
    mask_selected = apply_decision_rule(
        f, tau0, rule, data["p_hat"], problem, feasible, data["feasible_idx"]
    )
    mask_native = np.zeros(shape, dtype=bool)
    mask_native[feasible] = f > tau0
    rank = np.full(shape, -np.inf, dtype=float)
    rank[feasible] = f
    field = _base()._enforce_legal(problem, mask_selected, rank)
    final_mask = field > 0
    M = max(1, data["M"])
    changed_sel = float(np.count_nonzero((final_mask != mask_selected) & feasible)) / M
    changed_nat = float(np.count_nonzero((final_mask != mask_native) & feasible)) / M
    if diagnostics is not None:
        s_z = selected_config.get("s_z") if isinstance(selected_config, dict) else None
        diagnostics.update({
            "selected_rule": rule,
            "selected_config": selected_config,
            "selected_s_z": s_z,
            "cv_score": cv_score,
            "cv_evaluated": int(cv_evaluated),
            "deadline_stopped": bool(deadline_stopped),
            "fallback": False,
            "no_ore": False,
            "repaired": bool(changed_sel > REPAIR_FRACTION_THRESHOLD),
            "repair_changed_fraction": changed_sel,
            "native_changed_fraction": changed_nat,
            "wall_seconds": float(time.monotonic() - started) if started is not None else 0.0,
        })
    return field


def stretched_fallback(problem, data, diagnostics=None, started=None, *, cv_evaluated=0,
                       deadline_stopped=False, default_mode=False):
    """Numerical-failure fallback to stretched_nearest_class (§2.5)."""
    b = _base()
    field = b.stretched_nearest_class_default(problem) if default_mode else b.stretched_nearest_class(problem)
    if diagnostics is not None:
        diagnostics.update({
            "selected_rule": "fallback",
            "selected_config": None,
            "selected_s_z": None,
            "cv_score": None,
            "cv_evaluated": int(cv_evaluated),
            "deadline_stopped": bool(deadline_stopped),
            "fallback": True,
            "no_ore": False,
            "repaired": False,
            "repair_changed_fraction": 0.0,
            "native_changed_fraction": 0.0,
            "wall_seconds": float(time.monotonic() - started) if started is not None else 0.0,
        })
    return field


def solve_default(problem, budget, default_config, fit, tau0=0.0, *,
                  diagnostics=None, eval_chunk_size=EVAL_CHUNK_SIZE, fallback_configs=()):
    """Zero-CV, zero-grid-search solver: fixed default config + native threshold + _enforce_legal."""
    del budget
    started = time.monotonic()
    data = extract_public_data(problem)
    if not (data["y_b"] > 0.5).any():
        return no_ore_fallback(problem, data, diagnostics=diagnostics, started=started)

    for cfg in (default_config, *fallback_configs):
        try:
            pred_fn = fit(cfg, data["X_b"], data["y_b"], data["w_b"], data["X_c"], problem)
            f = predict_chunked(pred_fn, data["G"], chunk_size=eval_chunk_size)
            return finalize_field(
                problem,
                data,
                f,
                tau0,
                "native",
                selected_config=cfg,
                cv_score=None,
                cv_evaluated=0,
                deadline_stopped=False,
                diagnostics=diagnostics,
                started=started,
            )
        except Exception:
            continue

    return stretched_fallback(
        problem, data, diagnostics=diagnostics, started=started,
        cv_evaluated=0, deadline_stopped=False, default_mode=True,
    )


def _wall_deadline(budget, started):
    seconds = 120.0 if (budget is None or "wall_time_seconds" not in budget) else float(budget["wall_time_seconds"])
    if not np.isfinite(seconds) or seconds <= 0:
        seconds = 120.0
    return started + 0.5 * seconds


def _allowed_rules(decision_rule):
    if decision_rule is None or decision_rule == "cv":
        return ("native", "share")
    if decision_rule in ("native", "share"):
        return (decision_rule,)
    raise ValueError(f"unsupported decision_rule: {decision_rule!r}")


def tune_and_solve(problem, budget, configs, fit, tau0=0.0, *, decision_rule=None,
                   diagnostics=None, eval_chunk_size=EVAL_CHUNK_SIZE):
    """Common CV tuning + full refit + legalisation skeleton (baselines-lit.md §2.4-2.5, §3)."""
    started = time.monotonic()
    data = extract_public_data(problem)
    if not (data["y_b"] > 0.5).any():
        return no_ore_fallback(problem, data, diagnostics=diagnostics, started=started)

    rules = _allowed_rules(decision_rule)
    deadline = _wall_deadline(budget, started)
    folds = make_hole_folds(data, problem)

    best_idx = None
    best_rule = rules[0] if len(rules) == 1 else "share"
    best_score = -1.0
    cv_evaluated = 0
    deadline_stopped = False

    if folds:
        for idx, cfg in enumerate(configs):
            if time.monotonic() >= deadline:
                deadline_stopped = True
                break
            scores = evaluate_config_cv(cfg, fit, folds, data["S"], tau0, rules=rules, deadline=deadline)
            if scores is None:
                if time.monotonic() >= deadline:
                    deadline_stopped = True
                    break
                continue
            cv_evaluated += 1
            for r in rules:
                sc = scores[r]
                if sc > best_score + TIE_TOL:
                    best_score = sc
                    best_idx = idx
                    best_rule = r

    if best_idx is None:
        # No CV run (n_h < 2 / single class / deadline hit / all CV folds failed): try default then rest.
        order = list(range(len(configs)))
        best_score = None
        best_rule = rules[0] if len(rules) == 1 else "share"
    else:
        order = [best_idx] + [i for i in range(len(configs)) if i != best_idx]

    for idx in order:
        cfg = configs[idx]
        try:
            pred_fn = fit(cfg, data["X_b"], data["y_b"], data["w_b"], data["X_c"], problem)
            f = predict_chunked(pred_fn, data["G"], chunk_size=eval_chunk_size)
            return finalize_field(
                problem,
                data,
                f,
                tau0,
                best_rule,
                selected_config=cfg,
                cv_score=best_score if idx == best_idx else None,
                cv_evaluated=cv_evaluated,
                deadline_stopped=deadline_stopped,
                diagnostics=diagnostics,
                started=started,
            )
        except Exception:
            continue

    return stretched_fallback(
        problem, data, diagnostics=diagnostics, started=started,
        cv_evaluated=cv_evaluated, deadline_stopped=deadline_stopped,
    )


def denoise_binary_observations(problem):
    """Return (X_b, y_b_clean, w_b) with _denoise_hole_labels applied per drill hole when eps > 0."""
    b = _base()
    drill = b._drill_observation(problem)
    d_pts = np.asarray(drill["points"], dtype=float)
    d_lab = np.asarray(drill["labels"], dtype=float)
    d_grp = np.asarray(drill["groups"])
    d_w = float(drill.get("weight", 1.0))
    eps = float(drill.get("label_error", 0.0))

    d_bin = d_lab != 0.5
    x_parts, y_parts, w_parts = [], [], []
    if d_bin.any():
        bp = d_pts[d_bin]
        by = (d_lab[d_bin] > 0.5).copy()
        bg = d_grp[d_bin]
        if eps > 0.0:
            for gid in np.unique(bg):
                sel = bg == gid
                by[sel] = b._denoise_hole_labels(bp[sel, 2], by[sel])
        x_parts.append(bp)
        y_parts.append(by.astype(float))
        w_parts.append(np.full(len(bp), d_w, dtype=float))

    for obs in problem["observations"]:
        if obs is drill:
            continue
        oy = np.asarray(obs["labels"], dtype=float)
        ob = oy != 0.5
        if ob.any():
            n_ob = int(ob.sum())
            x_parts.append(np.asarray(obs["points"], dtype=float)[ob])
            y_parts.append((oy[ob] > 0.5).astype(float))
            w_parts.append(np.full(n_ob, float(obs.get("weight", 1.0)), dtype=float))

    if not x_parts:
        return np.zeros((0, 3), dtype=float), np.zeros(0, dtype=float), np.zeros(0, dtype=float)
    return np.concatenate(x_parts, axis=0), np.concatenate(y_parts, axis=0), np.concatenate(w_parts, axis=0)


def compute_sdf_targets(problem, s_z, *, center=None, h_hole=None):
    """Build signed-distance targets in stretched coordinates (baselines-lit.md §3 M1 / M5).

    Returns (X_stretched, values, weights) including binary samples and zero-valued contacts.
    Raises ValueError if cleaned binary labels have fewer than two classes.
    """
    b = _base()
    shape = np.asarray(problem["shape"], dtype=float)
    origin = np.asarray(problem["origin"], dtype=float)
    spacing = np.asarray(problem["spacing"], dtype=float)
    E = shape * spacing
    if center is None:
        center = origin + 0.5 * E
    drill = b._drill_observation(problem)
    d_pts = np.asarray(drill["points"], dtype=float)
    d_lab = np.asarray(drill["labels"], dtype=float)
    d_grp = np.asarray(drill["groups"])
    d_w = float(drill.get("weight", 1.0))
    if h_hole is None:
        n_h = len(np.unique(d_grp)) if len(d_grp) else 1
        h_hole = float(np.sqrt(E[0] * E[1] / max(1, n_h)))

    X_b, y_b, w_b = denoise_binary_observations(problem)
    ore = y_b > 0.5
    bar = ~ore
    if not ore.any() or not bar.any():
        raise ValueError("cleaned binary labels have only one class")

    X_c = d_pts[d_lab == 0.5]
    w_c = np.full(len(X_c), d_w, dtype=float)
    Xb_s = stretch_points(X_b, center, s_z)
    Xc_s = stretch_points(X_c, center, s_z)

    d_ore_opp = cKDTree(Xb_s[bar]).query(Xb_s[ore])[0]
    d_bar_opp = cKDTree(Xb_s[ore]).query(Xb_s[bar])[0]
    if len(Xc_s):
        tree_c = cKDTree(Xc_s)
        d_ore = np.minimum(d_ore_opp, tree_c.query(Xb_s[ore])[0])
        d_bar = np.minimum(d_bar_opp, tree_c.query(Xb_s[bar])[0])
    else:
        d_ore, d_bar = d_ore_opp, d_bar_opp

    cap = 2.0 * h_hole
    v_b = np.empty(len(Xb_s), dtype=float)
    v_b[ore] = np.minimum(d_ore, cap)
    v_b[bar] = -np.minimum(d_bar, cap)

    if len(Xc_s):
        X_all = np.concatenate([Xb_s, Xc_s], axis=0)
        v_all = np.concatenate([v_b, np.zeros(len(Xc_s), dtype=float)], axis=0)
        w_all = np.concatenate([w_b, w_c], axis=0)
    else:
        X_all, v_all, w_all = Xb_s, v_b, w_b
    return X_all, v_all, w_all


def merge_close_points(X, values, weights, tol):
    """Merge points closer than tol by connected components, keeping mean coords/values/weights."""
    n = len(X)
    if n <= 1 or tol <= 0.0:
        return X, values, weights
    pairs = cKDTree(X).query_pairs(r=float(tol))
    if not pairs:
        return X, values, weights
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in sorted(pairs):
        ri, rj = find(i), find(j)
        if ri != rj:
            if ri < rj:
                parent[rj] = ri
            else:
                parent[ri] = rj

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    if len(groups) == n:
        return X, values, weights
    out_X, out_v, out_w = [], [], []
    for root in sorted(groups):
        idx = groups[root]
        if len(idx) == 1:
            k = idx[0]
            out_X.append(X[k])
            out_v.append(values[k])
            out_w.append(weights[k])
        else:
            out_X.append(X[idx].mean(axis=0))
            out_v.append(float(np.mean(values[idx])))
            out_w.append(float(np.mean(weights[idx])))
    return np.asarray(out_X, dtype=float), np.asarray(out_v, dtype=float), np.asarray(out_w, dtype=float)
