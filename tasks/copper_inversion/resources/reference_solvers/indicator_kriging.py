"""Self-contained dual ordinary indicator kriging solver (indicator_kriging)."""

import heapq
import time
import numpy as np
from scipy import ndimage
from scipy.linalg import solve as linalg_solve
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from tasks.copper_inversion.v2.contract import validate
from tasks.copper_inversion.v2.geometry import canonicalize, components

EVAL_CHUNK_SIZE = 8192
TIE_TOL = 1e-12
SHARE_CLIP = (0.02, 0.65)
Z_STRETCH_GRID = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0)


def _ordered_configs(default, rest):
    out = [default]
    for cfg in rest:
        if cfg != default:
            out.append(cfg)
    return tuple(out)


INDICATOR_KRIGING_CONFIGS = _ordered_configs(
    {"s_z": 4, "rho": "exponential", "nu": 0.1},
    [
        {"s_z": s_z, "rho": rho, "nu": nu}
        for nu in (0.3, 0.1, 0.02)
        for rho in ("exponential", "spherical")
        for s_z in (4, 2, 8, 1)
    ],
)


def _grid_centers(problem):
    return np.stack(
        np.meshgrid(
            *[
                problem["origin"][i] + (np.arange(n) + 0.5) * problem["spacing"][i]
                for i, n in enumerate(problem["shape"])
            ],
            indexing="ij",
        ),
        axis=-1,
    )


def _drill_observation(problem):
    for obs in problem["observations"]:
        if obs["name"] == "drill":
            return obs
    return problem["observations"][0]


def _binary_observations(problem):
    pts, labs = [], []
    for obs in problem["observations"]:
        y = np.asarray(obs["labels"], dtype=float)
        b = y != 0.5
        if b.any():
            pts.append(np.asarray(obs["points"], dtype=float)[b])
            labs.append(y[b] > 0.5)
    if not pts:
        return np.zeros((0, 3), dtype=float), np.zeros(0, dtype=bool)
    return np.concatenate(pts), np.concatenate(labs)


def _drill_ore_share(problem):
    obs = _drill_observation(problem)
    y = np.asarray(obs["labels"], dtype=float)
    g = np.asarray(obs["groups"])
    b = y != 0.5
    if not b.any():
        return 0.0
    yb, gb = y[b] > 0.5, g[b]
    return float(np.mean([yb[gb == gid].mean() for gid in np.unique(gb)]))


def _volume_cell_bounds(problem, feasible):
    cell = float(np.prod(problem["spacing"]))
    bounds = problem["priors"]["hard_volume_bounds"]
    n_feas = int(feasible.sum())
    lo = 0 if bounds is None else int(np.ceil(bounds[0] / cell - 1e-9))
    hi = n_feas if bounds is None else min(n_feas, int(np.floor(bounds[1] / cell + 1e-9)))
    return lo, hi


def constrained_initial(problem, mask, rank):
    field = np.where(mask, 1.0, -1.0)
    try:
        canonicalize(problem, field)
        return field
    except ValueError:
        pass
    feasible = problem["domain_mask"] & ~problem["forbidden_mask"]
    labels, count = ndimage.label(feasible)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    cell = float(np.prod(problem["spacing"]))
    bounds = problem["priors"]["hard_volume_bounds"]
    lo = 0 if bounds is None else int(np.ceil(bounds[0] / cell - 1e-9))
    hi = int(feasible.sum()) if bounds is None else int(np.floor(bounds[1] / cell + 1e-9))
    maximum = problem["priors"]["max_components"]
    regions = sorted(range(1, count + 1), key=lambda i: (-sizes[i], i))
    if maximum is not None:
        regions = regions[:maximum]
    capacity = int(sum(sizes[i] for i in regions))
    target = min(max(int(mask.sum()), lo), hi, capacity)
    if target < lo:
        raise ValueError("public domain cannot satisfy initial volume/connectivity bounds")
    result = np.zeros(feasible.shape, bool)
    visited = np.zeros(feasible.shape, bool)
    filled = 0
    for region in regions:
        if filled >= target:
            break
        indices = np.argwhere(labels == region)
        start = tuple(indices[np.argmax(rank[labels == region])])
        heap = [(-float(rank[start]), start)]
        visited[start] = True
        while heap and filled < target:
            _, position = heapq.heappop(heap)
            result[position] = True
            filled += 1
            for axis in range(3):
                for direction in (-1, 1):
                    q = list(position)
                    q[axis] += direction
                    q = tuple(q)
                    if not 0 <= q[axis] < feasible.shape[axis] or visited[q] or labels[q] != region:
                        continue
                    visited[q] = True
                    heapq.heappush(heap, (-float(rank[q]), q))
    field = np.where(result, 1.0, -1.0)
    canonicalize(problem, field)
    return field


def _enforce_legal(problem, mask, rank):
    feasible = problem["domain_mask"] & ~problem["forbidden_mask"]
    mask = np.asarray(mask, dtype=bool) & feasible
    rank = np.asarray(rank, dtype=float)
    lo, hi = _volume_cell_bounds(problem, feasible)
    max_comp = problem["priors"]["max_components"]
    n_mask = int(mask.sum())
    if n_mask < lo or n_mask > hi:
        target = min(max(n_mask, lo), hi)
        if target == 0:
            mask = np.zeros_like(feasible)
        else:
            coords = np.argwhere(feasible)
            scores = rank[feasible]
            order = np.argsort(-scores, kind="mergesort")
            mask = np.zeros_like(feasible)
            chosen = coords[order[:target]]
            mask[tuple(chosen.T)] = True
    if max_comp is not None and mask.any():
        struct = ndimage.generate_binary_structure(3, 1)
        labels, count = ndimage.label(mask, structure=struct)
        if count > max_comp:
            sizes = np.bincount(labels.ravel())
            keep = sorted(
                range(1, count + 1),
                key=lambda i: (-int(sizes[i]), -float(rank[labels == i].max()), i),
            )[:max_comp]
            mask = np.isin(labels, keep)
            if int(mask.sum()) < lo:
                target = lo
                filled = int(mask.sum())
                visited = mask.copy()
                heap = []
                for pos in np.argwhere(mask):
                    p = tuple(pos)
                    for axis in range(3):
                        for d in (-1, 1):
                            q = list(p)
                            q[axis] += d
                            q = tuple(q)
                            if 0 <= q[axis] < feasible.shape[axis] and feasible[q] and not visited[q]:
                                visited[q] = True
                                heapq.heappush(heap, (-float(rank[q]), q))
                while heap and filled < target:
                    _, q = heapq.heappop(heap)
                    mask[q] = True
                    filled += 1
                    for axis in range(3):
                        for d in (-1, 1):
                            r = list(q)
                            r[axis] += d
                            r = tuple(r)
                            if 0 <= r[axis] < feasible.shape[axis] and feasible[r] and not visited[r]:
                                visited[r] = True
                                heapq.heappush(heap, (-float(rank[r]), r))
                if components(mask, problem["spacing"])[0] > max_comp:
                    mask = np.zeros_like(feasible)
    return constrained_initial(problem, mask, rank)


def _select_z_stretch_cv(problem):
    drill = _drill_observation(problem)
    y = np.asarray(drill["labels"], dtype=float)
    b = y != 0.5
    if not b.any():
        return 2.0
    d_pts = np.asarray(drill["points"], dtype=float)[b]
    d_lab = y[b] > 0.5
    d_grp = np.asarray(drill["groups"])[b]
    groups = np.unique(d_grp)
    if len(groups) < 2 or not d_lab.any() or d_lab.all():
        return 2.0
    extra_pts, extra_lab = [], []
    for obs in problem["observations"]:
        if obs is drill:
            continue
        oy = np.asarray(obs["labels"], dtype=float)
        ob = oy != 0.5
        if ob.any():
            extra_pts.append(np.asarray(obs["points"], dtype=float)[ob])
            extra_lab.append(oy[ob] > 0.5)
    e_pts = np.concatenate(extra_pts) if extra_pts else np.zeros((0, 3), dtype=float)
    e_lab = np.concatenate(extra_lab) if extra_lab else np.zeros(0, dtype=bool)
    best_zs, best_key = 2.0, (-1.0, -np.inf)
    for zs in Z_STRETCH_GRID:
        sc = np.array([1.0, 1.0, float(zs)])
        sd = d_pts * sc
        se = e_pts * sc
        tp = fp = fn = 0
        margins = []
        for g in groups:
            hold = d_grp == g
            tr_pts = np.concatenate([sd[~hold], se]) if len(se) else sd[~hold]
            tr_lab = np.concatenate([d_lab[~hold], e_lab]) if len(se) else d_lab[~hold]
            y_true = d_lab[hold]
            if tr_lab.any() and (~tr_lab).any():
                dp = cKDTree(tr_pts[tr_lab]).query(sd[hold])[0]
                dn = cKDTree(tr_pts[~tr_lab]).query(sd[hold])[0]
                pred = dp < dn
                contrast = (dn - dp) / (dn + dp + 1e-9)
                margins.append(np.where(y_true, contrast, -contrast))
            elif tr_lab.all():
                pred = np.ones(len(y_true), dtype=bool)
            else:
                pred = np.zeros(len(y_true), dtype=bool)
            tp += int(np.count_nonzero(pred & y_true))
            fp += int(np.count_nonzero(pred & ~y_true))
            fn += int(np.count_nonzero(~pred & y_true))
        if tp == 0:
            continue
        jaccard = tp / float(tp + fp + fn)
        margin = float(np.mean(np.concatenate(margins))) if margins else -np.inf
        key = (round(jaccard, 12), margin)
        if key > best_key:
            best_key, best_zs = key, float(zs)
    return best_zs


def stretched_nearest_class(problem):
    validate(problem)
    xyz = _grid_centers(problem)
    feasible = problem["domain_mask"] & ~problem["forbidden_mask"]
    pts, lab = _binary_observations(problem)
    if not lab.any():
        center = np.asarray(problem["origin"]) + 0.5 * np.asarray(problem["shape"]) * problem["spacing"]
        rank = -np.linalg.norm(xyz - center, axis=-1)
        return _enforce_legal(problem, np.zeros(tuple(problem["shape"]), bool), rank)
    zs = _select_z_stretch_cv(problem)
    sc = np.array([1.0, 1.0, zs])
    f_idx = np.argwhere(feasible)
    G = xyz[feasible] * sc
    X = pts * sc
    dp = cKDTree(X[lab]).query(G)[0]
    if (~lab).any():
        dn = cKDTree(X[~lab]).query(G)[0]
        score_f = (dn - dp) / (dn + dp + 1e-9)
    else:
        score_f = -dp
    rank = np.full(tuple(problem["shape"]), -np.inf, dtype=float)
    rank[feasible] = score_f
    lo, hi = _volume_cell_bounds(problem, feasible)
    share = _drill_ore_share(problem)
    target = int(round(share * len(G)))
    if target <= 0:
        mask = (rank > 0) & feasible if (~lab).any() else np.zeros(tuple(problem["shape"]), bool)
    else:
        target = min(max(target, max(1, lo)), hi)
        order = np.argsort(-score_f, kind="mergesort")
        mask = np.zeros(tuple(problem["shape"]), bool)
        mask[tuple(f_idx[order[:target]].T)] = True
    return _enforce_legal(problem, mask, rank)


def stretch_points(points, center, s_z):
    pts = np.asarray(points, dtype=float)
    if len(pts) == 0:
        return np.zeros((0, 3), dtype=float)
    sc = np.array([1.0, 1.0, float(s_z)], dtype=float)
    return (pts - center) * sc


def _symmetric_subsample_mask(feasible):
    shape = feasible.shape
    slices = tuple(slice((1 - n) % 3, n, 3) if n >= 3 else slice(None) for n in shape)
    sub = np.zeros(shape, dtype=bool)
    sub[slices] = True
    sub &= feasible
    return sub if sub.any() else feasible


def extract_public_data(problem):
    validate(problem)
    shape = tuple(int(n) for n in problem["shape"])
    origin = np.asarray(problem["origin"], dtype=float)
    spacing = np.asarray(problem["spacing"], dtype=float)
    E = np.asarray(shape, dtype=float) * spacing
    center = origin + 0.5 * E
    feasible = np.asarray(problem["domain_mask"], dtype=bool) & ~np.asarray(problem["forbidden_mask"], dtype=bool)
    xyz = _grid_centers(problem)
    feasible_idx = np.argwhere(feasible)
    G = xyz[feasible]
    sub_mask = _symmetric_subsample_mask(feasible)
    S = xyz[sub_mask]

    drill = _drill_observation(problem)
    d_pts = np.asarray(drill["points"], dtype=float)
    d_lab = np.asarray(drill["labels"], dtype=float)
    d_grp = np.asarray(drill["groups"])
    d_w = float(drill.get("weight", 1.0))
    eps = float(drill.get("label_error", 0.0))

    d_bin = d_lab != 0.5
    d_con = ~d_bin
    hole_ids = np.unique(d_grp) if len(d_grp) else np.zeros(0, dtype=np.int64)
    bin_hole_ids = np.unique(d_grp[d_bin]) if d_bin.any() else np.zeros(0, dtype=np.int64)

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
    grp_c = d_grp[d_con].astype(np.int64, copy=False) if d_con.any() else np.zeros(0, dtype=np.int64)

    return {
        "shape": shape,
        "E": E,
        "center": center,
        "hz": float(spacing[2]),
        "feasible": feasible,
        "feasible_idx": feasible_idx,
        "xyz": xyz,
        "G": G,
        "M": int(len(G)),
        "S": S,
        "hole_ids": hole_ids,
        "bin_hole_ids": bin_hole_ids,
        "X_b": X_b,
        "y_b": y_b,
        "w_b": w_b,
        "is_drill_b": is_drill_b,
        "grp_b": grp_b,
        "X_c": X_c,
        "grp_c": grp_c,
        "drill_weight": d_w,
        "eps": eps,
        "p_hat": float(_drill_ore_share(problem)),
    }


def _filtered_problem(problem, keep_hole_ids):
    drill = _drill_observation(problem)
    keep = np.isin(np.asarray(drill["groups"]), keep_hole_ids)
    new_obs = []
    for obs in problem["observations"]:
        if obs is drill:
            new_obs.append(
                {
                    **obs,
                    "points": np.asarray(obs["points"], dtype=float)[keep],
                    "labels": np.asarray(obs["labels"], dtype=float)[keep],
                    "groups": np.asarray(obs["groups"])[keep],
                }
            )
        else:
            new_obs.append(obs)
    return {**problem, "observations": new_obs}


def make_hole_folds(data, problem):
    is_drill_b = data["is_drill_b"]
    if not is_drill_b.any():
        return []
    drill_y = data["y_b"][is_drill_b] > 0.5
    bin_holes = data["bin_hole_ids"]
    if len(bin_holes) < 2 or not drill_y.any() or drill_y.all():
        return []

    hole_ids = data["hole_ids"]
    K = min(len(bin_holes), 6)
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
            p_raw = float(np.mean([(y_b[tr_drill_mask & (grp_b == h)] > 0.5).mean() for h in tr_bin_holes]))
        else:
            p_raw = SHARE_CLIP[0]
        p_train = float(np.clip(p_raw, SHARE_CLIP[0], SHARE_CLIP[1]))
        folds.append(
            {
                "X_b_tr": X_b[tr_mask],
                "y_b_tr": y_b[tr_mask],
                "w_b_tr": w_b[tr_mask],
                "X_c_tr": X_c[tr_c_mask],
                "X_val": X_b[val_mask],
                "y_val": y_b[val_mask] > 0.5,
                "p_train": p_train,
                "problem_tr": _filtered_problem(problem, tr_holes),
            }
        )
    return folds


def pooled_jaccard(tp, fp, fn):
    return float(tp) / float(tp + fp + fn) if tp > 0 else 0.0


def predict_chunked(predict, points, chunk_size=EVAL_CHUNK_SIZE):
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
    shape = feasible.shape
    mask = np.zeros(shape, dtype=bool)
    M = len(f)
    if M == 0:
        return mask
    if rule == "native":
        mask[feasible] = f > tau0
        return mask
    if rule == "share":
        lo, hi = _volume_cell_bounds(problem, feasible)
        if hi <= 0:
            return mask
        lo_eff = min(max(1, lo), hi)
        target = min(max(int(round(p_hat * M)), lo_eff), hi)
        order = np.argsort(-f, kind="mergesort")
        chosen = feasible_idx[order[:target]]
        mask[tuple(chosen.T)] = True
        return mask
    raise ValueError(f"unknown decision rule: {rule!r}")


def _kriging_corr(u, rho):
    u = np.maximum(np.asarray(u, dtype=float), 0.0)
    if rho == "exponential":
        return np.exp(-3.0 * u)
    if rho == "spherical":
        return np.where(u < 1.0, 1.0 - 1.5 * u + 0.5 * (u ** 3), 0.0)
    raise ValueError(f"unknown variogram model: {rho!r}")


def empirical_downhole_variogram(problem):
    drill = _drill_observation(problem)
    pts = np.asarray(drill["points"], dtype=float)
    lab = np.asarray(drill["labels"], dtype=float)
    grp = np.asarray(drill["groups"])
    b = lab != 0.5
    if not b.any():
        return np.zeros(0), np.zeros(0), np.zeros(0, dtype=int)
    bp, by, bg = pts[b], (lab[b] > 0.5).astype(float), grp[b]

    consec_dists = []
    hole_lens = []
    pair_dists = []
    pair_semis = []
    for gid in np.unique(bg):
        sel = bg == gid
        hp, hy = bp[sel], by[sel]
        if len(hp) < 2:
            continue
        order = np.argsort(hp[:, 2])
        hp, hy = hp[order], hy[order]
        diffs = np.linalg.norm(np.diff(hp, axis=0), axis=1)
        pos_diffs = diffs[diffs > 0.0]
        if len(pos_diffs):
            consec_dists.append(pos_diffs)
        hlen = float(np.linalg.norm(hp[-1] - hp[0]))
        if hlen > 0.0:
            hole_lens.append(hlen)
        iu, ju = np.triu_indices(len(hp), k=1)
        d = np.linalg.norm(hp[iu] - hp[ju], axis=1)
        s = 0.5 * ((hy[iu] - hy[ju]) ** 2)
        pair_dists.append(d)
        pair_semis.append(s)

    if not consec_dists or not hole_lens or not pair_dists:
        return np.zeros(0), np.zeros(0), np.zeros(0, dtype=int)
    delta = float(np.median(np.concatenate(consec_dists)))
    half_len = 0.5 * float(np.median(hole_lens))
    if delta <= 0.0 or half_len < delta:
        return np.zeros(0), np.zeros(0), np.zeros(0, dtype=int)

    max_k = int(np.floor(half_len / delta))
    if max_k < 1:
        return np.zeros(0), np.zeros(0), np.zeros(0, dtype=int)

    all_d = np.concatenate(pair_dists)
    all_s = np.concatenate(pair_semis)
    k_idx = np.rint(all_d / delta).astype(np.int64)
    valid = (k_idx >= 1) & (k_idx <= max_k) & (np.abs(all_d - k_idx * delta) <= 0.5 * delta)
    if not valid.any():
        return np.zeros(0), np.zeros(0), np.zeros(0, dtype=int)
    k_idx = k_idx[valid]
    all_s = all_s[valid]

    counts = np.bincount(k_idx, minlength=max_k + 1)[1:]
    sums = np.bincount(k_idx, weights=all_s, minlength=max_k + 1)[1:]
    keep = counts >= 30
    if int(keep.sum()) < 3:
        return np.zeros(0), np.zeros(0), np.zeros(0, dtype=int)
    ks = np.arange(1, max_k + 1, dtype=float)[keep]
    return ks * delta, sums[keep] / counts[keep], counts[keep]


def fit_vertical_range(h_lags, gamma_hat, n_pairs, rho, sigma2, c0, hz, E_z):
    if len(h_lags) < 3 or sigma2 <= 1e-12:
        return 0.25 * E_z
    lo_av = max(2.0 * hz, 1e-9)
    hi_av = max(E_z, lo_av * 1.001)
    av_grid = np.geomspace(lo_av, hi_av, 30)
    weights = n_pairs.astype(float) / (h_lags ** 2)
    sill = sigma2 - c0
    u = h_lags[:, None] / av_grid[None, :]
    gamma_m = c0 + sill * (1.0 - _kriging_corr(u, rho))
    sse = np.sum(weights[:, None] * ((gamma_hat[:, None] - gamma_m) ** 2), axis=0)
    return float(av_grid[int(np.argmin(sse))])


def fit_indicator_kriging(config, X_b, y_b, w_b, X_c, problem, *, center, E, hz, eps, drill_weight, variogram_cache):
    sigma2 = float(np.var(y_b))
    if sigma2 <= 1e-12:
        raise ValueError("constant binary indicators")

    nu = float(config["nu"])
    nu_eff = max(nu, 4.0 * eps * (1.0 - eps)) if eps > 0.0 else nu
    nu_eff = min(nu_eff, 0.99)
    c0 = nu_eff * sigma2
    sill = sigma2 - c0

    key = id(problem)
    if key in variogram_cache:
        h_lags, gamma_hat, n_pairs = variogram_cache[key]
    else:
        h_lags, gamma_hat, n_pairs = empirical_downhole_variogram(problem)
        variogram_cache[key] = (h_lags, gamma_hat, n_pairs)

    rho = config["rho"]
    a_v = fit_vertical_range(h_lags, gamma_hat, n_pairs, rho, sigma2, c0, hz, float(E[2]))
    s_z = float(config["s_z"])
    a_h = max(s_z * a_v, 1e-9)

    if len(X_c):
        X_all = np.concatenate([X_b, X_c], axis=0)
        y_all = np.concatenate([y_b, np.full(len(X_c), 0.5, dtype=float)], axis=0)
        w_all = np.concatenate([w_b, np.full(len(X_c), drill_weight, dtype=float)], axis=0)
    else:
        X_all, y_all, w_all = X_b, y_b, w_b

    X_s = stretch_points(X_all, center, s_z)
    n = len(X_s)
    D_mat = cdist(X_s, X_s) / a_h
    C = sill * _kriging_corr(D_mat, rho)
    diag_d = (c0 / np.maximum(w_all, 1e-12)) if c0 > 0.0 else np.full(n, 1e-10 * sigma2, dtype=float)
    C[np.diag_indices(n)] += diag_d

    A = np.empty((n + 1, n + 1), dtype=float)
    A[:n, :n] = C
    A[:n, n] = 1.0
    A[n, :n] = 1.0
    A[n, n] = 0.0
    rhs = np.empty(n + 1, dtype=float)
    rhs[:n] = y_all
    rhs[n] = 0.0

    sol = linalg_solve(A, rhs, assume_a="sym")
    if not np.isfinite(sol).all():
        raise ValueError("non-finite kriging solution")
    b_coef = sol[:n]
    mu = float(sol[n])

    def predict(points):
        pts_s = stretch_points(points, center, s_z)
        c_mat = sill * _kriging_corr(cdist(pts_s, X_s) / a_h, rho)
        return np.clip(c_mat @ b_coef + mu, 0.0, 1.0)

    return predict


def _wall_deadline(budget, started):
    seconds = 120.0 if (budget is None or "wall_time_seconds" not in budget) else float(budget["wall_time_seconds"])
    if not np.isfinite(seconds) or seconds <= 0:
        seconds = 120.0
    return started + 0.5 * seconds


def solve(problem, budget=None, seed=0):
    started = time.monotonic()
    data = extract_public_data(problem)
    if not (data["y_b"] > 0.5).any():
        rank = -np.linalg.norm(data["xyz"] - data["center"], axis=-1)
        return _enforce_legal(problem, np.zeros(data["shape"], dtype=bool), rank)

    center, E, hz = data["center"], data["E"], data["hz"]
    eps, drill_weight = data["eps"], data["drill_weight"]
    vcache = {}

    def fit(cfg, X_b, y_b, w_b, X_c, prob):
        return fit_indicator_kriging(
            cfg, X_b, y_b, w_b, X_c, prob,
            center=center, E=E, hz=hz, eps=eps, drill_weight=drill_weight, variogram_cache=vcache,
        )

    rules = ("native", "share")
    deadline = _wall_deadline(budget, started)
    folds = make_hole_folds(data, problem)

    best_idx = None
    best_rule = "share"
    best_score = -1.0

    if folds:
        for idx, cfg in enumerate(INDICATOR_KRIGING_CONFIGS):
            if time.monotonic() >= deadline:
                break
            scores = evaluate_config_cv(cfg, fit, folds, data["S"], 0.5, rules=rules, deadline=deadline)
            if scores is None:
                if time.monotonic() >= deadline:
                    break
                continue
            for r in rules:
                sc = scores[r]
                if sc > best_score + TIE_TOL:
                    best_score = sc
                    best_idx = idx
                    best_rule = r

    if best_idx is None:
        order = list(range(len(INDICATOR_KRIGING_CONFIGS)))
        best_rule = "share"
    else:
        order = [best_idx] + [i for i in range(len(INDICATOR_KRIGING_CONFIGS)) if i != best_idx]

    for idx in order:
        cfg = INDICATOR_KRIGING_CONFIGS[idx]
        try:
            pred_fn = fit(cfg, data["X_b"], data["y_b"], data["w_b"], data["X_c"], problem)
            f = predict_chunked(pred_fn, data["G"])
            f = np.asarray(f, dtype=float).reshape(-1)
            if len(f) != data["M"] or not np.isfinite(f).all():
                raise ValueError("non-finite field on feasible cells")
            mask_selected = apply_decision_rule(
                f, 0.5, best_rule, data["p_hat"], problem, data["feasible"], data["feasible_idx"]
            )
            rank = np.full(data["shape"], -np.inf, dtype=float)
            rank[data["feasible"]] = f
            return _enforce_legal(problem, mask_selected, rank)
        except Exception:
            continue

    return stretched_nearest_class(problem)
