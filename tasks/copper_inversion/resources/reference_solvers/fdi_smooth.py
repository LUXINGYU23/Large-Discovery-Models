"""Self-contained discrete smooth implicit solver (fdi_smooth)."""

import heapq
import time
import numpy as np
from scipy import ndimage, sparse
from scipy.sparse.linalg import LinearOperator, cg
from scipy.spatial import cKDTree
from tasks.copper_inversion.v2.contract import validate
from tasks.copper_inversion.v2.geometry import canonicalize, components

TIE_TOL = 1e-12
SHARE_CLIP = (0.02, 0.65)
Z_STRETCH_GRID = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0)


def _build_fdi_configs():
    default = {"s_z": 4, "lam": 0.1, "target": "sdf"}
    out = [default]
    for lam in (1.0, 0.1, 0.01):
        for s_z in (4, 2, 8, 1):
            for target in ("sdf", "sign"):
                cfg = {"s_z": s_z, "lam": lam, "target": target}
                if cfg != default:
                    out.append(cfg)
    return tuple(out)


FDI_CONFIGS = _build_fdi_configs()


def _grid_centers(problem):
    axes = [problem["origin"][i] + (np.arange(n) + 0.5) * problem["spacing"][i] for i, n in enumerate(problem["shape"])]
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)


def _drill_obs(problem):
    for obs in problem["observations"]:
        if obs["name"] == "drill":
            return obs
    return problem["observations"][0]


def _drill_ore_share(problem):
    obs = _drill_obs(problem)
    y = np.asarray(obs["labels"], dtype=float)
    g = np.asarray(obs["groups"])
    b = y != 0.5
    if not b.any():
        return 0.0
    yb, gb = y[b] > 0.5, g[b]
    return float(np.mean([yb[gb == gid].mean() for gid in np.unique(gb)]))


def _denoise_hole(z, y_bool):
    if len(y_bool) < 5:
        return y_bool
    order = np.argsort(z)
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))
    s = y_bool[order].copy()
    mid = s.copy()
    mid[1:-1] = (s[:-2].astype(int) + s[1:-1].astype(int) + s[2:].astype(int)) >= 2
    if s[0] and not s[1]:
        mid[0] = False
    if s[-1] and not s[-2]:
        mid[-1] = False
    cleaned = mid[inv]
    return cleaned if cleaned.any() else y_bool


def _vol_bounds(problem, feasible):
    cell = float(np.prod(problem["spacing"]))
    bounds = problem["priors"]["hard_volume_bounds"]
    n_feas = int(feasible.sum())
    lo = 0 if bounds is None else int(np.ceil(bounds[0] / cell - 1e-9))
    hi = n_feas if bounds is None else min(n_feas, int(np.floor(bounds[1] / cell + 1e-9)))
    return lo, hi


def _neighbors(pos, shape):
    for ax in range(3):
        for d in (-1, 1):
            v = pos[ax] + d
            if 0 <= v < shape[ax]:
                q = list(pos)
                q[ax] = v
                yield tuple(q)


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
    regions = sorted(range(1, count + 1), key=lambda i: (-sizes[i], i))[:maximum]
    target = min(max(int(mask.sum()), lo), hi, int(sum(sizes[i] for i in regions)))
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
            _, pos = heapq.heappop(heap)
            result[pos] = True
            filled += 1
            for q in _neighbors(pos, feasible.shape):
                if not visited[q] and labels[q] == region:
                    visited[q] = True
                    heapq.heappush(heap, (-float(rank[q]), q))
    field = np.where(result, 1.0, -1.0)
    canonicalize(problem, field)
    return field


def _enforce_legal(problem, mask, rank):
    feasible = problem["domain_mask"] & ~problem["forbidden_mask"]
    mask = np.asarray(mask, dtype=bool) & feasible
    rank = np.asarray(rank, dtype=float)
    lo, hi = _vol_bounds(problem, feasible)
    max_comp = problem["priors"]["max_components"]
    n_mask = int(mask.sum())
    if n_mask < lo or n_mask > hi:
        target = min(max(n_mask, lo), hi)
        mask = np.zeros_like(feasible)
        if target > 0:
            coords = np.argwhere(feasible)
            order = np.argsort(-rank[feasible], kind="mergesort")
            mask[tuple(coords[order[:target]].T)] = True
    if max_comp is not None and mask.any():
        labels, count = ndimage.label(mask, structure=ndimage.generate_binary_structure(3, 1))
        if count > max_comp:
            sizes = np.bincount(labels.ravel())
            keep = sorted(range(1, count + 1), key=lambda i: (-int(sizes[i]), -float(rank[labels == i].max()), i))[:max_comp]
            mask = np.isin(labels, keep)
            if int(mask.sum()) < lo:
                filled = int(mask.sum())
                visited = mask.copy()
                heap = []
                for pos in np.argwhere(mask):
                    for q in _neighbors(tuple(pos), feasible.shape):
                        if feasible[q] and not visited[q]:
                            visited[q] = True
                            heapq.heappush(heap, (-float(rank[q]), q))
                while heap and filled < lo:
                    _, q = heapq.heappop(heap)
                    mask[q] = True
                    filled += 1
                    for r in _neighbors(q, feasible.shape):
                        if feasible[r] and not visited[r]:
                            visited[r] = True
                            heapq.heappush(heap, (-float(rank[r]), r))
                if components(mask, problem["spacing"])[0] > max_comp:
                    mask = np.zeros_like(feasible)
    return constrained_initial(problem, mask, rank)


def _stretched_fallback(problem):
    validate(problem)
    xyz = _grid_centers(problem)
    feasible = problem["domain_mask"] & ~problem["forbidden_mask"]
    pts_l, lab_l = [], []
    for obs in problem["observations"]:
        y = np.asarray(obs["labels"], dtype=float)
        b = y != 0.5
        if b.any():
            pts_l.append(np.asarray(obs["points"], dtype=float)[b])
            lab_l.append(y[b] > 0.5)
    pts = np.concatenate(pts_l) if pts_l else np.zeros((0, 3), dtype=float)
    lab = np.concatenate(lab_l) if lab_l else np.zeros(0, dtype=bool)
    if not lab.any():
        center = np.asarray(problem["origin"]) + 0.5 * np.asarray(problem["shape"]) * problem["spacing"]
        return _enforce_legal(problem, np.zeros(tuple(problem["shape"]), bool), -np.linalg.norm(xyz - center, axis=-1))
    drill = _drill_obs(problem)
    dy = np.asarray(drill["labels"], dtype=float)
    db = dy != 0.5
    zs = 2.0
    if db.any():
        d_pts = np.asarray(drill["points"], dtype=float)[db]
        d_lab = dy[db] > 0.5
        d_grp = np.asarray(drill["groups"])[db]
        groups = np.unique(d_grp)
        if len(groups) >= 2 and d_lab.any() and not d_lab.all():
            ex_p, ex_l = [], []
            for obs in problem["observations"]:
                if obs is not drill:
                    oy = np.asarray(obs["labels"], dtype=float)
                    ob = oy != 0.5
                    if ob.any():
                        ex_p.append(np.asarray(obs["points"], dtype=float)[ob])
                        ex_l.append(oy[ob] > 0.5)
            e_pts = np.concatenate(ex_p) if ex_p else np.zeros((0, 3), dtype=float)
            e_lab = np.concatenate(ex_l) if ex_l else np.zeros(0, dtype=bool)
            best_key = (-1.0, -np.inf)
            for cand_zs in Z_STRETCH_GRID:
                sc = np.array([1.0, 1.0, float(cand_zs)])
                sd, se = d_pts * sc, e_pts * sc
                tp = fp = fn = 0
                margins = []
                for g in groups:
                    hold = d_grp == g
                    tr_p = np.concatenate([sd[~hold], se]) if len(se) else sd[~hold]
                    tr_y = np.concatenate([d_lab[~hold], e_lab]) if len(se) else d_lab[~hold]
                    yt = d_lab[hold]
                    if tr_y.any() and (~tr_y).any():
                        dp = cKDTree(tr_p[tr_y]).query(sd[hold])[0]
                        dn = cKDTree(tr_p[~tr_y]).query(sd[hold])[0]
                        pred = dp < dn
                        c = (dn - dp) / (dn + dp + 1e-9)
                        margins.append(np.where(yt, c, -c))
                    else:
                        pred = np.ones(len(yt), bool) if tr_y.all() else np.zeros(len(yt), bool)
                    tp += int(np.count_nonzero(pred & yt))
                    fp += int(np.count_nonzero(pred & ~yt))
                    fn += int(np.count_nonzero(~pred & yt))
                if tp > 0:
                    key = (round(tp / float(tp + fp + fn), 12), float(np.mean(np.concatenate(margins))) if margins else -np.inf)
                    if key > best_key:
                        best_key, zs = key, float(cand_zs)
    sc = np.array([1.0, 1.0, zs])
    f_idx = np.argwhere(feasible)
    G, X = xyz[feasible] * sc, pts * sc
    dp = cKDTree(X[lab]).query(G)[0]
    score_f = (cKDTree(X[~lab]).query(G)[0] - dp) / (cKDTree(X[~lab]).query(G)[0] + dp + 1e-9) if (~lab).any() else -dp
    rank = np.full(tuple(problem["shape"]), -np.inf, dtype=float)
    rank[feasible] = score_f
    lo, hi = _vol_bounds(problem, feasible)
    target = int(round(_drill_ore_share(problem) * len(G)))
    if target <= 0:
        mask = (rank > 0) & feasible if (~lab).any() else np.zeros(tuple(problem["shape"]), bool)
    else:
        target = min(max(target, max(1, lo)), hi)
        mask = np.zeros(tuple(problem["shape"]), bool)
        mask[tuple(f_idx[np.argsort(-score_f, kind="mergesort")[:target]].T)] = True
    return _enforce_legal(problem, mask, rank)


def stretch_points(points, center, s_z):
    pts = np.asarray(points, dtype=float)
    return np.zeros((0, 3), dtype=float) if len(pts) == 0 else (pts - center) * np.array([1.0, 1.0, float(s_z)])


def extract_public_data(problem):
    validate(problem)
    shape = tuple(int(n) for n in problem["shape"])
    origin = np.asarray(problem["origin"], dtype=float)
    spacing = np.asarray(problem["spacing"], dtype=float)
    E = np.asarray(shape, dtype=float) * spacing
    center = origin + 0.5 * E
    feasible = np.asarray(problem["domain_mask"], dtype=bool) & ~np.asarray(problem["forbidden_mask"], dtype=bool)
    xyz = _grid_centers(problem)
    sub = np.zeros(shape, dtype=bool)
    sub[tuple(slice((1 - n) % 3, n, 3) if n >= 3 else slice(None) for n in shape)] = True
    sub &= feasible
    S = xyz[sub if sub.any() else feasible]

    drill = _drill_obs(problem)
    d_pts = np.asarray(drill["points"], dtype=float)
    d_lab = np.asarray(drill["labels"], dtype=float)
    d_grp = np.asarray(drill["groups"])
    d_w = float(drill.get("weight", 1.0))
    d_bin = d_lab != 0.5
    d_con = ~d_bin
    hole_ids = np.unique(d_grp) if len(d_grp) else np.zeros(0, dtype=np.int64)
    bin_hole_ids = np.unique(d_grp[d_bin]) if d_bin.any() else np.zeros(0, dtype=np.int64)
    h_hole = float(np.sqrt(E[0] * E[1] / max(1, len(hole_ids))))

    xp, yp, wp, dp_flag, gp = [], [], [], [], []
    if d_bin.any():
        n_db = int(d_bin.sum())
        xp.append(d_pts[d_bin])
        yp.append((d_lab[d_bin] > 0.5).astype(float))
        wp.append(np.full(n_db, d_w, dtype=float))
        dp_flag.append(np.ones(n_db, dtype=bool))
        gp.append(d_grp[d_bin].astype(np.int64, copy=False))
    for obs in problem["observations"]:
        if obs is not drill:
            oy = np.asarray(obs["labels"], dtype=float)
            ob = oy != 0.5
            if ob.any():
                n_ob = int(ob.sum())
                xp.append(np.asarray(obs["points"], dtype=float)[ob])
                yp.append((oy[ob] > 0.5).astype(float))
                wp.append(np.full(n_ob, float(obs.get("weight", 1.0)), dtype=float))
                dp_flag.append(np.zeros(n_ob, dtype=bool))
                gp.append(np.full(n_ob, -1, dtype=np.int64))
    X_b = np.concatenate(xp, axis=0) if xp else np.zeros((0, 3), dtype=float)
    y_b = np.concatenate(yp, axis=0) if xp else np.zeros(0, dtype=float)
    w_b = np.concatenate(wp, axis=0) if xp else np.zeros(0, dtype=float)
    is_drill_b = np.concatenate(dp_flag, axis=0) if xp else np.zeros(0, dtype=bool)
    grp_b = np.concatenate(gp, axis=0) if xp else np.zeros(0, dtype=np.int64)
    X_c = d_pts[d_con] if d_con.any() else np.zeros((0, 3), dtype=float)
    G = xyz[feasible]
    return {
        "shape": shape, "origin": origin, "spacing": spacing, "center": center,
        "feasible": feasible, "feasible_idx": np.argwhere(feasible), "xyz": xyz,
        "G": G, "M": int(len(G)), "S": S, "hole_ids": hole_ids, "bin_hole_ids": bin_hole_ids,
        "h_hole": h_hole, "X_b": X_b, "y_b": y_b, "w_b": w_b, "is_drill_b": is_drill_b,
        "grp_b": grp_b, "X_c": X_c, "w_c": np.full(len(X_c), d_w, dtype=float),
        "p_hat": float(_drill_ore_share(problem)),
    }


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
    h2f = {int(h): int(r % K) for r, h in enumerate(bin_holes)}
    for r, h in enumerate(hole_ids):
        h2f.setdefault(int(h), int(r % K))
    grp_b, X_b, y_b = data["grp_b"], data["X_b"], data["y_b"]
    drill = _drill_obs(problem)
    d_groups = np.asarray(drill["groups"])
    folds = []
    for k in range(K):
        val_h = np.array([h for h in hole_ids if h2f[int(h)] == k], dtype=hole_ids.dtype)
        tr_h = np.array([h for h in hole_ids if h2f[int(h)] != k], dtype=hole_ids.dtype)
        val_m = is_drill_b & np.isin(grp_b, val_h)
        if not val_m.any():
            continue
        tr_d = (~val_m) & is_drill_b
        tr_bh = np.unique(grp_b[tr_d])
        p_raw = float(np.mean([(y_b[tr_d & (grp_b == h)] > 0.5).mean() for h in tr_bh])) if len(tr_bh) else SHARE_CLIP[0]
        keep = np.isin(d_groups, tr_h)
        new_obs = [
            {**o, "points": np.asarray(o["points"], float)[keep], "labels": np.asarray(o["labels"], float)[keep], "groups": np.asarray(o["groups"])[keep]}
            if o is drill else o
            for o in problem["observations"]
        ]
        folds.append({
            "X_val": X_b[val_m], "y_val": y_b[val_m] > 0.5,
            "p_train": float(np.clip(p_raw, SHARE_CLIP[0], SHARE_CLIP[1])),
            "problem_tr": {**problem, "observations": new_obs},
        })
    return folds


def denoise_binary_observations(problem):
    drill = _drill_obs(problem)
    d_pts = np.asarray(drill["points"], dtype=float)
    d_lab = np.asarray(drill["labels"], dtype=float)
    d_grp = np.asarray(drill["groups"])
    d_w = float(drill.get("weight", 1.0))
    eps = float(drill.get("label_error", 0.0))
    d_bin = d_lab != 0.5
    xp, yp, wp = [], [], []
    if d_bin.any():
        bp, by, bg = d_pts[d_bin], (d_lab[d_bin] > 0.5).copy(), d_grp[d_bin]
        if eps > 0.0:
            for gid in np.unique(bg):
                sel = bg == gid
                by[sel] = _denoise_hole(bp[sel, 2], by[sel])
        xp.append(bp)
        yp.append(by.astype(float))
        wp.append(np.full(len(bp), d_w, dtype=float))
    for obs in problem["observations"]:
        if obs is not drill:
            oy = np.asarray(obs["labels"], dtype=float)
            ob = oy != 0.5
            if ob.any():
                n_ob = int(ob.sum())
                xp.append(np.asarray(obs["points"], dtype=float)[ob])
                yp.append((oy[ob] > 0.5).astype(float))
                wp.append(np.full(n_ob, float(obs.get("weight", 1.0)), dtype=float))
    if not xp:
        return np.zeros((0, 3), dtype=float), np.zeros(0, dtype=float), np.zeros(0, dtype=float)
    return np.concatenate(xp, axis=0), np.concatenate(yp, axis=0), np.concatenate(wp, axis=0)


def _targets_for_problem(problem, target_kind, s_z, center, h_hole):
    if target_kind == "sdf":
        drill = _drill_obs(problem)
        d_pts = np.asarray(drill["points"], dtype=float)
        d_lab = np.asarray(drill["labels"], dtype=float)
        d_w = float(drill.get("weight", 1.0))
        X_b, y_b, w_b = denoise_binary_observations(problem)
        ore = y_b > 0.5
        bar = ~ore
        if not ore.any() or not bar.any():
            raise ValueError("cleaned binary labels have only one class")
        X_c = d_pts[d_lab == 0.5]
        w_c = np.full(len(X_c), d_w, dtype=float)
        Xb_s = stretch_points(X_b, center, s_z)
        Xc_s = stretch_points(X_c, center, s_z)
        d_ore = cKDTree(Xb_s[bar]).query(Xb_s[ore])[0]
        d_bar = cKDTree(Xb_s[ore]).query(Xb_s[bar])[0]
        if len(Xc_s):
            tc = cKDTree(Xc_s)
            d_ore = np.minimum(d_ore, tc.query(Xb_s[ore])[0])
            d_bar = np.minimum(d_bar, tc.query(Xb_s[bar])[0])
        cap = 2.0 * h_hole
        v_b = np.empty(len(Xb_s), dtype=float)
        v_b[ore] = np.minimum(d_ore, cap)
        v_b[bar] = -np.minimum(d_bar, cap)
        v_raw = np.concatenate([v_b, np.zeros(len(Xc_s), dtype=float)], axis=0) if len(Xc_s) else v_b
        w_all = np.concatenate([w_b, w_c], axis=0) if len(Xc_s) else w_b
        X_phys = np.concatenate([X_b, X_c], axis=0) if len(X_c) else X_b
        return X_phys, np.clip(v_raw / max(cap, 1e-12), -1.0, 1.0), w_all
    d = extract_public_data(problem)
    X_b, y_b, w_b, X_c, w_c = d["X_b"], d["y_b"], d["w_b"], d["X_c"], d["w_c"]
    v_b = np.where(y_b > 0.5, 1.0, -1.0)
    if len(X_c):
        return np.concatenate([X_b, X_c], 0), np.concatenate([v_b, np.zeros(len(X_c))], 0), np.concatenate([w_b, w_c], 0)
    return X_b, v_b, w_b


def _grid_geometry(origin, spacing, shape, *, coarse=False):
    origin = np.asarray(origin, dtype=float)
    spacing = np.asarray(spacing, dtype=float)
    shape = tuple(int(n) for n in shape)
    node_min = origin + 0.5 * spacing
    if not coarse:
        return shape, node_min, spacing
    c_shape = tuple(max(3, (n + 1) // 2) if n >= 3 else n for n in shape)
    if c_shape == shape:
        return shape, node_min, spacing
    span = (np.asarray(shape, dtype=float) - 1.0) * spacing
    return c_shape, node_min, span / np.maximum(np.asarray(c_shape, dtype=float) - 1.0, 1.0)


def trilinear_matrix(points, node_min, node_step, node_shape):
    pts = np.asarray(points, dtype=float)
    n_pts = len(pts)
    nx, ny, nz = (int(v) for v in node_shape)
    m_nodes = nx * ny * nz
    if n_pts == 0:
        return sparse.csr_matrix((0, m_nodes), dtype=float)
    q = (pts - np.asarray(node_min, dtype=float)) / np.asarray(node_step, dtype=float)
    i0 = np.clip(np.floor(q).astype(np.int64), 0, np.array([max(0, nx - 2), max(0, ny - 2), max(0, nz - 2)], np.int64))
    t = np.clip(q - i0, 0.0, 1.0)
    rows = np.empty(8 * n_pts, dtype=np.int64)
    cols = np.empty(8 * n_pts, dtype=np.int64)
    vals = np.empty(8 * n_pts, dtype=float)
    base = np.arange(n_pts, dtype=np.int64)
    slot = 0
    for dx in (0, 1):
        wx = t[:, 0] if dx else (1.0 - t[:, 0])
        ix = i0[:, 0] + dx
        for dy in (0, 1):
            wxy = wx * (t[:, 1] if dy else (1.0 - t[:, 1]))
            iy = i0[:, 1] + dy
            for dz in (0, 1):
                sl = slice(slot * n_pts, (slot + 1) * n_pts)
                rows[sl] = base
                cols[sl] = (ix * ny + iy) * nz + (i0[:, 2] + dz)
                vals[sl] = wxy * (t[:, 2] if dz else (1.0 - t[:, 2]))
                slot += 1
    return sparse.csr_matrix((vals, (rows, cols)), shape=(n_pts, m_nodes))


def regulariser_gram(node_shape, node_step, s_z):
    nx, ny, nz = (int(v) for v in node_shape)
    m_nodes = nx * ny * nz
    hs = (float(node_step[0]), float(node_step[1]), float(s_z) * float(node_step[2]))
    g2 = (hs[0] * hs[1] * hs[2]) ** (2.0 / 3.0)
    idx = np.arange(m_nodes, dtype=np.int64).reshape((nx, ny, nz))
    rc, cc, vc = [], [], []
    off = 0

    def add(nodes_list, weights):
        nonlocal off
        k = nodes_list[0].size
        if k:
            r = np.arange(off, off + k, dtype=np.int64)
            for arr, w in zip(nodes_list, weights):
                rc.append(r)
                cc.append(arr.ravel())
                vc.append(np.full(k, float(w), dtype=float))
            off += k

    dims = (nx, ny, nz)
    for ax in range(3):
        if dims[ax] >= 3:
            s = g2 / (hs[ax] * hs[ax])
            sl = lambda a: tuple(a if i == ax else slice(None) for i in range(3))
            add([idx[sl(slice(2, None))], idx[sl(slice(1, -1))], idx[sl(slice(None, -2))]], (s, -2.0 * s, s))
    rt2 = float(np.sqrt(2.0))
    for a, b in ((0, 1), (1, 2), (0, 2)):
        if dims[a] >= 2 and dims[b] >= 2:
            s = rt2 * g2 / (hs[a] * hs[b])
            sl2 = lambda sa, sb: tuple(sa if i == a else (sb if i == b else slice(None)) for i in range(3))
            hi, lo = slice(1, None), slice(None, -1)
            add([idx[sl2(hi, hi)], idx[sl2(hi, lo)], idx[sl2(lo, hi)], idx[sl2(lo, lo)]], (s, -s, -s, s))
    if off == 0:
        return sparse.csr_matrix((m_nodes, m_nodes), dtype=float)
    R = sparse.csr_matrix((np.concatenate(vc), (np.concatenate(rc), np.concatenate(cc))), shape=(off, m_nodes))
    return ((R.T @ R) * (1.0 / float(m_nodes))).tocsr()


def _solve_cg(AtWA, AtWv, RtR, lam, *, x0=None, rtol=1e-7):
    m_nodes = AtWA.shape[0]
    H = AtWA + float(lam) * RtR + 1e-14 * sparse.eye(m_nodes, format="csr")
    inv_d = 1.0 / np.maximum(H.diagonal(), 1e-14)
    pre = LinearOperator((m_nodes, m_nodes), matvec=lambda v: inv_d * v, dtype=float)
    u, info = cg(H, AtWv, x0=x0, rtol=rtol, atol=0.0, maxiter=3000, M=pre)
    if not np.isfinite(u).all() or info < 0:
        raise np.linalg.LinAlgError("FDI CG failure")
    if info > 0:
        rn = float(np.linalg.norm(AtWv))
        if rn > 0.0 and float(np.linalg.norm(H @ u - AtWv)) > 1e-3 * rn:
            raise np.linalg.LinAlgError("FDI CG did not converge")
    return u


def _apply_rule(f, rule, p_hat, problem, feasible, feasible_idx):
    mask = np.zeros(feasible.shape, dtype=bool)
    if len(f) == 0:
        return mask
    if rule == "native":
        mask[feasible] = f > 0.0
        return mask
    lo, hi = _vol_bounds(problem, feasible)
    if hi > 0:
        target = min(max(int(round(p_hat * len(f))), min(max(1, lo), hi)), hi)
        mask[tuple(feasible_idx[np.argsort(-f, kind="mergesort")[:target]].T)] = True
    return mask


def solve(problem, budget=None, seed=0):
    started = time.monotonic()
    data = extract_public_data(problem)
    if not (data["y_b"] > 0.5).any():
        return _enforce_legal(problem, np.zeros(data["shape"], bool), -np.linalg.norm(data["xyz"] - data["center"], axis=-1))

    secs = 120.0 if (budget is None or "wall_time_seconds" not in budget) else float(budget["wall_time_seconds"])
    deadline = started + 0.5 * (secs if np.isfinite(secs) and secs > 0 else 120.0)
    rules = ("native", "share")
    folds = make_hole_folds(data, problem)

    best_idx = None
    best_rule = "share"
    best_score = -1.0
    stopped = False

    if folds:
        c_shape, c_min, c_step = _grid_geometry(data["origin"], data["spacing"], data["shape"], coarse=True)
        A_S = trilinear_matrix(data["S"], c_min, c_step, c_shape)
        rtr_cache = {}
        fcache = [
            {"fold": f, "A_val": trilinear_matrix(f["X_val"], c_min, c_step, c_shape), "dm": {}, "tg": {}, "wm": {}}
            for f in folds
        ]
        for idx, cfg in enumerate(FDI_CONFIGS):
            if time.monotonic() >= deadline:
                break
            s_z, lam, t_kind = cfg["s_z"], cfg["lam"], cfg["target"]
            if s_z not in rtr_cache:
                rtr_cache[s_z] = regulariser_gram(c_shape, c_step, s_z)
            RtR = rtr_cache[s_z]
            counts = {r: [0, 0, 0] for r in rules}
            failed = False
            for fc in fcache:
                if time.monotonic() >= deadline:
                    stopped = failed = True
                    break
                fold = fc["fold"]
                tk = (t_kind, s_z) if t_kind == "sdf" else ("sign", 0)
                if tk not in fc["tg"]:
                    try:
                        fc["tg"][tk] = _targets_for_problem(fold["problem_tr"], t_kind, s_z, data["center"], data["h_hole"])
                    except Exception:
                        fc["tg"][tk] = None
                tgt = fc["tg"][tk]
                if tgt is None:
                    failed = True
                    break
                X_tr, v_tr, w_tr = tgt
                if t_kind not in fc["dm"]:
                    n_tr = len(X_tr)
                    A_tr = trilinear_matrix(X_tr, c_min, c_step, c_shape)
                    AtW = (A_tr.T @ sparse.diags(w_tr / float(n_tr), 0, shape=(n_tr, n_tr), format="csr")).tocsr()
                    fc["dm"][t_kind] = (AtW, (AtW @ A_tr).tocsr())
                AtW, AtWA = fc["dm"][t_kind]
                AtWv = np.asarray(AtW @ v_tr, dtype=float).reshape(-1)
                wk = (s_z, t_kind)
                try:
                    u_c = _solve_cg(AtWA, AtWv, RtR, lam, x0=fc["wm"].get(wk), rtol=1e-8)
                except Exception:
                    failed = True
                    break
                fc["wm"][wk] = u_c
                f_val = np.asarray(fc["A_val"] @ u_c, dtype=float).reshape(-1)
                f_S = np.asarray(A_S @ u_c, dtype=float).reshape(-1)
                yt = fold["y_val"]
                for r in rules:
                    if r == "native":
                        pred = f_val > 0.0
                    else:
                        q = float(np.quantile(f_S, 1.0 - fold["p_train"]))
                        pred = (f_val > q) if (np.any(f_S > q) or float(f_S.max()) <= float(f_S.min())) else (f_val >= q)
                    counts[r][0] += int(np.count_nonzero(pred & yt))
                    counts[r][1] += int(np.count_nonzero(pred & ~yt))
                    counts[r][2] += int(np.count_nonzero(~pred & yt))
            if failed:
                if stopped:
                    break
                continue
            for r in rules:
                tp, fp, fn = counts[r]
                sc = float(tp) / float(tp + fp + fn) if tp > 0 else 0.0
                if sc > best_score + TIE_TOL:
                    best_score, best_idx, best_rule = sc, idx, r

    order = list(range(len(FDI_CONFIGS))) if best_idx is None else [best_idx] + [i for i in range(len(FDI_CONFIGS)) if i != best_idx]
    for idx in order:
        cfg = FDI_CONFIGS[idx]
        try:
            X_f, v_f, w_f = _targets_for_problem(problem, cfg["target"], cfg["s_z"], data["center"], data["h_hole"])
            n_pts = len(X_f)
            if n_pts == 0:
                raise ValueError("empty FDI observations")
            n_sh, n_min, n_st = _grid_geometry(data["origin"], data["spacing"], data["shape"], coarse=False)
            A = trilinear_matrix(X_f, n_min, n_st, n_sh)
            AtW = (A.T @ sparse.diags(w_f / float(n_pts), 0, shape=(n_pts, n_pts), format="csr")).tocsr()
            u_full = _solve_cg((AtW @ A).tocsr(), np.asarray(AtW @ v_f, float).reshape(-1), regulariser_gram(n_sh, n_st, cfg["s_z"]), cfg["lam"], rtol=1e-7)
            f = np.asarray(u_full.reshape(data["shape"])[data["feasible"]], dtype=float).reshape(-1)
            if len(f) != data["M"] or not np.isfinite(f).all():
                raise ValueError("non-finite field")
            mask_sel = _apply_rule(f, best_rule, data["p_hat"], problem, data["feasible"], data["feasible_idx"])
            rank = np.full(data["shape"], -np.inf, dtype=float)
            rank[data["feasible"]] = f
            return _enforce_legal(problem, mask_sel, rank)
        except Exception:
            continue

    return _stretched_fallback(problem)
