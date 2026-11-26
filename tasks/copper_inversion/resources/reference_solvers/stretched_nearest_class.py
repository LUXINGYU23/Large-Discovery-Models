"""Self-contained z-stretched nearest-class contrast solver (stretched_nearest_class)."""

import heapq
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from tasks.copper_inversion.v2.contract import validate
from tasks.copper_inversion.v2.geometry import canonicalize, components

Z_STRETCH_GRID = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0)


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


def solve(problem, budget=None, seed=0):
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
