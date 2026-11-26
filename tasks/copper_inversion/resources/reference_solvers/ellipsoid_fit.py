"""Self-contained moment-matched 3D ellipsoid solver (ellipsoid_fit)."""

import heapq
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from tasks.copper_inversion.v2.contract import validate
from tasks.copper_inversion.v2.geometry import canonicalize, components


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


def _denoise_hole_labels(z, y_bool):
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


def solve(problem, budget=None, seed=0):
    validate(problem)
    xyz = _grid_centers(problem)
    feasible = problem["domain_mask"] & ~problem["forbidden_mask"]
    spacing = np.asarray(problem["spacing"], dtype=float)
    cell = float(np.prod(spacing))
    drill = _drill_observation(problem)
    dy = np.asarray(drill["labels"], dtype=float)
    db = dy != 0.5
    ore_pts = []
    if db.any():
        dp = np.asarray(drill["points"], dtype=float)[db]
        dl = dy[db] > 0.5
        dg = np.asarray(drill["groups"])[db]
        for gid in np.unique(dg):
            sel = dg == gid
            if not dl[sel].any():
                continue
            hp, hl = dp[sel], dl[sel]
            hl_clean = _denoise_hole_labels(hp[:, 2], hl)
            if hl_clean.any():
                ore_pts.append(hp[hl_clean])
    if ore_pts:
        P = np.concatenate(ore_pts)
    else:
        pts, lab = _binary_observations(problem)
        P = pts[lab]
    if not len(P):
        center = np.asarray(problem["origin"]) + 0.5 * np.asarray(problem["shape"]) * spacing
        rank = -np.linalg.norm(xyz - center, axis=-1)
        return _enforce_legal(problem, np.zeros(tuple(problem["shape"]), bool), rank)
    lo, hi = _volume_cell_bounds(problem, feasible)
    n_feas = int(feasible.sum())
    share = _drill_ore_share(problem)
    target = int(round(share * n_feas)) if share > 0 else max(1, lo)
    target = min(max(target, max(1, lo)), hi)
    target_vol = target * cell
    r_eq = float((3.0 * target_vol / (4.0 * np.pi)) ** (1.0 / 3.0))
    mu = P.mean(axis=0)
    cov = np.cov(P.T, bias=True) if len(P) >= 2 else np.zeros((3, 3), dtype=float)
    n_holes = max(1, len(np.unique(np.asarray(drill["groups"])[db]))) if db.any() else 1
    span_xy = np.asarray(problem["shape"][:2], dtype=float) * spacing[:2]
    h_hole = float(np.sqrt(span_xy[0] * span_xy[1] / n_holes))
    sigma_h = max(0.5 * float(np.hypot(spacing[0], spacing[1])), min(0.35 * h_hole, 0.5 * r_eq))
    sigma_z = max(0.5 * float(spacing[2]), 1e-6)
    cov = cov + np.diag([sigma_h ** 2, sigma_h ** 2, sigma_z ** 2])
    eigvals, eigvecs = np.linalg.eigh(cov)
    inv_cov = (eigvecs * (1.0 / np.maximum(eigvals, 1e-12))) @ eigvecs.T
    f_idx = np.argwhere(feasible)
    diff = xyz[feasible] - mu
    d2 = np.einsum("ni,ij,nj->n", diff, inv_cov, diff)
    rank = np.full(tuple(problem["shape"]), -np.inf, dtype=float)
    rank[feasible] = -d2
    order = np.argsort(d2, kind="mergesort")
    mask = np.zeros(tuple(problem["shape"]), bool)
    mask[tuple(f_idx[order[:target]].T)] = True
    return _enforce_legal(problem, mask, rank)
