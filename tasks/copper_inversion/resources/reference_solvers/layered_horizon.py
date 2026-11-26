"""Self-contained 2D top/bottom horizon interpolation solver (layered_horizon)."""

import heapq
import numpy as np
from scipy import ndimage
from scipy.interpolate import RBFInterpolator
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


def _hole_intervals(problem):
    drill = _drill_observation(problem)
    pts = np.asarray(drill["points"], dtype=float)
    y = np.asarray(drill["labels"], dtype=float)
    grp = np.asarray(drill["groups"])
    hz = float(problem["spacing"][2])
    ore_holes, barren_xy = [], []
    for gid in np.unique(grp):
        sel = grp == gid
        hp, hy = pts[sel], y[sel]
        b = hy != 0.5
        if not b.any():
            continue
        bp, by = hp[b], hy[b] > 0.5
        contacts_z = hp[~b, 2]
        order = np.argsort(bp[:, 2])
        bp, by = bp[order], by[order]
        by_clean = _denoise_hole_labels(bp[:, 2], by) if by.any() else by
        if not by_clean.any():
            barren_xy.append(bp[:, :2].mean(axis=0))
            continue
        best_run = None
        i = 0
        while i < len(by_clean):
            if not by_clean[i]:
                i += 1
                continue
            j = i
            while j < len(by_clean) and by_clean[j]:
                j += 1
            if best_run is None or (j - i) > (best_run[1] - best_run[0]):
                best_run = (i, j)
            i = j
        i0, i1 = best_run
        z_lo_ore, z_hi_ore = float(bp[i0, 2]), float(bp[i1 - 1, 2])
        if i0 > 0:
            z_below = float(bp[i0 - 1, 2])
            cand = contacts_z[(contacts_z >= z_below - hz) & (contacts_z <= z_lo_ore + hz)]
            z_bot = (
                float(cand[np.argmin(np.abs(cand - 0.5 * (z_below + z_lo_ore)))])
                if len(cand)
                else 0.5 * (z_below + z_lo_ore)
            )
        else:
            cand = contacts_z[(contacts_z <= z_lo_ore) & (contacts_z >= z_lo_ore - 2.0 * hz)]
            z_bot = float(cand.max()) if len(cand) else z_lo_ore - 0.5 * hz
        if i1 < len(by_clean):
            z_above = float(bp[i1, 2])
            cand = contacts_z[(contacts_z >= z_hi_ore - hz) & (contacts_z <= z_above + hz)]
            z_top = (
                float(cand[np.argmin(np.abs(cand - 0.5 * (z_hi_ore + z_above)))])
                if len(cand)
                else 0.5 * (z_hi_ore + z_above)
            )
        else:
            cand = contacts_z[(contacts_z >= z_hi_ore) & (contacts_z <= z_hi_ore + 2.0 * hz)]
            z_top = float(cand.min()) if len(cand) else z_hi_ore + 0.5 * hz
        xy_mean = bp[i0:i1, :2].mean(axis=0)
        ore_holes.append((float(xy_mean[0]), float(xy_mean[1]), min(z_bot, z_top), max(z_bot, z_top)))
    return ore_holes, (np.asarray(barren_xy, dtype=float) if barren_xy else np.zeros((0, 2), dtype=float))


def _interpolate_surface_2d(xy_pts, values, grid_xy, origin_xy, span_xy):
    n = len(xy_pts)
    if n == 1:
        return np.full(grid_xy.shape[:-1], float(values[0]), dtype=float)
    u_pts = (xy_pts - origin_xy) / span_xy
    u_grid = (grid_xy - origin_xy) / span_xy
    flat = u_grid.reshape(-1, 2)
    centered = u_pts - u_pts.mean(axis=0, keepdims=True)
    rank = int(np.linalg.matrix_rank(centered, tol=1e-6))
    try:
        if n >= 3 and rank >= 2:
            rbf = RBFInterpolator(u_pts, values, kernel="thin_plate_spline", smoothing=1e-3)
            return rbf(flat).reshape(grid_xy.shape[:-1])
        rbf = RBFInterpolator(u_pts, values, kernel="linear", degree=0, smoothing=1e-3)
        return rbf(flat).reshape(grid_xy.shape[:-1])
    except Exception:
        idx = cKDTree(u_pts).query(flat)[1]
        return np.asarray(values, dtype=float)[idx].reshape(grid_xy.shape[:-1])


def solve(problem, budget=None, seed=0):
    validate(problem)
    xyz = _grid_centers(problem)
    feasible = problem["domain_mask"] & ~problem["forbidden_mask"]
    ore_holes, barren_xy = _hole_intervals(problem)
    if not ore_holes:
        pts, lab = _binary_observations(problem)
        if lab.any():
            hz = float(problem["spacing"][2])
            ore_holes = [(float(p[0]), float(p[1]), float(p[2] - 0.5 * hz), float(p[2] + 0.5 * hz)) for p in pts[lab]]
        else:
            center = np.asarray(problem["origin"]) + 0.5 * np.asarray(problem["shape"]) * problem["spacing"]
            rank = -np.linalg.norm(xyz - center, axis=-1)
            return _enforce_legal(problem, np.zeros(tuple(problem["shape"]), bool), rank)
    arr = np.asarray(ore_holes, dtype=float)
    xy_ore, z_bot_pts, z_top_pts = arr[:, :2], arr[:, 2], arr[:, 3]
    origin_xy = np.asarray(problem["origin"][:2], dtype=float)
    span_xy = np.maximum(
        np.asarray(problem["shape"][:2], dtype=float) * np.asarray(problem["spacing"][:2], dtype=float), 1e-9
    )
    grid_xy = xyz[:, :, 0, :2]
    z_bot_surf = _interpolate_surface_2d(xy_ore, z_bot_pts, grid_xy, origin_xy, span_xy)
    z_top_surf = _interpolate_surface_2d(xy_ore, z_top_pts, grid_xy, origin_xy, span_xy)
    z_lo = np.minimum(z_bot_surf, z_top_surf)
    z_hi = np.maximum(z_bot_surf, z_top_surf)
    hz = float(problem["spacing"][2])
    med_thick = max(float(np.median(z_top_pts - z_bot_pts)), hz)
    z_grid = xyz[:, :, :, 2]
    s_vert = np.minimum(z_grid - z_lo[:, :, None], z_hi[:, :, None] - z_grid) / med_thick
    if len(barren_xy):
        flat_xy = grid_xy.reshape(-1, 2)
        d_ore = cKDTree(xy_ore).query(flat_xy)[0].reshape(grid_xy.shape[:-1])
        d_bar = cKDTree(barren_xy).query(flat_xy)[0].reshape(grid_xy.shape[:-1])
        h_scale = max(
            float(np.sqrt(span_xy[0] * span_xy[1] / max(1, len(xy_ore) + len(barren_xy)))),
            float(np.hypot(problem["spacing"][0], problem["spacing"][1])),
        )
        s_lat = (d_bar - d_ore) / h_scale
        rank = np.minimum(s_vert, s_lat[:, :, None])
    else:
        rank = s_vert
    mask = (rank > 0) & feasible
    return _enforce_legal(problem, mask, rank)
