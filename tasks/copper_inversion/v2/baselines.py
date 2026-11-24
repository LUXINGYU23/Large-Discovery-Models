"""Public-data-only development baselines and deterministic reference solvers.

Includes deterministic initialization, a Metropolis sampler, and three
deterministic reference solvers for headroom calibration (stretched
nearest-class, layered-horizon, and ellipsoid-fit). None read hidden truth.
"""

import argparse
import csv
import heapq
import json
import time
from pathlib import Path
import numpy as np
from scipy import ndimage
from scipy.interpolate import RBFInterpolator
from scipy.spatial import cKDTree
from .contract import validate
from .data import read_problem
from .geometry import canonicalize, components
from .metrics import compare
from .objective import score


def constrained_initial(problem, mask, rank):
    """Keep an already legal estimate; otherwise build a ranked connected body.

    This uses only observations and the public domain/prior, never hidden truth.
    It also makes a nonempty population prior explicit when no positive sample
    is observed, rather than silently returning an illegal empty initialization.
    """
    field=np.where(mask,1.,-1.)
    try:
        canonicalize(problem,field)
        return field
    except ValueError:
        pass
    feasible=problem['domain_mask'] & ~problem['forbidden_mask']
    labels,count=ndimage.label(feasible)
    sizes=np.bincount(labels.ravel());sizes[0]=0
    cell=float(np.prod(problem['spacing']));bounds=problem['priors']['hard_volume_bounds']
    lo=0 if bounds is None else int(np.ceil(bounds[0]/cell-1e-9))
    hi=int(feasible.sum()) if bounds is None else int(np.floor(bounds[1]/cell+1e-9))
    maximum=problem['priors']['max_components']
    regions=sorted(range(1,count+1),key=lambda i:(-sizes[i],i))
    if maximum is not None:regions=regions[:maximum]
    capacity=int(sum(sizes[i] for i in regions))
    target=min(max(int(mask.sum()),lo),hi,capacity)
    if target<lo:raise ValueError('public domain cannot satisfy initial volume/connectivity bounds')
    result=np.zeros(feasible.shape,bool);visited=np.zeros(feasible.shape,bool);filled=0
    for region in regions:
        if filled>=target:break
        indices=np.argwhere(labels==region)
        start=tuple(indices[np.argmax(rank[labels==region])]);heap=[(-float(rank[start]),start)];visited[start]=True
        while heap and filled<target:
            _,position=heapq.heappop(heap);result[position]=True;filled+=1
            for axis in range(3):
                for direction in (-1,1):
                    q=list(position);q[axis]+=direction;q=tuple(q)
                    if not 0<=q[axis]<feasible.shape[axis] or visited[q] or labels[q]!=region:continue
                    visited[q]=True;heapq.heappush(heap,(-float(rank[q]),q))
    field=np.where(result,1.,-1.)
    canonicalize(problem,field)
    return field


def initialize(problem):
    """Nearest positive/negative observations, with no truth-derived seed body."""
    validate(problem)
    xyz = np.stack(np.meshgrid(*[
        problem["origin"][i] + (np.arange(n) + .5) * problem["spacing"][i]
        for i, n in enumerate(problem["shape"])], indexing="ij"), axis=-1)
    points = np.concatenate([o["points"] for o in problem["observations"]])
    labels = np.concatenate([o["labels"] for o in problem["observations"]])
    positive, negative = points[labels == 1], points[labels == 0]
    feasible = problem["domain_mask"] & ~problem["forbidden_mask"]
    if not len(positive):
        mask = np.zeros(tuple(problem["shape"]), bool)
        center=np.asarray(problem['origin'])+.5*np.asarray(problem['shape'])*problem['spacing']
        rank=-np.linalg.norm(xyz-center,axis=-1)
    else:
        dp = cKDTree(positive).query(xyz.reshape(-1, 3))[0].reshape(problem["shape"])
        if len(negative):
            dn = cKDTree(negative).query(xyz.reshape(-1, 3))[0].reshape(problem["shape"])
            mask = dp < dn
            rank=dn-dp
        else:
            # A public resolution-based radius, not a hidden volume estimate.
            mask = dp <= 2 * np.linalg.norm(problem["spacing"])
            rank=-dp
        mask &= feasible
    return constrained_initial(problem,mask,rank)


def deterministic(problem, budget, seed):
    return initialize(problem)


def metropolis(problem, budget, seed, *, temperature=.02, radius_fraction=.1):
    """Symmetric local occupancy flips, minimizing the same public objective."""
    started = time.monotonic()
    seconds = float(budget["wall_time_seconds"])
    if not np.isfinite(seconds) or seconds <= 0 or temperature <= 0 or radius_fraction <= 0:
        raise ValueError("invalid sampler budget or parameters")
    # Reserve time for process teardown/serialization; same convention for LDM.
    deadline = started + .9 * seconds
    rng = np.random.default_rng(seed)
    field = initialize(problem)
    current = score(problem, field)["loss_total"]
    best, best_loss = field.copy(), current
    feasible = problem["domain_mask"] & ~problem["forbidden_mask"]
    centers = np.argwhere(feasible)
    coords = np.indices(problem["shape"], dtype=float)
    spacing = np.asarray(problem["spacing"])
    length = float(np.linalg.norm(np.asarray(problem["shape"]) * spacing))
    proposals = accepted = 0
    while time.monotonic() < deadline:
        center = centers[rng.integers(len(centers))]
        radius = max(float(spacing.min()), length * radius_fraction * rng.uniform(.25, 1.))
        patch = sum(((coords[i] - center[i]) * spacing[i])**2 for i in range(3)) <= radius**2
        patch &= feasible
        proposal = field.copy(); proposal[patch] *= -1
        proposals += 1
        try:
            loss = score(problem, proposal)["loss_total"]
        except ValueError:
            continue
        if np.log(max(rng.random(), np.finfo(float).tiny)) < min(0., (current - loss) / temperature):
            field, current = proposal, loss
            accepted += 1
            if loss < best_loss:
                best, best_loss = proposal.copy(), loss
    return best, dict(proposals=proposals, accepted=accepted, loss_total=best_loss,
                     wall_seconds=time.monotonic() - started)


Z_STRETCH_GRID = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0)


def _grid_centers(problem):
    return np.stack(np.meshgrid(*[
        problem["origin"][i] + (np.arange(n) + .5) * problem["spacing"][i]
        for i, n in enumerate(problem["shape"])], indexing="ij"), axis=-1)


def _drill_observation(problem):
    for obs in problem["observations"]:
        if obs["name"] == "drill":
            return obs
    return problem["observations"][0]


def _binary_observations(problem):
    pts, labs = [], []
    for obs in problem["observations"]:
        y = np.asarray(obs["labels"], dtype=float)
        b = y != .5
        if b.any():
            pts.append(np.asarray(obs["points"], dtype=float)[b])
            labs.append(y[b] > .5)
    if not pts:
        return np.zeros((0, 3), dtype=float), np.zeros(0, dtype=bool)
    return np.concatenate(pts), np.concatenate(labs)


def _drill_ore_share(problem):
    obs = _drill_observation(problem)
    y = np.asarray(obs["labels"], dtype=float)
    g = np.asarray(obs["groups"])
    b = y != .5
    if not b.any():
        return 0.0
    yb, gb = y[b] > .5, g[b]
    return float(np.mean([yb[gb == gid].mean() for gid in np.unique(gb)]))


def _denoise_hole_labels(z, y_bool):
    """Remove isolated single-sample label flips along a sorted borehole."""
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


def _enforce_legal(problem, mask, rank):
    """Enforce feasibility, volume bounds, and max_components while preserving ranked multi-lobe masks."""
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
                # Grow from the kept components in rank order to meet the hard volume floor.
                target = lo
                filled = int(mask.sum())
                visited = mask.copy()
                heap = []
                for pos in np.argwhere(mask):
                    p = tuple(pos)
                    for axis in range(3):
                        for d in (-1, 1):
                            q = list(p); q[axis] += d; q = tuple(q)
                            if 0 <= q[axis] < feasible.shape[axis] and feasible[q] and not visited[q]:
                                visited[q] = True
                                heapq.heappush(heap, (-float(rank[q]), q))
                while heap and filled < target:
                    _, q = heapq.heappop(heap)
                    mask[q] = True
                    filled += 1
                    for axis in range(3):
                        for d in (-1, 1):
                            r = list(q); r[axis] += d; r = tuple(r)
                            if 0 <= r[axis] < feasible.shape[axis] and feasible[r] and not visited[r]:
                                visited[r] = True
                                heapq.heappush(heap, (-float(rank[r]), r))
                # Merging two components while growing is allowed (component count only decreases).
                if components(mask, problem["spacing"])[0] > max_comp:
                    mask = np.zeros_like(feasible)
    return constrained_initial(problem, mask, rank)


def _select_z_stretch_cv(problem):
    """Choose one vertical stretch factor by leave-one-hole-out CV on public drill holes."""
    drill = _drill_observation(problem)
    y = np.asarray(drill["labels"], dtype=float)
    b = y != .5
    if not b.any():
        return 2.0
    d_pts = np.asarray(drill["points"], dtype=float)[b]
    d_lab = y[b] > .5
    d_grp = np.asarray(drill["groups"])[b]
    groups = np.unique(d_grp)
    if len(groups) < 2 or not d_lab.any() or d_lab.all():
        return 2.0
    extra_pts, extra_lab = [], []
    for obs in problem["observations"]:
        if obs is drill:
            continue
        oy = np.asarray(obs["labels"], dtype=float)
        ob = oy != .5
        if ob.any():
            extra_pts.append(np.asarray(obs["points"], dtype=float)[ob])
            extra_lab.append(oy[ob] > .5)
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
        iou = tp / float(tp + fp + fn)
        margin = float(np.mean(np.concatenate(margins))) if margins else -np.inf
        key = (round(iou, 12), margin)
        if key > best_key:
            best_key, best_zs = key, float(zs)
    return best_zs


def stretched_nearest_class(problem, budget=None, seed=0):
    """Reference solver 1: z-stretched nearest-class contrast cut at public drill ore share.

    One free parameter (z_stretch), chosen by leave-one-hole-out CV on public holes only.
    """
    validate(problem)
    xyz = _grid_centers(problem)
    feasible = problem["domain_mask"] & ~problem["forbidden_mask"]
    pts, lab = _binary_observations(problem)
    if not lab.any():
        center = np.asarray(problem["origin"]) + .5 * np.asarray(problem["shape"]) * problem["spacing"]
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


def _hole_intervals(problem):
    """Extract per-hole (x, y, z_bot, z_top) for ore-bearing drill holes and (x, y) for barren holes."""
    drill = _drill_observation(problem)
    pts = np.asarray(drill["points"], dtype=float)
    y = np.asarray(drill["labels"], dtype=float)
    grp = np.asarray(drill["groups"])
    hz = float(problem["spacing"][2])
    ore_holes, barren_xy = [], []
    for gid in np.unique(grp):
        sel = grp == gid
        hp, hy = pts[sel], y[sel]
        b = hy != .5
        if not b.any():
            continue
        bp, by = hp[b], hy[b] > .5
        contacts_z = hp[~b, 2]
        order = np.argsort(bp[:, 2])
        bp, by = bp[order], by[order]
        by_clean = _denoise_hole_labels(bp[:, 2], by) if by.any() else by
        if not by_clean.any():
            barren_xy.append(bp[:, :2].mean(axis=0))
            continue
        # Find the longest contiguous ore run along this borehole.
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
            z_bot = float(cand[np.argmin(np.abs(cand - .5 * (z_below + z_lo_ore)))]) if len(cand) else .5 * (z_below + z_lo_ore)
        else:
            cand = contacts_z[(contacts_z <= z_lo_ore) & (contacts_z >= z_lo_ore - 2.0 * hz)]
            z_bot = float(cand.max()) if len(cand) else z_lo_ore - .5 * hz
        if i1 < len(by_clean):
            z_above = float(bp[i1, 2])
            cand = contacts_z[(contacts_z >= z_hi_ore - hz) & (contacts_z <= z_above + hz)]
            z_top = float(cand[np.argmin(np.abs(cand - .5 * (z_hi_ore + z_above)))]) if len(cand) else .5 * (z_hi_ore + z_above)
        else:
            cand = contacts_z[(contacts_z >= z_hi_ore) & (contacts_z <= z_hi_ore + 2.0 * hz)]
            z_top = float(cand.min()) if len(cand) else z_hi_ore + .5 * hz
        xy_mean = bp[i0:i1, :2].mean(axis=0)
        ore_holes.append((float(xy_mean[0]), float(xy_mean[1]), min(z_bot, z_top), max(z_bot, z_top)))
    return ore_holes, (np.asarray(barren_xy, dtype=float) if barren_xy else np.zeros((0, 2), dtype=float))


def _interpolate_surface_2d(xy_pts, values, grid_xy, origin_xy, span_xy):
    """Deterministic scale-invariant 2D horizon interpolation (thin-plate spline with safe fallback)."""
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


def layered_horizon(problem, budget=None, seed=0):
    """Reference solver 2: interpolate per-hole top/bottom ore intervals into 2D horizons."""
    validate(problem)
    xyz = _grid_centers(problem)
    feasible = problem["domain_mask"] & ~problem["forbidden_mask"]
    ore_holes, barren_xy = _hole_intervals(problem)
    if not ore_holes:
        # Fallback if drill had no ore but another modality (e.g. outcrop) has positive samples.
        pts, lab = _binary_observations(problem)
        if lab.any():
            hz = float(problem["spacing"][2])
            ore_holes = [(float(p[0]), float(p[1]), float(p[2] - .5 * hz), float(p[2] + .5 * hz)) for p in pts[lab]]
        else:
            center = np.asarray(problem["origin"]) + .5 * np.asarray(problem["shape"]) * problem["spacing"]
            rank = -np.linalg.norm(xyz - center, axis=-1)
            return _enforce_legal(problem, np.zeros(tuple(problem["shape"]), bool), rank)
    arr = np.asarray(ore_holes, dtype=float)
    xy_ore, z_bot_pts, z_top_pts = arr[:, :2], arr[:, 2], arr[:, 3]
    origin_xy = np.asarray(problem["origin"][:2], dtype=float)
    span_xy = np.maximum(np.asarray(problem["shape"][:2], dtype=float) * np.asarray(problem["spacing"][:2], dtype=float), 1e-9)
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
        h_scale = max(float(np.sqrt(span_xy[0] * span_xy[1] / max(1, len(xy_ore) + len(barren_xy)))),
                      float(np.hypot(problem["spacing"][0], problem["spacing"][1])))
        s_lat = (d_bar - d_ore) / h_scale
        rank = np.minimum(s_vert, s_lat[:, :, None])
    else:
        rank = s_vert
    mask = (rank > 0) & feasible
    return _enforce_legal(problem, mask, rank)


def ellipsoid_fit(problem, budget=None, seed=0):
    """Reference solver 3: fit an oriented 3D ellipsoid/slab to public ore points at the public ore share."""
    validate(problem)
    xyz = _grid_centers(problem)
    feasible = problem["domain_mask"] & ~problem["forbidden_mask"]
    spacing = np.asarray(problem["spacing"], dtype=float)
    cell = float(np.prod(spacing))
    # Prefer denoised drill ore points; fall back to all binary positive observations.
    drill = _drill_observation(problem)
    dy = np.asarray(drill["labels"], dtype=float)
    db = dy != .5
    ore_pts = []
    if db.any():
        dp = np.asarray(drill["points"], dtype=float)[db]
        dl = dy[db] > .5
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
        center = np.asarray(problem["origin"]) + .5 * np.asarray(problem["shape"]) * spacing
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


REFERENCE_SOLVERS = {
    "initial": deterministic,
    "deterministic": deterministic,
    "stretched_nearest_class": stretched_nearest_class,
    "layered_horizon": layered_horizon,
    "ellipsoid_fit": ellipsoid_fit,
}


def _resolve_manifest(release_dir, split):
    release_dir = Path(release_dir)
    if split not in ("train", "train_s", "calibration"):
        raise ValueError(f"unsupported split for reference evaluation: {split!r}")
    for candidate in (release_dir / split / "manifest.json", release_dir / "manifests" / f"{split}.json"):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"manifest for split {split!r} not found under {release_dir}")


def _family_map(release_dir, split, manifest):
    release_dir = Path(release_dir)
    cases = manifest["cases"]
    # 1. Direct family/strata field in manifest cases.
    direct = {}
    for c in cases:
        fam = c.get("family") or (c.get("strata") or {}).get("family")
        if fam:
            direct[c["case_id"]] = str(fam)
    if len(direct) == len(cases):
        return direct
    # 2. strata.csv (rsi-r2 layout).
    strata_csv = release_dir / "strata.csv"
    if strata_csv.is_file():
        by_case, by_parent = {}, {}
        with strata_csv.open(newline="") as fh:
            for row in csv.DictReader(fh):
                if row.get("family"):
                    if row.get("case_id"):
                        by_case[row["case_id"]] = row["family"]
                    if row.get("parent_id"):
                        by_parent[row["parent_id"]] = row["family"]
        found = {c["case_id"]: by_case.get(c["case_id"]) or by_parent.get(c.get("parent_id")) for c in cases}
        if all(found.values()):
            return found
    # 3. Sibling dataset/id/plan.json + private-manifest.json (rsi-r1 layout).
    dataset_dir = release_dir.parent / "dataset"
    if dataset_dir.is_dir():
        by_parent = {}
        for part in ("id", "ood"):
            plan_path = dataset_dir / part / "plan.json"
            if not plan_path.is_file():
                continue
            plan = json.loads(plan_path.read_text())
            for entry in plan.get("parents", []):
                name = f"{entry['split']}-{entry['index']:02}"
                pm = dataset_dir / part / "packs" / name / "private-manifest.json"
                if pm.is_file():
                    by_parent[json.loads(pm.read_text())["parent_digest"]] = entry["family"]
        found = {c["case_id"]: by_parent.get(c.get("parent_id")) for c in cases}
        if all(found.values()):
            return found
    # 4. release/evidence/layout.json assignment ordering.
    layout_path = release_dir / "evidence" / "layout.json"
    if layout_path.is_file():
        layout = json.loads(layout_path.read_text())
        base_split = "train" if split == "train_s" else split
        rows = [r for r in layout.get("assignment", []) if r.get("split") == base_split]
        if split == "train_s":
            short_names = set(layout.get("short_tier", {}).get("parents", []))
            rows = [r for r in rows if r.get("parent") in short_names]
        if len(rows) == len(cases) and all(r.get("selected_modality") == c.get("modality") for r, c in zip(rows, cases)):
            return {c["case_id"]: r["family"] for c, r in zip(cases, rows)}
    return {c["case_id"]: direct.get(c["case_id"], "unknown") for c in cases}


def evaluate_release(release_dir, split, solver_name):
    """Evaluate a deterministic reference solver on a train or calibration split (aggregates only)."""
    if solver_name not in REFERENCE_SOLVERS:
        raise ValueError(f"unknown solver: {solver_name!r}")
    manifest_path = _resolve_manifest(release_dir, split)
    manifest = json.loads(manifest_path.read_text())
    root = manifest_path.parent
    families = _family_map(release_dir, split, manifest)
    solver = REFERENCE_SOLVERS[solver_name]
    budget = {"wall_time_seconds": 120.0}
    per_family = {}
    all_rows = []
    started = time.monotonic()
    for case in manifest["cases"]:
        problem = read_problem(root / case["public"])
        truth = np.load(root / case["truth"] / "truth.npy", allow_pickle=False)
        field = solver(problem, budget, 0)
        stats = compare(problem, field, truth)
        fam = families[case["case_id"]]
        per_family.setdefault(fam, []).append(stats)
        all_rows.append(stats)
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
    return {
        "release": str(Path(release_dir)),
        "split": split,
        "solver": solver_name,
        "overall": summarize(all_rows),
        "family_macro_iou": family_macro_iou,
        "by_family": by_family,
        "wall_seconds": round(wall, 3),
    }


def _format_table(reports):
    lines = [
        f"{'solver':<24} {'family':<14} {'n':>3} {'mean_iou':>9} {'std_iou':>8} {'min_iou':>8} {'surf_f1':>8} {'vol_err':>8}"
    ]
    lines.append("-" * len(lines[0]))
    for rep in reports:
        s = rep["solver"]
        for fam, row in rep["by_family"].items():
            lines.append(
                f"{s:<24} {fam:<14} {row['n']:>3d} {row['mean_iou']:>9.4f} {row['std_iou']:>8.4f} "
                f"{row['min_iou']:>8.4f} {row['mean_surface_f1']:>8.4f} {row['mean_rel_vol_err']:>8.4f}"
            )
        ov = rep["overall"]
        lines.append(
            f"{s:<24} {'OVERALL':<14} {ov['n']:>3d} {ov['mean_iou']:>9.4f} {ov['std_iou']:>8.4f} "
            f"{ov['min_iou']:>8.4f} {ov['mean_surface_f1']:>8.4f} {ov['mean_rel_vol_err']:>8.4f}"
        )
        lines.append(f"{s:<24} {'MACRO_FAMILY':<14} {ov['n']:>3d} {rep['family_macro_iou']:>9.4f} "
                     f"{'':>8} {'':>8} {'':>8} {'':>8}  ({rep['wall_seconds']:.2f}s)")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate deterministic reference solvers on train or calibration splits.")
    parser.add_argument("--release", type=Path, required=True, help="Release directory containing split manifests.")
    parser.add_argument("--split", choices=("train", "train_s", "calibration"), required=True)
    parser.add_argument("--solver", choices=("initial", "deterministic", "stretched_nearest_class",
                                             "layered_horizon", "ellipsoid_fit", "all"), required=True)
    parser.add_argument("--json", action="store_true", help="Print aggregate results as JSON.")
    args = parser.parse_args(argv)
    names = ["initial", "stretched_nearest_class", "layered_horizon", "ellipsoid_fit"] if args.solver == "all" else [args.solver]
    reports = [evaluate_release(args.release, args.split, name) for name in names]
    if args.json:
        print(json.dumps(reports[0] if len(reports) == 1 else reports, sort_keys=True, indent=2))
    else:
        print(_format_table(reports))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
