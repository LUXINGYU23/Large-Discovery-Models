"""Copper v2 round-9 main candidate 3: Refinement: T0=3e-5 annealing at leader radii."""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import time
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from scipy.ndimage import map_coordinates
from tasks.copper_inversion.v2.objective import score
from tasks.copper_inversion.v2.geometry import canonicalize
from tasks.copper_inversion.v2.baselines import initialize

CFG = {'time': 0.68, 'radii': (0.05, 0.075, 0.105, 0.13, 0.17), 'jitter': 0.35, 'mode': 'anneal', 'seed_offset': 11, 'fast_legal': True, 'fast_loss': True, 'T0': 3e-05, 'shell_frac': 0.012}
_ST = ndimage.generate_binary_structure(3, 1)


def _prune(mask, maximum):
    if maximum is None or not mask.any():
        return mask
    lab, n = ndimage.label(mask, structure=_ST)
    if n <= maximum:
        return mask
    sizes = np.bincount(lab.ravel()); sizes[0] = 0
    keep = np.argsort(sizes[1:])[::-1][:maximum] + 1
    return np.isin(lab, keep)


def _add(mask, rank, feasible, need):
    need = int(min(max(need, 0), int((feasible & ~mask).sum())))
    if need <= 0:
        return mask
    idx = np.argwhere(feasible & ~mask)
    if not len(idx):
        return mask
    sr = (-ndimage.distance_transform_edt(~mask) * 1e9 + np.asarray(rank, float)
          if mask.any() else np.asarray(rank, float))
    out = mask.copy()
    out[tuple(idx[np.argsort(sr[tuple(idx.T)])[::-1][:need]].T)] = True
    return out


def _remove(mask, rank, excess):
    excess = int(max(excess, 0))
    if excess <= 0 or not mask.any():
        return mask
    idx = np.argwhere(mask); vals = np.asarray(rank, float)[tuple(idx.T)]
    out = mask.copy(); out[tuple(idx[np.argsort(vals)[:excess]].T)] = False
    return out


def _repair(problem, mask, rank):
    feasible = problem["domain_mask"] & ~problem["forbidden_mask"]
    pri = problem["priors"]; m = np.asarray(mask, bool) & feasible
    m = _prune(m, pri["max_components"])
    cell = float(np.prod(np.asarray(problem["spacing"], float)))
    hb = pri["hard_volume_bounds"]
    lo = 0 if hb is None else int(np.ceil(hb[0] / cell - 1e-9))
    hi = int(feasible.sum()) if hb is None else int(np.floor(hb[1] / cell + 1e-9))
    lo = max(0, min(lo, int(feasible.sum()))); hi = max(lo, min(hi, int(feasible.sum())))
    if pri["fragment_weight"] > 0 and pri["fragment_volume"] > 0 and m.any():
        lab, _ = ndimage.label(m, structure=_ST)
        sizes = np.bincount(lab.ravel()) * cell; sizes[0] = 0
        keep = np.where(sizes >= pri["fragment_volume"])[0]; keep = keep[keep > 0]
        m2 = np.isin(lab, keep)
        if int(m2.sum()) >= lo:
            m = m2
    for _ in range(6):
        m = _prune(m, pri["max_components"]); n = int(m.sum())
        if n < lo:
            m = _add(m, rank, feasible, lo - n)
        elif n > hi:
            m = _remove(m, rank, n - hi)
        else:
            break
    if pri["volume_weight"] > 0 and pri["volume_bounds"] is not None:
        n = int(m.sum()); vb = pri["volume_bounds"]
        slo = max(lo, min(int(np.ceil(vb[0] / cell - 1e-9)), hi))
        shi = min(hi, max(lo, int(np.floor(vb[1] / cell + 1e-9))))
        if n < slo:
            m = _add(m, rank, feasible, slo - n)
        elif n > shi:
            m = _remove(m, rank, n - shi)
    m = _prune(m, pri["max_components"])
    if int(m.sum()) < lo:
        m = _add(m, rank, feasible, lo - int(m.sum()))
        m = _prune(m, pri["max_components"])
    return m


def _rank(problem):
    sp = np.asarray(problem["spacing"], float); org = np.asarray(problem["origin"], float)
    xyz = np.stack(np.meshgrid(*[org[i] + (np.arange(n) + .5) * sp[i]
                                 for i, n in enumerate(problem["shape"])],
                                indexing="ij"), axis=-1)
    pts = np.concatenate([np.asarray(o["points"], float) for o in problem["observations"]])
    lab = np.concatenate([np.asarray(o["labels"], float) for o in problem["observations"]])
    pos, neg = pts[lab == 1], pts[lab == 0]
    if len(pos):
        dp = cKDTree(pos).query(xyz.reshape(-1, 3))[0].reshape(tuple(problem["shape"]))
        if len(neg):
            dn = cKDTree(neg).query(xyz.reshape(-1, 3))[0].reshape(tuple(problem["shape"]))
            return dn - dp
        return -dp
    center = org + .5 * np.asarray(problem["shape"]) * sp
    return -np.linalg.norm(xyz - center, axis=-1)


def _gm(values, groups):
    if not len(values):
        return 0.
    return float(np.mean([values[groups == g].mean() for g in np.unique(groups)]))


def _fast_loss(problem, field):
    pri = problem["priors"]
    if pri["area_weight"] or pri["volume_weight"] or pri["fragment_weight"]:
        return float(score(problem, field)["loss_total"])
    mask, phi = canonicalize(problem, field)
    total_weight = sum(o["weight"] for o in problem["observations"])
    binary_total = 0.; contact_total = 0.
    for obs in problem["observations"]:
        coords = ((obs["points"] - np.asarray(problem["origin"])) /
                  problem["spacing"] - .5).T
        distance = map_coordinates(phi, coords, order=1, mode="nearest")
        y, groups = obs["labels"], obs["groups"]
        binary, contact = y != .5, y == .5
        signed = (2 * y[binary] - 1) * distance[binary] / obs["transition_scale"]
        log_correct = -np.logaddexp(0., -signed)
        eps = obs["label_error"]
        nll = -log_correct if eps == 0 else -np.logaddexp(
            np.log(eps), np.log1p(-2 * eps) + log_correct)
        b = _gm(nll, groups[binary])
        residual = np.abs(distance[contact] / obs["contact_sigma"])
        huber = np.where(residual <= 1, .5 * residual ** 2, residual - .5)
        c = _gm(huber, groups[contact])
        weight = obs["weight"] / total_weight
        binary_total += weight * b; contact_total += weight * c
    return float(binary_total + contact_total)


def _score(problem, field):
    try:
        val = _fast_loss(problem, field) if CFG.get("fast_loss", False) else \
            float(score(problem, field)["loss_total"])
        return val if np.isfinite(val) else np.inf
    except Exception:
        return np.inf


def _initial(problem, rank):
    try:
        field = np.asarray(initialize(problem), float); val = _score(problem, field)
        if np.isfinite(val):
            return field, val
    except Exception:
        pass
    feasible = problem["domain_mask"] & ~problem["forbidden_mask"]
    m = _repair(problem, np.zeros(tuple(problem["shape"]), bool), rank)
    f = np.where(m, 1., -1.)
    return f, _score(problem, f)


def _proposal(problem, field, center, radius, coords, sp, length, feasible, rng):
    d2 = (((coords[0] - center[0]) * sp[0]) ** 2 +
          ((coords[1] - center[1]) * sp[1]) ** 2 +
          ((coords[2] - center[2]) * sp[2]) ** 2)
    mode = CFG["mode"]
    if mode == "shell":
        thick = max(float(sp.min()), float(CFG.get("shell_frac", .012)) * length)
        thick = min(thick, .70 * radius)
        patch = (d2 <= radius ** 2) & (d2 >= max(0., radius - thick) ** 2)
    elif mode == "halfspace":
        n = rng.normal(size=3); nn = float(np.linalg.norm(n))
        if nn <= 1e-12:
            return None
        n = n / nn
        dot = ((coords[0] - center[0]) * n[0] + (coords[1] - center[1]) * n[1] +
               (coords[2] - center[2]) * n[2])
        patch = (d2 <= radius ** 2) & (dot >= 0)
    else:
        patch = d2 <= radius ** 2
    patch &= feasible
    if not patch.any():
        return None
    prop = field.copy(); prop[patch] *= -1
    return prop


def solve(problem, budget, seed):
    started = time.monotonic()
    try:
        secs = float(budget.get("wall_time_seconds", 120))
    except Exception:
        secs = 120.
    if not np.isfinite(secs) or secs <= 0:
        secs = 120.
    small = int(np.prod(problem["shape"])) < 5000
    deadline = started + max(.5, (1.5 if small else float(CFG["time"]) * secs))
    rank = _rank(problem); field, cur = _initial(problem, rank); base = field.copy()
    feasible = problem["domain_mask"] & ~problem["forbidden_mask"]
    centers = np.argwhere(feasible)
    if not len(centers):
        return np.asarray(field, float)
    coords = np.indices(tuple(problem["shape"]), dtype=float)
    sp = np.asarray(problem["spacing"], float)
    length = float(np.linalg.norm(np.asarray(problem["shape"]) * sp))
    rng = np.random.default_rng(int(seed) + int(CFG.get("seed_offset", 0)))
    radii = tuple(float(r) for r in CFG["radii"])
    best, best_l = field.copy(), cur
    while time.monotonic() < deadline:
        center = centers[rng.integers(len(centers))]
        radius = max(float(sp.min()), length * radii[int(rng.integers(len(radii)))] *
                     rng.uniform(float(CFG["jitter"]), 1.0))
        prop = _proposal(problem, field, center, radius, coords, sp, length,
                         feasible, rng)
        if prop is None:
            continue
        if CFG.get("fast_legal", False):
            val = _score(problem, prop)
            if not np.isfinite(val):
                prop = np.where(_repair(problem, prop > 0, prop), 1., -1.)
                val = _score(problem, prop)
        else:
            prop = np.where(_repair(problem, prop > 0, prop), 1., -1.)
            val = _score(problem, prop)
        if not np.isfinite(val):
            continue
        accept = val < cur
        if CFG["mode"] == "anneal" and not accept:
            temp = (float(CFG.get("T0", 1e-5)) *
                    max(0.0, 1.0 - (time.monotonic() - started) /
                        max(deadline - started, 1e-6)) + 1e-12)
            accept = rng.random() < np.exp(min(0.0, (cur - val) / temp))
        if accept:
            field, cur = prop, val
            if val < best_l:
                best, best_l = prop.copy(), val
    for candidate in (best, base):
        try:
            canonicalize(problem, candidate)
            return np.asarray(candidate, dtype=np.float64)
        except Exception:
            pass
    try:
        fb = np.asarray(initialize(problem), float); canonicalize(problem, fb)
        return fb
    except Exception:
        m = np.zeros(tuple(problem["shape"]), bool); m[tuple(centers[0])] = True
        return np.where(m, 1., -1.).astype(np.float64)
