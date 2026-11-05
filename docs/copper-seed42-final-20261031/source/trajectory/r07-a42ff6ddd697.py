"""Copper v2 round-6 candidate: baseline-seeded objective-greedy patch refinement."""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import time
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from tasks.copper_inversion.v2.objective import score
from tasks.copper_inversion.v2.geometry import canonicalize
from tasks.copper_inversion.v2.baselines import initialize

# Replaced per candidate by the generator.
CFG = {'time': 0.68, 'radii': (0.05, 0.075, 0.105, 0.13, 0.17), 'jitter': 0.35, 'phases': None, 'seed_offset': 0, 'note': 'upper_17_68'}

_ST = ndimage.generate_binary_structure(3, 1)


def _prune(mask, maximum):
    if maximum is None or not mask.any():
        return mask
    lab, n = ndimage.label(mask, structure=_ST)
    if n <= maximum:
        return mask
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    keep = np.argsort(sizes[1:])[::-1][:maximum] + 1
    return np.isin(lab, keep)


def _add(mask, rank, feasible, need):
    need = int(min(max(need, 0), int((feasible & ~mask).sum())))
    if need <= 0:
        return mask
    idx = np.argwhere(feasible & ~mask)
    if not len(idx):
        return mask
    if mask.any():
        dist = ndimage.distance_transform_edt(~mask)
        score_rank = -dist * 1e9 + np.asarray(rank, float)
    else:
        score_rank = np.asarray(rank, float)
    vals = score_rank[tuple(idx.T)]
    out = mask.copy()
    out[tuple(idx[np.argsort(vals)[::-1][:need]].T)] = True
    return out


def _remove(mask, rank, excess):
    excess = int(max(excess, 0))
    if excess <= 0 or not mask.any():
        return mask
    idx = np.argwhere(mask)
    vals = np.asarray(rank, float)[tuple(idx.T)]
    out = mask.copy()
    out[tuple(idx[np.argsort(vals)[:excess]].T)] = False
    return out


def _repair(problem, mask, rank):
    feasible = problem["domain_mask"] & ~problem["forbidden_mask"]
    pri = problem["priors"]
    m = np.asarray(mask, bool) & feasible
    maxc = pri["max_components"]
    m = _prune(m, maxc)
    cell = float(np.prod(np.asarray(problem["spacing"], float)))
    hb = pri["hard_volume_bounds"]
    lo = 0 if hb is None else int(np.ceil(hb[0] / cell - 1e-9))
    hi = int(feasible.sum()) if hb is None else int(np.floor(hb[1] / cell + 1e-9))
    lo = max(0, min(lo, int(feasible.sum())))
    hi = max(lo, min(hi, int(feasible.sum())))

    fw, fv = pri["fragment_weight"], pri["fragment_volume"]
    if fw > 0 and fv > 0 and m.any():
        lab, _ = ndimage.label(m, structure=_ST)
        sizes = np.bincount(lab.ravel()) * cell
        sizes[0] = 0
        keep = np.where(sizes >= fv)[0]
        keep = keep[keep > 0]
        m2 = np.isin(lab, keep)
        if int(m2.sum()) >= lo:
            m = m2

    for _ in range(6):
        m = _prune(m, maxc)
        n = int(m.sum())
        if n < lo:
            m = _add(m, rank, feasible, lo - n)
        elif n > hi:
            m = _remove(m, rank, n - hi)
        else:
            break

    vb, vw = pri["volume_bounds"], pri["volume_weight"]
    if vw > 0 and vb is not None:
        n = int(m.sum())
        slo = max(lo, min(int(np.ceil(vb[0] / cell - 1e-9)), hi))
        shi = min(hi, max(lo, int(np.floor(vb[1] / cell + 1e-9))))
        if n < slo:
            m = _add(m, rank, feasible, slo - n)
        elif n > shi:
            m = _remove(m, rank, n - shi)

    m = _prune(m, maxc)
    if int(m.sum()) < lo:
        m = _add(m, rank, feasible, lo - int(m.sum()))
        m = _prune(m, maxc)
    return m


def _rank(problem):
    sp = np.asarray(problem["spacing"], float)
    org = np.asarray(problem["origin"], float)
    xyz = np.stack(np.meshgrid(*[org[i] + (np.arange(n) + .5) * sp[i]
                                 for i, n in enumerate(problem["shape"])], indexing="ij"), axis=-1)
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


def _score(problem, field):
    try:
        val = float(score(problem, field)["loss_total"])
        return val if np.isfinite(val) else np.inf
    except Exception:
        return np.inf


def _initial(problem, rank):
    try:
        field = np.asarray(initialize(problem), float)
        val = _score(problem, field)
        if np.isfinite(val):
            return field, val
    except Exception:
        pass
    feasible = problem["domain_mask"] & ~problem["forbidden_mask"]
    m = _repair(problem, np.zeros(tuple(problem["shape"]), bool), rank)
    f = np.where(m, 1., -1.)
    return f, _score(problem, f)


def _search(problem, field, cur, rank, feasible, centers, coords, sp, length,
            deadline, rng):
    best = field.copy()
    best_l = cur
    start = time.monotonic()
    total = max(deadline - start, 1e-6)
    phases = CFG["phases"]
    if phases is None:
        phases = ((1.0, CFG["radii"]),)
    for frac_end, radii in phases:
        phase_end = start + total * float(frac_end)
        while time.monotonic() < min(phase_end, deadline):
            center = centers[rng.integers(len(centers))]
            rfrac = float(radii[int(rng.integers(len(radii)))])
            radius = max(float(sp.min()), length * rfrac *
                         rng.uniform(float(CFG["jitter"]), 1.0))
            patch = (((coords[0] - center[0]) * sp[0]) ** 2 +
                     ((coords[1] - center[1]) * sp[1]) ** 2 +
                     ((coords[2] - center[2]) * sp[2]) ** 2) <= radius ** 2
            patch &= feasible
            if not patch.any():
                continue
            prop = field.copy()
            prop[patch] *= -1
            m = _repair(problem, prop > 0, prop)
            prop = np.where(m, 1., -1.)
            # score() canonicalizes internally; avoid a redundant distance transform.
            val = _score(problem, prop)
            if val < cur:
                field, cur = prop, val
                if val < best_l:
                    best, best_l = prop.copy(), val
    return best, best_l


def solve(problem, budget, seed):
    started = time.monotonic()
    try:
        secs = float(budget.get("wall_time_seconds", 120))
    except Exception:
        secs = 120.
    if not np.isfinite(secs) or secs <= 0:
        secs = 120.
    small = int(np.prod(problem["shape"])) < 5000
    cap = 1.5 if small else float(CFG["time"]) * secs
    deadline = started + max(.5, cap)
    try:
        rank = _rank(problem)
        field, cur = _initial(problem, rank)
        feasible = problem["domain_mask"] & ~problem["forbidden_mask"]
        centers = np.argwhere(feasible)
        if not len(centers):
            canonicalize(problem, field)
            return np.asarray(field, float)
        coords = np.indices(tuple(problem["shape"]), dtype=float)
        sp = np.asarray(problem["spacing"], float)
        length = float(np.linalg.norm(np.asarray(problem["shape"]) * sp))
        best, best_l = field.copy(), cur
        rng = np.random.default_rng(int(seed) + int(CFG["seed_offset"]))
        f, l = _search(problem, field, cur, rank, feasible, centers, coords,
                       sp, length, deadline, rng)
        if l < best_l:
            best, best_l = f, l
        try:
            canonicalize(problem, best)
            return best.astype(np.float64)
        except Exception:
            canonicalize(problem, field)
            return np.asarray(field, float)
    except Exception:
        try:
            fallback = np.asarray(initialize(problem), float)
            canonicalize(problem, fallback)
            return fallback
        except Exception:
            feasible = problem["domain_mask"] & ~problem["forbidden_mask"]
            idx = np.argwhere(feasible)
            m = np.zeros(tuple(problem["shape"]), bool)
            if len(idx):
                m[tuple(idx[0])] = True
            return np.where(m, 1., -1.).astype(np.float64)
