"""Copper v2 candidate: baseline-seeded objective-greedy patch refinement."""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import time
import numpy as np
from scipy import ndimage
from tasks.copper_inversion.v2.objective import score
from tasks.copper_inversion.v2.geometry import canonicalize
from tasks.copper_inversion.v2.baselines import initialize

_STRUCT = ndimage.generate_binary_structure(3, 1)

# ---- configuration (replaced per candidate) ----
TIME_FRAC = 0.68
RADIUS_FRAC = 0.075
JITTER_LOW = 0.35
SCHEDULE = None
FAST = True
SEED_OFFSET = 0
N_RUNS = 1


def _prune(mask, maximum):
    if maximum is None or not mask.any():
        return mask
    lab, n = ndimage.label(mask, structure=_STRUCT)
    if n <= maximum:
        return mask
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    keep = np.argsort(sizes[1:])[::-1][:maximum] + 1
    return np.isin(lab, keep)


def _repair(problem, mask, rank):
    feasible = problem["domain_mask"] & ~problem["forbidden_mask"]
    m = np.asarray(mask, bool) & feasible
    pri = problem["priors"]
    m = _prune(m, pri["max_components"])
    cell = float(np.prod(np.asarray(problem["spacing"], float)))
    bounds = pri["hard_volume_bounds"]
    lo = 0 if bounds is None else int(np.ceil(bounds[0] / cell - 1e-9))
    hi = int(feasible.sum()) if bounds is None else int(np.floor(bounds[1] / cell + 1e-9))
    for _ in range(4):
        n = int(m.sum())
        if n > hi:
            idx = np.argwhere(m)
            vals = np.asarray(rank, float)[tuple(idx.T)]
            m[tuple(idx[np.argsort(vals)[:n - hi]].T)] = False
            m = _prune(m, pri["max_components"])
        elif n < lo:
            idx = np.argwhere(feasible & ~m)
            if not len(idx):
                break
            if m.any():
                dist = ndimage.distance_transform_edt(~m)
                add = -dist * 1e9 + np.asarray(rank, float)
            else:
                add = np.asarray(rank, float)
            vals = add[tuple(idx.T)]
            m[tuple(idx[np.argsort(vals)[::-1][:lo - n]].T)] = True
            m = _prune(m, pri["max_components"])
        else:
            break
    return m


def _rank(problem):
    sp = np.asarray(problem["spacing"], float)
    org = np.asarray(problem["origin"], float)
    xyz = np.stack(np.meshgrid(*[org[i] + (np.arange(n) + .5) * sp[i]
                                 for i, n in enumerate(problem["shape"])], indexing="ij"), axis=-1)
    pts = np.concatenate([o["points"] for o in problem["observations"]])
    lab = np.concatenate([o["labels"] for o in problem["observations"]])
    from scipy.spatial import cKDTree
    pos, neg = pts[lab == 1], pts[lab == 0]
    if len(pos):
        dp = cKDTree(pos).query(xyz.reshape(-1, 3))[0].reshape(problem["shape"])
        if len(neg):
            dn = cKDTree(neg).query(xyz.reshape(-1, 3))[0].reshape(problem["shape"])
            return dn - dp
        return -dp
    center = org + .5 * np.asarray(problem["shape"]) * sp
    return -np.linalg.norm(xyz - center, axis=-1)


def _score(problem, field):
    try:
        v = float(score(problem, field)["loss_total"])
        return v if np.isfinite(v) else np.inf
    except Exception:
        return np.inf


def _search(problem, field, cur, rank, feasible, centers, coords, sp, length,
            deadline, rng, schedule):
    best = field.copy()
    best_l = cur
    phase_start = time.monotonic()
    total = max(deadline - phase_start, 1e-6)
    if schedule is None:
        schedule = ((1.0, RADIUS_FRAC),)
    for frac_end, rfrac in schedule:
        phase_end = phase_start + total * frac_end
        while time.monotonic() < min(phase_end, deadline):
            center = centers[rng.integers(len(centers))]
            radius = max(float(sp.min()), length * rfrac * rng.uniform(JITTER_LOW, 1.0))
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
            if FAST:
                val = _score(problem, prop)
            else:
                try:
                    canonicalize(problem, prop)
                    val = _score(problem, prop)
                except Exception:
                    continue
            if val < cur:
                field, cur = prop, val
                if val < best_l:
                    best, best_l = prop.copy(), val
    return best, best_l


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


def solve(problem, budget, seed):
    started = time.monotonic()
    try:
        secs = float(budget.get("wall_time_seconds", 120))
    except Exception:
        secs = 120.
    if not np.isfinite(secs) or secs <= 0:
        secs = 120.
    small = int(np.prod(problem["shape"])) < 5000
    cap = 1.5 if small else TIME_FRAC * secs
    deadline = started + max(.5, cap)
    rank = _rank(problem)
    field, cur = _initial(problem, rank)
    feasible = problem["domain_mask"] & ~problem["forbidden_mask"]
    centers = np.argwhere(feasible)
    if not len(centers):
        return np.asarray(field, float)
    coords = np.indices(tuple(problem["shape"]), dtype=float)
    sp = np.asarray(problem["spacing"], float)
    length = float(np.linalg.norm(np.asarray(problem["shape"]) * sp))
    best, best_l = field.copy(), cur
    search_start = time.monotonic()
    span = max(deadline - search_start, 1e-6)
    runs = max(int(N_RUNS), 1)
    for i in range(runs):
        run_deadline = search_start + span * (i + 1) / runs
        rng = np.random.default_rng(int(seed) + int(SEED_OFFSET) + 1009 * i)
        f, l = _search(problem, field, cur, rank, feasible, centers, coords,
                       sp, length, run_deadline, rng, SCHEDULE)
        if l < best_l:
            best, best_l = f.copy(), l
    try:
        canonicalize(problem, best)
        return best.astype(np.float64)
    except Exception:
        return np.asarray(field, float)
