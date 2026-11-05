"""Copper v2 r10 c5: ellipsoid shape-space moves (competitive alternative/control)."""
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

CFG = {'search': 'anneal', 'move': 'sphere', 'time': .94, 'anneal_frac': .68,
       'radii': (.05, .075, .105, .13, .17), 'jitter': .35, 'T0': 3e-5,
       'seed_offset': 11, 'slab_frac': .03, 'sphere_mix': .5, 'cycles': 3,
       'rbf': False, 'polish': True, 'temp_mults': (1., .45, 1.8)}
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
    if mask.any():
        vals = (-ndimage.distance_transform_edt(~mask) * 1e9 +
                np.asarray(rank, float))[tuple(idx.T)]
    else:
        vals = np.asarray(rank, float)[tuple(idx.T)]
    out = mask.copy()
    out[tuple(idx[np.argsort(vals)[::-1][:need]].T)] = True
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


def _grid(problem):
    sp = np.asarray(problem["spacing"], float); org = np.asarray(problem["origin"], float)
    return np.stack(np.meshgrid(*[org[i] + (np.arange(n) + .5) * sp[i]
                                  for i, n in enumerate(problem["shape"])],
                                 indexing="ij"), axis=-1)


def _rank(problem):
    xyz = _grid(problem)
    pts = np.concatenate([np.asarray(o["points"], float) for o in problem["observations"]])
    lab = np.concatenate([np.asarray(o["labels"], float) for o in problem["observations"]])
    pos, neg = pts[lab == 1], pts[lab == 0]
    if len(pos):
        dp = cKDTree(pos).query(xyz.reshape(-1, 3))[0].reshape(tuple(problem["shape"]))
        if len(neg):
            dn = cKDTree(neg).query(xyz.reshape(-1, 3))[0].reshape(tuple(problem["shape"]))
            return dn - dp
        return -dp
    center = np.asarray(problem["origin"], float) + .5 * np.asarray(problem["shape"]) * np.asarray(problem["spacing"], float)
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
        val = _fast_loss(problem, field)
        return val if np.isfinite(val) else np.inf
    except Exception:
        return np.inf


def _rbf_seed(problem, rank, deadline):
    try:
        from scipy.interpolate import RBFInterpolator
        pts = np.concatenate([np.asarray(o["points"], float) for o in problem["observations"]])
        lab = np.concatenate([np.asarray(o["labels"], float) for o in problem["observations"]])
        if len(pts) < 4 or not np.any(lab == 1) or not np.any(lab == 0):
            return None
        og = np.asarray(problem["origin"], float)
        X = pts - og
        scale = np.maximum(np.ptp(X, axis=0), 1e-9)
        val = np.where(lab == .5, 0., np.where(lab == 1., 1., -1.))
        G = (_grid(problem) - og) / scale
        flat = G.reshape(-1, 3)
        best = None
        for kind, sm in (("thin_plate_spline", 1e-2), ("thin_plate_spline", 1e-1),
                         ("multiquadric", 1e-2), ("inverse_quadratic", 1e-2)):
            if time.monotonic() >= deadline:
                break
            try:
                rbf = RBFInterpolator(X / scale, val, kernel=kind, smoothing=sm, degree=1)
                out = rbf(flat).reshape(tuple(problem["shape"]))
            except Exception:
                continue
            if not np.isfinite(out).all():
                continue
            for off in (0., .15, -.15, .35, -.35):
                m = _repair(problem, out > off, rank)
                f = np.where(m, 1., -1.)
                v = _score(problem, f)
                if np.isfinite(v) and (best is None or v < best[0]):
                    best = (v, f)
        return best
    except Exception:
        return None


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


def _proposal(problem, field, rank, grad, center, radius, coords, sp, length,
              feasible, rng, move):
    d2 = (((coords[0] - center[0]) * sp[0]) ** 2 +
          ((coords[1] - center[1]) * sp[1]) ** 2 +
          ((coords[2] - center[2]) * sp[2]) ** 2)
    if move == 'topology':
        lab, ncomp = ndimage.label(field > 0, structure=_ST)
        if ncomp > 1 and rng.random() < .35:
            ids = np.arange(1, ncomp + 1)
            sizes = np.bincount(lab.ravel())[1:]
            pick = ids[int(np.argmin(sizes))] if rng.random() < .5 else ids[rng.integers(ncomp)]
            prop = field.copy(); prop[lab == pick] = -1.
            return prop
        pos = center
        axes = np.array([radius * rng.uniform(.55, 1.35) for _ in range(3)])
        patch = (((coords[0] - pos[0]) * sp[0] / axes[0]) ** 2 +
                 ((coords[1] - pos[1]) * sp[1] / axes[1]) ** 2 +
                 ((coords[2] - pos[2]) * sp[2] / axes[2]) ** 2) <= 1.
        patch &= feasible
        if not patch.any():
            return None
        prop = field.copy(); prop[patch] = 1.
        return prop
    if move == 'ellipsoid':
        axes = np.array([radius * rng.uniform(.55, 1.45) for _ in range(3)])
        patch = (((coords[0] - center[0]) * sp[0] / axes[0]) ** 2 +
                 ((coords[1] - center[1]) * sp[1] / axes[1]) ** 2 +
                 ((coords[2] - center[2]) * sp[2] / axes[2]) ** 2) <= 1.
    elif move == 'slab' and rng.random() >= CFG.get('sphere_mix', .5):
        idx = tuple(np.rint(center).astype(int))
        n = np.array([grad[0][idx], grad[1][idx], grad[2][idx]], float)
        ln = float(np.linalg.norm(n))
        if ln <= 1e-12:
            patch = d2 <= radius ** 2
        else:
            n = n / ln
            dot = ((coords[0] - center[0]) * sp[0] * n[0] +
                   (coords[1] - center[1]) * sp[1] * n[1] +
                   (coords[2] - center[2]) * sp[2] * n[2])
            thick = max(float(sp.min()), float(CFG.get('slab_frac', .03)) * length)
            if rng.random() < .5:
                patch = (d2 <= radius ** 2) & (np.abs(dot) <= thick)
            else:
                patch = (d2 <= radius ** 2) & (dot >= 0.)
    else:
        patch = d2 <= radius ** 2
    patch &= feasible
    if not patch.any():
        return None
    prop = field.copy(); prop[patch] *= -1.
    return prop


def _anneal(problem, rank, field, cur, start, end, rng, t0, coords, sp,
            length, feasible, centers, grad, move, best, best_l):
    while time.monotonic() < end:
        center = centers[rng.integers(len(centers))]
        radius = max(float(sp.min()), length * CFG['radii'][int(rng.integers(len(CFG['radii'])))] *
                     rng.uniform(float(CFG['jitter']), 1.))
        prop = _proposal(problem, field, rank, grad, center, radius, coords, sp,
                         length, feasible, rng, move)
        if prop is None:
            continue
        val = _score(problem, prop)
        if not np.isfinite(val):
            prop = np.where(_repair(problem, prop > 0, rank), 1., -1.)
            val = _score(problem, prop)
        if not np.isfinite(val):
            continue
        accept = val < cur
        if not accept:
            temp = t0 * max(0., 1. - (time.monotonic() - start) /
                            max(end - start, 1e-6)) + 1e-12
            accept = rng.random() < np.exp(min(0., (cur - val) / temp))
        if accept:
            field, cur = prop, val
            if val < best_l:
                best, best_l = prop.copy(), val
    return field, cur, best, best_l


def _polish(problem, rank, field, cur, end, rng, coords, sp, feasible, best, best_l):
    while time.monotonic() < end:
        mask = field > 0
        boundary = ((ndimage.binary_dilation(mask, structure=_ST) & ~mask) |
                    (mask & ndimage.binary_dilation(~mask, structure=_ST))) & feasible
        idx = np.argwhere(boundary)
        if not len(idx):
            break
        c = idx[rng.integers(len(idx))]
        r = int(rng.integers(1, 3))
        sl = tuple(slice(max(0, int(c[i]) - r), min(field.shape[i], int(c[i]) + r + 1))
                   for i in range(3))
        prop = field.copy(); prop[sl] *= -1.
        val = _score(problem, prop)
        if not np.isfinite(val):
            continue
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
    work_end = started + max(.5, (.35 if small else float(CFG["time"]) * secs))
    rank = _rank(problem); field, cur = _initial(problem, rank); best = field.copy(); best_l = cur
    try:
        if CFG.get('rbf', False):
            rb = _rbf_seed(problem, rank, started + .18 * secs)
            if rb is not None and rb[0] < best_l:
                field, cur = rb; best, best_l = rb[1].copy(), rb[0]
    except Exception:
        pass
    feasible = problem["domain_mask"] & ~problem["forbidden_mask"]
    centers = np.argwhere(feasible)
    if not len(centers):
        return np.asarray(field, float)
    coords = np.indices(tuple(problem["shape"]), dtype=float)
    sp = np.asarray(problem["spacing"], float)
    length = float(np.linalg.norm(np.asarray(problem["shape"]) * sp))
    gs = tuple(1. / np.maximum(sp, 1e-12))
    grad = np.gradient(np.asarray(rank, float), *sp)
    rng = np.random.default_rng(int(seed) + int(CFG.get('seed_offset', 0)))
    search = CFG.get('search', 'anneal')
    move = CFG.get('move', 'sphere')
    if search == 'cycle':
        cyc = int(CFG.get('cycles', 3)); seg = max(.1, (work_end - started) / cyc)
        for c in range(cyc):
            end = min(work_end, started + seg * (c + 1))
            field, cur = best.copy(), best_l
            tm = float(CFG.get('temp_mults', (1., .45, 1.8))[c % 3])
            field, cur, best, best_l = _anneal(
                problem, rank, field, cur, time.monotonic(), end,
                np.random.default_rng(int(seed) + int(CFG.get('seed_offset', 0)) + 97 * c),
                float(CFG['T0']) * tm, coords, sp, length, feasible, centers, grad, move,
                best, best_l)
            if c == cyc - 1 and CFG.get('polish', False) and time.monotonic() < end:
                best, best_l = _polish(problem, rank, best, best_l, end, rng, coords,
                                       sp, feasible, best, best_l)
    elif search == 'portfolio':
        specs = (('sphere', 1.4e-5), ('slab', 3e-5), ('topology', 1e-5))
        seg = max(.1, (work_end - started) / (len(specs) + .8))
        for c, (mv, tj) in enumerate(specs):
            end = min(work_end, started + seg * (c + 1))
            f0 = best.copy() if c == 0 else np.asarray(initialize(problem), float)
            l0 = _score(problem, f0)
            if not np.isfinite(l0):
                f0, l0 = best.copy(), best_l
            f0, l0, best, best_l = _anneal(
                problem, rank, f0, l0, time.monotonic(), end,
                np.random.default_rng(int(seed) + int(CFG.get('seed_offset', 0)) + 131 * c),
                tj, coords, sp, length, feasible, centers, grad, mv, best, best_l)
        if CFG.get('polish', False):
            best, best_l = _polish(problem, rank, best, best_l, work_end, rng, coords,
                                   sp, feasible, best, best_l)
    else:
        anneal_end = (started + float(CFG.get('anneal_frac', .68)) * secs
                      if CFG.get('polish', False) else work_end)
        anneal_end = min(anneal_end, work_end)
        field, cur, best, best_l = _anneal(
            problem, rank, field, cur, started, anneal_end, rng, float(CFG['T0']),
            coords, sp, length, feasible, centers, grad, move, best, best_l)
        if CFG.get('polish', False) and time.monotonic() < work_end:
            best, best_l = _polish(problem, rank, best, best_l, work_end, rng, coords,
                                   sp, feasible, best, best_l)
    for candidate in (best, field):
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
