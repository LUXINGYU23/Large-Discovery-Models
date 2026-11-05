"""centre-weighted RFF ensemble"""
import time
import numpy as np
from scipy import ndimage
from scipy.spatial.distance import cdist
from tasks.copper_inversion.v2.baselines import initialize
from tasks.copper_inversion.v2.objective import score
from tasks.copper_inversion.v2.geometry import canonicalize

CFG = {'lam': 0.045, 'pw': 1.2, 'ensemble': True, 'restarts': 1, 'stage_a': 0.0, 'sigma': 0.0, 'temp': 0.001, 'soft': False, 'D': 128, 'ls': 1.05, 'ls_list': (0.75, 1.05, 1.4), 'ens_w': (0.25, 0.5, 0.25), 'seed_off': 233, 'qs': (0.1, 0.18, 0.26, 0.34, 0.42, 0.52, 0.64, 0.76)}


def _sec(budget):
    if isinstance(budget, dict):
        return float(budget.get("wall_time_seconds", 60.0))
    return float(budget)


def _field(mask):
    return np.where(mask, 1.0, -1.0).astype(np.float64)


def _score(problem, mask):
    try:
        return float(score(problem, _field(mask))["loss_total"])
    except Exception:
        return float("inf")


def _valid(problem, mask):
    try:
        canonicalize(problem, _field(mask))
        return True
    except Exception:
        return False


def _legal(problem, values, feasible):
    vals = np.asarray(values, float)
    mask = (vals > 0.0) & feasible
    cell = float(np.prod(np.asarray(problem["spacing"], float)))
    priors = problem["priors"]
    N = int(feasible.sum())
    bounds = priors.get("hard_volume_bounds")
    if bounds is None:
        lo, hi = 0, N
    else:
        lo = int(np.ceil(float(bounds[0]) / cell - 1e-9))
        hi = int(np.floor(float(bounds[1]) / cell + 1e-9))
        lo = max(0, min(lo, N))
        hi = max(0, min(hi, N))
        lo = min(lo, hi)
    maxc = priors.get("max_components")
    st = ndimage.generate_binary_structure(3, 1)
    for _ in range(40):
        if maxc is not None and mask.any():
            lab, ncomp = ndimage.label(mask, structure=st)
            if ncomp > int(maxc):
                sizes = np.bincount(lab.ravel())
                sizes[0] = 0
                keep = np.argsort(sizes)[::-1][:int(maxc)]
                keep = keep[keep > 0]
                mask = np.isin(lab, keep) if len(keep) else np.zeros_like(mask)
        npos = int(mask.sum())
        if npos > hi:
            pos = np.argwhere(mask)
            pv = vals[tuple(pos.T)]
            drop = pos[np.argsort(pv)[:npos - hi]]
            mask[tuple(drop.T)] = False
            continue
        if npos < lo:
            need = lo - npos
            cand = np.argwhere(feasible & ~mask)
            if len(cand) == 0:
                break
            cv = vals[tuple(cand.T)]
            dil = (ndimage.binary_dilation(mask, structure=st, iterations=1)
                   if mask.any() else mask)
            key = cv + 0.08 * dil[tuple(cand.T)].astype(float)
            if len(cand) > need:
                choose = np.argsort(key)[::-1][:need]
                mask[tuple(cand[choose].T)] = True
            else:
                mask |= feasible
            continue
        break
    return mask


def _observations(problem):
    pts = np.concatenate([np.asarray(o["points"], float) for o in problem["observations"]])
    labels = np.concatenate([np.asarray(o["labels"], float) for o in problem["observations"]])
    wgt = np.concatenate([np.full(len(o["labels"]), float(o["weight"]))
                          for o in problem["observations"]])
    eps = np.concatenate([np.full(len(o["labels"]), float(o["label_error"]))
                          for o in problem["observations"]])
    return pts, labels, wgt, eps


def _basis(problem, rng, cfg, ls):
    feasible = problem["domain_mask"] & ~problem["forbidden_mask"]
    spacing = np.asarray(problem["spacing"], float)
    origin = np.asarray(problem["origin"], float)
    idx = np.argwhere(feasible)
    P = origin + (idx + 0.5) * spacing
    mu = P.mean(0)
    sd = P.std(0) + 1e-12
    D = max(8, int(cfg["D"]))
    if cfg.get("multiscale"):
        h = max(4, D // 2)
        W1 = rng.normal(0.0, 1.0 / ls, (3, h))
        b1 = rng.uniform(0.0, 2.0 * np.pi, h)
        W2 = rng.normal(0.0, 1.0 / (2.2 * ls), (3, h))
        b2 = rng.uniform(0.0, 2.0 * np.pi, h)

        def design(Q):
            X = (np.asarray(Q, float) - mu) / sd
            F = (np.sqrt(2.0 / h) * np.cos(X @ W1 + b1) +
                 np.sqrt(2.0 / h) * np.cos(X @ W2 + b2)) / np.sqrt(2.0)
            return np.concatenate([F, X, np.ones((len(X), 1))], axis=1)
    else:
        W = rng.normal(0.0, 1.0 / ls, (3, D))
        b = rng.uniform(0.0, 2.0 * np.pi, D)

        def design(Q):
            X = (np.asarray(Q, float) - mu) / sd
            F = np.sqrt(2.0 / D) * np.cos(X @ W + b)
            return np.concatenate([F, X, np.ones((len(X), 1))], axis=1)

    return design, P, idx


def _rff_fit(problem, rng, cfg, ls):
    design, P, idx = _basis(problem, rng, cfg, ls)
    pts, labels, wgt, eps = _observations(problem)
    A = design(pts)
    if cfg.get("soft"):
        target = np.where(labels > 0.75, 1.0 - eps,
                          np.where(labels < 0.25, eps, 0.5))
        wgt = wgt * np.maximum(0.25, 1.0 - 2.0 * eps)
    else:
        target = labels.astype(float)
    row = wgt * np.where(target > 0.55, float(cfg["pw"]), 1.0)
    sw = np.sqrt(np.maximum(row, 1e-9))
    Aw = A * sw[:, None]
    yw = target * sw
    n = Aw.shape[1]
    reg = float(cfg["lam"]) * np.eye(n)
    reg[-1, -1] *= 0.1
    try:
        coef = np.linalg.solve(Aw.T @ Aw + reg, Aw.T @ yw)
    except Exception:
        coef = np.linalg.lstsq(Aw.T @ Aw + reg, Aw.T @ yw, rcond=None)[0]
    return design, P, idx, coef


def _predict(problem, rng, cfg):
    feasible = problem["domain_mask"] & ~problem["forbidden_mask"]
    shape = tuple(problem["shape"])
    if cfg.get("kernel"):
        spacing = np.asarray(problem["spacing"], float)
        origin = np.asarray(problem["origin"], float)
        idx = np.argwhere(feasible)
        P = origin + (idx + 0.5) * spacing
        pts, labels, wgt, eps = _observations(problem)
        mu = pts.mean(0)
        sd = pts.std(0) + 1e-12
        Z = (pts - mu) / sd
        ls = float(cfg["ls"])
        K = np.exp(-cdist(Z, Z, "sqeuclidean") / (2.0 * ls * ls))
        sw = np.sqrt(np.maximum(wgt, 1e-9))
        A = K * sw[:, None]
        target = labels.astype(float)
        try:
            coef = np.linalg.solve(A.T @ A + float(cfg["lam"]) * np.eye(len(Z)),
                                   A.T @ (target * sw))
        except Exception:
            coef = np.linalg.lstsq(A.T @ A + float(cfg["lam"]) * np.eye(len(Z)),
                                   A.T @ (target * sw), rcond=None)[0]
        zf = np.empty(len(P), float)
        Qn = (P - mu) / sd
        batch = 16000
        for a in range(0, len(P), batch):
            Kq = np.exp(-cdist(Qn[a:a + batch], Z, "sqeuclidean") / (2.0 * ls * ls))
            zf[a:a + batch] = Kq @ coef
        z = np.full(shape, -1.0, float)
        z[tuple(idx.T)] = zf
        return z, P, idx, None
    if cfg.get("ensemble"):
        zs = []
        P = idx = None
        for ls in cfg["ls_list"]:
            design, P, idx, coef = _rff_fit(problem, rng, cfg, float(ls))
            zf = np.asarray(design(P) @ coef, float)
            zs.append((zf - zf.mean()) / (zf.std() + 1e-12))
        if cfg.get("ens_w"):
            zf = np.average(zs, axis=0, weights=cfg["ens_w"])
        else:
            zf = np.mean(zs, axis=0)
        z = np.full(shape, -1.0, float)
        z[tuple(idx.T)] = zf
        return z, P, idx, None
    design, P, idx, coef = _rff_fit(problem, rng, cfg, float(cfg["ls"]))
    z = np.full(shape, -1.0, float)
    z[tuple(idx.T)] = np.asarray(design(P) @ coef, float)
    return z, P, idx, design


def solve(problem, budget, seed):
    started = time.monotonic()
    seconds = _sec(budget)
    if not np.isfinite(seconds) or seconds <= 0:
        seconds = 60.0
    deadline = started + 0.94 * seconds
    rng = np.random.default_rng((int(seed) + int(CFG.get("seed_off", 0))) & 0x7fffffff)
    feasible = problem["domain_mask"] & ~problem["forbidden_mask"]
    shape = tuple(problem["shape"])
    try:
        fallback = initialize(problem)
        best_mask = fallback > 0
        best_loss = float(score(problem, fallback)["loss_total"])
    except Exception:
        fallback = None
        best_mask = np.zeros(shape, bool)
        best_loss = float("inf")

    restarts = max(1, int(CFG.get("restarts", 1)))
    for r in range(restarts):
        if time.monotonic() >= deadline - 0.5:
            break
        now = time.monotonic()
        r_end = started + (deadline - started) * (r + 1) / restarts
        try:
            z, P, idx, design = _predict(problem, rng, CFG)
        except Exception:
            continue
        zf = z[tuple(idx.T)]
        local_mask = best_mask.copy()
        local_loss = best_loss
        chosen = float(np.quantile(zf, 0.42))
        for q in list(CFG["qs"]) + [0.0]:
            if time.monotonic() >= r_end - 0.25:
                break
            t = float(np.quantile(zf, q)) if q else 0.0
            m = _legal(problem, z - t, feasible)
            lval = _score(problem, m)
            if lval < local_loss:
                local_loss = lval
                local_mask = m.copy()
                chosen = t
        if local_loss < best_loss:
            best_loss, best_mask = local_loss, local_mask.copy()
        cur_z = z - chosen
        cur_mask = _legal(problem, cur_z, feasible)
        cur_loss = _score(problem, cur_mask)
        if cur_loss < best_loss:
            best_loss, best_mask = cur_loss, cur_mask.copy()
        F = None
        if design is not None:
            try:
                F = np.asarray(design(P), float)
            except Exception:
                F = None
        if F is not None:
            a_end = min(r_end - 0.15,
                        time.monotonic() + float(CFG["stage_a"]) * (r_end - time.monotonic()))
            while time.monotonic() < a_end:
                prog = min(1.0, max(0.0, (time.monotonic() - now) /
                                    max(1e-6, r_end - now)))
                sigma = float(CFG["sigma"]) * (1.0 - 0.85 * prog) + 0.003
                dw = rng.normal(0.0, sigma, F.shape[1])
                znew = cur_z.copy()
                znew[tuple(idx.T)] += F @ dw
                mnew = _legal(problem, znew, feasible)
                lnew = _score(problem, mnew)
                temp = float(CFG["temp"]) * (1.0 - prog) + 2e-5
                accept = lnew <= cur_loss
                if not accept and np.isfinite(lnew):
                    accept = rng.random() < np.exp(max(-60.0, -(lnew - cur_loss) / temp))
                if accept:
                    cur_z, cur_mask, cur_loss = znew, mnew, lnew
                if lnew < best_loss:
                    best_loss, best_mask = lnew, mnew.copy()
        cur_mask = best_mask.copy()
        cur_loss = best_loss
        spacing = np.asarray(problem["spacing"], float)
        coords = np.indices(shape, dtype=float)
        st = ndimage.generate_binary_structure(3, 1)
        band = None
        age = 10 ** 9
        while time.monotonic() < r_end - 0.08:
            age += 1
            if band is None or age > 35:
                dil = ndimage.binary_dilation(cur_mask, structure=st, iterations=2)
                ero = ndimage.binary_erosion(cur_mask, structure=st, iterations=1)
                band = np.argwhere((dil & ~ero) & feasible)
                if len(band) == 0:
                    band = np.argwhere(feasible)
                age = 0
            c = band[rng.integers(len(band))]
            rad = max(float(spacing.min()) * 0.8,
                      rng.uniform(0.8, 1.8) * float(np.mean(spacing)))
            patch = (((coords[0] - c[0]) * spacing[0]) ** 2 +
                     ((coords[1] - c[1]) * spacing[1]) ** 2 +
                     ((coords[2] - c[2]) * spacing[2]) ** 2) <= rad * rad
            patch &= feasible
            if not patch.any():
                continue
            prop = cur_mask.copy()
            prop[patch] = ~prop[patch]
            if not _valid(problem, prop):
                continue
            lnew = _score(problem, prop)
            if np.isfinite(lnew) and lnew <= cur_loss:
                cur_mask, cur_loss = prop, lnew
            if np.isfinite(lnew) and lnew < best_loss:
                best_loss, best_mask = lnew, prop.copy()

    if best_mask is None or not np.isfinite(best_loss):
        if fallback is not None:
            return fallback
        return _field(np.zeros(shape, bool))
    try:
        out = _field(best_mask)
        canonicalize(problem, out)
        return out
    except Exception:
        if fallback is not None:
            return fallback
        return _field(np.zeros(shape, bool))
