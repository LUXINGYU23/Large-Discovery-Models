"""Geological inversion by bounded implicit-surface model selection.

The solver builds several smooth implicit fields that honour the drillhole
labels (0 = exterior, 1 = interior) and force outcrop/contact samples near the
zero level set.  Each field is passed through the same signed-distance
reinitialisation used by the host, its observation residual is measured, and
the best field is refined by a bounded pattern search.  No target array is
stored: everything is derived from the problem inputs.
"""

import itertools
import time

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.interpolate import RBFInterpolator

try:
    import skfmm
except Exception:  # pragma: no cover
    skfmm = None


# ----------------------------------------------------------------------------
# input helpers
# ----------------------------------------------------------------------------
def _grid(problem):
    shape = tuple(int(n) for n in problem["shape"])
    origin = np.asarray(problem["origin"], dtype=float)
    spacing = np.asarray(problem["spacing"], dtype=float)
    axes = [origin[i] + (np.arange(n, dtype=float) + 0.5) * spacing[i]
            for i, n in enumerate(shape)]
    mesh = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    return shape, origin, spacing, mesh


def _gather(problem):
    coords, values = [], []
    origin = np.asarray(problem["origin"], dtype=float)
    spacing = np.asarray(problem["spacing"], dtype=float)
    for obs in problem.get("observations", []):
        vals = np.asarray(obs["values"], dtype=float)
        mask = np.isfinite(vals)
        if mask.any():
            coords.append(origin + (np.argwhere(mask) + 0.5) * spacing)
            values.append(vals[mask])
    if not coords:
        return np.zeros((0, len(problem["shape"]))), np.zeros(0)
    return np.concatenate(coords, 0), np.concatenate(values, 0)


def _norm_params(mesh, pts):
    flat = mesh.reshape(-1, mesh.shape[-1])
    lo, hi = flat.min(0), flat.max(0)
    if len(pts):
        centre = pts.mean(0)
        span = float(np.max(pts.max(0) - pts.min(0)))
        if span <= 1e-12:
            span = float(np.max(hi - lo))
    else:
        centre = 0.5 * (lo + hi)
        span = float(np.max(hi - lo))
    if not np.isfinite(span) or span <= 0:
        span = 1.0
    return centre, span


def _features(u, degree):
    u = np.asarray(u, dtype=float)
    if u.ndim == 1:
        u = u[None, :]
    cols = [np.ones(len(u))]
    dim = u.shape[1]
    for d in range(1, degree + 1):
        for combo in itertools.combinations_with_replacement(range(dim), d):
            col = np.ones(len(u))
            for j in combo:
                col = col * u[:, j]
            cols.append(col)
    return np.stack(cols, axis=1)


# ----------------------------------------------------------------------------
# candidate fields
# ----------------------------------------------------------------------------
def _fit_logistic(pts_u, signs, contacts_u, degree, l2=1e-6, cw=100.0):
    """Max-margin (logistic) polynomial with contact equality penalty."""
    X = _features(pts_u, degree)
    Xc = _features(contacts_u, degree) if len(contacts_u) else None

    def fun(w):
        z = signs * (X @ w)
        loss = np.logaddexp(0.0, -z).sum()
        g = X.T @ (-signs * expit(-z))
        if Xc is not None:
            fc = Xc @ w
            loss += cw * float(fc @ fc)
            g = g + 2.0 * cw * (Xc.T @ fc)
        if l2:
            loss += l2 * float(w[1:] @ w[1:])
            g[1:] += 2.0 * l2 * w[1:]
        return loss, g

    try:
        res = minimize(fun, np.zeros(X.shape[1]), jac=True, method="L-BFGS-B",
                       options={"maxiter": 800, "ftol": 1e-13, "gtol": 1e-9})
        return res.x
    except Exception:
        return np.zeros(X.shape[1])


def _shepard(grid_u, pts_u, target, power=2.0, chunk=4000, deadline=None):
    flat = grid_u.reshape(-1, grid_u.shape[-1])
    out = np.empty(len(flat), dtype=float)
    good = 0
    for s in range(0, len(flat), chunk):
        if deadline is not None and time.perf_counter() > deadline and good:
            break
        q = flat[s:s + chunk]
        d = np.linalg.norm(q[:, None, :] - pts_u[None, :, :], axis=2)
        w = 1.0 / (np.maximum(d, 0.0) ** power + 1e-9)
        out[s:s + chunk] = (w @ target) / np.maximum(w.sum(1), 1e-300)
        good = s + len(q)
    if good < len(flat):
        out[good:] = out[good - 1] if good else 0.0
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out.reshape(grid_u.shape[:-1])


def _contact_correct(field, grid_u, contacts_u, contact_values):
    """Local Gaussian RBF correction that puts the field on contacts."""
    if len(contacts_u) == 0 or contact_values is None:
        return field
    n = min(len(contacts_u), 800)
    contacts_u = contacts_u[:n]
    fc = np.asarray(contact_values, dtype=float)[:n]
    d = np.linalg.norm(contacts_u[:, None, :] - contacts_u[None, :, :], axis=2)
    off = d[d > 0]
    sigma = float(np.median(off)) / 6.0 if off.size else 0.05
    sigma = min(max(sigma, 0.02), 0.12)
    A = np.exp(-0.5 * (d / sigma) ** 2) + 1e-10 * np.eye(n)
    try:
        a = np.linalg.solve(A, -fc)
    except Exception:
        a = np.linalg.lstsq(A, -fc, rcond=None)[0]
    flat = grid_u.reshape(-1, grid_u.shape[-1])
    out = field.reshape(-1).copy()
    out += np.exp(-0.5 * ((flat[:, None, :] - contacts_u[None, :, :])
                          / sigma) ** 2).sum(axis=2) @ a
    return out.reshape(field.shape)


def _plane_through_contacts(contacts_u):
    n = len(contacts_u)
    if n < 3:
        return None
    c = contacts_u.mean(0)
    try:
        _, sv, vt = np.linalg.svd(contacts_u - c, full_matrices=False)
    except Exception:
        return None
    if len(sv) < 2 or sv[1] <= 1e-9 * max(sv[0], 1e-12):
        return None
    return vt[-1], c


# ----------------------------------------------------------------------------
# forward check / residual used only for model selection
# ----------------------------------------------------------------------------
def _loss(problem, field):
    if not isinstance(field, np.ndarray) or not np.isfinite(field).all():
        return np.inf
    if not ((field > 0).any() and (field < 0).any()):
        return np.inf
    if skfmm is None:
        return np.inf
    spacing = tuple(float(s) for s in problem["spacing"])
    try:
        phi = skfmm.distance(np.ascontiguousarray(field, dtype=np.float64),
                             dx=spacing)
    except Exception:
        return np.inf
    if isinstance(phi, np.ma.MaskedArray) or not np.isfinite(phi).all():
        return np.inf
    total = 0.0
    for obs in problem.get("observations", []):
        vals = np.asarray(obs["values"], dtype=float)
        mask = np.isfinite(vals)
        y = vals[mask]
        d = phi[mask]
        w = float(obs.get("weight", 1.0))
        logistic = (np.logaddexp(0.0, d[y == 0]).sum()
                    + np.logaddexp(0.0, -d[y > 0]).sum())
        contact = phi[vals == 0.5]
        bias = float(contact.mean() ** 2) if contact.size else 0.0
        variance = float(np.mean(contact ** 2)) if contact.size else 0.0
        total += w * (float(logistic) + bias + variance)
    if problem.get("sketch") is not None:
        try:
            total += _sketch_loss(problem["sketch"], phi, spacing,
                                  problem["origin"])
        except Exception:
            return np.inf
    return float(total)


def _sketch_loss(reference, field, spacing, origin):
    coords = np.stack(
        [c.reshape(-1, order="F") for c in np.meshgrid(
            *[origin[i] + (np.arange(n) + 0.5) * spacing[i]
              for i, n in enumerate(field.shape)], indexing="ij")], axis=1)
    a = coords[np.asarray(reference).reshape(-1, order="F") == 1]
    b = coords[field.reshape(-1, order="F") > 0]
    if len(a) < 2 or len(b) < 2:
        return np.inf
    a = a - a.mean(0)
    b = b - b.mean(0)
    an, bn = np.linalg.norm(a), np.linalg.norm(b)
    if not an or not bn:
        return np.inf
    a, b = a / an, b / bn
    n = max(len(a), len(b))
    a = np.pad(a, ((0, n - len(a)), (0, 0)))
    b = np.pad(b, ((0, n - len(b)), (0, 0)))
    sv = np.linalg.svd(a.T @ b, compute_uv=False)
    return float(5.0 * (1.0 - sv.sum() ** 2))


# ----------------------------------------------------------------------------
# main entry point
# ----------------------------------------------------------------------------
def solve(problem, budget, seed):
    t0 = time.perf_counter()
    try:
        wall = float(budget.get("wall_time_seconds", 60.0))
    except Exception:
        wall = 60.0
    deadline = t0 + max(5.0, 0.70 * wall)

    shape, origin, spacing, mesh = _grid(problem)
    pts, values = _gather(problem)
    np.random.seed(int(seed) % (2 ** 31 - 1))

    initial = np.asarray(problem.get("initial_level_set"), dtype=float)
    if initial.shape != shape:
        initial = np.zeros(shape, dtype=float)

    centre, scale = _norm_params(mesh, pts)
    u_pts = (pts - centre) / scale if pts.size else pts
    u_grid = (mesh - centre) / scale
    is_neg = values == 0
    is_pos = values == 1
    is_con = values == 0.5
    labels = np.where(is_pos, 1.0, -1.0)
    con_u = u_pts[is_con]
    target = np.where(is_pos, 1.0, np.where(is_neg, -1.0, 0.0))

    # physical grid indices of the contact samples (for exact correction)
    obs_idx = []
    for obs in problem.get("observations", []):
        vals = np.asarray(obs["values"], dtype=float)
        obs_idx.append(np.argwhere(np.isfinite(vals)))
    all_idx = (np.concatenate(obs_idx, 0) if obs_idx
               else np.zeros((0, len(shape)), int))
    con_idx = all_idx[is_con] if len(all_idx) == len(values) else None

    candidates = []

    def add(field):
        if field is None:
            return
        field = np.asarray(field, dtype=float)
        if field.shape == shape and np.isfinite(field).all():
            candidates.append(field.copy())

    add(initial)

    # ---- polynomial max-margin fields -------------------------------------
    if len(u_pts) >= 2:
        for degree in (1, 2, 3):
            if degree == 3 and len(u_pts) < 10:
                continue
            for l2 in (1e-2, 1e-6):
                if time.perf_counter() > deadline:
                    break
                w = _fit_logistic(u_pts, labels, con_u, degree, l2=l2)
                base = (_features(u_grid.reshape(-1, u_grid.shape[-1]),
                                  degree) @ w).reshape(shape)
                add(base)
                if len(con_u) and con_idx is not None:
                    fv = np.array([base[tuple(ix)] for ix in con_idx])
                    add(_contact_correct(base, u_grid, con_u, fv))

    # ---- RBF interpolants through all labels/contacts ----------------------
    if len(u_pts) >= 2:
        for kernel in ("linear", "thin_plate_spline", "cubic", "multiquadric"):
            if time.perf_counter() > deadline:
                break
            smooths = ((0.0,) if kernel == "multiquadric"
                       else (0.0, 0.01, 0.05, 0.2))
            for smoothing in smooths:
                if time.perf_counter() > deadline:
                    break
                try:
                    kw = dict(epsilon=1.0) if kernel == "multiquadric" else {}
                    rbf = RBFInterpolator(u_pts, target, kernel=kernel,
                                          degree=1, smoothing=smoothing, **kw)
                    add(rbf(u_grid.reshape(-1, u_grid.shape[-1])).reshape(shape))
                except Exception:
                    pass

    # ---- inverse-distance (Shepard) interpolants --------------------------
    if len(u_pts) >= 2:
        for power in (2.0, 4.0):
            if time.perf_counter() > deadline:
                break
            add(_shepard(u_grid, u_pts, target, power=power,
                         deadline=deadline))

    # ---- plane through contacts -------------------------------------------
    plane = _plane_through_contacts(con_u)
    if plane is not None:
        normal, c = plane
        add(((u_grid.reshape(-1, u_grid.shape[-1]) - c) @ normal).reshape(shape))

    # ---- evaluate candidates with the public forward model ----------------
    best_field, best_loss, checked = None, np.inf, 0
    for field in candidates:
        if time.perf_counter() > deadline and checked > 3:
            break
        sd = float(np.std(field))
        if not np.isfinite(sd) or sd <= 0:
            continue
        fn = field / sd
        for shift in (-0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75):
            if time.perf_counter() > deadline and checked > 6:
                break
            trial = fn + shift
            checked += 1
            loss = _loss(problem, trial)
            if loss < best_loss:
                best_loss, best_field = loss, trial

    # fine shift refinement around the best coarse candidate
    if best_field is not None:
        for delta in (-0.125, -0.0625, -0.03125, 0.03125, 0.0625, 0.125):
            if time.perf_counter() > deadline:
                break
            trial = best_field + delta
            loss = _loss(problem, trial)
            if loss < best_loss:
                best_loss, best_field = loss, trial

    # ---- bounded local shape refinement -----------------------------------
    try:
        if best_field is not None:
            flat = u_grid.reshape(-1, u_grid.shape[-1])
            basis = [np.ones(len(flat))]
            for j in range(flat.shape[1]):
                basis.append(flat[:, j])
            if flat.shape[1] >= 3:
                basis.append(flat[:, 2] ** 2)
            basis = [b.reshape(shape) for b in basis]
            cur, cur_loss, evals = best_field.copy(), best_loss, 0
            stop = False
            for step in (0.12, 0.06, 0.03, 0.015, 0.0075):
                moved, rounds = True, 0
                while moved and rounds < 3 and not stop:
                    moved = False
                    rounds += 1
                    for b in basis:
                        for sgn in (1.0, -1.0):
                            if (time.perf_counter() > deadline
                                    or evals >= 180):
                                stop = True
                                break
                            trial = cur + sgn * step * b
                            evals += 1
                            loss = _loss(problem, trial)
                            if loss < cur_loss - 1e-10:
                                cur, cur_loss = trial, loss
                                moved = True
                        if stop:
                            break
                if stop:
                    break
            if cur_loss < best_loss:
                best_field, best_loss = cur, cur_loss
    except Exception:
        pass

    # ---- per-contact local bump refinement --------------------------------
    try:
        if best_field is not None and len(con_u):
            cur, cur_loss = best_field.copy(), best_loss
            evals = 0
            for sigma in (0.2, 0.12):
                if time.perf_counter() > deadline or evals >= 160:
                    break
                bumps = []
                for cu in con_u:
                    g = np.exp(-np.sum((u_grid - cu) ** 2, axis=-1)
                               / (2.0 * sigma * sigma))
                    gs = float(np.std(g))
                    if np.isfinite(gs) and gs > 0:
                        bumps.append(g / gs)
                if not bumps:
                    continue
                amps = np.zeros(len(bumps))
                off = 0.0
                base_cur = cur.copy()
                for step in (0.03125, 0.015625, 0.0078125, 0.00390625):
                    if time.perf_counter() > deadline or evals >= 160:
                        break
                    for _ in range(2):
                        moved = False
                        for j in range(len(bumps)):
                            for sgn in (1.0, -1.0):
                                if time.perf_counter() > deadline or evals >= 160:
                                    break
                                trial_amps = amps.copy()
                                trial_amps[j] += sgn * step
                                trial = base_cur + off
                                for a, g in zip(trial_amps, bumps):
                                    trial = trial + a * g
                                evals += 1
                                loss = _loss(problem, trial)
                                if loss < cur_loss - 1e-12:
                                    amps = trial_amps
                                    cur, cur_loss = trial, loss
                                    moved = True
                        for sgn in (1.0, -1.0):
                            if time.perf_counter() > deadline or evals >= 160:
                                break
                            trial_off = off + sgn * step
                            trial = base_cur + trial_off
                            for a, g in zip(amps, bumps):
                                trial = trial + a * g
                            evals += 1
                            loss = _loss(problem, trial)
                            if loss < cur_loss - 1e-12:
                                off = trial_off
                                cur, cur_loss = trial, loss
                                moved = True
                        if not moved:
                            break
            if cur_loss < best_loss:
                best_field, best_loss = cur, cur_loss
    except Exception:
        pass

    if best_field is None or not np.isfinite(best_field).all() \
            or not ((best_field > 0).any() and (best_field < 0).any()):
        f = initial
        if not np.isfinite(f).all() or not ((f > 0).any() and (f < 0).any()):
            f = np.zeros(shape, dtype=float)
            if len(u_pts):
                f = _shepard(u_grid, u_pts, target, power=2.0)
            if not ((f > 0).any() and (f < 0).any()):
                f = np.indices(shape, dtype=float)[0] - (shape[0] - 1) / 2.0
        best_field = f

    return np.ascontiguousarray(best_field, dtype=np.float64)
