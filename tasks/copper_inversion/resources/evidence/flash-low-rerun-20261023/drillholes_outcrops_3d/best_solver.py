"""Smooth implicit-geometry inversion with polynomial fits and a smoothed
graph-logistic refinement.

The sparse signed observations are fitted by a compact portfolio of smooth
implicit polynomial models (degree 2-4 logistic and least-squares fits).
Contacts are treated as zero-level constraints.  All candidates are scored
with the published logistic plus contact objective after the same skfmm
reinitialisation used by the host, and the lowest-loss valid field is returned.
If even the best polynomial is poor, it is further refined by graph-Laplacian
regularised logistic IRLS.  The supplied initial level set is a fallback.
"""

import time
import itertools

import numpy as np
import skfmm
from scipy import sparse
from scipy.sparse.linalg import cg
from scipy.optimize import minimize
from scipy.special import expit


# ---------------------------------------------------------------- utilities
def _grid_coords(problem):
    shape = tuple(int(n) for n in problem["shape"])
    origin = np.asarray(problem["origin"], dtype=float)
    spacing = np.asarray(problem["spacing"], dtype=float)
    axes = [origin[i] + (np.arange(n, dtype=float) + 0.5) * spacing[i]
            for i, n in enumerate(shape)]
    mesh = np.meshgrid(*axes, indexing="ij")
    return np.stack([m.reshape(-1, order="F") for m in mesh], axis=1), mesh


def _observations(problem, coords):
    """Collect observations in the same Fortran cell order as coords/phi."""
    pts, vals, wts = [], [], []
    for o in problem.get("observations", []):
        v = np.asarray(o["values"], dtype=float)
        m = np.isfinite(v)
        if o.get("mask") is not None:
            m &= np.asarray(o["mask"], dtype=bool)
        mf = m.reshape(-1, order="F")
        if not np.any(mf):
            continue
        pts.append(coords[mf])
        vals.append(v.reshape(-1, order="F")[mf])
        wts.append(np.full(int(mf.sum()), float(o["weight"])))
    if pts:
        return np.vstack(pts), np.concatenate(vals), np.concatenate(wts)
    return np.zeros((0, coords.shape[1])), np.zeros(0), np.zeros(0)


def _poly_features(points, center, scale, degree):
    u = (points - center) / scale
    cols = [np.ones(len(points))]
    d = points.shape[1]
    for deg in range(1, degree + 1):
        for combo in itertools.product(range(deg + 1), repeat=d):
            if sum(combo) != deg:
                continue
            term = np.ones(len(points))
            for j, k in enumerate(combo):
                if k:
                    term = term * u[:, j] ** k
            cols.append(term)
    return np.stack(cols, axis=1)


def _sketch_loss(reference, field, spacing, origin):
    """Exact copy of the published 2-D sketch term."""
    coords = np.stack(
        [c.reshape(-1, order="F") for c in np.meshgrid(
            *[origin[i] + (np.arange(n) + 0.5) * spacing[i]
              for i, n in enumerate(field.shape)], indexing="ij")], axis=1)
    a = coords[reference.reshape(-1, order="F") == 1]
    b = coords[field.reshape(-1, order="F") > 0]
    if len(a) < 2 or len(b) < 2:
        return np.inf
    a = a - a.mean(0)
    b = b - b.mean(0)
    an = np.linalg.norm(a)
    bn = np.linalg.norm(b)
    if not an or not bn:
        return np.inf
    a = a / an
    b = b / bn
    n = max(len(a), len(b))
    a = np.pad(a, ((0, n - len(a)), (0, 0)))
    b = np.pad(b, ((0, n - len(b)), (0, 0)))
    singular = np.linalg.svd(a.T @ b, compute_uv=False)
    return float(5 * (1 - singular.sum() ** 2))


def _score(problem, field, obs):
    """Exact published loss for one candidate; inf when the geometry is invalid."""
    field = np.asarray(field, dtype=np.float64)
    if field.shape != tuple(problem["shape"]) or not np.isfinite(field).all():
        return np.inf
    if not (np.any(field > 0) and np.any(field < 0)):
        return np.inf
    try:
        phi = np.asarray(skfmm.distance(np.ascontiguousarray(field),
                                        dx=tuple(problem["spacing"])),
                         dtype=np.float64)
    except Exception:
        return np.inf
    if not np.isfinite(phi).all():
        return np.inf
    total = 0.0
    flat_phi = phi.reshape(-1, order="F")
    for o in problem.get("observations", []):
        v = np.asarray(o["values"], dtype=float)
        m = np.isfinite(v)
        if o.get("mask") is not None:
            m &= np.asarray(o["mask"], dtype=bool)
        mf = m.reshape(-1, order="F")
        if not np.any(mf):
            continue
        y = v.reshape(-1, order="F")[mf]
        d = flat_phi[mf]
        logistic = np.logaddexp(0.0, d[y == 0]).sum() + np.logaddexp(0.0, -d[y > 0]).sum()
        c = d[y == 0.5]
        bias = float(c.mean() ** 2) if c.size else 0.0
        variance = float(np.mean(c ** 2)) if c.size else 0.0
        total += float(o["weight"]) * (logistic + bias + variance)
    if problem.get("sketch") is not None:
        total += _sketch_loss(np.asarray(problem["sketch"]),
                              phi, problem["spacing"], problem["origin"])
    return float(total)


# --------------------------------------------------------------- candidates
def _poly_ls(points, vals, wts, center, scale, degree, ridge, contact_mult):
    X = _poly_features(points, center, scale, degree)
    w = wts.copy()
    w[vals == 0.5] *= contact_mult
    t = np.where(vals > 0.5, 1.0, np.where(vals < 0.5, -1.0, 0.0))
    reg = np.full(X.shape[1], ridge, dtype=float)
    reg[0] *= 0.05
    A = X.T @ (w[:, None] * X) + np.diag(reg)
    b = X.T @ (w * t)
    try:
        p = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        p = np.linalg.lstsq(A, b, rcond=None)[0]
    return p


def _poly_logistic(points, vals, wts, center, scale, degree, ridge, contact_mult):
    X = _poly_features(points, center, scale, degree)
    s = np.where(vals >= 0.5, 1.0, -1.0)  # contacts count as positive labels
    cw = np.where(vals == 0.5, contact_mult * wts, 0.0)
    t = np.where(vals > 0.5, 1.0, np.where(vals < 0.5, -1.0, 0.0))
    w0 = np.where(vals == 0.5, np.maximum(contact_mult, 1.0) * wts, wts)
    reg0 = np.full(X.shape[1], 1e-6)
    reg0[0] *= 0.05
    try:
        p0 = np.linalg.solve(X.T @ (w0[:, None] * X) + np.diag(reg0), X.T @ (w0 * t))
    except np.linalg.LinAlgError:
        p0 = np.zeros(X.shape[1])
    p0[0] = 0.0

    def fun(p):
        F = X @ p
        z = -s * F
        loss = np.sum(wts * np.logaddexp(0.0, z))
        loss += np.sum(cw * F * F)
        loss += ridge * np.sum(p[1:] ** 2)
        return loss

    def jac(p):
        F = X @ p
        z = -s * F
        sig = 0.5 * (1.0 + np.tanh(0.5 * z))
        coef = -s * wts * sig + 2.0 * cw * F
        g = X.T @ coef
        g[1:] += 2.0 * ridge * p[1:]
        return g

    try:
        res = minimize(fun, p0, jac=jac, method="L-BFGS-B",
                       options={"maxiter": 200, "ftol": 1e-12, "gtol": 1e-7})
        p = res.x
    except Exception:
        p = p0
    return p


def _graph_laplacian(shape):
    shape = tuple(int(n) for n in shape)
    n = int(np.prod(shape))
    L = sparse.csr_matrix((n, n))
    for ax, m in enumerate(shape):
        e = np.ones(m)
        Dax = sparse.diags([-e[:-1], 2 * e, -e[:-1]], [-1, 0, 1],
                           shape=(m, m), format="csr")
        mats = [Dax if j == ax else sparse.eye(mm, format="csr")
                for j, mm in enumerate(shape)]
        K = mats[0]
        for mat in mats[1:]:
            K = sparse.kron(K, mat, format="csr")
        L = L + K
    return L


def _smooth_irls(problem, base_field, points, vals, wts, lam, contact_mult,
                 max_iter=30):
    """Damped IRLS for lam*||grad f||^2 + weighted logistic + contact penalty."""
    shape = tuple(int(n) for n in problem["shape"])
    n = int(np.prod(shape))
    if n > 350000 or len(vals) == 0:
        return None
    origin = np.asarray(problem["origin"], dtype=float)
    spacing = np.asarray(problem["spacing"], dtype=float)
    grid_idx = np.floor((points - origin) / spacing).astype(int)
    grid_idx = np.clip(grid_idx, 0, np.asarray(shape) - 1)
    ids = np.ravel_multi_index(tuple(grid_idx.T), shape, order="F")
    s = np.where(vals >= 0.5, 1.0, -1.0)
    contact = (vals == 0.5).astype(float)
    cw = contact_mult * wts * contact

    L = _graph_laplacian(shape)
    f = np.asarray(base_field, dtype=float).reshape(-1, order="F").copy()
    scale = float(np.max(np.abs(f)))
    if not np.isfinite(scale) or scale < 1e-9:
        return None
    f *= 4.0 / scale

    def objective(x):
        xi = x[ids]
        val = np.sum(wts * np.logaddexp(0.0, -s * xi))
        val += np.sum(cw * xi * xi)
        val += lam * float(x @ (L @ x))
        return val

    deg = 2.0 * len(shape)
    for _ in range(max_iter):
        fi = f[ids]
        q = expit(s * fi)
        g = (lam * (L @ f)
             + np.bincount(ids, -s * wts * (1.0 - q), minlength=n)
             + np.bincount(ids, 2.0 * cw * fi, minlength=n))
        h = (np.bincount(ids, wts * q * (1.0 - q), minlength=n)
             + np.bincount(ids, 2.0 * cw, minlength=n))
        H = lam * L + sparse.diags(h + 1e-7)
        M = sparse.diags(1.0 / (h + 1e-7 + lam * deg))
        try:
            delta, _ = cg(H, -g, rtol=1e-6, atol=0.0, maxiter=200, M=M)
        except Exception:
            break
        if not np.isfinite(delta).all():
            break
        o0 = objective(f)
        alpha = 1.0
        accepted = False
        for _ls in range(12):
            fn = f + alpha * delta
            on = objective(fn)
            if np.isfinite(on) and on <= o0 + 1e-4 * alpha * float(g @ delta):
                accepted = True
                break
            alpha *= 0.5
        if not accepted:
            fn = f + 0.05 * delta
        f = fn
        if np.linalg.norm(alpha * delta) < 1e-5:
            break

    if not np.isfinite(f).all():
        return None
    return f.reshape(shape, order="F")


# --------------------------------------------------------------------- solve
last_report = []


def solve(problem, budget, seed):
    global last_report
    t0 = time.monotonic()
    wall = float(budget.get("wall_time_seconds", 120.0)) if isinstance(budget, dict) else 120.0
    deadline = t0 + max(1.0, 0.86 * wall)
    shape = tuple(int(n) for n in problem["shape"])
    coords, mesh = _grid_coords(problem)
    points, vals, wts = _observations(problem, coords)
    obs = (points, vals, wts)

    candidates = []
    init = np.asarray(problem.get("initial_level_set"), dtype=np.float64)
    if init.shape == shape:
        candidates.append(("initial", init.copy()))

    if len(vals):
        if np.any(vals == 0.5):
            center = points[vals == 0.5].mean(0)
        else:
            center = points.mean(0)
        span = np.ptp(coords, axis=0)
        scale = max(float(np.max(span)) * 0.5, 1e-6)

        def add_poly(name, p, degree):
            Xg = _poly_features(coords, center, scale, degree)
            field = (Xg @ p).reshape(shape, order="F")
            candidates.append((name, field))

        # logistic models: degree 4 with weak contact anchoring is usually the
        # best compromise between label separation and smooth extrapolation.
        plans = [
            (4, (0.0, 1e-11, 1e-8, 1e-6), (0.02, 0.05, 0.1, 0.2, 0.5, 1.0)),
            (3, (1e-8, 1e-5, 1e-3), (0.05, 0.2, 1.0, 5.0, 20.0)),
            (2, (1e-8, 1e-3), (0.05, 1.0, 20.0)),
        ]
        for degree, ridges, cms in plans:
            for ridge in ridges:
                for cm in cms:
                    if time.monotonic() > deadline:
                        break
                    try:
                        p = _poly_logistic(points, vals, wts, center, scale,
                                           degree, ridge, cm)
                        add_poly("logit_d%d_r%g_c%g" % (degree, ridge, cm), p, degree)
                    except Exception:
                        pass
        # a few direct least-squares shapes as diversity / fallback
        for degree in (2, 3):
            for ridge in (1e-4, 1e-2):
                for cm in (1.0, 8.0, 40.0):
                    if time.monotonic() > deadline:
                        break
                    try:
                        p = _poly_ls(points, vals, wts, center, scale,
                                     degree, ridge, cm)
                        add_poly("ls_d%d_r%g_c%g" % (degree, ridge, cm), p, degree)
                    except Exception:
                        pass

    best_field = None
    best_loss = np.inf
    report = []
    for name, field in candidates:
        if field.shape != shape:
            continue
        loss = _score(problem, field, obs)
        if np.isfinite(loss):
            report.append((name, {"loss_total": loss}))
        if loss < best_loss:
            best_loss = loss
            best_field = np.asarray(field, dtype=np.float64).copy()

    # Only invoke the more expensive IRLS stage when the polynomial fit is
    # poor; it can over-sharpen a good smooth solution.
    if best_field is not None and best_loss > 80.0 and time.monotonic() < deadline:
        for lam in (0.003, 0.01, 0.03):
            if time.monotonic() > deadline:
                break
            try:
                f = _smooth_irls(problem, best_field, points, vals, wts,
                                 lam, 5.0, max_iter=30)
                if f is None:
                    continue
                loss = _score(problem, f, obs)
                if np.isfinite(loss):
                    report.append(("irls_l%g" % lam, {"loss_total": loss}))
                if loss < best_loss:
                    best_loss = loss
                    best_field = np.asarray(f, dtype=np.float64).copy()
            except Exception:
                pass

    if best_field is None or not (np.any(best_field > 0) and np.any(best_field < 0)):
        xx = np.stack([m.reshape(-1, order="F") for m in mesh], axis=1)
        c = xx.mean(0)
        s = np.max(np.ptp(xx, axis=0)) * 0.25 + 1e-6
        best_field = (s * s - np.sum((xx - c) ** 2, axis=1)).reshape(shape, order="F")

    best_field = np.ascontiguousarray(best_field, dtype=np.float64)
    if not (np.isfinite(best_field).all() and np.any(best_field > 0)
            and np.any(best_field < 0)):
        best_field = np.asarray(problem["initial_level_set"], dtype=np.float64).copy()

    last_report = report
    return best_field
