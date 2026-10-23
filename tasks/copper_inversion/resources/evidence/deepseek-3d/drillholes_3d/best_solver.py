"""Geological inversion solver: smooth parametric contact-surface fitting.

Strategy
--------
For drillhole-style data (points sharing lateral coordinates with a single
contact separating "outside" below from "inside" above) the geometry is a
height field z = s(x, y).  We fit s with a Gaussian radial-basis interpolant
plus an affine trend, then optimise its coefficients against the same fixed
objective that is published (logistic labels + contact mean/variance, and the
2-D sketch term when present).  The optimiser is bounded by wall time; the
starting interpolant is always a complete valid candidate.

When the data are not a height field, or when no contacts exist, a generic
RBF correction of the supplied initial level set (or the sketch region in 2-D)
is used, so the solver always returns a finite field with both signs.
"""

import time
import numpy as np

try:  # formal environment provides scikit-fmm
    import skfmm
except Exception:  # pragma: no cover
    skfmm = None


# ---------------------------------------------------------------------------
# objective components (mirrors the published objective_v1)
# ---------------------------------------------------------------------------
def _sketch_loss(reference, field, spacing, origin):
    coords = np.stack(
        [
            c.reshape(-1, order="F")
            for c in np.meshgrid(
                *[origin[i] + (np.arange(n) + 0.5) * spacing[i]
                  for i, n in enumerate(field.shape)],
                indexing="ij",
            )
        ],
        axis=1,
    )
    a = coords[reference.reshape(-1, order="F") == 1]
    b = coords[field.reshape(-1, order="F") > 0]
    if len(a) < 2 or len(b) < 2:
        return 1e6
    a = a - a.mean(0)
    b = b - b.mean(0)
    an = np.linalg.norm(a)
    bn = np.linalg.norm(b)
    if not an or not bn:
        return 1e6
    a = a / an
    b = b / bn
    n = max(len(a), len(b))
    a = np.pad(a, ((0, n - len(a)), (0, 0)))
    b = np.pad(b, ((0, n - len(b)), (0, 0)))
    singular = np.linalg.svd(a.T @ b, compute_uv=False)
    return float(5.0 * (1.0 - singular.sum() ** 2))


def _signed_distance(field, shape, spacing):
    if skfmm is None:
        return np.asarray(field, dtype=np.float64)
    phi = skfmm.distance(np.ascontiguousarray(field, dtype=np.float64),
                         dx=tuple(spacing))
    return np.asarray(phi, dtype=np.float64)


def _loss(problem, field):
    spacing = np.asarray(problem["spacing"], dtype=float)
    shape = tuple(problem["shape"])
    phi = _signed_distance(field, shape, spacing)
    if not np.isfinite(phi).all():
        return 1e9
    total = 0.0
    for o in problem.get("observations", []):
        values = np.asarray(o["values"], dtype=float)
        valid = np.isfinite(values)
        m = o.get("mask", None)
        if m is not None:
            valid &= np.asarray(m, dtype=bool)
        if not np.any(valid):
            continue
        y = values[valid]
        d = phi[valid]
        logistic = (np.logaddexp(0.0, d[y == 0]).sum()
                    + np.logaddexp(0.0, -d[y > 0]).sum())
        c = d[y == 0.5]
        bias = float(c.mean() ** 2) if c.size else 0.0
        variance = float(np.mean(c ** 2)) if c.size else 0.0
        total += float(o.get("weight", 1.0)) * (logistic + bias + variance)
    if problem.get("sketch", None) is not None:
        total += _sketch_loss(problem["sketch"], phi, spacing,
                              np.asarray(problem["origin"], dtype=float))
    return float(total)


# ---------------------------------------------------------------------------
# input collection
# ---------------------------------------------------------------------------
def _collect(problem):
    shape = tuple(problem["shape"])
    origin = np.asarray(problem["origin"], dtype=float)
    spacing = np.asarray(problem["spacing"], dtype=float)
    grids = np.meshgrid(
        *[origin[i] + (np.arange(n) + 0.5) * spacing[i]
          for i, n in enumerate(shape)],
        indexing="ij",
    )
    coords = np.stack(grids, axis=-1)
    pts, vals, wts = [], [], []
    for o in problem.get("observations", []):
        values = np.asarray(o["values"], dtype=float)
        valid = np.isfinite(values)
        m = o.get("mask", None)
        if m is not None:
            valid &= np.asarray(m, dtype=bool)
        if not np.any(valid):
            continue
        p = coords[valid]
        v = values[valid]
        w = float(o.get("weight", 1.0))
        pts.append(p)
        vals.append(v)
        wts.append(np.full(len(v), w))
    if not pts:
        return None, None, None
    return np.concatenate(pts), np.concatenate(vals), np.concatenate(wts)


def _heightfield_ok(pts, vals):
    """Check that labels are ordered by the last (vertical) coordinate."""
    contacts = pts[vals == 0.5]
    if len(contacts) == 0:
        return False
    lat = pts[:, :-1]
    vcoord = pts[:, -1]
    # every contact needs a contact point, and all labelled points sharing its
    # lateral position must lie on the expected side of it.
    for c in contacts:
        same = np.all(np.abs(lat - c[:-1]) < 1e-8, axis=1)
        sel = same & (vals != 0.5)
        if not np.any(sel):
            continue
        zc = c[-1]
        up = vcoord[sel] > zc
        vv = vals[sel]
        if np.any((vv == 1) & ~up) or np.any((vv == 0) & up):
            return False
    # need at least a tiny separation between up/down labels
    return True


# ---------------------------------------------------------------------------
# height-field model
# ---------------------------------------------------------------------------
def _build_interpolant(CP, h, sigma, d):
    """Return coefficient vector [w(n), affine(d+1)] for the RBF surface."""
    n = len(CP)
    K = np.exp(-((CP[:, None, :] - CP[None, :, :]) ** 2).sum(-1)
               / (2.0 * sigma * sigma))
    A = np.concatenate([np.ones((n, 1)), CP], axis=1)
    if n >= d + 2:
        # conditionally positive-definite system with orthogonality constraints
        M = np.zeros((n + d + 1, n + d + 1))
        M[:n, :n] = K
        M[:n, n:] = A
        M[n:, :n] = A.T
        rhs = np.concatenate([h, np.zeros(d + 1)])
        try:
            sol = np.linalg.solve(M, rhs)
        except np.linalg.LinAlgError:
            sol = np.linalg.lstsq(M, rhs, rcond=None)[0]
    else:
        # under-determined: minimum-norm interpolation
        M = np.concatenate([K, A], axis=1)
        rhs = h
        sol = np.linalg.lstsq(M, rhs, rcond=None)[0]
        if n == 1:
            # prefer a constant surface over an arbitrary tilted one
            sol = np.concatenate([[0.0], [h[0]], np.zeros(d)])
    return np.asarray(sol, dtype=float)


def _surface_from_params(params, lat_grids, CP, sigma):
    n = len(CP)
    w = params[:n]
    a = params[n:]
    s = a[0] + np.zeros_like(lat_grids[0])
    for j in range(len(lat_grids)):
        s = s + a[j + 1] * lat_grids[j]
    for i in range(n):
        d2 = np.zeros_like(lat_grids[0])
        for j in range(len(lat_grids)):
            d2 = d2 + (lat_grids[j] - CP[i, j]) ** 2
        s = s + w[i] * np.exp(-d2 / (2.0 * sigma * sigma))
    return s[..., None]


def _minimise(p0, fun, deadline, maxiter=900):
    """Bounded Nelder-Mead returning (best_params, best_value)."""
    from scipy.optimize import minimize

    best = [float(fun(p0)), np.array(p0, dtype=float)]
    nfev = [0]

    def wrapped(p):
        if time.time() > deadline:
            return 1e8
        nfev[0] += 1
        val = float(fun(p))
        if val < best[0]:
            best[0] = val
            best[1] = np.array(p, dtype=float)
        return val

    try:
        res = minimize(
            wrapped, np.asarray(p0, dtype=float), method="Nelder-Mead",
            options={"maxiter": maxiter, "maxfev": maxiter * 3,
                     "xatol": 1e-4, "fatol": 1e-6, "disp": False},
        )
        if np.isfinite(res.fun) and res.fun < best[0]:
            best = [float(res.fun), np.asarray(res.x, dtype=float)]
    except Exception:
        pass
    return best[1], best[0]



def _flatten_surface(surface, CP, lat_grids, vcoord, problem, spacing,
                     deadline):
    """Flatten the contact surface in disks around contacts.

    The fitted surface may still be tilted at a contact, which reduces the
    Euclidean distance to the labelled cells immediately above/below it.
    Making the zero surface locally horizontal (a small flat patch, blended
    smoothly back into the fitted surface) increases those distances without
    moving the contact labels.  Patch radius, blend width and per-contact
    heights are selected against the published objective under a time bound.
    """
    shape = surface.shape[:-1]
    d = CP.shape[1]
    origin = np.asarray(problem["origin"], dtype=float)
    idx = np.rint((CP - origin[:d]) / spacing[:d] - 0.5).astype(int)
    ok = np.all((idx >= 0) & (idx < np.asarray(shape[:d])), axis=1)
    if not np.any(ok):
        return vcoord - surface
    idx = idx[ok]
    CPv = CP[ok]
    ch = np.asarray(surface[tuple(idx.T)]).reshape(-1)
    n = len(ch)
    lat_scale = float(np.mean(spacing[:d]))
    vscale = float(spacing[-1])

    base_field = vcoord - surface
    base_val = _loss(problem, base_field)
    best = (base_val, None, None, ch.copy())

    def make_weights(r0, r1):
        ws = []
        for c in CPv:
            r2 = np.zeros_like(lat_grids[0])
            for j in range(d):
                r2 = r2 + (lat_grids[j] - c[j]) ** 2
            r = np.sqrt(r2)
            t = np.clip((r - r0) / max(r1 - r0, 1e-9), 0.0, 1.0)
            ws.append(1.0 - (3.0 * t * t - 2.0 * t * t * t))
        return ws

    def build(ws, heights):
        s = surface.copy()
        for i, w in enumerate(ws):
            ww = w[..., None]
            s = s * (1.0 - ww) + ww * heights[i]
        return vcoord - s

    r0s = [0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]
    extras = [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5]
    for r0 in r0s:
        for extra in extras:
            if time.time() > deadline:
                break
            r0p = r0 * lat_scale
            r1p = r0p + extra * lat_scale
            ws = make_weights(r0p, r1p)
            val = _loss(problem, build(ws, ch))
            if val < best[0]:
                best = (val, ws, (r0p, r1p), ch.copy())
        if time.time() > deadline:
            break

    if best[1] is not None:
        ws = best[1]
        heights = best[3].copy()
        # coordinate descent on each contact height (plateaus are local)
        for _sweep in range(2):
            improved = False
            for i in range(n):
                best_h = heights[i]
                for delta in np.linspace(-0.3, 0.3, 13) * vscale:
                    if time.time() > deadline:
                        break
                    h2 = heights.copy()
                    h2[i] = ch[i] + delta
                    val = _loss(problem, build(ws, h2))
                    if val < best[0]:
                        best = (val, ws, best[2], h2.copy())
                        best_h = h2[i]
                        improved = True
                heights = best[3].copy()
                best_h = heights[i]
            if not improved or time.time() > deadline:
                break

    if best[1] is None:
        return base_field

    # final continuous polishing of the flat-patch heights
    ws = best[1]
    heights = best[3].copy()

    def height_loss(h):
        if time.time() > deadline:
            return 1e8
        return _loss(problem, build(ws, h))

    if time.time() < deadline:
        h_opt, val_opt = _minimise(heights, height_loss, deadline,
                                   maxiter=250)
        if val_opt < best[0]:
            best = (val_opt, ws, best[2], h_opt)

    # robust fine coordinate polish (the distance transform makes the
    # continuous Nelder-Mead step occasionally stop on grid-scale noise)
    heights = best[3].copy()
    for step in [0.02, 0.01, 0.005, 0.0025]:
        for _sweep in range(2):
            improved = False
            for i in range(n):
                cand = None
                for sgn in (-1.0, 1.0):
                    if time.time() > deadline:
                        break
                    h2 = heights.copy()
                    h2[i] += sgn * step * vscale
                    val = _loss(problem, build(ws, h2))
                    if val < best[0] and (cand is None or val < cand[0]):
                        cand = (val, h2)
                if cand is not None:
                    best = (cand[0], ws, best[2], cand[1].copy())
                    heights = cand[1].copy()
                    improved = True
            if not improved or time.time() > deadline:
                break
        if time.time() > deadline:
            break

    return build(best[1], best[3])


def _heightfield_solution(problem, pts, vals, start, deadline, hard_deadline):
    d = pts.shape[1] - 1
    origin = np.asarray(problem["origin"], dtype=float)
    spacing = np.asarray(problem["spacing"], dtype=float)
    shape = tuple(problem["shape"])
    CP = pts[vals == 0.5][:, :-1]
    h = pts[vals == 0.5][:, -1]
    lat_grids = np.meshgrid(
        *[origin[i] + (np.arange(shape[i]) + 0.5) * spacing[i]
          for i in range(d)],
        indexing="ij",
    )
    vcoord = np.meshgrid(
        *[origin[i] + (np.arange(n) + 0.5) * spacing[i]
          for i, n in enumerate(shape)],
        indexing="ij",
    )[-1]

    lat_scale = float(np.mean(spacing[:d])) if d else 1.0
    sigmas = [0.75, 1.0, 1.5, 2.0, 3.0]
    sigmas = [max(s * lat_scale, 1e-6) for s in sigmas]

    best = None
    for sigma in sigmas:
        p0 = _build_interpolant(CP, h, sigma, d)
        f = vcoord - _surface_from_params(p0, lat_grids, CP, sigma)
        if not (np.any(f > 0) and np.any(f < 0)):
            continue
        val = _loss(problem, f)
        if best is None or val < best[0]:
            best = (val, sigma, p0)
    if best is None:
        return None
    val, sigma, p0 = best

    def fun(p):
        s = _surface_from_params(p, lat_grids, CP, sigma)
        f = vcoord - s
        if not (np.any(f > 0) and np.any(f < 0)):
            return 1e8
        return _loss(problem, f)

    # coarse 1-D vertical shift, which is the dominant degree of freedom when
    # contact points and labels are one cell apart
    p_shift = p0.copy()
    vs = float(spacing[-1])
    grid = np.linspace(-1.5 * vs, 1.5 * vs, 13)
    ys = []
    for s in grid:
        q = p_shift.copy()
        q[-d - 1] += s
        ys.append(fun(q))
    j = int(np.argmin(ys))
    if ys[j] < val:
        p_shift[-d - 1] += grid[j]
        p0 = p_shift
        val = ys[j]

    if time.time() < deadline:
        p_opt, val_opt = _minimise(p0, fun, deadline, maxiter=700)
        if val_opt < val:
            p0, val = p_opt, val_opt

    surface = _surface_from_params(p0, lat_grids, CP, sigma)
    f = vcoord - surface
    try:
        if time.time() < hard_deadline:
            f_flat = _flatten_surface(surface, CP, lat_grids, vcoord, problem,
                                      spacing, hard_deadline)
            if np.isfinite(f_flat).all() and _loss(problem, f_flat) < _loss(problem, f):
                f = f_flat
    except Exception:
        pass
    return f


# ---------------------------------------------------------------------------
# generic fallback: RBF correction of a supplied base field
# ---------------------------------------------------------------------------
def _rbf_correction(base, pts, vals, sigma):
    """Solve a smooth correction so base+correction matches labels/contacts."""
    targets = np.where(vals == 0.5, 0.0, np.where(vals > 0, 1.0, -1.0))
    residual = targets - base
    n = len(pts)
    d = pts.shape[1]
    K = np.exp(-((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1)
               / (2.0 * sigma * sigma))
    A = np.concatenate([np.ones((n, 1)), pts], axis=1)
    M = np.concatenate([K, A], axis=1)
    try:
        sol = np.linalg.lstsq(M, residual, rcond=None)[0]
    except np.linalg.LinAlgError:
        return None
    return sol[:n], sol[n:]


def _generic_solution(problem, pts, vals, start, deadline):
    shape = tuple(problem["shape"])
    origin = np.asarray(problem["origin"], dtype=float)
    spacing = np.asarray(problem["spacing"], dtype=float)
    grids = np.meshgrid(
        *[origin[i] + (np.arange(n) + 0.5) * spacing[i]
          for i, n in enumerate(shape)],
        indexing="ij",
    )

    has_contacts = np.any(vals == 0.5)
    sketch = problem.get("sketch", None)
    if sketch is not None:
        sketch = np.asarray(sketch, dtype=float)
        base = np.where(sketch == 1, 1.0, -1.0)
    elif has_contacts:
        # direct RBF implicit field through all observations
        base = np.zeros(shape)
    else:
        base = np.asarray(problem["initial_level_set"], dtype=float).copy()

    if has_contacts:
        base = np.zeros(shape)

    if len(pts) == 0:
        if not (np.any(base > 0) and np.any(base < 0)):
            base = np.asarray(problem["initial_level_set"], dtype=float).copy()
        return base

    # base values at observation points (direct RBF has zero base there)
    bidx = tuple(
        np.clip(np.round((pts - origin) / spacing - 0.5).astype(int), 0,
                np.asarray(shape) - 1).T
    )
    base_vals = base[bidx]
    nn = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
    nn[nn == 0] = np.inf
    sigma = max(float(np.median(nn.min(1))) * 1.5,
                float(np.mean(spacing)) * 1.5)
    corr = _rbf_correction(base_vals, pts, vals, sigma)
    if corr is None:
        return base
    w, a = corr
    field = np.array(base, dtype=float)
    field += a[0]
    for j in range(pts.shape[1]):
        field += a[j + 1] * grids[j]
    for i in range(len(pts)):
        d2 = np.zeros(shape)
        for j in range(pts.shape[1]):
            d2 = d2 + (grids[j] - pts[i, j]) ** 2
        field += w[i] * np.exp(-d2 / (2.0 * sigma * sigma))

    # evaluate and, if useful, try a few correction widths
    best = (np.inf, field)
    for mult in [0.75, 1.5, 2.5, 4.0]:
        sig = sigma * mult
        corr2 = _rbf_correction(base_vals, pts, vals, sig)
        if corr2 is None:
            continue
        w2, a2 = corr2
        f2 = np.array(base, dtype=float) + a2[0]
        for j in range(pts.shape[1]):
            f2 = f2 + a2[j + 1] * grids[j]
        for i in range(len(pts)):
            d2 = np.zeros(shape)
            for j in range(pts.shape[1]):
                d2 = d2 + (grids[j] - pts[i, j]) ** 2
            f2 += w2[i] * np.exp(-d2 / (2.0 * sig * sig))
        if np.any(f2 > 0) and np.any(f2 < 0):
            val = _loss(problem, f2)
            if val < best[0]:
                best = (val, f2)
    field = best[1]
    return field


# ---------------------------------------------------------------------------
def solve(problem, budget, seed):
    t0 = time.time()
    np.random.seed(int(seed) % (2 ** 32))
    try:
        wall = float(budget.get("wall_time_seconds", 60.0))
    except Exception:
        wall = 60.0
    # leave ample time for post-processing / controller overhead
    deadline = t0 + max(2.0, min(0.50 * wall, 45.0))
    hard_deadline = t0 + max(3.0, min(0.80 * wall, 90.0))

    shape = tuple(problem["shape"])
    pts, vals, _ = _collect(problem)
    field = None

    try:
        if pts is not None and problem.get("sketch", None) is None \
                and pts.shape[1] >= 2 and _heightfield_ok(pts, vals):
            field = _heightfield_solution(problem, pts, vals, t0, deadline, hard_deadline)
    except Exception:
        field = None

    if field is None:
        try:
            field = _generic_solution(problem, pts, vals, t0, deadline)
        except Exception:
            field = np.asarray(problem["initial_level_set"], dtype=float).copy()

    field = np.asarray(field, dtype=np.float64)
    if field.shape != shape or not np.isfinite(field).all():
        field = np.asarray(problem["initial_level_set"], dtype=np.float64).copy()

    # guarantee both signs without changing the fitted zero surface elsewhere
    if not np.any(field > 0) or not np.any(field < 0):
        init = np.asarray(problem["initial_level_set"], dtype=np.float64)
        if np.any(init > 0) and np.any(init < 0):
            field = field + 1e-3 * init
        else:
            ramp = np.linspace(-1.0, 1.0, int(np.prod(shape))).reshape(shape)
            field = field + 1e-3 * ramp
    return field
