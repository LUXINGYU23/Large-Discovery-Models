"""Candidate 3: Chains the rotated anisotropic compact C2 stage with a second anisofine refinement.

Base: compact-plateau + Gaussian-bump stages; this variant chains refinement modes
and keeps the best exact-loss field as fallback.

Stage A fits the per-contact compact plateau surface, stage B adds pair-midpoint bumps.
Stage C refines the signed-distance field with coordinate descent on quadratic plus
Gaussian pair, triple and hole bases (smooth, non-compact corrections). Exact
objective_v1 scoring retains every earlier stage as fallback.
"""
import numpy as np
import time as _pytime
_DEADLINE = None
try:
    import skfmm
except Exception:
    skfmm = None
try:
    from scipy.optimize import minimize
except Exception:
    minimize = None


def _coords(shape, origin, spacing):
    axes = [origin[i] + (np.arange(n) + 0.5) * spacing[i] for i, n in enumerate(shape)]
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)


def _dist(field, spacing):
    f = np.ascontiguousarray(field, dtype=np.float64)
    return skfmm.distance(f, dx=tuple(spacing)) if skfmm is not None else f


def _sketch_loss(ref, phi, spacing, origin):
    coords = np.stack([c.reshape(-1, order="F") for c in np.meshgrid(
        *[origin[i] + (np.arange(n) + 0.5) * spacing[i] for i, n in enumerate(phi.shape)],
        indexing="ij")], axis=1)
    a = coords[np.asarray(ref).reshape(-1, order="F") == 1]
    b = coords[phi.reshape(-1, order="F") > 0]
    if len(a) < 2 or len(b) < 2:
        raise ValueError("degenerate sketch")
    a = a - a.mean(0); b = b - b.mean(0)
    an, bn = np.linalg.norm(a), np.linalg.norm(b)
    if not an or not bn:
        raise ValueError("zero sketch")
    a, b = a / an, b / bn
    n = max(len(a), len(b))
    a = np.pad(a, ((0, n - len(a)), (0, 0)))
    b = np.pad(b, ((0, n - len(b)), (0, 0)))
    return float(5 * (1 - np.linalg.svd(a.T @ b, compute_uv=False).sum() ** 2))


def _loss(problem, field):
    shape = tuple(problem["shape"])
    spacing = np.asarray(problem["spacing"], float)
    phi = _dist(np.asarray(field, float).reshape(shape), spacing)
    if not np.isfinite(phi).all():
        raise ValueError("nonfinite")
    total = 0.0
    for o in problem.get("observations") or []:
        values = np.asarray(o["values"], float)
        w = float(o.get("weight", 1.0))
        fin = np.isfinite(values)
        d, y = phi[fin], values[fin]
        contact = phi[values == 0.5]
        logi = np.logaddexp(0, d[y == 0]).sum() + np.logaddexp(0, -d[y > 0]).sum()
        bias = float(contact.mean() ** 2) if contact.size else 0.0
        var = float(np.mean(contact ** 2)) if contact.size else 0.0
        total += w * (logi + bias + var)
    ref = problem.get("sketch")
    if ref is not None:
        try:
            total += _sketch_loss(np.asarray(ref), phi, spacing,
                                  np.asarray(problem["origin"], float))
        except Exception:
            total += 1e6
    return float(total)


def _labels(problem, origin, spacing):
    pts, t, w = [], [], []
    for o in problem.get("observations") or []:
        v = np.asarray(o["values"], float)
        idx = np.argwhere(np.isfinite(v))
        if not len(idx):
            continue
        vv = v[np.isfinite(v)]
        pts.append(np.stack([origin[i] + (idx[:, i] + 0.5) * spacing[i]
                             for i in range(idx.shape[1])], axis=1))
        t.append(np.where(vv > 0.75, 1.0, np.where(vv < 0.25, -1.0, 0.0)))
        w.append(np.full(len(vv), float(o.get("weight", 1.0))))
    if not pts:
        return None
    return np.concatenate(pts), np.concatenate(t), np.concatenate(w)


def _analytic_heights(C, P, t):
    zc = C[:, 2]
    lab = np.argmin(((P[:, None, :2] - C[None, :, :2]) ** 2).sum(-1), 1)
    def fun(h):
        phi = P[:, 2] - h[lab]
        delta = zc - h
        logi = (np.logaddexp(0, phi[t == -1.0]).sum()
                + np.logaddexp(0, -phi[t == 0.0]).sum()
                + np.logaddexp(0, -phi[t == 1.0]).sum())
        return float(logi + np.mean(delta) ** 2 + np.mean(delta ** 2))
    if minimize is not None:
        try:
            o = minimize(fun, zc + 0.25, method="Powell",
                         bounds=[(z - 1.5, z + 0.5) for z in zc],
                         options=dict(maxfev=220, xtol=1e-4, ftol=1e-8))
            return np.asarray(o.x, float)
        except Exception:
            pass
    return zc + 0.25


def _rbf_field(P, t, w, X, shape):
    span = float(np.max(np.ptp(X, axis=0))) or 1.0
    l = 0.11 * span
    d2 = ((P[:, None] - P[None, :]) ** 2).sum(-1)
    K = np.exp(-d2 / (2 * l * l))
    try:
        c = np.linalg.solve(K + 0.05 * np.eye(len(P)), t * w)
    except np.linalg.LinAlgError:
        c = np.linalg.lstsq(K + 0.05 * np.eye(len(P)), t * w, rcond=None)[0]
    pred = np.empty(len(X))
    for s in range(0, len(X), 20000):
        E = X[s:s + 20000]
        pred[s:s + 20000] = np.exp(
            -((E[:, None] - P[None, :]) ** 2).sum(-1) / (2 * l * l)) @ c
    return pred.reshape(shape)


def solve(problem, budget, seed):
    global _DEADLINE
    shape = tuple(problem["shape"])
    origin = np.asarray(problem["origin"], float)
    spacing = np.asarray(problem["spacing"], float)
    coords = _coords(shape, origin, spacing)
    X = coords.reshape(-1, len(shape))
    init = np.asarray(problem["initial_level_set"], float)
    init = np.where(np.isfinite(init), init, 0.0)
    if not (np.any(init > 0) and np.any(init < 0)):
        init = coords[..., 0] - np.mean(coords[..., 0])

    best, best_field = np.inf, np.ascontiguousarray(init.reshape(shape))
    def take(field):
        return np.ascontiguousarray(np.asarray(field, float).reshape(shape))
    def consider(field):
        nonlocal best, best_field
        f = take(field)
        if not np.isfinite(f).all() or not (np.any(f > 0) and np.any(f < 0)):
            return np.inf
        try:
            l = _loss(problem, f)
        except Exception:
            return np.inf
        if np.isfinite(l) and l < best:
            best, best_field = l, f
        return l

    _wall = float(budget.get("wall_time_seconds", 120.0)) if isinstance(budget, dict) else 120.0
    _DEADLINE = _pytime.monotonic() + 0.985 * max(30.0, _wall)
    consider(init)
    lab = _labels(problem, origin, spacing)
    if lab is not None:
        P, t, w = lab
        C = P[t == 0]
        if len(shape) == 3 and 2 <= len(C) <= 16 and len(P) >= 3:
            zc = C[:, 2]
            n = len(C)
            h0 = _analytic_heights(C, P, t)
            def base(x):
                h = x[:n]; g = float(x[n])
                R = np.exp(np.clip(x[n + 1:2 * n + 1], np.log(1.5), np.log(40.0)))
                ox = x[2 * n + 1:3 * n + 1]; oy = x[3 * n + 1:4 * n + 1]
                pp = np.clip(x[4 * n + 1:5 * n + 1], 1.5, 8.0)
                dxm = X[:, 0:1] - (C[None, :, 0] + ox[None, :])
                dym = X[:, 1:2] - (C[None, :, 1] + oy[None, :])
                dc = np.sqrt(dxm ** 2 + dym ** 2)
                wgt = np.maximum(1.0 - dc / R[None, :], 0.0) ** pp[None, :]
                s = wgt.sum(1, keepdims=True)
                bad = s[:, 0] <= 0
                if np.any(bad):
                    wgt[bad, :] = 0.0
                    wgt[bad, np.argmin(dc[bad], 1)] = 1.0
                    s = wgt.sum(1, keepdims=True)
                return X[:, 2] - ((wgt @ h) / s[:, 0] + g)
            def score_x(x):
                f = base(x)
                if not (f > 0).any() or not (f < 0).any(): return 1e9
                try: return _loss(problem, take(f))
                except Exception: return 1e9
            seed = (np.inf, None)
            for R0 in (4.0, 6.0, 8.0, 11.0, 15.0):
                for hh in (h0, zc + 0.25):
                    x0 = np.r_[hh, 0.0, np.full(n, np.log(R0)),
                               np.zeros(n), np.zeros(n), np.full(n, 4.0)]
                    l = score_x(x0)
                    if l < seed[0]: seed = (l, x0)
            xb = seed[1]
            if minimize is not None and xb is not None:
                lo = ([(z - 1.5, z + 0.5) for z in zc] + [(-0.6, 0.6)]
                      + [(np.log(1.5), np.log(40.0))] * n
                      + [(-2.5, 2.5)] * (2 * n) + [(1.5, 8.0)] * n)
                try:
                    opt = minimize(score_x, xb, method="Powell", bounds=lo,
                                   options=dict(maxfev=450, maxiter=130,
                                                xtol=1e-5, ftol=1e-10))
                    xb = opt.x
                    consider(base(xb))
                except Exception:
                    pass
            pairs = [(i, j) for i in range(n) for j in range(i + 1, n)][:12]
            if pairs:
                mids = np.array([0.5 * (C[i, :2] + C[j, :2]) for i, j in pairs])
                D = np.array([np.linalg.norm(C[i, :2] - C[j, :2]) for i, j in pairs])
                dm = np.sqrt((X[:, 0:1] - mids[None, :, 0]) ** 2
                             + (X[:, 1:2] - mids[None, :, 1]) ** 2)
                def cfield(c):
                    f0 = base(xb)
                    fr = float(np.clip(c[-1], 0.20, 0.49))
                    r = np.maximum(fr * D, 1e-6)
                    ker = np.maximum(1.0 - dm / r[None, :], 0.0) ** 2
                    return f0 - (ker @ c[:-1])
                def cfun(c):
                    f = cfield(c)
                    if not (f > 0).any() or not (f < 0).any(): return 1e9
                    try: return _loss(problem, take(f))
                    except Exception: return 1e9
                try:
                    co = minimize(cfun, np.r_[np.zeros(len(pairs)), 0.45],
                                  method="Powell",
                                  bounds=[(-2.0, 2.0)] * len(pairs) + [(0.20, 0.49)],
                                  options=dict(maxfev=200, maxiter=70,
                                               xtol=1e-5, ftol=1e-10))
                    consider(cfield(co.x))
                except Exception:
                    pass
        consider(_rbf_field(P, t, w, X, shape))
        if len(shape) == 3 and len(C) >= 2:
            phi0 = _dist(best_field, spacing)
            xs = coords[:, :, 0, 0]; ys = coords[:, :, 0, 1]
            bases = []
            def gauss(d, r): return np.exp(-0.5 * (d / r) ** 2)
            xn = (xs - np.mean(xs)) / (np.std(xs) + 1e-9)
            yn = (ys - np.mean(ys)) / (np.std(ys) + 1e-9)
            bases += [np.ones_like(xn), xn, yn, xn * xn, yn * yn, xn * yn]
            pairs = [(i, j) for i in range(len(C)) for j in range(i + 1, len(C))][:12]
            mids = np.array([0.5 * (C[i, :2] + C[j, :2]) for i, j in pairs])
            pd = np.array([np.linalg.norm(C[i, :2] - C[j, :2]) for i, j in pairs])
            for q in range(len(pairs)):
                d = np.sqrt((xs - mids[q, 0]) ** 2 + (ys - mids[q, 1]) ** 2)
                bases.append(gauss(d, 0.30 * pd[q]))
            triples = [(i, j, k) for i in range(len(C)) for j in range(i + 1, len(C))
                       for k in range(j + 1, len(C))][:4]
            for tr in triples:
                ctr = C[list(tr), :2].mean(0)
                d = np.sqrt((xs - ctr[0]) ** 2 + (ys - ctr[1]) ** 2)
                bases.append(gauss(d, 3.0))
            for q in range(len(C)):
                d = np.sqrt((xs - C[q, 0]) ** 2 + (ys - C[q, 1]) ** 2)
                bases.append(gauss(d, 3.0))
            B = np.stack([np.asarray(b, float).reshape(-1) for b in bases], 0)
            if B.shape[0] > 0 and np.isfinite(B).all():
                aa = np.zeros(B.shape[0])
                def cfield_c(aa):
                    c2 = (B.T @ aa).reshape(shape[0], shape[1])
                    return phi0 - c2[:, :, None]
                def score_c(aa):
                    f = cfield_c(aa)
                    if not np.isfinite(f).all() or not (np.any(f > 0) and np.any(f < 0)):
                        return 1e9
                    try:
                        return _loss(problem, take(f))
                    except Exception:
                        return 1e9
                cur = score_c(aa)
                for st in (0.15, 0.08, 0.04, 0.02, 0.01, 0.005):
                    for _pass in range(2):
                        imp = False
                        for ii in range(len(aa)):
                            for ss in (st, -st):
                                a2 = aa.copy(); a2[ii] += ss
                                L = score_c(a2)
                                if L < cur - 1e-10:
                                    cur = L; aa = a2; imp = True
                        if not imp:
                            break
                consider(cfield_c(aa))

    ref = problem.get("sketch")
    if ref is not None:
        basef = np.where(np.asarray(ref) > 0.5, 1.0, -1.0)
        consider(basef)
        consider(basef + 0.25 * best_field / (np.std(best_field) + 1e-9))
    try:
        best_field = _grid_correct(problem, best_field, shape, origin, spacing, 11, 9, 600, (0.1, 0.05, 0.02, 0.01, 0.005, 0.002), 'height')
    except Exception:
        pass
    try:
        if lab is not None and len(lab[0]) >= 8 and len(shape) == 3:
            best_field = _polish_fast(problem, best_field)
    except Exception:
        pass
    _refine_modes = [('tilt', 320), ('pairmid', 200), ('wendl', 220), ('wendl2', 180), ('anisorot', 200), ('anisofine', 150)]
    for _rm, _rc in _refine_modes:
        try:
            _pf = _post_refine(problem, best_field, shape, origin, spacing, _rm, _rc)
            if _pf is not None:
                _pl = _loss(problem, _pf)
                _bl = _loss(problem, best_field)
                if np.isfinite(_pl) and _pl < _bl:
                    best_field = _pf
        except Exception:
            pass
    return best_field.astype(np.float64)


def _grid_correct(problem, field, shape, origin, spacing, nx, ny, cap, steps, mode):
    """Bicubic 2-D height correction on the zero level set, exact-loss coordinate descent."""
    if len(shape) != 3:
        return field
    try:
        from scipy.interpolate import RectBivariateSpline
        sp = np.asarray(spacing, float)
        org = np.asarray(origin, float)
        shape = tuple(shape)
        Xd = org[0] + (np.arange(shape[0]) + 0.5) * sp[0]
        Yd = org[1] + (np.arange(shape[1]) + 0.5) * sp[1]
        Zd = org[2] + (np.arange(shape[2]) + 0.5) * sp[2]
        XXd, YYd = np.meshgrid(Xd, Yd, indexing='ij')
        base = np.ascontiguousarray(np.asarray(field, float).reshape(shape))
        if mode == 'dist':
            seedf = _dist(base, spacing)
        elif mode == 'raw':
            seedf = base
        else:
            h0 = np.empty((shape[0], shape[1]), float)
            for i in range(shape[0]):
                for j in range(shape[1]):
                    col = base[i, j, :]
                    pos = np.nonzero(col > 0)[0]
                    if pos.size == 0:
                        h0[i, j] = Zd[0] - 0.5 * sp[2]
                    elif pos[0] == 0:
                        h0[i, j] = Zd[0]
                    else:
                        k = int(pos[0])
                        c0, c1 = float(col[k - 1]), float(col[k])
                        if c1 > c0:
                            h0[i, j] = Zd[k - 1] + (-c0) / (c1 - c0) * sp[2]
                        else:
                            h0[i, j] = Zd[k]
            seedf = Zd[None, None, :] - h0[:, :, None]
        xg = np.linspace(Xd[0], Xd[-1], nx)
        yg = np.linspace(Yd[0], Yd[-1], ny)
        C = np.zeros((nx, ny), float)
        cnt = [0]

        def ev(Cg):
            spl = RectBivariateSpline(xg, yg, Cg, kx=min(3, nx - 1), ky=min(3, ny - 1))
            cc = spl(XXd[:, 0], YYd[0, :])
            F = seedf - cc[:, :, None]
            if not np.isfinite(F).all() or not (np.any(F > 0) and np.any(F < 0)):
                return np.inf
            try:
                return _loss(problem, F)
            except Exception:
                return np.inf

        cur = ev(C)
        if not np.isfinite(cur):
            return base
        try:
            orig = _loss(problem, base)
        except Exception:
            orig = np.inf
        done = False
        for step in steps:
            for _pass in range(2):
                imp = False
                for i in range(nx):
                    for j in range(ny):
                        for s in (step, -step):
                            if cnt[0] >= cap:
                                done = True
                                break
                            cnt[0] += 1
                            C2 = C.copy()
                            C2[i, j] += s
                            L = ev(C2)
                            if L < cur - 1e-12:
                                cur = L
                                C = C2
                                imp = True
                        if done:
                            break
                    if done:
                        break
                if done or not imp:
                    break
            if done:
                break
        spl = RectBivariateSpline(xg, yg, C, kx=min(3, nx - 1), ky=min(3, ny - 1))
        cc = spl(XXd[:, 0], YYd[0, :])
        F = seedf - cc[:, :, None]
        if (np.isfinite(cur) and cur < orig - 1e-12 and np.isfinite(F).all()
                and np.any(F > 0) and np.any(F < 0)):
            return F
        return base
    except Exception:
        return field


def _polish_fast(problem, field):
    """Two-scale exact-loss polish: coarse multi-bump basis, then a richer basis."""
    import numpy as _np
    fld = _np.ascontiguousarray(_np.asarray(field, dtype=_np.float64))
    shape = tuple(problem.get("shape", fld.shape))
    if len(shape) != 3 or skfmm is None or problem.get("sketch") is not None:
        return fld
    try:
        origin = _np.asarray(problem["origin"], float)
        spacing = _np.asarray(problem["spacing"], float)
        axes = [origin[i] + (_np.arange(n) + 0.5) * spacing[i] for i, n in enumerate(shape)]
        c3 = _np.stack(_np.meshgrid(*axes, indexing="ij"), axis=-1)
        X2 = c3[:, :, 0, 0]; Y2 = c3[:, :, 0, 1]
        pts, tg = [], []
        for o in problem.get("observations") or []:
            v = _np.asarray(o["values"], float)
            idx = _np.argwhere(_np.isfinite(v))
            if not len(idx):
                continue
            vv = v[_np.isfinite(v)]
            pts.append(_np.stack([origin[i] + (idx[:, i] + 0.5) * spacing[i]
                                  for i in range(3)], axis=1))
            tg.append(_np.where(vv > 0.75, 1.0, _np.where(vv < 0.25, -1.0, 0.0)))
        if not pts:
            return fld
        P = _np.concatenate(pts, 0); T = _np.concatenate(tg, 0)
        C = P[T == 0.0]
        if len(C) < 2 or len(P) < 3:
            return fld
        vals_all = [(_np.asarray(o["values"], float), float(o.get("weight", 1.0)))
                    for o in problem.get("observations") or []]

        def ploss(f):
            phi = skfmm.distance(_np.ascontiguousarray(f, dtype=_np.float64),
                                 dx=tuple(spacing))
            if not _np.isfinite(phi).all():
                return 1e18
            tot = 0.0
            for v, w in vals_all:
                fin = _np.isfinite(v)
                d = phi[fin]; y = v[fin]; ct = phi[v == 0.5]
                lg = (_np.logaddexp(0, d[y == 0]).sum()
                      + _np.logaddexp(0, -d[y > 0]).sum())
                b = float(ct.mean() ** 2) if ct.size else 0.0
                q = float(_np.mean(ct ** 2)) if ct.size else 0.0
                tot += w * (float(lg) + b + q)
            return tot

        n = len(C)
        XF = X2.reshape(-1); YF = Y2.reshape(-1)
        xn = (X2 - X2.mean()) / (X2.std() + 1e-9)
        yn = (Y2 - Y2.mean()) / (Y2.std() + 1e-9)
        qb = [_np.ones_like(XF), xn.reshape(-1), yn.reshape(-1),
              (xn * xn).reshape(-1), (yn * yn).reshape(-1), (xn * yn).reshape(-1)]

        def gs(mx, my, sig):
            return _np.exp(-0.5 * ((XF - mx) ** 2 + (YF - my) ** 2) / sig ** 2)

        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)][:12]
        paird = [float(_np.linalg.norm(C[i, :2] - C[j, :2])) for i, j in pairs]
        mids = [0.5 * (C[i, :2] + C[j, :2]) for i, j in pairs]
        B1 = list(qb)
        for q in range(min(n, 8)):
            for sig in (0.8, 1.2, 2.0, 3.0, 5.0):
                B1.append(gs(C[q, 0], C[q, 1], sig))
        for (i, j), dd, (mx, my) in zip(pairs, paird, mids):
            for fr in (0.25, 0.40):
                sig = max(0.6, fr * dd)
                g = gs(mx, my, sig)
                B1.append(g); B1.append(g * (XF - mx) / sig); B1.append(g * (YF - my) / sig)
        B2 = list(qb)
        for q in range(min(n, 8)):
            for sig in (0.5, 0.8, 1.2, 1.8, 2.5, 3.5, 5.0, 7.0):
                B2.append(gs(C[q, 0], C[q, 1], sig))
        for (i, j), dd, (mx, my) in zip(pairs, paird, mids):
            for fr in (0.20, 0.30, 0.45):
                sig = max(0.6, fr * dd)
                g = gs(mx, my, sig)
                B2.append(g); B2.append(g * (XF - mx) / sig); B2.append(g * (YF - my) / sig)
        nu = min(n, 5)
        triples = [(i, j, k) for i in range(nu) for j in range(i + 1, nu)
                   for k in range(j + 1, nu)][:4]
        for tr in triples:
            cx = C[list(tr), 0].mean(); cy = C[list(tr), 1].mean()
            for sig in (2.0, 4.0):
                B2.append(gs(cx, cy, sig))

        def run(F0, B, steps):
            Bm = _np.stack([_np.asarray(b, float).reshape(-1) for b in B], 0)
            nb = Bm.shape[0]

            def field_of(c):
                return F0 - (Bm.T @ c).reshape(shape[0], shape[1])[:, :, None]

            c = _np.zeros(nb); cur = ploss(F0); bestF = F0; bestL = cur
            for st in steps:
                imp = False
                for i in range(nb):
                    for ss in (st, -st):
                        c2 = c.copy(); c2[i] += ss
                        L = ploss(field_of(c2))
                        if _np.isfinite(L) and L < cur - 1e-12:
                            cur = L; c = c2; imp = True
                if imp and cur < bestL:
                    bestL = cur; bestF = field_of(c)
                if not imp:
                    break
            return bestF, bestL

        F1, L1 = run(fld, B1, (0.02, 0.008, 0.003, 0.001))
        F2, L2 = run(F1, B2, (0.015, 0.006, 0.002, 0.001))
        if L2 < ploss(fld):
            return _np.ascontiguousarray(F2, dtype=_np.float64)
        return fld
    except Exception:
        return fld


def _post_refine(problem, field, shape, origin, spacing, mode, cap):
    """Extra exact-loss local correction stage appended after the incumbent pipeline."""
    import numpy as _np
    try:
        fld = _np.ascontiguousarray(_np.asarray(field, float).reshape(shape))
        if len(shape) != 3 or skfmm is None or problem.get('sketch') is not None:
            return fld
        if mode == 'finegrid':
            return _grid_correct(problem, fld, shape, origin, spacing, 17, 13, cap,
                                 (0.05, 0.02, 0.01, 0.005, 0.002, 0.001), 'height')
        origin = _np.asarray(origin, float); spacing = _np.asarray(spacing, float)
        axes = [origin[i] + (_np.arange(n) + 0.5) * spacing[i] for i, n in enumerate(shape)]
        c3 = _np.stack(_np.meshgrid(*axes, indexing='ij'), axis=-1)
        X2 = c3[:, :, 0, 0]; Y2 = c3[:, :, 0, 1]
        pts, tg = [], []
        for o in problem.get('observations') or []:
            v = _np.asarray(o['values'], float)
            idx = _np.argwhere(_np.isfinite(v))
            if not len(idx):
                continue
            vv = v[_np.isfinite(v)]
            pts.append(_np.stack([origin[i] + (idx[:, i] + 0.5) * spacing[i]
                                  for i in range(3)], axis=1))
            tg.append(_np.where(vv > 0.75, 1.0, _np.where(vv < 0.25, -1.0, 0.0)))
        if not pts:
            return fld
        P = _np.concatenate(pts, 0); T = _np.concatenate(tg, 0)
        C = P[T == 0.0]
        if len(C) < 2 or len(C) > 16 or len(P) < 3:
            return fld
        phi0 = _dist(fld, spacing)
        if not _np.isfinite(phi0).all():
            return fld
        XF = X2.reshape(-1); YF = Y2.reshape(-1)
        xn = (X2 - X2.mean()) / (X2.std() + 1e-9)
        yn = (Y2 - Y2.mean()) / (Y2.std() + 1e-9)
        bases = [np.ones_like(XF), xn.reshape(-1), yn.reshape(-1),
                 (xn * xn).reshape(-1), (yn * yn).reshape(-1), (xn * yn).reshape(-1)]
        def gs(mx, my, sig):
            return _np.exp(-0.5 * ((XF - mx) ** 2 + (YF - my) ** 2) / sig ** 2)
        n = len(C)
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)][:12]
        paird = [float(_np.linalg.norm(C[i, :2] - C[j, :2])) for i, j in pairs]
        mids = [0.5 * (C[i, :2] + C[j, :2]) for i, j in pairs]
        if mode == 'narrow':
            for q in range(min(n, 8)):
                for sig in (0.45, 0.70, 1.10, 1.70, 2.60):
                    bases.append(gs(C[q, 0], C[q, 1], sig))
            for (i, j), dd, (mx, my) in zip(pairs, paird, mids):
                for fr in (0.10, 0.18, 0.28):
                    sig = max(0.45, fr * dd)
                    g = gs(mx, my, sig)
                    bases.append(g)
                    bases.append(g * (XF - mx) / sig)
                    bases.append(g * (YF - my) / sig)
        elif mode == 'tilt':
            for q in range(min(n, 8)):
                for sig in (1.4, 2.4, 3.8):
                    g = gs(C[q, 0], C[q, 1], sig)
                    dx = (XF - C[q, 0]) / sig; dy = (YF - C[q, 1]) / sig
                    bases.append(g); bases.append(g * dx); bases.append(g * dy)
                    bases.append(g * (dx * dx - dy * dy) * 0.5)
                    bases.append(g * dx * dy)
            for (i, j), dd, (mx, my) in zip(pairs, paird, mids):
                sig = max(0.6, 0.30 * dd)
                g = gs(mx, my, sig)
                bases.append(g)
                bases.append(g * (XF - mx) / sig)
                bases.append(g * (YF - my) / sig)
        elif mode == 'pairmid':
            for (i, j), dd, (mx, my) in zip(pairs, paird, mids):
                for fr in (0.08, 0.14, 0.22, 0.32):
                    sig = max(0.45, fr * dd)
                    g = gs(mx, my, sig)
                    bases.append(g)
                    bases.append(g * (XF - mx) / sig)
                    bases.append(g * (YF - my) / sig)
                for sig in (0.6, 1.2):
                    g = gs(mx, my, sig)
                    bases.append(g)
        elif mode == 'rot':
            angs = (0.0, 0.7853981633974483, 1.5707963267948966)
            for q in range(min(n, 3)):
                for sig in (0.8, 1.8, 3.2):
                    for th in angs:
                        ct, st = np.cos(th), np.sin(th)
                        dx = (XF - C[q, 0]) / sig
                        dy = (YF - C[q, 1]) / sig
                        u = ct * dx + st * dy
                        v = -st * dx + ct * dy
                        g = _np.exp(-0.5 * (u * u + v * v))
                        bases.append(g)
                        bases.append(g * u)
                        bases.append(g * v)
                        bases.append(g * (u * u - v * v) * 0.5)
                        bases.append(g * u * v)
            for (i, j), dd, (mx, my) in zip(pairs, paird, mids):
                for th in angs:
                    sig = max(0.6, 0.28 * dd)
                    ct, st = np.cos(th), np.sin(th)
                    dx = (XF - mx) / sig
                    dy = (YF - my) / sig
                    u = ct * dx + st * dy
                    v = -st * dx + ct * dy
                    g = _np.exp(-0.5 * (u * u + v * v))
                    bases.append(g); bases.append(g * u); bases.append(g * v)
        elif mode == 'micro':
            for q in range(min(n, 6)):
                for sig in (0.35, 0.55, 0.85, 1.3):
                    bases.append(gs(C[q, 0], C[q, 1], sig))
                rr = _np.sqrt((XF - C[q, 0]) ** 2 + (YF - C[q, 1]) ** 2)
                for rad in (0.9, 1.6, 2.8):
                    bases.append(_np.maximum(1.0 - rr / rad, 0.0) ** 2)
            for (i, j), dd, (mx, my) in zip(pairs, paird, mids):
                for fr in (0.07, 0.13, 0.22, 0.34):
                    sig = max(0.4, fr * dd)
                    g = gs(mx, my, sig)
                    bases.append(g)
                    bases.append(g * (XF - mx) / sig)
                    bases.append(g * (YF - my) / sig)
        elif mode == 'wendl':
            def _c2(rr, rad):
                qq = _np.maximum(1.0 - rr / rad, 0.0)
                return (qq ** 4) * (4.0 * rr / rad + 1.0)
            for q in range(min(n, 6)):
                rr = _np.sqrt((XF - C[q, 0]) ** 2 + (YF - C[q, 1]) ** 2)
                for rad in (0.8, 1.4, 2.4, 3.8):
                    w = _c2(rr, rad)
                    bases.append(w)
                    bases.append(w * (XF - C[q, 0]))
                    bases.append(w * (YF - C[q, 1]))
            for (i, j), dd, (mx, my) in zip(pairs, paird, mids):
                rr = _np.sqrt((XF - mx) ** 2 + (YF - my) ** 2)
                for fr in (0.15, 0.28, 0.45):
                    rad = max(0.6, fr * dd)
                    w = _c2(rr, rad)
                    bases.append(w)
                    bases.append(w * (XF - mx))
                    bases.append(w * (YF - my))

        elif mode == 'wendl2':
            def _c2b(rr, rad):
                qq = _np.maximum(1.0 - rr / rad, 0.0)
                return (qq ** 4) * (4.0 * rr / rad + 1.0)
            for q in range(min(n, 4)):
                dx = XF - C[q, 0]; dy = YF - C[q, 1]
                for sx, sy in ((0.6, 1.8), (1.2, 3.0), (2.2, 4.5)):
                    rr = _np.sqrt((dx / sx) ** 2 + (dy / sy) ** 2)
                    for rad in (1.0, 1.8, 3.0):
                        w = _c2b(rr, rad)
                        bases.append(w)
                        bases.append(w * dx / sx)
                        bases.append(w * dy / sy)
            for (i, j), dd, (mx, my) in zip(pairs, paird, mids):
                dx = XF - mx; dy = YF - my
                for fr in (0.12, 0.22, 0.36):
                    rad = max(0.6, fr * dd)
                    rr = _np.sqrt(dx * dx + dy * dy)
                    w = _c2b(rr, rad)
                    bases.append(w); bases.append(w * dx); bases.append(w * dy)

        elif mode == 'aniso4':
            def _c4a(rr, rad):
                qq = _np.maximum(1.0 - rr / rad, 0.0)
                s = rr / rad
                return (qq ** 6) * (35.0 * s * s + 18.0 * s + 3.0)
            for q in range(min(n, 6)):
                dx = XF - C[q, 0]; dy = YF - C[q, 1]
                for sx, sy in ((0.6, 1.8), (1.4, 3.6)):
                    rr = _np.sqrt((dx / sx) ** 2 + (dy / sy) ** 2)
                    for rad in (1.2, 2.8):
                        w = _c4a(rr, rad)
                        bases.append(w)
                        bases.append(w * dx / sx)
                        bases.append(w * dy / sy)
            for (i, j), dd, (mx, my) in zip(pairs, paird, mids):
                rr = _np.sqrt((XF - mx) ** 2 + (YF - my) ** 2)
                for fr in (0.18, 0.36, 0.55):
                    rad = max(0.6, fr * dd)
                    w = _c4a(rr, rad)
                    bases.append(w)
                    bases.append(w * (XF - mx))
                    bases.append(w * (YF - my))

        elif mode == 'anisotor' or mode == 'anisorot':
            def _c2r(rr, rad):
                qq = _np.maximum(1.0 - rr / rad, 0.0)
                return (qq ** 4) * (4.0 * rr / rad + 1.0)
            for q in range(min(n, 4)):
                dx = XF - C[q, 0]; dy = YF - C[q, 1]
                for th in (0.0, 0.7853981633974483, 1.5707963267948966):
                    ct, st = _np.cos(th), _np.sin(th)
                    u = ct * dx / 0.9 + st * dy / 2.4
                    v = -st * dx / 0.9 + ct * dy / 2.4
                    rr = _np.sqrt(u * u + v * v)
                    for rad in (1.2, 2.6):
                        w = _c2r(rr, rad)
                        bases.append(w); bases.append(w * u); bases.append(w * v)
            for (i, j), dd, (mx, my) in zip(pairs, paird, mids):
                dx = XF - mx; dy = YF - my
                for th in (0.0, 0.7853981633974483, 1.5707963267948966):
                    ct, st = _np.cos(th), _np.sin(th)
                    u = ct * dx / 0.9 + st * dy / 2.4
                    v = -st * dx / 0.9 + ct * dy / 2.4
                    rr = _np.sqrt(u * u + v * v)
                    for fr in (0.20, 0.40):
                        w = _c2r(rr, max(0.6, fr * dd))
                        bases.append(w); bases.append(w * (XF - mx)); bases.append(w * (YF - my))

        elif mode == 'anisofine':
            def _c2n(rr, rad):
                qq = _np.maximum(1.0 - rr / rad, 0.0)
                return (qq ** 4) * (4.0 * rr / rad + 1.0)
            for q in range(min(n, 4)):
                dx = XF - C[q, 0]; dy = YF - C[q, 1]
                for sx, sy in ((0.6, 1.8), (1.2, 3.0), (2.2, 4.5)):
                    rr = _np.sqrt((dx / sx) ** 2 + (dy / sy) ** 2)
                    for rad in (1.0, 1.8, 3.0):
                        w = _c2n(rr, rad)
                        bases.append(w); bases.append(w * dx / sx); bases.append(w * dy / sy)
            for (i, j), dd, (mx, my) in zip(pairs, paird, mids):
                dx = XF - mx; dy = YF - my
                for fr in (0.12, 0.22, 0.36):
                    rr = _np.sqrt(dx * dx + dy * dy)
                    w = _c2n(rr, max(0.6, fr * dd))
                    bases.append(w); bases.append(w * dx); bases.append(w * dy)

        elif mode == 'anisoq':
            def _c2q(rr, rad):
                qq = _np.maximum(1.0 - rr / rad, 0.0)
                return (qq ** 4) * (4.0 * rr / rad + 1.0)
            for q in range(min(n, 4)):
                dx = XF - C[q, 0]; dy = YF - C[q, 1]
                for sx, sy in ((0.7, 2.0), (1.5, 3.8)):
                    u = dx / sx; v = dy / sy
                    rr = _np.sqrt(u * u + v * v)
                    for rad in (1.1, 2.4):
                        w = _c2q(rr, rad)
                        bases.append(w); bases.append(w * u); bases.append(w * v)
                        bases.append(w * (u * u - v * v) * 0.5); bases.append(w * u * v)
            for (i, j), dd, (mx, my) in zip(pairs, paird, mids):
                dx = XF - mx; dy = YF - my
                for fr in (0.16, 0.34):
                    rr = _np.sqrt(dx * dx + dy * dy)
                    w = _c2q(rr, max(0.6, fr * dd))
                    bases.append(w); bases.append(w * dx); bases.append(w * dy)
        else:
            return fld
        B = _np.stack([_np.asarray(b, float).reshape(-1) for b in bases], 0)
        if B.shape[0] == 0 or not _np.isfinite(B).all():
            return fld
        nb = B.shape[0]
        def field_of(c):
            return phi0 - (B.T @ c).reshape(shape[0], shape[1])[:, :, None]
        c = _np.zeros(nb)
        cur = float(_loss(problem, fld))
        bestF = fld; bestL = cur
        steps = ((0.001, 0.0005, 0.0002, 0.0001, 0.00005) if mode == 'wendlfine'
                 else (0.05, 0.02, 0.008, 0.003, 0.001))
        cnt = [0]
        for st in steps:
            imp = False
            for ii in range(nb):
                for ss in (st, -st):
                    if _DEADLINE is not None and _pytime.monotonic() > _DEADLINE:
                        return _np.ascontiguousarray(bestF, dtype=_np.float64)
                    if cnt[0] >= cap:
                        break
                    cnt[0] += 1
                    c2 = c.copy(); c2[ii] += ss
                    F = field_of(c2)
                    if not (F > 0).any() or not (F < 0).any():
                        continue
                    try:
                        L = float(_loss(problem, F))
                    except Exception:
                        continue
                    if _np.isfinite(L) and L < cur - 1e-12:
                        cur = L; c = c2; imp = True
                        if L < bestL - 1e-12:
                            bestL = L; bestF = F
                if cnt[0] >= cap:
                    break
            if cnt[0] >= cap or not imp:
                break
        return _np.ascontiguousarray(bestF, dtype=_np.float64)
    except Exception:
        return np.ascontiguousarray(np.asarray(field, float).reshape(shape))
