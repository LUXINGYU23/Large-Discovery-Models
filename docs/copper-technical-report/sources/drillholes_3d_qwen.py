"""Per-hole rotated broad/narrow exponential mixture PLUS a per-hole flat-top kernel.

Champion-relative change: the measured best source (a8d76fa0, loss 6.514931)
owns a rotated broad/narrow exponential mixture per drillhole. This variant adds a
third kernel per hole, w_p = plate_i * exp(-r_i**q_i) with q_i in [1.2,8], i.e. a
nearly flat plateau out to r~1 followed by a steep flank (the mechanism of the
6.5149314 sibling 4dd7e8f0), while keeping the per-hole rotation. plate_i=0
reproduces the parent exactly, so the family strictly contains it.
"""

import time
import numpy as np
import skfmm
from scipy.optimize import minimize

MODE = "plateau"


def _gather(problem):
    shape = tuple(int(n) for n in problem["shape"])
    origin = np.asarray(problem["origin"], float)
    spacing = np.asarray(problem["spacing"], float)
    idxs, vals, obsl = [], [], []
    for o in problem["observations"]:
        v = np.asarray(o["values"], float)
        fin = np.isfinite(v)
        if fin.any():
            ii = np.argwhere(fin)
            idxs.append(ii)
            vals.append(v[fin])
            obsl.append((ii, v[fin], float(o.get("weight", 1.0))))
    if not vals:
        return None
    idx = np.concatenate(idxs).astype(np.int64)
    val = np.concatenate(vals)
    coords = origin + (idx + 0.5) * spacing
    return shape, origin, spacing, idx, val, coords, obsl


def _holes(idx, val, coords):
    ndim = idx.shape[1]
    groups = {}
    for n, row in enumerate(idx):
        groups.setdefault(tuple(int(t) for t in row[:ndim - 1]), []).append(n)
    out = []
    for key in sorted(groups):
        mem = np.asarray(groups[key])
        lab = val[mem]
        zz = coords[mem, -1]
        cm = np.abs(lab - 0.5) < 1e-9
        if not cm.any():
            return None
        c = float(zz[cm].mean())
        pos = zz[lab > 0.5]
        neg = zz[lab == 0.0]
        if pos.size == 0 or pos.min() <= c:
            return None
        if neg.size and neg.max() >= c:
            return None
        gap = float(np.min(c - neg)) if neg.size else 1.0
        out.append((coords[mem[0], :-1].copy(), c, max(min(gap, 1.0), 1e-6)))
    return out


def _loss_phi(phi, obsl):
    tot = 0.0
    for ii, vv, w in obsl:
        d = phi[tuple(ii.T)]
        out = float(np.logaddexp(0.0, d[vv == 0.0]).sum())
        out += float(np.logaddexp(0.0, -d[vv > 0.0]).sum())
        c = d[np.abs(vv - 0.5) < 1e-9]
        if c.size:
            out += float(c.mean() ** 2) + float(np.mean(c ** 2))
        tot += w * out
    return tot


def _sign_ok(f, idx, val):
    d = f[tuple(idx.T)]
    ext = np.abs(val - 0.0) < 1e-12
    pos = val > 0.5
    con = np.abs(val - 0.5) < 1e-9
    return (np.isfinite(f).all() and np.all(d[pos] > 1e-6)
            and np.all(d[ext] < -1e-6)
            and (not con.any() or np.all(np.abs(d[con]) < 0.999)))


def _fit(evalf, x0, bounds, tlim, t0, rng):
    lo = np.array([b[0] for b in bounds], float)
    hi = np.array([b[1] for b in bounds], float)
    nd = len(x0)
    best = [np.inf, None]

    def left():
        return tlim - (time.time() - t0)

    def ev(x):
        if left() <= 0.0:
            return 1e3
        x = np.clip(np.asarray(x, float), lo, hi)
        if not np.isfinite(x).all():
            return 1e3
        try:
            L = float(evalf(x))
        except Exception:
            return 1e3
        if L < best[0]:
            best[0] = L
            best[1] = x.copy()
        return L

    ev(x0)
    for frac in (0.5, 0.25, 0.75):
        if left() <= 2.0:
            break
        ev(lo + frac * (hi - lo))
    for _ in range(2):
        if best[1] is None:
            break
        for i in range(nd):
            if left() <= 2.0:
                break
            b = best[1]
            span = hi[i] - lo[i]
            for v in (lo[i] + 0.2 * span, 0.5 * (lo[i] + hi[i]),
                      hi[i] - 0.2 * span):
                t = b.copy()
                t[i] = np.clip(v, lo[i], hi[i])
                ev(t)
    rep = 0
    while best[1] is not None and left() > 0.34 * tlim and rep < 5:
        try:
            minimize(ev, best[1].copy(), method="Powell", bounds=bounds,
                     options=dict(maxiter=60, xtol=1e-6, ftol=1e-11))
        except Exception:
            pass
        if best[1] is None or left() <= 0.34 * tlim:
            break
        amp = 0.09 * (hi - lo) * (0.55 ** rep)
        x = np.clip(best[1] + rng.uniform(-1.0, 1.0, nd) * amp, lo, hi)
        ev(x)
        rep += 1
    if best[1] is not None and left() > 0.20 * tlim:
        try:
            minimize(ev, best[1].copy(), method="Nelder-Mead",
                     options=dict(maxiter=4000, xatol=1e-6, fatol=1e-12,
                                  adaptive=True))
        except Exception:
            pass
    if best[1] is not None:
        for rnd in range(9):
            if left() <= 0.0:
                break
            x = best[1].copy()
            step = 0.06 * (hi - lo) * (0.5 ** rnd)
            improved = False
            for j in range(nd):
                if left() <= 0.0:
                    break
                bL = best[0]
                pick = None
                for s in (1.0, -1.0):
                    t = x.copy()
                    t[j] = np.clip(t[j] + s * step[j], lo[j], hi[j])
                    if t[j] == x[j]:
                        continue
                    L = ev(t)
                    if L < bL - 1e-13:
                        bL = L
                        pick = t
                if pick is not None:
                    x = pick
                    improved = True
                elif best[1] is not None:
                    x = best[1].copy()
            if not improved:
                break
    return best[1]


def solve(problem, budget, seed):
    initial = np.asarray(problem["initial_level_set"], float).copy()
    try:
        got = _gather(problem)
        if got is None:
            return initial
        shape, origin, spacing, idx, val, coords, obsl = got
        if len(shape) != 3 or not (val > 0.5).any():
            return initial
        holes = _holes(idx, val, coords)
        if not holes:
            return initial
        axes = [origin[i] + (np.arange(shape[i]) + 0.5) * spacing[i]
                for i in range(3)]
        mesh = np.meshgrid(*axes, indexing="ij")
        X, Y, Z = mesh[0], mesh[1], mesh[2]
        nh = len(holes)
        if isinstance(budget, dict):
            wall = float(budget.get("wall_time_seconds", 60) or 60)
        else:
            wall = float(budget or 60)
        small = val.size < 12 or int(np.prod(shape)) < 2000
        tlim = min(0.80 * wall, 95.0)
        if small:
            tlim = min(tlim, 2.5)
        t0 = time.time()
        rng = np.random.default_rng(int(seed) if seed is not None else 0)
        hx = float(np.mean([h[0][0] for h in holes]))
        hy = float(np.mean([h[0][1] for h in holes]))
        L = 0.5 * max(1.0, float(np.ptp(X)), float(np.ptp(Y)))
        xc = (X - hx) / L
        yc = (Y - hy) / L
        WARP = (xc, yc, xc * xc, xc * yc, yc * yc,
                xc ** 3, xc * xc * yc, xc * yc * yc, yc ** 3)
        na = len(WARP)
        E = 5 * nh + na + 1 + nh

        def build(x):
            us = x[:nh]
            Tx = np.maximum(x[nh:2 * nh], 1e-3)
            Ty = np.maximum(x[2 * nh:3 * nh], 1e-3)
            mix = np.clip(x[3 * nh:4 * nh], 0.0, 1.0)
            rho = np.clip(x[4 * nh:5 * nh], 0.12, 0.9)
            th = x[5 * nh + 1 + na:5 * nh + 1 + na + nh]
            ex1 = x[E:E + nh]
            ex2 = x[E + nh:E + 2 * nh]
            num = np.zeros(shape, float)
            den = np.zeros(shape, float)
            uus, vvs, rrs, dds = [], [], [], []
            for k, (hc, cz, gap) in enumerate(holes):
                dx = X - hc[0]
                dy = Y - hc[1]
                ct = np.cos(th[k])
                st = np.sin(th[k])
                uu = ct * dx + st * dy
                vv = -st * dx + ct * dy
                r = np.sqrt((uu / Tx[k]) ** 2 + (vv / Ty[k]) ** 2)
                wb = np.exp(-r) + 1e-18
                wn = np.exp(-np.maximum(r / rho[k], 0.0)) + 1e-18
                if MODE == "plateau":
                    w = (mix[k] * wb + (1.0 - mix[k]) * wn
                         + ex1[k] * np.exp(-np.power(np.maximum(r, 1e-6),
                                                     ex2[k])))
                elif MODE == "narrow2":
                    wn2 = np.exp(-np.maximum(r / ex2[k], 0.0)) + 1e-18
                    w = (mix[k] * wb + (1.0 - mix[k])
                         * (ex1[k] * wn + (1.0 - ex1[k]) * wn2))
                else:
                    w = mix[k] * wb + (1.0 - mix[k]) * wn
                num += w * (cz - us[k])
                den += w
                uus.append(uu)
                vvs.append(vv)
                rrs.append(r)
                dds.append((dx, dy))
            h = num / den
            if MODE == "tilt":
                for k in range(nh):
                    h = h + (ex1[k] * uus[k] + ex2[k] * vvs[k]) * np.exp(-rrs[k])
            elif MODE == "bump":
                for k in range(nh):
                    d2 = dds[k][0] ** 2 + dds[k][1] ** 2
                    h = h + ex1[k] * np.exp(-d2 / (2.0 * ex2[k] ** 2 + 1e-12))
            for k in range(na):
                h = h + x[5 * nh + k] * WARP[k]
            h = h + x[5 * nh + na]
            return Z - h

        def evalf(x):
            try:
                f = build(x)
            except Exception:
                return 1e3
            if not np.isfinite(f).all() or not _sign_ok(f, idx, val):
                return 1e3
            phi = skfmm.distance(np.ascontiguousarray(f, np.float64),
                                 dx=tuple(float(s) for s in spacing))
            phi = np.asarray(phi, float)
            if not np.isfinite(phi).all():
                return 1e3
            return _loss_phi(phi, obsl)

        base = ([(0.0, 0.7)] * nh + [(0.30, 3.0)] * 2 * nh
                + [(0.02, 0.98)] * nh + [(0.15, 0.85)] * nh
                + [(-0.30, 0.30)] * na + [(-0.6, 0.6)]
                + [(-1.5708, 1.5708)] * nh)
        bx0 = [0.25] * nh + [1.4] * 2 * nh + [0.5] * nh + [0.5] * nh \
            + [0.0] * na + [0.0] + [0.0] * nh
        if MODE == "plateau":
            bounds = tuple(list(base) + [(0.0, 1.5)] * nh
                           + [(1.2, 8.0)] * nh)
            x0 = np.array(bx0 + [0.0] * nh + [3.0] * nh, float)
        elif MODE == "narrow2":
            bounds = tuple(list(base) + [(0.0, 1.0)] * nh
                           + [(0.08, 0.9)] * nh)
            x0 = np.array(bx0 + [1.0] * nh + [0.2] * nh, float)
        elif MODE == "tilt":
            bounds = tuple(list(base) + [(-0.35, 0.35)] * 2 * nh)
            x0 = np.array(bx0 + [0.0] * 2 * nh, float)
        else:
            bounds = tuple(list(base) + [(-0.5, 0.5)] * nh
                           + [(0.5, 3.0)] * nh)
            x0 = np.array(bx0 + [0.0] * nh + [1.5] * nh, float)
        assert len(bounds) == len(x0)
        xb = _fit(evalf, x0, bounds, tlim, t0, rng)
        if xb is not None:
            f = build(xb)
            if np.isfinite(f).all() and _sign_ok(f, idx, val):
                return f
        num = np.zeros(shape, float)
        den = np.zeros(shape, float)
        for (hc, cz, gap) in holes:
            d2 = (X - hc[0]) ** 2 + (Y - hc[1]) ** 2
            w = 1.0 / (d2 + 1e-9)
            num += w * (cz - 0.25 * min(1.0, gap))
            den += w
        f = Z - num / den
        if np.isfinite(f).all() and _sign_ok(f, idx, val):
            return f
    except Exception:
        pass
    return initial
