"""phase1 0.35; explicit multiscale grid: coarse 6x8x6 grid (30% of the window, step 1.5/0.75/0.35) then fine 10x12x8 grid (70%) with an extended step ladder ending at 0.03.

numpy + scipy.ndimage + scikit-fmm; wall-clock bounded; output always finite with
both signs and returned as a reinitialised signed-distance field.
"""

import time
import numpy as np
import skfmm
from scipy.ndimage import gaussian_filter, distance_transform_edt

CFG = {'front': 'edt', 'polish': ['c', 'b', 'l'], 'shares': [0.30, 0.70, 1.0], 'nodes': (10, 12, 8), 'steps': (1.0, 0.5, 0.25, 0.12, 0.06, 0.03), 'phase1': 0.33}


def signed_distance(field, spacing):
    return skfmm.distance(
        np.ascontiguousarray(field, dtype=np.float64),
        dx=tuple(float(s) for s in spacing),
    )


def _loss_and_grad(phi, observations):
    total = 0.0
    grads = {}
    for o in observations:
        v = np.asarray(o["values"], dtype=float)
        if v.shape != phi.shape:
            continue
        m = np.isfinite(v)
        if not np.any(m):
            continue
        w = float(o["weight"])
        d = phi[m]
        y = v[m]
        lidx = np.flatnonzero(m.ravel())
        is0 = y == 0
        ispos = y > 0
        val = np.logaddexp(0.0, d[is0]).sum() + np.logaddexp(0.0, -d[ispos]).sum()
        g = np.zeros_like(d)
        g[is0] = 1.0 / (1.0 + np.exp(-d[is0]))
        g[ispos] = -1.0 / (1.0 + np.exp(d[ispos]))
        con = y == 0.5
        if np.any(con):
            dc = d[con]
            n = dc.size
            mean = dc.mean()
            val += float(mean ** 2) + float(np.mean(dc ** 2))
            g[con] += 2.0 * (mean + dc) / n
        total += w * float(val)
        for i, gi in zip(lidx, w * g):
            grads[i] = grads.get(i, 0.0) + float(gi)
    return total, grads


def _collect(problem):
    shape = tuple(int(n) for n in problem["shape"])
    ncell = int(np.prod(shape))
    ones, zeros, conts = [], [], []
    for o in problem["observations"]:
        v = np.asarray(o["values"], dtype=float)
        if v.shape != shape:
            continue
        m = np.isfinite(v)
        w = float(o["weight"])
        if not np.any(m):
            continue
        lab = v.ravel()[np.flatnonzero(m.ravel())]
        idx = np.flatnonzero(m.ravel())
        for arr, sel in ((ones, lab > 0.5), (zeros, lab == 0.0), (conts, lab == 0.5)):
            if np.any(sel):
                arr.append((idx[sel], w))
    return shape, ncell, ones, zeros, conts


def _edt_field(shape, ones, zeros, spacing):
    f = np.zeros(shape)
    for idx, w in ones:
        m = np.zeros(shape, dtype=bool)
        m.ravel()[idx] = True
        f -= w * distance_transform_edt(~m, sampling=spacing)
    for idx, w in zeros:
        m = np.zeros(shape, dtype=bool)
        m.ravel()[idx] = True
        f += w * distance_transform_edt(~m, sampling=spacing)
    return f


def _kde_field(shape, ncell, ones, zeros, conts, sigma, contact_w):
    a1 = np.zeros(ncell)
    a0 = np.zeros(ncell)
    for idx, w in ones:
        a1[idx] += w
    for idx, w in zeros:
        a0[idx] += w
    for idx, w in conts:
        a1[idx] += contact_w * w
        a0[idx] += contact_w * w
    if a1.sum() > 0:
        a1 /= a1.sum()
    if a0.sum() > 0:
        a0 /= a0.sum()
    return (gaussian_filter(a1.reshape(shape), sigma, mode="nearest")
            - gaussian_filter(a0.reshape(shape), sigma, mode="nearest"))


def _basis_global(shape, spacing, origin):
    """Global smooth basis: constant, per-axis tilt/bowl, cross terms, 10 lobes."""
    g = np.indices(shape, dtype=float)
    xyz = [origin[k] + (g[k] + 0.5) * spacing[k] for k in range(len(shape))]
    u = []
    for k in range(len(shape)):
        n = shape[k]
        c = origin[k] + (n / 2.0) * spacing[k]
        h = max(1e-9, (n / 2.0) * spacing[k])
        u.append((xyz[k] - c) / h)
    B = [np.ones(shape)]
    for k in range(len(shape)):
        B.append(u[k] - u[k].mean())
    for k in range(len(shape)):
        t = u[k] ** 2
        B.append(t - t.mean())
    if len(shape) == 3:
        for k, l in ((0, 1), (0, 2), (1, 2)):
            t = u[k] * u[l]
            B.append(t - t.mean())
        ext = [xyz[k].max() - xyz[k].min() for k in range(3)]
        lobes = []
        for fx in (0.28, 0.62):
            for fy in (0.25, 0.75):
                for fz in (0.2, 0.75):
                    lobes.append((xyz[0].min() + fx * ext[0],
                                  xyz[1].min() + fy * ext[1],
                                  xyz[2].min() + fz * ext[2]))
        sig = 0.30 * max(ext)
        for (cx, cy, cz) in lobes:
            r2 = ((xyz[0] - cx) ** 2 + (xyz[1] - cy) ** 2 + (xyz[2] - cz) ** 2) / (2 * sig * sig)
            B.append(np.exp(-r2))
    return B


def _interp_ops(shape, nodes):
    ops = []
    for k in range(len(shape)):
        n = shape[k]
        g = nodes[k]
        x = (np.arange(n) + 0.5) / n * (g - 1)
        i0 = np.minimum(x.astype(int), g - 1)
        i1 = np.minimum(i0 + 1, g - 1)
        rs = [1] * len(shape)
        rs[k] = n
        ops.append((i0, i1, (x - i0).reshape(rs)))
    return ops


def _apply_grid(ctrl, ops):
    o = ctrl
    for k, (i0, i1, w) in enumerate(ops):
        o = np.take(o, i0, k) * (1 - w) + np.take(o, i1, k) * w
    return o


def polish_global(base, Lf, spacing, deadline):
    B = _basis_global(base.shape, spacing, _ORIGIN[0])
    best, bestL = base, Lf(base)
    c = np.zeros(len(B))
    for st in (0.4, 0.2, 0.1, 0.05, 0.025):
        for _pass in range(2):
            improved = False
            for j in range(len(B)):
                if time.perf_counter() > deadline:
                    return best, bestL
                for s in (st, -st, 2 * st, -2 * st):
                    trial = c.copy()
                    trial[j] += s
                    f = base.copy()
                    for ci, bi in zip(trial, B):
                        if ci != 0.0:
                            f -= ci * bi
                    Lv = Lf(f)
                    if np.isfinite(Lv) and Lv < bestL - 1e-7:
                        bestL, best, c = Lv, signed_distance(f, spacing), trial
                        improved = True
                        break
            if not improved:
                break
    return best, bestL


def polish_grid(base, Lf, spacing, deadline, nodes, steps):
    ops = _interp_ops(base.shape, nodes)
    best, bestL = base, Lf(base)
    c = np.zeros(nodes)
    for st in steps:
        for _sweep in range(2):
            improved = False
            for i in range(nodes[0]):
                for j in range(nodes[1]):
                    for k in range(nodes[2] if len(nodes) == 3 else 1):
                        if time.perf_counter() > deadline:
                            return best, bestL
                        for s in (st, -st):
                            trial = c.copy()
                            if len(nodes) == 3:
                                trial[i, j, k] += s
                            else:
                                trial[i, j] += s
                            f = base - _apply_grid(trial, ops)
                            Lv = Lf(f)
                            if np.isfinite(Lv) and Lv < bestL - 1e-7:
                                bestL, best, c = Lv, signed_distance(f, spacing), trial
                                improved = True
                                break
            if not improved:
                break
    return best, bestL


_ORIGIN = [np.zeros(3)]
_RAMP = [None]


def solve(problem, budget, seed):
    t0 = time.perf_counter()
    wall = float(budget.get("wall_time_seconds", 60.0)) if budget else 60.0
    hard = t0 + max(5.0, 0.95 * wall)
    t_phase1 = t0 + max(3.0, CFG['phase1'] * wall)

    shape, ncell, ones, zeros, conts = _collect(problem)
    obs = problem["observations"]
    spacing = np.array(problem["spacing"], dtype=float).ravel()
    _ORIGIN[0] = np.array(problem.get("origin", [0.0] * len(shape)), dtype=float).ravel()
    if _ORIGIN[0].size != len(shape):
        _ORIGIN[0] = np.zeros(len(shape))
    init = problem.get("initial_level_set", None)
    init = np.asarray(init, dtype=float) if init is not None else np.zeros(shape)
    if init.shape != shape or not np.all(np.isfinite(init)):
        init = np.zeros(shape)
    if not ones and not zeros:
        return np.ascontiguousarray(init, dtype=np.float64)

    lidx_all = []
    for o in obs:
        v = np.asarray(o["values"], dtype=float)
        if v.shape == shape:
            m = np.isfinite(v)
            if np.any(m):
                lidx_all.append(np.flatnonzero(m.ravel()))
    lidx = np.concatenate(lidx_all) if lidx_all else np.zeros(0, dtype=int)
    ulin = np.unique(lidx) if lidx.size else np.zeros(0, dtype=int)
    cidx = np.concatenate([c[0] for c in conts]) if conts else np.zeros(0, dtype=int)

    best_f, best_loss = None, np.inf

    def evaluate(f):
        nonlocal best_f, best_loss
        if not (np.any(f > 0) and np.any(f < 0)):
            return None, np.inf
        phi = signed_distance(f, spacing)
        loss, _ = _loss_and_grad(phi, obs)
        if np.isfinite(loss) and loss < best_loss:
            best_loss = loss
            best_f = phi.copy()
        return phi, loss

    def refine(f, t_end, max_reject=3):
        f = signed_distance(f, spacing)
        lr, sig_r, fails = 0.35, 1.5, 0
        while time.perf_counter() < t_end and fails < max_reject:
            phi = signed_distance(f, spacing)
            cur, grads = _loss_and_grad(phi, obs)
            if not np.isfinite(cur):
                break
            g = np.zeros(ncell)
            if ulin.size:
                gv = np.fromiter((grads.get(int(i), 0.0) for i in ulin), float, ulin.size)
                g[ulin] = gv
            gs = gaussian_filter(g.reshape(shape), sig_r, mode="nearest")
            gs = gs / (np.abs(gs).max() + 1e-12)
            _, loss_t = evaluate(f - lr * gs)
            if loss_t <= cur:
                f = signed_distance(f - lr * gs, spacing)
                fails = 0
                lr = min(0.8, lr * 1.15)
            else:
                fails += 1
                lr *= 0.55
                if fails >= 2:
                    sig_r = max(0.8, sig_r * 0.7)
                if lr < 1e-3:
                    break
        return f

    def scan(f0, levels):
        if not (np.any(f0 > 0) and np.any(f0 < 0)):
            return None
        ref = f0.ravel()[ulin] if ulin.size else f0.ravel()
        spread = float(np.std(ref)) + 1e-12
        tau0 = float(np.median(f0.ravel()[cidx])) if cidx.size else float(np.median(ref))
        best = None
        for k in levels:
            _, loss = evaluate(f0 - tau0 - k * 0.22 * spread)
            if np.isfinite(loss) and (best is None or loss < best[0]):
                best = (loss, f0 - tau0 - k * 0.22 * spread)
        return None if best is None else best[1]

    def front_v4(t_end):
        f = scan(_edt_field(shape, ones, zeros, spacing), list(range(-6, 7)))
        if f is not None:
            refine(f, t_end)
        for sigma in (1.6, 1.0, 2.6):
            if time.perf_counter() > t_end:
                break
            g = scan(_kde_field(shape, ncell, ones, zeros, conts, sigma, 1.0), list(range(-4, 5)))
            if g is not None:
                refine(g, t_end)

    def front_edt(t_end):
        f = scan(_edt_field(shape, ones, zeros, spacing), list(range(-8, 9)))
        if f is not None:
            refine(f, t_end)

    def front_kde(t_end):
        for sigma in (2.2, 1.4, 0.9, 1.8, 1.1):
            if time.perf_counter() > t_end:
                break
            g = scan(_kde_field(shape, ncell, ones, zeros, conts, sigma, 1.4), list(range(-7, 8)))
            if g is not None:
                refine(g, t_end)

    if CFG["front"] == "v4":
        front_v4(t_phase1)
    elif CFG["front"] == "edt":
        front_edt(t_phase1)
    else:
        front_kde(t_phase1)

    if best_f is not None:
        base = best_f.copy()
        for k in range(-8, 9):
            evaluate(base + k * 0.25)

    def Lf(f):
        if not (np.any(f > 0) and np.any(f < 0)):
            return np.inf
        return _loss_and_grad(signed_distance(f, spacing), obs)[0]

    _RAMP[0] = _deep_ramp(problem, shape)
    obs_w = []
    for o in obs:
        o2 = dict(o)
        if "drill" in str(o.get("name", "")).lower():
            o2["weight"] = float(o.get("weight", 1.0)) * 2.5
        obs_w.append(o2)

    def Lw(f):
        if not (np.any(f > 0) and np.any(f < 0)):
            return np.inf
        return _loss_and_grad(signed_distance(f, spacing), obs_w)[0]

    try:
        schedule = CFG["polish"]
        shares = CFG.get("shares", [1.0] * len(schedule))
        for si, stage in enumerate(schedule):
            if time.perf_counter() > hard - 1.0 or best_f is None:
                break
            bud = max(1.0, shares[min(si, len(shares) - 1)] * (hard - time.perf_counter()))
            if stage == "a":
                f, lv = polish_global(best_f, Lf, spacing, time.perf_counter() + bud)
            elif stage == "b":
                f, lv = polish_grid(best_f, Lf, spacing, time.perf_counter() + bud,
                                    CFG["nodes"], CFG["steps"])
            elif stage == "c":
                f, lv = polish_grid(best_f, Lf, spacing, time.perf_counter() + bud,
                                    (6, 8, 6), (1.5, 0.75, 0.35))
            elif stage == "g":
                f, lv = stage_growth(best_f, Lf, spacing, _RAMP[0],
                                     time.perf_counter() + bud)
            elif stage == "l":
                f, lv = polish_grid_masked(best_f, Lf, spacing,
                                           time.perf_counter() + bud,
                                           (8, 10, 6), (1.0, 0.5, 0.25, 0.12),
                                           _RAMP[0])
            elif stage == "w":
                f, _lvw = polish_grid(best_f, Lw, spacing, time.perf_counter() + bud,
                                      (8, 10, 6), (1.0, 0.5, 0.25))
                lv = Lf(f)
            else:
                f = refine(best_f, min(hard, time.perf_counter() + 0.5 * bud))
                lv = Lf(f)
                if not np.isfinite(lv) or lv >= best_loss:
                    continue
            if np.isfinite(lv) and lv < best_loss:
                evaluate(f)
    except Exception:
        pass

    out = best_f if best_f is not None else init
    if (not np.all(np.isfinite(out))) or not (np.any(out > 0) and np.any(out < 0)):
        out = init
    return np.ascontiguousarray(out, dtype=np.float64)


# ---- new deep-region stages (added round 20) -------------------------------
def _deep_ramp(problem, shape):
    """Index-space ramp that is 0 at the shallowest zero label and 1 at the
    opposite (deep) end of the axis where the contacts live.  Growing the body
    along this ramp cannot push a zero-positive violation: zeros want phi<0 and
    the ramp is zero exactly at their level."""
    zaxis = len(shape) - 1
    idx = np.indices(shape, dtype=float)[zaxis]
    kz = []          # zero-labelled cells (outcrop surface)
    kc = []          # contact cells (deep)
    for o in problem["observations"]:
        v = np.asarray(o["values"], dtype=float)
        if v.shape != shape:
            continue
        m = np.isfinite(v)
        if np.any(m & (v == 0.0)):
            kz.append(float(idx[m & (v == 0.0)].mean()))
        if np.any(m & (v == 0.5)):
            kc.append(float(idx[m & (v == 0.5)].mean()))
    if not kz:
        return np.zeros(shape)
    k0 = float(np.mean(kz))
    kmax = float(shape[zaxis] - 1)
    if kc and float(np.mean(kc)) < k0:
        span = max(1e-9, k0)
        ramp = np.clip((k0 - idx) / span, 0.0, 1.0)
    else:
        span = max(1e-9, kmax - k0)
        ramp = np.clip((idx - k0) / span, 0.0, 1.0)
    return ramp


def stage_growth(base_f, Lf, spacing, ramp, deadline):
    best, bestL = base_f, Lf(base_f)
    for al in (-2.0, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0):
        if time.perf_counter() > deadline:
            break
        f = base_f + al * ramp
        Lv = Lf(f)
        if np.isfinite(Lv) and Lv < bestL - 1e-7:
            bestL, best = Lv, signed_distance(f, spacing)
    return best, bestL


def polish_grid_masked(base_f, Lf, spacing, deadline, nodes, steps, mask):
    ops = _interp_ops(base_f.shape, nodes)
    best, bestL = base_f, Lf(base_f)
    c = np.zeros(nodes)
    for st in steps:
        for _sweep in range(2):
            improved = False
            for i in range(nodes[0]):
                for j in range(nodes[1]):
                    for k in range(nodes[2] if len(nodes) == 3 else 1):
                        if time.perf_counter() > deadline:
                            return best, bestL
                        for s in (st, -st):
                            trial = c.copy()
                            if len(nodes) == 3:
                                trial[i, j, k] += s
                            else:
                                trial[i, j] += s
                            f = base_f - _apply_grid(trial, ops) * mask
                            Lv = Lf(f)
                            if np.isfinite(Lv) and Lv < bestL - 1e-7:
                                bestL, best, c = Lv, signed_distance(f, spacing), trial
                                improved = True
                                break
            if not improved:
                break
    return best, bestL
