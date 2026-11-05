# candidate 2: base + extended init scan + dilation + short contact descent
"""Candidate: anisotropic density grid + boosting + progressive flip sweeps.

Start field: best of an anisotropic Gaussian label-density grid (spatial scales
1.3/1.5, z/x ratios 1.5/1.75/2.0, negative weight beta=0.95, three threshold
offsets).  Then one residual-boosting round (Gaussian bumps at the worst label
cells, four amplitudes), coarse in-place flip sweeps at thr 0.8, a medium sweep
at 1.6, Gaussian warps at the worst labels, per-contact value surgery, a wide
sweep at 2.5 and ball (1-neighbourhood) moves.

Everything runs in-process (no worker pool).  Stages are fault isolated and the
best valid field is kept; the result is always finite with both signs.
"""

import time
import itertools
import numpy as np
from scipy.ndimage import gaussian_filter

try:
    import skfmm
except Exception:  # pragma: no cover
    skfmm = None

BIG = 1.0e18


def _sd(field, spacing):
    return np.asarray(skfmm.distance(np.ascontiguousarray(field, dtype=np.float64),
                                    dx=tuple(float(s) for s in spacing)),
                      dtype=np.float64)


def _sketch_loss(reference, phi, spacing, origin):
    coords = np.stack([c.reshape(-1, order="F") for c in np.meshgrid(
        *[origin[i] + (np.arange(n) + 0.5) * spacing[i]
          for i, n in enumerate(phi.shape)], indexing="ij")], axis=1)
    a = coords[reference.reshape(-1, order="F") == 1]
    b = coords[phi.reshape(-1, order="F") > 0]
    if len(a) < 2 or len(b) < 2:
        return 50.0
    a = a - a.mean(0)
    b = b - b.mean(0)
    an = np.linalg.norm(a)
    bn = np.linalg.norm(b)
    if not an or not bn:
        return 50.0
    a = a / an
    b = b / bn
    n = max(len(a), len(b))
    a = np.pad(a, ((0, n - len(a)), (0, 0)))
    b = np.pad(b, ((0, n - len(b)), (0, 0)))
    sv = np.linalg.svd(a.T @ b, compute_uv=False)
    return float(5 * (1 - sv.sum() ** 2))


def _loss(problem, field):
    if field is None:
        return BIG
    f = np.asarray(field, dtype=np.float64)
    if not np.isfinite(f).all() or not (f > 0).any() or not (f < 0).any():
        return BIG
    try:
        phi = _sd(f, problem["spacing"])
    except Exception:
        return BIG
    if not np.isfinite(phi).all():
        return BIG
    total = 0.0
    for o in problem["observations"]:
        v = np.asarray(o["values"], dtype=float)
        m = np.isfinite(v)
        if not m.any():
            continue
        d = phi[m]
        y = v[m]
        lg = 0.0
        if (y == 0).any():
            lg += float(np.logaddexp(0.0, d[y == 0]).sum())
        if (y > 0).any():
            lg += float(np.logaddexp(0.0, -d[y > 0]).sum())
        c = phi[v == 0.5]
        if c.size:
            lg += float(c.mean() ** 2) + float(np.mean(c ** 2))
        total += float(o.get("weight", 1.0)) * lg
    sk = problem.get("sketch")
    if sk is not None:
        total += _sketch_loss(np.asarray(sk, dtype=float), phi,
                              tuple(problem["spacing"]), tuple(problem["origin"]))
    return float(total) if np.isfinite(total) else BIG


def _fallback(problem):
    shape = tuple(int(n) for n in problem["shape"])
    init = problem.get("initial_level_set")
    if init is not None:
        init = np.asarray(init, dtype=float)
        if (init.shape == shape and np.isfinite(init).all()
                and (init > 0).any() and (init < 0).any()):
            return init.copy()
    g = np.indices(shape, dtype=float)
    out = np.zeros(shape)
    for i, n in enumerate(shape):
        out = out + (g[i] - (n - 1) / 2.0) / max(n, 1)
    out -= np.median(out)
    if not (out > 0).any():
        out[tuple(n // 2 for n in shape)] = 1.0
    if not (out < 0).any():
        out[tuple(0 for _ in shape)] = -1.0
    return out


_WORKER = {}


def _winit(prob):
    _WORKER['p'] = prob


def _wloss(f):
    try:
        return _loss(_WORKER['p'], f)
    except Exception:
        return BIG


def _make_ev(problem, workers=2):
    """Exact loss evaluator; candidate fields are scored in worker processes.

    skfmm's distance solve is the whole cost of this solver, so two worker
    processes (the formal budget is 2 CPU) nearly double the number of exact
    evaluations that fit in the wall budget.  Falls back to in-process scoring
    whenever a pool cannot be created or a map fails.
    """
    seq = lambda fields: [_loss(problem, f) for f in fields]
    try:
        if int(np.prod(problem["shape"])) < 20000:
            return seq, None
        import multiprocessing as _mp
        import atexit as _atexit
        ctx = _mp.get_context('fork')
        pool = ctx.Pool(workers, initializer=_winit, initargs=(problem,))
        _atexit.register(pool.terminate)

        def ev(fields):
            fields = list(fields)
            if len(fields) < 2:
                return [_loss(problem, f) for f in fields]
            try:
                return [float(v) for v in pool.map(_wloss, fields, chunksize=1)]
            except Exception:
                return [_loss(problem, f) for f in fields]
        return ev, pool
    except Exception:
        return seq, None




def _sigm(s, nd, fz):
    if nd >= 3:
        return (s, s, s * fz)
    return (s,) * nd


def solve(problem, budget, seed=47):
    t0 = time.time()
    try:
        wall = float((budget or {}).get("wall_time_seconds", 60) or 60)
    except Exception:
        wall = 60.0
    shape = tuple(int(n) for n in problem["shape"])
    spacing = tuple(float(s) for s in problem["spacing"])
    nd = len(shape)
    nl = int(np.prod(shape))
    horizon = min(0.55 * wall, 6.0) if nl < 6000 else 0.82 * wall
    deadline = t0 + max(1.0, horizon)
    span = deadline - t0

    _pool = None
    ev, _pool = _make_ev(problem, 2)

    bestF = None
    bestL = BIG
    try:
        Wp = np.zeros(shape)
        Wn = np.zeros(shape)
        obs = []
        contacts = []
        have_pos = have_neg = False
        for o in problem["observations"]:
            v = np.asarray(o["values"], dtype=float)
            m = np.isfinite(v)
            w = float(o.get("weight", 1.0))
            if not m.any():
                continue
            idx = np.flatnonzero(m.reshape(-1))
            obs.append((idx, v.reshape(-1)[idx].astype(np.float64), w))
            Wp[m & (v > 0.5)] += w
            Wn[m & (v < 0.5)] += w
            if np.any(m & (v > 0.5)):
                have_pos = True
            if np.any(m & (v < 0.5)):
                have_neg = True
            for c in np.argwhere(m & (v == 0.5)):
                contacts.append(tuple(int(z) for z in c))
        sk = problem.get("sketch")
        if sk is not None:
            sk = np.asarray(sk, dtype=float)
            if sk.shape == shape:
                ws = max(sum(float(o.get("weight", 1.0))
                             for o in problem["observations"]), 1.0)
                Wp = Wp + (sk == 1) * (0.5 * ws)
                Wn = Wn + (sk == 0) * (0.25 * ws)
                if np.any(sk == 1):
                    have_pos = True
                if np.any(sk == 0):
                    have_neg = True
        if not have_pos or not have_neg:
            return _fallback(problem)

        sref = float(np.mean(spacing)) or 1.0
        u = t0 + 0.10 * span
        best = (BIG, None)
        for sig in (1.3, 1.5):
            for fz in ((1.5, 1.75, 2.0) if nd >= 3 else (1.0,)):
                for off in (0.0, 0.12, -0.12):
                    ss = _sigm(sig * sref, nd, fz)
                    G = gaussian_filter(Wp, ss, mode="constant") - 0.95 * \
                        gaussian_filter(Wn, ss, mode="constant")
                    if off:
                        G = G - off * float(np.std(G) + 1e-12)
                    if not ((G > 0).any() and (G < 0).any()):
                        continue
                    L = _loss(problem, G)
                    if L < best[0]:
                        best = (L, G)
                    if time.time() > u and best[1] is not None:
                        break
        # ---- extended anisotropic density init scan ----
        try:
            u2 = t0 + 0.16 * span
            for sig2 in (1.2, 1.4, 1.6):
                for fz2 in ((1.4, 1.6, 1.8, 2.1) if nd >= 3 else (1.0,)):
                    for beta2 in (0.85, 1.0, 1.1):
                        ss2 = _sigm(sig2 * sref, nd, fz2)
                        G0 = (gaussian_filter(Wp, ss2, mode="constant")
                              - beta2 * gaussian_filter(Wn, ss2, mode="constant"))
                        for off2 in (0.0, 0.1, -0.1):
                            if time.time() > u2:
                                break
                            G = G0 - off2 * float(np.std(G0) + 1e-12)
                            if not ((G > 0).any() and (G < 0).any()):
                                continue
                            L = _loss(problem, G)
                            if L < best[0]:
                                best = (L, G)
        except Exception:
            pass
        if best[1] is None:
            return _fallback(problem)
        F, cur = best[1], best[0]
        bestF, bestL = np.array(F, copy=True), float(cur)
        gi = np.indices(shape, dtype=float)

        # ---- global amplitude (power) renormalisation scan ----
        try:
            for pp in (1.10, 1.25, 1.40):
                G = np.sign(F) * np.abs(F) ** pp
                if not ((G > 0).any() and (G < 0).any()):
                    continue
                L = _loss(problem, G)
                if L < cur - 1e-9:
                    F, cur = G, L
                    if L < bestL:
                        bestF, bestL = np.array(F, copy=True), float(L)
        except Exception:
            pass

        def cands(thr, pen_contacts=False):
            phi = _sd(F, spacing)
            flat = phi.reshape(-1)
            out = []
            for idx, y, w in obs:
                d = flat[idx]
                sgn = np.where(y > 0, 1.0, -1.0)
                pen = np.where(sgn > 0, np.logaddexp(0.0, -d),
                               np.logaddexp(0.0, d))
                if pen_contacts:
                    pen = pen + np.where(y == 0.5, 2.0 * d * d, 0.0)
                sel = (np.sign(d) != sgn) | (np.abs(d) < thr)
                for k in np.flatnonzero(sel):
                    out.append((float(pen[k]) * w, int(idx[k]), float(sgn[k]),
                                float(d[k])))
            out.sort(key=lambda z: -z[0])
            return out

        # ---- global distance dilation/erosion sweep ----
        try:
            for dd in (0.6, 0.35, 0.2, -0.2, -0.35, -0.6, 1.0, -1.0):
                if time.time() > t0 + 0.30 * span:
                    break
                G = _sd(F, spacing) + float(dd)
                if not ((G > 0).any() and (G < 0).any()):
                    continue
                L = _loss(problem, G)
                if L < cur - 1e-9:
                    F, cur = G, L
                    if L < bestL:
                        bestF, bestL = np.array(F, copy=True), float(L)
        except Exception:
            pass
        # ---- residual boosting ----
        try:
            u = t0 + 0.22 * span
            for g in (1.2, 1.8, 2.6):
                for _rep in range(2):
                    if time.time() > u:
                        raise StopIteration
                    cl = cands(1.0, True)
                    sc = float(np.std(F)) + 1e-12
                    for _p, fi, sgn, _v in cl[:18]:
                        if time.time() > u:
                            break
                        c = np.unravel_index(fi, shape)
                        d2 = np.zeros(shape)
                        for k in range(nd):
                            d2 = d2 + (gi[k] - c[k]) ** 2
                        bump = np.exp(-d2 / (2.0 * g * g))
                        props = [F + sgn * amp * sc * bump
                                 for amp in (0.1, 0.25, 0.5, 0.9)]
                        vals = ev(props)
                        j = int(np.argmin(vals))
                        if vals[j] < cur - 1e-9:
                            F, cur = props[j], vals[j]
        except Exception:
            pass

        # ---- in-place flip sweeps ----
        def sweep(until, thr, amps):
            nonlocal F, cur
            u = t0 + until * span
            while time.time() < u:
                cl = cands(thr)
                if not cl:
                    cl = cands(1e9)
                if not cl:
                    break
                sc = float(np.std(F)) + 1e-12
                acc = False
                for i in range(0, len(cl), 2):
                    if time.time() > u:
                        break
                    props = []
                    for _p, fi, sgn, _v in cl[i:i + 2]:
                        for a in amps:
                            G = np.array(F, copy=True).reshape(-1)
                            G[fi] = sgn * a * sc
                            props.append(G.reshape(shape))
                    vals = ev(props)
                    j = int(np.argmin(vals))
                    if vals[j] < cur - 1e-9:
                        F, cur = props[j], vals[j]
                        acc = True
                if not acc:
                    break

        # ---- second residual-boosting round (steeper widths, small amps) ----
        try:
            u = t0 + 0.36 * span
            for g in (1.0, 2.0, 3.2):
                if time.time() > u:
                    break
                cl = cands(1.4, True)
                sc = float(np.std(F)) + 1e-12
                for _p, fi, sgn, _v in cl[:14]:
                    if time.time() > u:
                        break
                    c = np.unravel_index(fi, shape)
                    d2 = np.zeros(shape)
                    for k in range(nd):
                        d2 = d2 + (gi[k] - c[k]) ** 2
                    bump = np.exp(-d2 / (2.0 * g * g))
                    props = [F + sgn * amp * sc * bump
                             for amp in (0.2, 0.5, 1.0)]
                    vals = ev(props)
                    j = int(np.argmin(vals))
                    if vals[j] < cur - 1e-9:
                        F, cur = props[j], vals[j]
        except Exception:
            pass
        try:
            sweep(0.40, 0.8, (1.0, 2.0, 4.0))
        except Exception:
            pass
        try:
            sweep(0.55, 1.6, (2.0, 4.0, 8.0))
        except Exception:
            pass


        # ---- margin push: non-flipping additive increments at near labels ----
        def margin_push(until, thr, amps):
            nonlocal F, cur
            u = t0 + until * span
            while time.time() < u:
                cl = cands(thr)
                if not cl:
                    break
                sc = float(np.std(F)) + 1e-12
                acc = False
                for i in range(0, len(cl), 2):
                    if time.time() > u:
                        break
                    props = []
                    for _p, fi, sgn, _v in cl[i:i + 2]:
                        for a in amps:
                            G = np.array(F, copy=True).reshape(-1)
                            G[fi] = G[fi] + sgn * a * sc
                            props.append(G.reshape(shape))
                    props = [p for p in props if (p > 0).any() and (p < 0).any()]
                    if not props:
                        continue
                    vals = ev(props)
                    j = int(np.argmin(vals))
                    if vals[j] < cur - 1e-9:
                        F, cur = props[j], vals[j]
                        acc = True
                if not acc:
                    break

        try:
            margin_push(0.62, 1.6, (0.35, 0.7, 1.4))
        except Exception:
            pass
        # ---- label warps ----
        try:
            u = t0 + 0.64 * span
            for sg in (1.1, 1.7):
                if time.time() > u:
                    raise StopIteration
                cl = cands(1e9, True)[:16]
                amp = float(np.std(F)) + 1e-12
                for _p, fi, sgn, _v in cl:
                    if time.time() > u:
                        break
                    c = np.unravel_index(fi, shape)
                    d2 = np.zeros(shape)
                    for k in range(nd):
                        d2 = d2 + (gi[k] - c[k]) ** 2
                    bump = np.exp(-d2 / (2.0 * sg * sg))
                    props = [F + sgn * d * amp * bump
                             for d in (0.6, -0.6, 0.25, -0.25)]
                    props = [p for p in props if (p > 0).any() and (p < 0).any()]
                    if not props:
                        continue
                    vals = ev(props)
                    j = int(np.argmin(vals))
                    if vals[j] < cur - 1e-9:
                        F, cur = props[j], vals[j]
        except Exception:
            pass

        # ---- extended multi-width contact descent ----
        try:
            u2 = t0 + 0.70 * span
            if contacts:
                sc = float(np.std(F)) + 1e-12
                for sg in (0.9, 1.4, 2.1):
                    if time.time() > u2:
                        break
                    basis = []
                    for c in contacts:
                        d2 = np.zeros(shape)
                        for k in range(nd):
                            d2 = d2 + (gi[k] - c[k]) ** 2
                        basis.append(np.exp(-d2 / (2.0 * sg * sg)))
                    st = 1.0
                    for _r in range(2):
                        for B in basis:
                            for d in (st, -st):
                                if time.time() > u2:
                                    break
                                G = F + d * sc * B
                                if not ((G > 0).any() and (G < 0).any()):
                                    continue
                                L = _loss(problem, G)
                                if L < cur - 1e-9:
                                    F, cur = G, L
                                    if L < bestL:
                                        bestF, bestL = np.array(F, copy=True), float(L)
                        st *= 0.5
        except Exception:
            pass
        # ---- contact surgery ----
        try:
            u = t0 + 0.75 * span
            for _rep in range(3):
                for c in contacts:
                    if time.time() > u:
                        raise StopIteration
                    sc = float(np.std(F)) + 1e-12
                    props = []
                    for v in (0.0, 0.01 * sc, -0.01 * sc, 0.03 * sc, -0.03 * sc):
                        G = np.array(F, copy=True)
                        G[c] = v
                        props.append(G)
                    vals = ev(props)
                    j = int(np.argmin(vals))
                    if vals[j] < cur - 1e-9:
                        F, cur = props[j], vals[j]
        except Exception:
            pass

        try:
            margin_push(0.935, 2.5, (0.35, 0.7, 1.4))
        except Exception:
            pass

        if F is not None and np.isfinite(cur) and cur < bestL \
                and (F > 0).any() and (F < 0).any():
            bestF, bestL = np.array(F, copy=True), float(cur)

        try:
            sweep(0.90, 2.5, (1.5, 3.0, 6.0, 12.0))
        except Exception:
            pass
        try:
            sweep(0.94, 0.4, (0.6, 1.2, 2.4))
        except Exception:
            pass
        if F is not None and np.isfinite(cur) and cur < bestL \
                and (F > 0).any() and (F < 0).any():
            bestF, bestL = np.array(F, copy=True), float(cur)

        # ---- ball moves ----
        try:
            u = t0 + 0.97 * span
            offs = [dd for dd in itertools.product(range(-1, 2), repeat=nd)
                    if sum(abs(c) for c in dd) <= 1]
            while time.time() < u:
                cl = cands(2.5)
                if not cl:
                    break
                sc = float(np.std(F)) + 1e-12
                acc = False
                for i in range(0, min(len(cl), 80), 2):
                    if time.time() > u:
                        break
                    props = []
                    for _p, fi, sgn, _v in cl[i:i + 2]:
                        c = np.array(np.unravel_index(fi, shape))
                        for amp in (2.0, 4.0):
                            G = np.array(F, copy=True)
                            for dd in offs:
                                q = tuple(np.clip(c + np.array(dd), 0,
                                                  np.array(shape) - 1))
                                G[q] = sgn * amp * sc
                            props.append(G)
                    vals = ev(props)
                    j = int(np.argmin(vals))
                    if vals[j] < cur - 1e-9:
                        F, cur = props[j], vals[j]
                        acc = True
                if not acc:
                    break
        except Exception:
            pass
        try:
            margin_push(0.995, 1.0, (0.2, 0.45, 0.9))
        except Exception:
            pass
        if F is not None and np.isfinite(cur) and cur < bestL \
                and (F > 0).any() and (F < 0).any():
            bestF, bestL = np.array(F, copy=True), float(cur)
    except Exception:
        pass

    try:
        if _pool is not None:
            _pool.terminate()
    except Exception:
        pass
    out = np.ascontiguousarray(bestF if bestF is not None else _fallback(problem),
                               dtype=np.float64)
    if (out.shape != shape or not np.isfinite(out).all()
            or not (out > 0).any() or not (out < 0).any()):
        out = np.ascontiguousarray(_fallback(problem), dtype=np.float64)
    return out
