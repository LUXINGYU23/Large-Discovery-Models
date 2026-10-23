"""Geological implicit-field inversion.

The geometry is represented by a scalar potential whose zero isosurface is the
geological contact.  Sparse drillhole/outcrop labels and contact points are
turned into signed sources.  Several complementary interpolation models are
constructed:

* a Euclidean-distance (Voronoi) potential from the labelled voxels, lightly
  smoothed to round the separating surface;
* a Gaussian kernel-density potential from the same signed sources;
* local Gaussian margin corrections around observed points that remain close
  to the separating surface;
* a graph-harmonic potential field with a weak prior from the initial level set.

Every candidate is a smooth, data-driven implicit field.  The candidate whose
reconstructed signed-distance geometry best matches the fixed public objective
(logistic label loss + contact distance + sketch if present) is returned.  The
search is deterministic and bounded by the supplied wall-time budget.
"""

import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy import ndimage


def _sketch_loss(reference, field, spacing, origin):
    coords = np.stack(
        [
            c.reshape(-1, order="F")
            for c in np.meshgrid(
                *[
                    origin[i] + (np.arange(n) + 0.5) * spacing[i]
                    for i, n in enumerate(field.shape)
                ],
                indexing="ij",
            )
        ],
        axis=1,
    )
    a = coords[reference.reshape(-1, order="F") == 1]
    b = coords[field.reshape(-1, order="F") > 0]
    if len(a) < 2 or len(b) < 2:
        return 1.0e6
    a = a - a.mean(0)
    b = b - b.mean(0)
    an = np.linalg.norm(a)
    bn = np.linalg.norm(b)
    if not an or not bn:
        return 1.0e6
    a = a / an
    b = b / bn
    n = max(len(a), len(b))
    a = np.pad(a, ((0, n - len(a)), (0, 0)))
    b = np.pad(b, ((0, n - len(b)), (0, 0)))
    singular = np.linalg.svd(a.T @ b, compute_uv=False)
    return float(5 * (1 - singular.sum() ** 2))


def _proxy_score(problem, field):
    """Fixed-objective proxy computed on the reconstructed distance field."""
    import skfmm

    shape = tuple(problem["shape"])
    spacing = tuple(float(s) for s in problem["spacing"])
    try:
        phi = skfmm.distance(
            np.ascontiguousarray(field, dtype=np.float64), dx=spacing
        )
    except Exception:
        return float("inf")
    if not np.isfinite(phi).all():
        return float("inf")
    total = 0.0
    for obs in problem.get("observations", []) or []:
        values = np.asarray(obs.get("values"))
        if values.shape != shape:
            continue
        mask = np.isfinite(values)
        y = values[mask]
        d = phi[mask]
        contact = phi[values == 0.5]
        logistic = np.logaddexp(0, d[y == 0]).sum() + np.logaddexp(
            0, -d[y > 0]
        ).sum()
        bias = float(contact.mean() ** 2) if contact.size else 0.0
        variance = float(np.mean(contact ** 2)) if contact.size else 0.0
        total += float(obs.get("weight", 1.0)) * (logistic + bias + variance)
    sketch = problem.get("sketch")
    if sketch is not None:
        try:
            total += _sketch_loss(
                np.asarray(sketch),
                phi,
                np.asarray(problem["spacing"], float),
                np.asarray(problem["origin"], float),
            )
        except Exception:
            total += 1.0e6
    return float(total)


def solve(problem, budget, seed):
    t_start = time.time()
    shape = tuple(int(s) for s in problem["shape"])
    ndim = len(shape)
    spacing = np.asarray(problem.get("spacing", [1.0] * ndim), dtype=np.float64)
    spacing = np.where(np.isfinite(spacing) & (spacing > 0), spacing, 1.0)
    n = int(np.prod(shape))
    idx = np.arange(n, dtype=np.int64).reshape(shape)

    # ---- signed training points ------------------------------------------
    pos = []
    neg = []
    contact = []
    obs_idx = []
    obs_val = []
    obs_w = []
    for obs in problem.get("observations", []) or []:
        values = np.asarray(obs.get("values"))
        if values.shape != shape:
            continue
        mask = np.isfinite(values)
        if not np.any(mask):
            continue
        pts = np.argwhere(mask)
        vals = values[mask]
        ow = float(obs.get("weight", 1.0))
        if not np.isfinite(ow) or ow <= 0:
            ow = 1.0
        pos.append(pts[vals > 0.5])
        neg.append(pts[vals < 0.5])
        contact.append(pts[vals == 0.5])
        obs_idx.append(pts)
        obs_val.append(vals)
        obs_w.append(np.full(len(vals), ow))
    pos = np.concatenate(pos) if pos else np.empty((0, ndim), dtype=np.int64)
    neg = np.concatenate(neg) if neg else np.empty((0, ndim), dtype=np.int64)
    contact = (
        np.concatenate(contact)
        if contact
        else np.empty((0, ndim), dtype=np.int64)
    )
    obs_idx = (
        np.concatenate(obs_idx)
        if obs_idx
        else np.empty((0, ndim), dtype=np.int64)
    )
    obs_val = np.concatenate(obs_val) if obs_val else np.empty(0)
    obs_w = np.concatenate(obs_w) if obs_w else np.empty(0)
    cmask = np.zeros(shape, dtype=bool)
    if len(contact):
        cmask[tuple(contact.T)] = True

    # physical smoothing widths (same units as spacing)
    sigmas = [0.6, 0.8, 0.9, 1.0, 1.1, 1.3, 1.5]
    sig_idx = [
        tuple(float(sg) / float(spacing[i]) for i in range(ndim))
        for sg in sigmas
    ]

    candidates = []  # (name, field)
    base09 = None

    # ---- distance-transform (Voronoi) potentials -------------------------
    if len(pos) and len(neg):
        pi = np.ones(shape, dtype=bool)
        ni = np.ones(shape, dtype=bool)
        pi[tuple(pos.T)] = False
        ni[tuple(neg.T)] = False
        dp = ndimage.distance_transform_edt(pi, sampling=spacing)
        dn = ndimage.distance_transform_edt(ni, sampling=spacing)
        ratios = [(1.0, 1.0), (1.0, 0.9), (1.05, 1.0), (1.1, 1.0),
                  (1.0, 1.1), (1.15, 1.0)]
        for a, b in ratios:
            base = a * dn - b * dp
            for sg, si in zip(sigmas, sig_idx):
                f = ndimage.gaussian_filter(base, si, mode="nearest")
                if cmask.any():
                    f[cmask] = 0.0
                candidates.append((f"dist_{a}_{b}_{sg}", f))
                if a == 1.0 and b == 1.0 and sg == 0.9:
                    base09 = f.copy()

        # Gaussian kernel-density potential from the same signed sources
        raw = np.zeros(shape, dtype=np.float64)
        raw[tuple(pos.T)] = 1.0
        raw[tuple(neg.T)] = -1.0
        for sg, si in zip(sigmas, sig_idx):
            f = ndimage.gaussian_filter(raw, si, mode="nearest")
            if cmask.any():
                f[cmask] = 0.0
            candidates.append((f"kde_{sg}", f))

    # ---- local margin corrections around remaining low-margin labels -----
    if base09 is not None and len(obs_idx):
        try:
            import skfmm

            phi09 = skfmm.distance(
                np.ascontiguousarray(base09, dtype=np.float64),
                dx=tuple(spacing),
            )
            dd = phi09[tuple(obs_idx.T)]
            cost = np.where(
                obs_val < 0.5,
                np.logaddexp(0, dd),
                np.logaddexp(0, -dd),
            ) * obs_w
            cost[obs_val == 0.5] = np.log(2.0) * obs_w[obs_val == 0.5]
            sgn = np.where(
                obs_val > 0.5, 1.0, np.where(obs_val < 0.5, -1.0, 0.0)
            )
            for thr in (0.15, 0.2, 0.25, 0.3, 0.4):
                sel = (cost > thr) & (obs_val != 0.5)
                if not np.any(sel):
                    continue
                src = np.zeros(shape, dtype=np.float64)
                np.add.at(src, tuple(obs_idx[sel].T), sgn[sel])
                for r_phys in (0.9, 1.1, 1.2, 1.4, 1.6):
                    si = tuple(float(r_phys) / float(spacing[i])
                               for i in range(ndim))
                    bump = ndimage.gaussian_filter(src, si, mode="nearest")
                    for amp in (2.0, 4.0, 6.0, 8.0):
                        f = base09 + amp * bump
                        if cmask.any():
                            f[cmask] = 0.0
                        candidates.append(
                            (f"bump_{thr}_{r_phys}_{amp}", f)
                        )
        except Exception:
            pass

    # ---- graph-harmonic potentials ---------------------------------------
    rows, cols, vals = [], [], []
    diag = np.zeros(n, dtype=np.float64)
    for ax in range(ndim):
        if shape[ax] < 2:
            continue
        w = 1.0 / (spacing[ax] * spacing[ax])
        s1 = [slice(None)] * ndim
        s2 = [slice(None)] * ndim
        s1[ax] = slice(0, shape[ax] - 1)
        s2[ax] = slice(1, shape[ax])
        a = idx[tuple(s1)].ravel()
        b = idx[tuple(s2)].ravel()
        m = a.size
        rows.append(np.concatenate([a, b]))
        cols.append(np.concatenate([b, a]))
        vals.append(np.full(2 * m, w, dtype=np.float64))
        diag += np.bincount(a, minlength=n) * w
        diag += np.bincount(b, minlength=n) * w
    rows.append(np.arange(n, dtype=np.int64))
    cols.append(np.arange(n, dtype=np.int64))
    vals.append(diag)
    L = sp.coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n, n),
    ).tocsr()

    init = problem.get("initial_level_set")
    prior = np.zeros(n, dtype=np.float64)
    if isinstance(init, np.ndarray) and init.shape == shape:
        p = np.asarray(init, dtype=np.float64).ravel()
        p = np.where(np.isfinite(p), p, 0.0)
        s = float(np.max(np.abs(p)))
        if s > 0:
            p = p / s
        prior = np.clip(p, -1.0, 1.0)

    target_sum = np.zeros(n, dtype=np.float64)
    data_w = np.zeros(n, dtype=np.float64)
    for obs in problem.get("observations", []) or []:
        values = np.asarray(obs.get("values"))
        if values.shape != shape:
            continue
        mask = np.isfinite(values)
        if not np.any(mask):
            continue
        t = np.where(values[mask] == 0.5, 0.0,
                     np.where(values[mask] > 0.5, 1.0, -1.0))
        flat = idx[mask]
        ow = float(obs.get("weight", 1.0))
        if not np.isfinite(ow) or ow <= 0:
            ow = 1.0
        np.add.at(target_sum, flat, ow * t)
        np.add.at(data_w, flat, ow)
    has_data = data_w > 0
    tgt = np.zeros(n, dtype=np.float64)
    tgt[has_data] = target_sum[has_data] / data_w[has_data]

    harm_cache = {}
    for lam, hard, eps in [
        (50.0, True, 1.0e-3), (8.0, True, 1.0e-3), (2.0, False, 1.0e-3),
        (50.0, False, 1.0e-3), (8.0, False, 1.0e-3), (2.0, True, 5.0e-2),
    ]:
        key = (lam, eps)
        if key not in harm_cache:
            w_data = lam * data_w
            A = L + sp.diags(w_data + eps, format="csr")
            b = w_data * tgt + eps * prior
            try:
                f, _ = spla.cg(
                    A, b, x0=prior.copy(), rtol=1.0e-9, atol=1.0e-12,
                    maxiter=min(1500, max(200, 5 * n)),
                )
                if not np.all(np.isfinite(f)):
                    f = prior.copy()
            except Exception:
                f = b / np.maximum(A.diagonal(), 1.0e-12)
            harm_cache[key] = np.asarray(f, dtype=np.float64).ravel()
        f = harm_cache[key].copy()
        if hard:
            f[has_data] = tgt[has_data]
        f = f.reshape(shape)
        if not np.all(np.isfinite(f)):
            f = np.nan_to_num(f, nan=0.0, posinf=1.0, neginf=-1.0)
        candidates.append((f"harm_{lam}_{hard}_{eps}", f))

    # ---- bounded, data-driven model selection ----------------------------
    budget_wall = 120.0
    if isinstance(budget, dict):
        try:
            budget_wall = float(budget.get("wall_time_seconds", budget_wall))
        except Exception:
            pass
    abort_after = t_start + 0.55 * max(budget_wall, 1.0)

    best_f = None
    best_score = float("inf")
    for _name, f in candidates:
        if not (np.any(f > 0) and np.any(f < 0)):
            continue
        try:
            score = _proxy_score(problem, f)
        except Exception:
            score = float("inf")
        if score < best_score:
            best_score = score
            best_f = f.copy()
        if time.time() > abort_after:
            break
    if best_f is not None:
        return np.ascontiguousarray(best_f, dtype=np.float64)

    # ---- guaranteed finite sign-changing fallback ------------------------
    f = prior.reshape(shape) if prior.size == n else np.zeros(shape)
    f = np.asarray(f, dtype=np.float64)
    if not (np.any(f > 0) and np.any(f < 0)):
        acc = np.zeros(shape, dtype=np.float64)
        for ax in range(ndim):
            v = ((np.arange(shape[ax]) - (shape[ax] - 1) / 2.0)
                 / max(shape[ax] / 3.0, 1.0)).reshape(
                [-1 if i == ax else 1 for i in range(ndim)])
            acc = acc + v * v
        f = 1.0 - acc
        if not (np.any(f > 0) and np.any(f < 0)):
            f = idx.astype(np.float64) - n / 2.0
    return np.ascontiguousarray(f, dtype=np.float64)
