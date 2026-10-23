import numpy as np
from scipy import ndimage


def _distance_difference(positive, negative, spacing):
    if positive.any() and negative.any():
        dp = ndimage.distance_transform_edt(~positive, sampling=spacing)
        dn = ndimage.distance_transform_edt(~negative, sampling=spacing)
        return dn - dp
    return None


def solve(problem, budget, seed):
    shape = tuple(int(x) for x in problem["shape"])
    ndim = len(shape)
    spacing = tuple(float(x) for x in problem.get("spacing", [1.0] * ndim))
    positive = np.zeros(shape, dtype=bool)
    negative = np.zeros(shape, dtype=bool)
    observations = problem.get("observations", [])
    weights = []
    max_weight = max((float(o.get("weight", 1.0)) for o in observations), default=1.0)
    for o in observations:
        v = np.asarray(o["values"])
        m = np.asarray(o.get("mask", np.isfinite(v)), dtype=bool) & np.isfinite(v)
        rel = np.sqrt(max(float(o.get("weight", 1.0)), 0.0) / max(max_weight, 1e-12))
        if "outcrop" in str(o.get("name", "")).lower() and ndim == 3:
            rel = 0.30
        weights.append(rel)
        positive |= m & (v == 1)
        negative |= m & (v == 0)

    raw = np.zeros(shape, dtype=np.float64)
    density = np.zeros(shape, dtype=np.float64)
    for o, rel in zip(observations, weights):
        v = np.asarray(o["values"])
        m = np.asarray(o.get("mask", np.isfinite(v)), dtype=bool) & np.isfinite(v)
        ordinary = m & (v != 0.5)
        raw[ordinary] += np.where(v[ordinary] == 1, 1.0, -1.0) * rel
        density[ordinary] += rel
        contacts = m & (v == 0.5)
        raw[contacts] += 3.0 * 0.15
        density[contacts] += 3.0

    nearest = _distance_difference(positive, negative, spacing)
    if nearest is not None:
        if ndim == 3:
            r1, r2 = (1.3, 1.3, 1.4), (2.6, 2.6, 1.8)
            correction, blend = 0.04, 0.475
        else:
            r1, r2 = (1.1,) * ndim, (1.8,) * ndim
            correction, blend = 0.06, 0.50
        sig1 = tuple(max(r / s, 0.35) for r, s in zip(r1, spacing))
        sig2 = tuple(max(r / s, 0.45) for r, s in zip(r2, spacing))
        smooth1 = ndimage.gaussian_filter(raw, sig1)
        smooth1 /= ndimage.gaussian_filter(density, sig1) + 1e-9
        smooth2 = ndimage.gaussian_filter(raw, sig2)
        smooth2 /= ndimage.gaussian_filter(density, sig2) + 1e-9
        field = blend * smooth1 + (1.0 - blend) * smooth2 + correction * nearest
    else:
        field = np.asarray(problem.get("initial_level_set", np.zeros(shape)), dtype=float).copy()

    sketch = problem.get("sketch", None)
    if ndim == 2 and sketch is not None:
        sk = np.asarray(sketch)
        sf = _distance_difference(sk == 1, sk != 1, spacing)
        if sf is not None:
            scale = max(float(np.median(np.abs(sf))), 1.0)
            field += 0.12 * np.tanh(sf / scale)

    field = np.asarray(field, dtype=np.float64)
    field[~np.isfinite(field)] = 0.0
    # A mild sign-preserving power adjusts the sub-cell zero crossing used by
    # skfmm without changing the learned classification.
    if ndim == 3 and nearest is not None:
        field = np.sign(field) * np.abs(field) ** 1.22
    if not np.any(field > 0) or not np.any(field < 0):
        init = np.asarray(problem.get("initial_level_set", np.zeros(shape)), dtype=float)
        if init.shape == shape and np.any(init > 0) and np.any(init < 0):
            field = init.copy()
        else:
            grid = np.indices(shape, dtype=float)
            center = np.array([(n - 1) / 2.0 for n in shape])
            q = sum(((grid[i] - center[i]) / max(shape[i] / 3.0, 1.0)) ** 2 for i in range(ndim))
            field = 1.0 - q
    return np.asarray(field, dtype=np.float64)
