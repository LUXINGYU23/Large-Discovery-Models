"""Trusted geometry accuracy; import only in the independent scorer."""

import numpy as np
from scipy.spatial import cKDTree
from .geometry import canonicalize


def interface_points(mask, domain, spacing, origin):
    # Interior face centers, excluding model-domain cuts. No artificial padded shell.
    points, weights = [], []
    spacing = np.asarray(spacing)
    for axis in range(3):
        a, b = [slice(None)] * 3, [slice(None)] * 3
        a[axis], b[axis] = slice(None, -1), slice(1, None)
        a, b = tuple(a), tuple(b)
        index = np.argwhere((mask[a] != mask[b]) & domain[a] & domain[b]).astype(float)
        index += .5
        index[:, axis] += .5
        points.append(np.asarray(origin) + index * spacing)
        weights.append(np.full(len(index), np.prod(spacing) / spacing[axis]))
    return np.concatenate(points), np.concatenate(weights)


def compare(problem, field, truth):
    mask, _ = canonicalize(problem, field)
    if not isinstance(truth, np.ndarray) or truth.dtype != bool or truth.shape != mask.shape:
        raise ValueError("invalid trusted truth")
    if np.any(truth & (~problem["domain_mask"] | problem["forbidden_mask"])):
        raise ValueError("trusted truth violates public feasible region")
    union = int(np.count_nonzero(mask | truth))
    iou = float(np.count_nonzero(mask & truth) / union) if union else 1.
    p, pw = interface_points(mask, problem["domain_mask"], problem["spacing"], problem["origin"])
    t, tw = interface_points(truth, problem["domain_mask"], problem["spacing"], problem["origin"])
    tolerance = max(float(np.linalg.norm(problem["spacing"])),
                    max(o["contact_sigma"] for o in problem["observations"]))
    cap = float(np.linalg.norm(np.asarray(problem["shape"]) * problem["spacing"]))
    if len(p) and len(t):
        dp, dt = cKDTree(t).query(p)[0], cKDTree(p).query(t)[0]
        precision, recall = np.average(dp <= tolerance, weights=pw), np.average(dt <= tolerance, weights=tw)
        f1 = float(2 * precision * recall / (precision + recall)) if precision + recall else 0.
        distances, weights = np.concatenate([dp, dt]), np.concatenate([pw, tw])
        average = float(np.average(distances, weights=weights))
        order = np.argsort(distances)
        hd95 = float(distances[order][np.searchsorted(np.cumsum(weights[order]), .95 * weights.sum())])
    else:
        identical = np.array_equal(mask, truth)
        f1, average, hd95 = (1., 0., 0.) if identical else (0., cap, cap)
    return dict(iou=iou, surface_f1=f1, surface_tolerance=tolerance,
                mean_surface_distance=average, hd95=hd95,
                relative_volume_error=float(abs(int(mask.sum()) - int(truth.sum())) / max(1, int(truth.sum()))))
