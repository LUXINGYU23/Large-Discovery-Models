"""objective_v1 adapted from Xiaolong Wei's MIT-licensed loss_functions.py.

Contact labels intentionally contribute to both positive logistic and contact terms.
"""

import numpy as np
from .geometry import signed_distance


def binary_loss(phi, values, weight):
    mask = np.isfinite(values)
    y = values[mask]
    d = phi[mask]
    contact = phi[values == 0.5]
    logistic = np.logaddexp(0, d[y == 0]).sum() + np.logaddexp(0, -d[y > 0]).sum()
    bias = float(contact.mean() ** 2) if contact.size else 0.0
    variance = float(np.mean(contact**2)) if contact.size else 0.0
    return float(weight * (logistic + bias + variance))


def sketch_loss(reference, field, spacing, origin):
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
        raise ValueError("degenerate sketch geometry")
    a = a - a.mean(0)
    b = b - b.mean(0)
    an = np.linalg.norm(a)
    bn = np.linalg.norm(b)
    if not an or not bn:
        raise ValueError("zero sketch norm")
    a = a / an
    b = b / bn
    n = max(len(a), len(b))
    a = np.pad(a, ((0, n - len(a)), (0, 0)))
    b = np.pad(b, ((0, n - len(b)), (0, 0)))
    singular = np.linalg.svd(a.T @ b, compute_uv=False)
    return float(5 * (1 - singular.sum() ** 2))


def score(problem, field):
    phi = signed_distance(field, problem["shape"], problem["spacing"])
    parts = {
        f"loss_{o['name']}": binary_loss(phi, o["values"], o["weight"])
        for o in problem["observations"]
    }
    if problem["sketch"] is not None:
        parts["loss_sketch"] = sketch_loss(
            problem["sketch"], phi, problem["spacing"], problem["origin"]
        )
    total = float(sum(parts.values()))
    if not np.isfinite(total):
        raise ValueError("nonfinite objective")
    contacts = np.concatenate(
        [phi[o["values"] == 0.5] for o in problem["observations"]]
    )
    mismatch = sum(
        int(np.sum((phi[o["values"] == 0] > 0)))
        + int(np.sum((phi[o["values"] == 1] <= 0)))
        for o in problem["observations"]
    )
    return {
        **parts,
        "loss_total": total,
        "research_utility": -total,
        "positive_voxels": int((phi > 0).sum()),
        "label_mismatches": mismatch,
        "contact_rmse": float(np.sqrt(np.mean(contacts**2))) if contacts.size else 0.0,
    }
