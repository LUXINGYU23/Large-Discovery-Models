"""Public data objective. This module never accepts hidden geology."""

import numpy as np
from scipy.ndimage import map_coordinates
from .geometry import canonicalize, diagnostics


def grouped_mean(values, groups):
    return float(np.mean([values[groups == group].mean() for group in np.unique(groups)])) if len(values) else 0.


def score(problem, field):
    mask, phi = canonicalize(problem, field)
    result = diagnostics(problem, mask)
    binary_total = contact_total = 0.
    total_weight = sum(o["weight"] for o in problem["observations"])
    for obs in problem["observations"]:
        coords = ((obs["points"] - np.asarray(problem["origin"])) / problem["spacing"] - .5).T
        distance = map_coordinates(phi, coords, order=1, mode="nearest")
        y, groups = obs["labels"], obs["groups"]
        binary, contact = y != .5, y == .5
        signed = (2 * y[binary] - 1) * distance[binary] / obs["transition_scale"]
        log_correct = -np.logaddexp(0., -signed)
        epsilon = obs["label_error"]
        nll = -log_correct if epsilon == 0 else -np.logaddexp(np.log(epsilon), np.log1p(-2*epsilon) + log_correct)
        b = grouped_mean(nll, groups[binary])
        residual = np.abs(distance[contact] / obs["contact_sigma"])
        huber = np.where(residual <= 1, .5 * residual**2, residual - .5)
        c = grouped_mean(huber, groups[contact])
        weight = obs["weight"] / total_weight
        binary_total += weight * b
        contact_total += weight * c
        result[f"{obs['name']}_binary_nll"] = b
        result[f"{obs['name']}_contact_huber"] = c
        result[f"{obs['name']}_label_error"] = grouped_mean(((distance[binary] > 0) != y[binary]).astype(float), groups[binary])
        result[f"{obs['name']}_contact_rmse"] = float(np.sqrt(np.mean(distance[contact]**2))) if contact.any() else 0.
    result.update(loss_binary=binary_total, loss_contact=contact_total)
    result["loss_total"] = binary_total + contact_total + sum(
        problem["priors"][f"{name}_weight"] * result[f"regularizer_{name}"]
        for name in ("area", "volume", "fragment"))
    if not np.isfinite(result["loss_total"]):
        raise ValueError("nonfinite public objective")
    return result
