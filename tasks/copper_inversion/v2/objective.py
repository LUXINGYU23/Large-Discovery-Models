"""Public data objective. This module never accepts hidden geology."""

import numpy as np
from scipy.ndimage import map_coordinates
from .contract import LEGACY_OBJECTIVE, OBJECTIVE, OBJECTIVE_VERSIONS
from .geometry import canonicalize, diagnostics

K_TOL = .5


def grouped_mean(values, groups):
    return float(np.mean([values[groups == group].mean() for group in np.unique(groups)])) if len(values) else 0.


def _score_v3(problem, mask, phi, result, *, k_tol=K_TOL):
    origin = np.asarray(problem["origin"], dtype=float)
    spacing = np.asarray(problem["spacing"], dtype=float)
    disc = .5 * float(np.linalg.norm(spacing))
    tp = fp = fn = 0.
    for obs in problem["observations"]:
        pts = np.asarray(obs["points"], dtype=float)
        coords = ((pts - origin) / spacing - .5).T
        distance = map_coordinates(phi, coords, order=1, mode="nearest")
        idx = np.clip(np.rint(coords.T).astype(int), 0, np.asarray(mask.shape) - 1)
        pred = mask[tuple(idx.T)]
        y, groups = np.asarray(obs["labels"], dtype=float), np.asarray(obs["groups"])
        binary, contact = y != .5, y == .5
        sigma_meas = float(np.sqrt(max(0., float(obs["transition_scale"])**2 - disc**2)))
        delta = float(np.hypot(float(k_tol) * spacing[2], sigma_meas))
        keep = binary.copy()
        if contact.any() and delta > 0.:
            for grp in np.unique(groups[contact]):
                c_pts = pts[contact & (groups == grp)]
                m = np.flatnonzero(binary & (groups == grp))
                if len(m):
                    d = np.min(np.linalg.norm(pts[m][:, None, :] - c_pts[None, :, :], axis=2), axis=1)
                    keep[m[d < delta]] = False
        p_keep, y_keep = pred[keep], y[keep] > .5
        n_pos = float(np.count_nonzero(p_keep))
        n_neg = float(np.count_nonzero(~p_keep))
        a = float(np.count_nonzero(p_keep & y_keep))
        c = float(np.count_nonzero(~p_keep & y_keep))
        eps = float(obs["label_error"])
        if eps > 0.:
            denom = 1. - 2. * eps
            a = float(np.clip((a - eps * n_pos) / denom, 0., n_pos))
            c = float(np.clip((c - eps * n_neg) / denom, 0., n_neg))
        w = float(obs["weight"])
        mod_tp, mod_fp, mod_fn = w * a, w * (n_pos - a), w * c
        tp += mod_tp
        fp += mod_fp
        fn += mod_fn
        result[f"{obs['name']}_tp"] = mod_tp
        result[f"{obs['name']}_fp"] = mod_fp
        result[f"{obs['name']}_fn"] = mod_fn
        result[f"{obs['name']}_kept_samples"] = int(np.count_nonzero(keep))
        result[f"{obs['name']}_label_error"] = grouped_mean((pred[binary] != (y[binary] > .5)).astype(float), groups[binary])
        result[f"{obs['name']}_contact_rmse"] = float(np.sqrt(np.mean(distance[contact]**2))) if contact.any() else 0.
    jaccard = float(1. - (tp + 1.) / (tp + fp + fn + 1.))
    result.update(objective_version=OBJECTIVE, loss_jaccard=jaccard, tp=float(tp), fp=float(fp), fn=float(fn))
    result["loss_total"] = float(jaccard + sum(
        problem["priors"][f"{name}_weight"] * result[f"regularizer_{name}"]
        for name in ("area", "volume", "fragment")))
    if not np.isfinite(result["loss_total"]):
        raise ValueError("nonfinite public objective")
    return result


def _score_v2(problem, phi, result):
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


def score(problem, field, *, objective_version=None, k_tol=K_TOL):
    mask, phi = canonicalize(problem, field)
    version = problem["objective_version"] if objective_version is None else objective_version
    if version not in OBJECTIVE_VERSIONS:
        raise ValueError("unsupported objective_version")
    result = diagnostics(problem, mask)
    if version == LEGACY_OBJECTIVE:
        return _score_v2(problem, phi, result)
    return _score_v3(problem, mask, phi, result, k_tol=k_tol)
