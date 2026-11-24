"""Published in-container baselines (baselines-lit.md §3: M1, M2, M3, M6, M8-kNN).

NumPy/SciPy only; deterministic; no file or network access; public inputs only.
"""

import time
import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.linalg import cholesky, solve, solve_triangular
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.special import expit

from . import baseline_harness as harness


def _ordered_configs(default, rest):
    out = [default]
    for cfg in rest:
        if cfg != default:
            out.append(cfg)
    return tuple(out)


# ---------------------------------------------------------------------------
# M1. Signed-distance RBF implicit model — rbf_sdf
# ---------------------------------------------------------------------------

RBF_SDF_CONFIGS = _ordered_configs(
    {"s_z": 4, "kernel": "linear", "alpha": 1.0},
    [
        {"s_z": s_z, "kernel": kernel, "alpha": alpha}
        for alpha in (4.0, 1.0, 0.0)
        for kernel in ("linear", "cubic")
        for s_z in (4, 2, 8, 1)
    ],
)


def fit_rbf_sdf(config, X_b, y_b, w_b, X_c, problem, *, center=None, E=None, hbar=None, h_hole=None):
    shape = np.asarray(problem["shape"], dtype=float)
    origin = np.asarray(problem["origin"], dtype=float)
    spacing = np.asarray(problem["spacing"], dtype=float)
    if E is None:
        E = shape * spacing
    if center is None:
        center = origin + 0.5 * E
    if hbar is None:
        hbar = float(np.mean(spacing))

    s_z = config["s_z"]
    kernel = config["kernel"]
    alpha = float(config["alpha"])

    X_all, v_all, w_all = harness.compute_sdf_targets(problem, s_z, center=center, h_hole=h_hole)
    tol = 1e-6 * float(np.linalg.norm(E))
    X_m, v_m, w_m = harness.merge_close_points(X_all, v_all, w_all, tol)

    kernel_unit = hbar if kernel == "linear" else hbar ** 3
    s = alpha * kernel_unit
    smoothing = np.full(len(X_m), s, dtype=float) / np.maximum(w_m, 1e-12)
    try:
        rbf = RBFInterpolator(X_m, v_m, kernel=kernel, degree=1, smoothing=smoothing)
    except np.linalg.LinAlgError:
        if alpha == 0.0:
            retry_s = np.full(len(X_m), 1e-8 * kernel_unit, dtype=float) / np.maximum(w_m, 1e-12)
            rbf = RBFInterpolator(X_m, v_m, kernel=kernel, degree=1, smoothing=retry_s)
        else:
            raise

    def predict(points):
        return rbf(harness.stretch_points(points, center, s_z))

    return predict


def rbf_sdf(problem, budget=None, seed=0, *, decision_rule=None, diagnostics=None):
    """M1: Signed-distance RBF implicit model (Cowan et al. 2003; Carr et al. 2001)."""
    data = harness.extract_public_data(problem)
    center, E, hbar, h_hole = data["center"], data["E"], data["hbar"], data["h_hole"]

    def fit(cfg, X_b, y_b, w_b, X_c, prob):
        return fit_rbf_sdf(cfg, X_b, y_b, w_b, X_c, prob, center=center, E=E, hbar=hbar, h_hole=h_hole)

    return harness.tune_and_solve(
        problem, budget, RBF_SDF_CONFIGS, fit, tau0=0.0,
        decision_rule=decision_rule, diagnostics=diagnostics,
    )


# ---------------------------------------------------------------------------
# M2. Indicator kriging — indicator_kriging
# ---------------------------------------------------------------------------

INDICATOR_KRIGING_CONFIGS = _ordered_configs(
    {"s_z": 4, "rho": "exponential", "nu": 0.1},
    [
        {"s_z": s_z, "rho": rho, "nu": nu}
        for nu in (0.3, 0.1, 0.02)
        for rho in ("exponential", "spherical")
        for s_z in (4, 2, 8, 1)
    ],
)


def _kriging_corr(u, rho):
    u = np.maximum(np.asarray(u, dtype=float), 0.0)
    if rho == "exponential":
        return np.exp(-3.0 * u)
    if rho == "spherical":
        return np.where(u < 1.0, 1.0 - 1.5 * u + 0.5 * (u ** 3), 0.0)
    raise ValueError(f"unknown variogram model: {rho!r}")


def empirical_downhole_variogram(problem):
    """Experimental down-hole indicator semivariogram from binary drill samples."""
    drill = harness._base()._drill_observation(problem)
    pts = np.asarray(drill["points"], dtype=float)
    lab = np.asarray(drill["labels"], dtype=float)
    grp = np.asarray(drill["groups"])
    b = lab != 0.5
    if not b.any():
        return np.zeros(0), np.zeros(0), np.zeros(0, dtype=int)
    bp, by, bg = pts[b], (lab[b] > 0.5).astype(float), grp[b]

    consec_dists = []
    hole_lens = []
    pair_dists = []
    pair_semis = []
    for gid in np.unique(bg):
        sel = bg == gid
        hp, hy = bp[sel], by[sel]
        if len(hp) < 2:
            continue
        order = np.argsort(hp[:, 2])
        hp, hy = hp[order], hy[order]
        diffs = np.linalg.norm(np.diff(hp, axis=0), axis=1)
        pos_diffs = diffs[diffs > 0.0]
        if len(pos_diffs):
            consec_dists.append(pos_diffs)
        hlen = float(np.linalg.norm(hp[-1] - hp[0]))
        if hlen > 0.0:
            hole_lens.append(hlen)
        iu, ju = np.triu_indices(len(hp), k=1)
        d = np.linalg.norm(hp[iu] - hp[ju], axis=1)
        s = 0.5 * ((hy[iu] - hy[ju]) ** 2)
        pair_dists.append(d)
        pair_semis.append(s)

    if not consec_dists or not hole_lens or not pair_dists:
        return np.zeros(0), np.zeros(0), np.zeros(0, dtype=int)
    delta = float(np.median(np.concatenate(consec_dists)))
    half_len = 0.5 * float(np.median(hole_lens))
    if delta <= 0.0 or half_len < delta:
        return np.zeros(0), np.zeros(0), np.zeros(0, dtype=int)

    max_k = int(np.floor(half_len / delta))
    if max_k < 1:
        return np.zeros(0), np.zeros(0), np.zeros(0, dtype=int)

    all_d = np.concatenate(pair_dists)
    all_s = np.concatenate(pair_semis)
    # Assign each pair to nearest lag k*delta within tolerance delta/2.
    k_idx = np.rint(all_d / delta).astype(np.int64)
    valid = (k_idx >= 1) & (k_idx <= max_k) & (np.abs(all_d - k_idx * delta) <= 0.5 * delta)
    if not valid.any():
        return np.zeros(0), np.zeros(0), np.zeros(0, dtype=int)
    k_idx = k_idx[valid]
    all_s = all_s[valid]

    counts = np.bincount(k_idx, minlength=max_k + 1)[1:]
    sums = np.bincount(k_idx, weights=all_s, minlength=max_k + 1)[1:]
    keep = counts >= 30
    if int(keep.sum()) < 3:
        return np.zeros(0), np.zeros(0), np.zeros(0, dtype=int)
    ks = np.arange(1, max_k + 1, dtype=float)[keep]
    return ks * delta, sums[keep] / counts[keep], counts[keep]


def fit_vertical_range(h_lags, gamma_hat, n_pairs, rho, sigma2, c0, hz, E_z):
    """Fit vertical range a_v on 30 log-spaced values in [2*hz, E_z] by weighted least squares."""
    if len(h_lags) < 3 or sigma2 <= 1e-12:
        return 0.25 * E_z
    lo_av = max(2.0 * hz, 1e-9)
    hi_av = max(E_z, lo_av * 1.001)
    av_grid = np.geomspace(lo_av, hi_av, 30)
    weights = n_pairs.astype(float) / (h_lags ** 2)
    sill = sigma2 - c0
    u = h_lags[:, None] / av_grid[None, :]
    gamma_m = c0 + sill * (1.0 - _kriging_corr(u, rho))
    sse = np.sum(weights[:, None] * ((gamma_hat[:, None] - gamma_m) ** 2), axis=0)
    return float(av_grid[int(np.argmin(sse))])


def fit_indicator_kriging(config, X_b, y_b, w_b, X_c, problem, *, center=None, E=None, hz=None,
                          eps=None, drill_weight=None, variogram_cache=None):
    shape = np.asarray(problem["shape"], dtype=float)
    origin = np.asarray(problem["origin"], dtype=float)
    spacing = np.asarray(problem["spacing"], dtype=float)
    if E is None:
        E = shape * spacing
    if center is None:
        center = origin + 0.5 * E
    if hz is None:
        hz = float(spacing[2])
    drill = harness._base()._drill_observation(problem)
    if eps is None:
        eps = float(drill.get("label_error", 0.0))
    if drill_weight is None:
        drill_weight = float(drill.get("weight", 1.0))

    sigma2 = float(np.var(y_b))
    if sigma2 <= 1e-12:
        raise ValueError("constant binary indicators")

    nu = float(config["nu"])
    nu_eff = max(nu, 4.0 * eps * (1.0 - eps)) if eps > 0.0 else nu
    nu_eff = min(nu_eff, 0.99)
    c0 = nu_eff * sigma2
    sill = sigma2 - c0

    key = id(problem)
    if variogram_cache is not None and key in variogram_cache:
        h_lags, gamma_hat, n_pairs = variogram_cache[key]
    else:
        h_lags, gamma_hat, n_pairs = empirical_downhole_variogram(problem)
        if variogram_cache is not None:
            variogram_cache[key] = (h_lags, gamma_hat, n_pairs)

    rho = config["rho"]
    a_v = fit_vertical_range(h_lags, gamma_hat, n_pairs, rho, sigma2, c0, hz, float(E[2]))
    s_z = float(config["s_z"])
    a_h = max(s_z * a_v, 1e-9)

    if len(X_c):
        X_all = np.concatenate([X_b, X_c], axis=0)
        y_all = np.concatenate([y_b, np.full(len(X_c), 0.5, dtype=float)], axis=0)
        w_all = np.concatenate([w_b, np.full(len(X_c), drill_weight, dtype=float)], axis=0)
    else:
        X_all, y_all, w_all = X_b, y_b, w_b

    X_s = harness.stretch_points(X_all, center, s_z)
    n = len(X_s)
    D_mat = cdist(X_s, X_s) / a_h
    C = sill * _kriging_corr(D_mat, rho)
    diag_d = (c0 / np.maximum(w_all, 1e-12)) if c0 > 0.0 else np.full(n, 1e-10 * sigma2, dtype=float)
    C[np.diag_indices(n)] += diag_d

    A = np.empty((n + 1, n + 1), dtype=float)
    A[:n, :n] = C
    A[:n, n] = 1.0
    A[n, :n] = 1.0
    A[n, n] = 0.0
    rhs = np.empty(n + 1, dtype=float)
    rhs[:n] = y_all
    rhs[n] = 0.0

    sol = solve(A, rhs, assume_a="sym")
    if not np.isfinite(sol).all():
        raise ValueError("non-finite kriging solution")
    b_coef = sol[:n]
    mu = float(sol[n])

    def predict(points):
        pts_s = harness.stretch_points(points, center, s_z)
        c_mat = sill * _kriging_corr(cdist(pts_s, X_s) / a_h, rho)
        return np.clip(c_mat @ b_coef + mu, 0.0, 1.0)

    return predict


def indicator_kriging(problem, budget=None, seed=0, *, decision_rule=None, diagnostics=None):
    """M2: Dual ordinary indicator kriging with down-hole variogram fit (Journel 1983)."""
    data = harness.extract_public_data(problem)
    center, E, hz = data["center"], data["E"], data["hz"]
    eps, drill_weight = data["eps"], data["drill_weight"]
    vcache = {}

    def fit(cfg, X_b, y_b, w_b, X_c, prob):
        return fit_indicator_kriging(
            cfg, X_b, y_b, w_b, X_c, prob,
            center=center, E=E, hz=hz, eps=eps, drill_weight=drill_weight, variogram_cache=vcache,
        )

    return harness.tune_and_solve(
        problem, budget, INDICATOR_KRIGING_CONFIGS, fit, tau0=0.5,
        decision_rule=decision_rule, diagnostics=diagnostics,
    )


# ---------------------------------------------------------------------------
# M3. Kernel SVM on coordinates — svm_rbf
# ---------------------------------------------------------------------------

SVM_RBF_CONFIGS = _ordered_configs(
    {"s_z": 4, "ell_mult": 1.0, "C": 1.0},
    [
        {"s_z": s_z, "ell_mult": ell_mult, "C": C}
        for C in (1.0, 100.0)
        for ell_mult in (2.0, 1.0, 0.5)
        for s_z in (4, 2, 8, 1)
    ],
)


def svm_kkt_residual(Q, alpha, upper):
    """Maximum KKT violation for min 0.5 a^T Q a - 1^T a s.t. 0 <= a_i <= upper_i."""
    g = Q @ alpha - 1.0
    upper = np.asarray(upper, dtype=float)
    tol = 1e-9 * np.maximum(1.0, upper)
    v_lo = np.where(alpha < upper - tol, np.maximum(0.0, -g), 0.0)
    v_hi = np.where(alpha > tol, np.maximum(0.0, g), 0.0)
    return float(max(np.max(v_lo, initial=0.0), np.max(v_hi, initial=0.0)))


def solve_svm_dual(X_s, y_b, w_b, ell, C):
    """Solve the bias-absorbed RBF SVM dual by L-BFGS-B."""
    y_tilde = 2.0 * np.asarray(y_b, dtype=float) - 1.0
    if not (y_tilde > 0.0).any() or not (y_tilde < 0.0).any():
        raise ValueError("SVM requires both classes")
    ell = max(float(ell), 1e-9)
    inv_2ell2 = 1.0 / (2.0 * ell * ell)
    K = np.exp(-cdist(X_s, X_s, metric="sqeuclidean") * inv_2ell2)
    Q = (y_tilde[:, None] * y_tilde[None, :]) * (K + 1.0)
    upper = float(C) * np.asarray(w_b, dtype=float)
    bounds = list(zip(np.zeros(len(upper), dtype=float), upper))

    def fun(alpha):
        Qa = Q @ alpha
        return 0.5 * float(np.dot(alpha, Qa)) - float(np.sum(alpha)), Qa - 1.0

    alpha0 = np.zeros(len(upper), dtype=float)
    res = minimize(
        fun, alpha0, method="L-BFGS-B", jac=True, bounds=bounds,
        options={"maxiter": 2000, "ftol": 1e-10, "gtol": 1e-7},
    )
    alpha = np.asarray(res.x, dtype=float)
    if not np.isfinite(alpha).all():
        raise ValueError("non-finite SVM dual solution")
    # If large C stopped early on relative ftol with KKT residual > 1e-4, polish tightly.
    if svm_kkt_residual(Q, alpha, upper) > 1e-4:
        res2 = minimize(
            fun, alpha, method="L-BFGS-B", jac=True, bounds=bounds,
            options={"maxiter": 2000, "ftol": 1e-14, "gtol": 1e-8},
        )
        if np.isfinite(res2.x).all():
            alpha = np.asarray(res2.x, dtype=float)
    return alpha, Q, upper, y_tilde


def fit_svm_rbf(config, X_b, y_b, w_b, X_c, problem, *, center=None, h_hole=None):
    shape = np.asarray(problem["shape"], dtype=float)
    origin = np.asarray(problem["origin"], dtype=float)
    spacing = np.asarray(problem["spacing"], dtype=float)
    E = shape * spacing
    if center is None:
        center = origin + 0.5 * E
    if h_hole is None:
        drill = harness._base()._drill_observation(problem)
        n_h = len(np.unique(drill["groups"])) if len(drill["groups"]) else 1
        h_hole = float(np.sqrt(E[0] * E[1] / max(1, n_h)))

    s_z = float(config["s_z"])
    ell = max(float(config["ell_mult"]) * h_hole, 1e-9)
    C = float(config["C"])
    X_s = harness.stretch_points(X_b, center, s_z)
    alpha, _, _, y_tilde = solve_svm_dual(X_s, y_b, w_b, ell, C)

    coef = alpha * y_tilde
    bias = float(np.sum(coef))
    sv = np.abs(coef) > 1e-12
    X_sv = X_s[sv]
    coef_sv = coef[sv]
    inv_2ell2 = 1.0 / (2.0 * ell * ell)

    def predict(points):
        pts_s = harness.stretch_points(points, center, s_z)
        if len(X_sv) == 0:
            return np.full(len(pts_s), bias, dtype=float)
        K_star = np.exp(-cdist(pts_s, X_sv, metric="sqeuclidean") * inv_2ell2)
        return K_star @ coef_sv + bias

    return predict


def svm_rbf(problem, budget=None, seed=0, *, decision_rule=None, diagnostics=None):
    """M3: Kernel SVM on stretched coordinates (Smirnoff et al. 2008; Cortes & Vapnik 1995)."""
    data = harness.extract_public_data(problem)
    center, h_hole = data["center"], data["h_hole"]

    def fit(cfg, X_b, y_b, w_b, X_c, prob):
        return fit_svm_rbf(cfg, X_b, y_b, w_b, X_c, prob, center=center, h_hole=h_hole)

    return harness.tune_and_solve(
        problem, budget, SVM_RBF_CONFIGS, fit, tau0=0.0,
        decision_rule=decision_rule, diagnostics=diagnostics,
    )


# ---------------------------------------------------------------------------
# M6. Gaussian-process classification (Laplace) — gp_classifier
# ---------------------------------------------------------------------------

GP_CLASSIFIER_CONFIGS = _ordered_configs(
    {"s_z": 4, "ell_mult": 1.0, "sigma_f2": 4.0},
    [
        {"s_z": s_z, "ell_mult": ell_mult, "sigma_f2": sigma_f2}
        for sigma_f2 in (4.0, 25.0)
        for ell_mult in (2.0, 1.0, 0.5)
        for s_z in (4, 2, 8, 1)
    ],
)


def _matern32(dist, ell, sigma_f2):
    u = np.sqrt(3.0) * (dist / max(float(ell), 1e-9))
    return float(sigma_f2) * (1.0 + u) * np.exp(-u)


def fit_gp_laplace(config, X_b, y_b, w_b, X_c, problem, *, center=None, h_hole=None):
    """Rasmussen & Williams Alg 3.1 Laplace GP classifier with soft targets and non-zero prior mean.

    Returns (predict, log_evidence). predict(points, return_var=False) returns latent mean
    (or (mean, var) if return_var=True).
    """
    b = harness._base()
    shape = np.asarray(problem["shape"], dtype=float)
    origin = np.asarray(problem["origin"], dtype=float)
    spacing = np.asarray(problem["spacing"], dtype=float)
    E = shape * spacing
    if center is None:
        center = origin + 0.5 * E
    drill = b._drill_observation(problem)
    if h_hole is None:
        n_h = len(np.unique(drill["groups"])) if len(drill["groups"]) else 1
        h_hole = float(np.sqrt(E[0] * E[1] / max(1, n_h)))
    eps = float(drill.get("label_error", 0.0))
    drill_weight = float(drill.get("weight", 1.0))

    e = max(eps, 0.01)
    t_b = np.where(y_b > 0.5, 1.0 - e, e)
    if len(X_c):
        X_all = np.concatenate([X_b, X_c], axis=0)
        t_all = np.concatenate([t_b, np.full(len(X_c), 0.5, dtype=float)], axis=0)
        w_all = np.concatenate([w_b, np.full(len(X_c), drill_weight, dtype=float)], axis=0)
    else:
        X_all, t_all, w_all = X_b, t_b, w_b
    if len(X_all) == 0:
        raise ValueError("no samples for GP classifier")

    p_clip = float(np.clip(b._drill_ore_share(problem), harness.SHARE_CLIP[0], harness.SHARE_CLIP[1]))
    m = float(np.log(p_clip / (1.0 - p_clip)))

    s_z = float(config["s_z"])
    ell = max(float(config["ell_mult"]) * h_hole, 1e-9)
    sigma_f2 = float(config["sigma_f2"])
    X_s = harness.stretch_points(X_all, center, s_z)
    n = len(X_s)

    K = _matern32(cdist(X_s, X_s), ell, sigma_f2)
    K[np.diag_indices(n)] += 1e-10 * sigma_f2

    def _log_lik(g_vec):
        return float(np.sum(w_all * (t_all * g_vec - np.logaddexp(0.0, g_vec))))

    h = np.zeros(n, dtype=float)
    a = np.zeros(n, dtype=float)
    g = np.full(n, m, dtype=float)
    obj = _log_lik(g)

    for _ in range(30):
        pi = expit(g)
        W_diag = w_all * pi * (1.0 - pi)
        sW = np.sqrt(np.maximum(W_diag, 0.0))
        B = (sW[:, None] * K) * sW[None, :]
        B[np.diag_indices(n)] += 1.0
        L = cholesky(B, lower=True)
        grad = w_all * (t_all - pi)
        b_vec = W_diag * h + grad
        v = solve_triangular(L, sW * (K @ b_vec), lower=True)
        a_new = b_vec - sW * solve_triangular(L.T, v, lower=False)
        h_new = K @ a_new
        g_new = m + h_new
        obj_new = -0.5 * float(np.dot(a_new, h_new)) + _log_lik(g_new)

        if not np.isfinite(obj_new) or obj_new < obj - 1e-12:
            step = 0.5
            accepted = False
            while step >= 1.0 / 1024.0:
                a_try = (1.0 - step) * a + step * a_new
                h_try = (1.0 - step) * h + step * h_new
                g_try = m + h_try
                obj_try = -0.5 * float(np.dot(a_try, h_try)) + _log_lik(g_try)
                if np.isfinite(obj_try) and obj_try >= obj - 1e-12:
                    a_new, h_new, g_new, obj_new = a_try, h_try, g_try, obj_try
                    accepted = True
                    break
                step *= 0.5
            if not accepted:
                break

        diff = abs(obj_new - obj)
        a, h, g, obj = a_new, h_new, g_new, obj_new
        if diff < 1e-6:
            break

    pi = expit(g)
    W_diag = w_all * pi * (1.0 - pi)
    sW = np.sqrt(np.maximum(W_diag, 0.0))
    B = (sW[:, None] * K) * sW[None, :]
    B[np.diag_indices(n)] += 1.0
    L = cholesky(B, lower=True)
    log_q = -0.5 * float(np.dot(a, h)) + _log_lik(g) - float(np.sum(np.log(np.diag(L))))
    if not np.isfinite(log_q):
        raise ValueError("non-finite GP log evidence")
    coef = w_all * (t_all - pi)

    def predict(points, return_var=False):
        pts_s = harness.stretch_points(points, center, s_z)
        K_star = _matern32(cdist(pts_s, X_s), ell, sigma_f2)
        mean = m + K_star @ coef
        if not return_var:
            return mean
        v = solve_triangular(L, sW[:, None] * K_star.T, lower=True)
        var = np.maximum(sigma_f2 - np.sum(v * v, axis=0), 0.0)
        return mean, var

    return predict, float(log_q)


def gp_classifier(problem, budget=None, seed=0, *, decision_rule=None, diagnostics=None):
    """M6: Laplace GP classification with approximate marginal-likelihood model selection."""
    started = time.monotonic()
    data = harness.extract_public_data(problem)
    if not (data["y_b"] > 0.5).any():
        return harness.no_ore_fallback(problem, data, diagnostics=diagnostics, started=started)

    rules = harness._allowed_rules(decision_rule)
    deadline = harness._wall_deadline(budget, started)
    center, h_hole = data["center"], data["h_hole"]

    # Stage 1: select hyperparameters by largest approximate log evidence on full data.
    best_cfg = None
    best_pred = None
    best_ev = -np.inf
    n_eval = 0
    deadline_stopped = False
    for cfg in GP_CLASSIFIER_CONFIGS:
        if n_eval > 0 and time.monotonic() >= deadline:
            deadline_stopped = True
            break
        try:
            pred_fn, log_q = fit_gp_laplace(
                cfg, data["X_b"], data["y_b"], data["w_b"], data["X_c"], problem,
                center=center, h_hole=h_hole,
            )
        except Exception:
            continue
        n_eval += 1
        if log_q > best_ev + harness.TIE_TOL:
            best_ev = log_q
            best_cfg = cfg
            best_pred = pred_fn

    if best_pred is None:
        return harness.stretched_fallback(
            problem, data, diagnostics=diagnostics, started=started,
            cv_evaluated=n_eval, deadline_stopped=deadline_stopped,
        )

    # Stage 2: select decision rule by one pass of common CV on the winning hyperparameters.
    best_rule = rules[0] if len(rules) == 1 else "share"
    cv_score = None
    if len(rules) > 1:
        folds = harness.make_hole_folds(data, problem)
        if folds and time.monotonic() < deadline:
            def _fit_pred_only(cfg, X_b, y_b, w_b, X_c, prob):
                pred_fn, _ = fit_gp_laplace(cfg, X_b, y_b, w_b, X_c, prob, center=center, h_hole=h_hole)
                return pred_fn

            scores = harness.evaluate_config_cv(
                best_cfg, _fit_pred_only, folds, data["S"], tau0=0.0, rules=rules, deadline=deadline
            )
            if scores is not None:
                best_rule = "native"
                cv_score = scores["native"]
                if scores["share"] > cv_score + harness.TIE_TOL:
                    best_rule = "share"
                    cv_score = scores["share"]
            elif time.monotonic() >= deadline:
                deadline_stopped = True

    try:
        f = harness.predict_chunked(best_pred, data["G"])
        field = harness.finalize_field(
            problem,
            data,
            f,
            0.0,
            best_rule,
            selected_config=best_cfg,
            cv_score=cv_score,
            cv_evaluated=n_eval,
            deadline_stopped=deadline_stopped,
            diagnostics=diagnostics,
            started=started,
        )
        if diagnostics is not None:
            diagnostics["log_evidence"] = float(best_ev)
        return field
    except Exception:
        return harness.stretched_fallback(
            problem, data, diagnostics=diagnostics, started=started,
            cv_evaluated=n_eval, deadline_stopped=deadline_stopped,
        )


# ---------------------------------------------------------------------------
# M8 (in-container). k-nearest neighbours / IDW on coordinates — knn_idw
# ---------------------------------------------------------------------------

KNN_IDW_CONFIGS = tuple(
    {"s_z": s_z, "k": k}
    for k in (64, 16, 4, 2)
    for s_z in (4, 2, 8, 1)
)


def fit_knn_idw(config, X_b, y_b, w_b, X_c, problem, *, center=None):
    if len(X_b) == 0:
        raise ValueError("no binary samples for knn_idw")
    if center is None:
        shape = np.asarray(problem["shape"], dtype=float)
        origin = np.asarray(problem["origin"], dtype=float)
        spacing = np.asarray(problem["spacing"], dtype=float)
        center = origin + 0.5 * shape * spacing

    s_z = float(config["s_z"])
    k_eff = min(int(config["k"]), len(X_b))
    X_s = harness.stretch_points(X_b, center, s_z)
    tree = cKDTree(X_s)
    y_arr = np.asarray(y_b, dtype=float)
    w_arr = np.asarray(w_b, dtype=float)

    def predict(points):
        pts_s = harness.stretch_points(points, center, s_z)
        dists, idx = tree.query(pts_s, k=k_eff)
        if k_eff == 1:
            dists = dists[:, None]
            idx = idx[:, None]
        w_nb = w_arr[idx]
        y_nb = y_arr[idx]
        zero = dists <= 1e-15
        has_zero = np.any(zero, axis=1)
        out = np.empty(len(pts_s), dtype=float)
        if np.any(~has_zero):
            nz = ~has_zero
            inv = w_nb[nz] / (dists[nz] ** 2)
            out[nz] = np.sum(inv * y_nb[nz], axis=1) / np.sum(inv, axis=1)
        if np.any(has_zero):
            hz_mask = has_zero
            wz = np.where(zero[hz_mask], w_nb[hz_mask], 0.0)
            out[hz_mask] = np.sum(wz * y_nb[hz_mask], axis=1) / np.sum(wz, axis=1)
        return out

    return predict


def knn_idw(problem, budget=None, seed=0, *, decision_rule=None, diagnostics=None):
    """M8 (in-container): Distance-weighted k-nearest neighbours (Cover & Hart 1967; Shepard 1968)."""
    data = harness.extract_public_data(problem)
    center = data["center"]

    def fit(cfg, X_b, y_b, w_b, X_c, prob):
        return fit_knn_idw(cfg, X_b, y_b, w_b, X_c, prob, center=center)

    return harness.tune_and_solve(
        problem, budget, KNN_IDW_CONFIGS, fit, tau0=0.5,
        decision_rule=decision_rule, diagnostics=diagnostics,
    )


# ---------------------------------------------------------------------------
# Native-rule ("as published") variants and registry
# ---------------------------------------------------------------------------

def _make_native_wrapper(fn):
    def _native(problem, budget=None, seed=0, *, diagnostics=None):
        return fn(problem, budget, seed, decision_rule="native", diagnostics=diagnostics)
    _native.__name__ = f"{fn.__name__}_native"
    _native.__doc__ = f"{fn.__doc__} (decision rule forced to native)."
    return _native


rbf_sdf_native = _make_native_wrapper(rbf_sdf)
indicator_kriging_native = _make_native_wrapper(indicator_kriging)
svm_rbf_native = _make_native_wrapper(svm_rbf)
gp_classifier_native = _make_native_wrapper(gp_classifier)
knn_idw_native = _make_native_wrapper(knn_idw)

LIT_SOLVER_NAMES = (
    "rbf_sdf",
    "indicator_kriging",
    "svm_rbf",
    "gp_classifier",
    "knn_idw",
)

LIT_NATIVE_SOLVER_NAMES = (
    "rbf_sdf_native",
    "indicator_kriging_native",
    "svm_rbf_native",
    "gp_classifier_native",
    "knn_idw_native",
)

LIT_REFERENCE_SOLVERS = {
    "rbf_sdf": rbf_sdf,
    "indicator_kriging": indicator_kriging,
    "svm_rbf": svm_rbf,
    "gp_classifier": gp_classifier,
    "knn_idw": knn_idw,
    "rbf_sdf_native": rbf_sdf_native,
    "indicator_kriging_native": indicator_kriging_native,
    "svm_rbf_native": svm_rbf_native,
    "gp_classifier_native": gp_classifier_native,
    "knn_idw_native": knn_idw_native,
}
