"""M5 Discrete smooth implicit interpolation (fdi_smooth) after Grose et al. (2021).

In-container solver: imports only time, numpy, scipy, and v2 baseline_harness.
"""

import time
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, cg
from . import baseline_harness as harness


def _build_fdi_configs():
    default = {"s_z": 4, "lam": 0.1, "target": "sdf"}
    configs = [default]
    for lam in (1.0, 0.1, 0.01):
        for s_z in (4, 2, 8, 1):
            for target in ("sdf", "sign"):
                cfg = {"s_z": s_z, "lam": lam, "target": target}
                if cfg != default:
                    configs.append(cfg)
    return tuple(configs)


FDI_CONFIGS = _build_fdi_configs()


def _grid_geometry(origin, spacing, shape, *, coarse=False):
    origin = np.asarray(origin, dtype=float)
    spacing = np.asarray(spacing, dtype=float)
    shape = tuple(int(n) for n in shape)
    node_min = origin + 0.5 * spacing
    if not coarse:
        return shape, node_min, spacing
    c_shape = tuple(max(3, (n + 1) // 2) if n >= 3 else n for n in shape)
    if c_shape == shape:
        return shape, node_min, spacing
    span = (np.asarray(shape, dtype=float) - 1.0) * spacing
    denom = np.maximum(np.asarray(c_shape, dtype=float) - 1.0, 1.0)
    return c_shape, node_min, span / denom


def trilinear_matrix(points, node_min, node_step, node_shape):
    """Build N x M sparse CSR trilinear interpolation matrix on a regular 3D node grid."""
    pts = np.asarray(points, dtype=float)
    n_pts = len(pts)
    nx, ny, nz = (int(v) for v in node_shape)
    m_nodes = nx * ny * nz
    if n_pts == 0:
        return sparse.csr_matrix((0, m_nodes), dtype=float)

    q = (pts - np.asarray(node_min, dtype=float)) / np.asarray(node_step, dtype=float)
    upper = np.array([max(0, nx - 2), max(0, ny - 2), max(0, nz - 2)], dtype=np.int64)
    i0 = np.clip(np.floor(q).astype(np.int64), 0, upper)
    t = np.clip(q - i0, 0.0, 1.0)

    rows = np.empty(8 * n_pts, dtype=np.int64)
    cols = np.empty(8 * n_pts, dtype=np.int64)
    vals = np.empty(8 * n_pts, dtype=float)
    base_rows = np.arange(n_pts, dtype=np.int64)

    slot = 0
    for dx in (0, 1):
        wx = t[:, 0] if dx else (1.0 - t[:, 0])
        ix = i0[:, 0] + dx
        for dy in (0, 1):
            wxy = wx * (t[:, 1] if dy else (1.0 - t[:, 1]))
            iy = i0[:, 1] + dy
            for dz in (0, 1):
                wxyz = wxy * (t[:, 2] if dz else (1.0 - t[:, 2]))
                iz = i0[:, 2] + dz
                sl = slice(slot * n_pts, (slot + 1) * n_pts)
                rows[sl] = base_rows
                cols[sl] = (ix * ny + iy) * nz + iz
                vals[sl] = wxyz
                slot += 1

    return sparse.csr_matrix((vals, (rows, cols)), shape=(n_pts, m_nodes))


def regulariser_gram(node_shape, node_step, s_z):
    """Return R^T R / M for 3D bending energy (d_xx, d_yy, d_zz, sqrt(2)*d_xy, sqrt(2)*d_yz, sqrt(2)*d_xz)."""
    nx, ny, nz = (int(v) for v in node_shape)
    m_nodes = nx * ny * nz
    hx = float(node_step[0])
    hy = float(node_step[1])
    hz = float(s_z) * float(node_step[2])
    g2 = (hx * hy * hz) ** (2.0 / 3.0)

    idx = np.arange(m_nodes, dtype=np.int64).reshape((nx, ny, nz))
    row_chunks, col_chunks, val_chunks = [], [], []
    row_offset = 0

    def add_stencil(nodes_list, weights):
        nonlocal row_offset
        k = nodes_list[0].size
        if k == 0:
            return
        r = np.arange(row_offset, row_offset + k, dtype=np.int64)
        for node_arr, w in zip(nodes_list, weights):
            row_chunks.append(r)
            col_chunks.append(node_arr.ravel())
            val_chunks.append(np.full(k, float(w), dtype=float))
        row_offset += k

    if nx >= 3:
        s = g2 / (hx * hx)
        add_stencil([idx[2:, :, :], idx[1:-1, :, :], idx[:-2, :, :]], (s, -2.0 * s, s))
    if ny >= 3:
        s = g2 / (hy * hy)
        add_stencil([idx[:, 2:, :], idx[:, 1:-1, :], idx[:, :-2, :]], (s, -2.0 * s, s))
    if nz >= 3:
        s = g2 / (hz * hz)
        add_stencil([idx[:, :, 2:], idx[:, :, 1:-1], idx[:, :, :-2]], (s, -2.0 * s, s))

    rt2 = float(np.sqrt(2.0))
    if nx >= 2 and ny >= 2:
        s = rt2 * g2 / (hx * hy)
        add_stencil([idx[1:, 1:, :], idx[1:, :-1, :], idx[:-1, 1:, :], idx[:-1, :-1, :]], (s, -s, -s, s))
    if ny >= 2 and nz >= 2:
        s = rt2 * g2 / (hy * hz)
        add_stencil([idx[:, 1:, 1:], idx[:, 1:, :-1], idx[:, :-1, 1:], idx[:, :-1, :-1]], (s, -s, -s, s))
    if nx >= 2 and nz >= 2:
        s = rt2 * g2 / (hx * hz)
        add_stencil([idx[1:, :, 1:], idx[1:, :, :-1], idx[:-1, :, 1:], idx[:-1, :, :-1]], (s, -s, -s, s))

    if row_offset == 0:
        return sparse.csr_matrix((m_nodes, m_nodes), dtype=float)

    R = sparse.csr_matrix(
        (np.concatenate(val_chunks), (np.concatenate(row_chunks), np.concatenate(col_chunks))),
        shape=(row_offset, m_nodes),
    )
    return ((R.T @ R) * (1.0 / float(m_nodes))).tocsr()


def _solve_cg(AtWA_over_N, AtWv_over_N, RtR_over_M, lam, *, x0=None, rtol=1e-7):
    m_nodes = AtWA_over_N.shape[0]
    H = AtWA_over_N + float(lam) * RtR_over_M + 1e-14 * sparse.eye(m_nodes, format="csr")
    inv_diag = 1.0 / np.maximum(H.diagonal(), 1e-14)
    precond = LinearOperator((m_nodes, m_nodes), matvec=lambda v: inv_diag * v, dtype=float)
    u, info = cg(H, AtWv_over_N, x0=x0, rtol=rtol, atol=0.0, maxiter=3000, M=precond)
    if not np.isfinite(u).all():
        raise np.linalg.LinAlgError("non-finite FDI CG iterate")
    if info > 0:
        rhs_norm = float(np.linalg.norm(AtWv_over_N))
        res_norm = float(np.linalg.norm(H @ u - AtWv_over_N))
        if rhs_norm > 0.0 and res_norm > 1e-3 * rhs_norm:
            raise np.linalg.LinAlgError(f"FDI CG did not converge (residual={res_norm:.3e})")
    elif info < 0:
        raise np.linalg.LinAlgError(f"FDI CG breakdown (info={info})")
    return u


def fit_fdi_grid(points, values, weights, origin, spacing, shape, *, s_z=1.0, lam=0.1, coarse=False, x0=None, rtol=1e-7):
    """Fit discrete smooth implicit field u on grid nodes; returns (u, node_shape, node_min, node_step)."""
    pts = np.asarray(points, dtype=float)
    vals = np.asarray(values, dtype=float).reshape(-1)
    w = np.asarray(weights, dtype=float).reshape(-1)
    n_pts = len(pts)
    if n_pts == 0:
        raise ValueError("empty FDI observations")
    node_shape, node_min, node_step = _grid_geometry(origin, spacing, shape, coarse=coarse)
    A = trilinear_matrix(pts, node_min, node_step, node_shape)
    W_over_N = sparse.diags(w / float(n_pts), 0, shape=(n_pts, n_pts), format="csr")
    AtW = (A.T @ W_over_N).tocsr()
    AtWA = (AtW @ A).tocsr()
    AtWv = np.asarray(AtW @ vals, dtype=float).reshape(-1)
    RtR = regulariser_gram(node_shape, node_step, s_z)
    u = _solve_cg(AtWA, AtWv, RtR, lam, x0=x0, rtol=rtol)
    return u, node_shape, node_min, node_step


def _targets_for_problem(problem, target_kind, s_z, center, h_hole):
    if target_kind == "sdf":
        _, v_raw, w_all = harness.compute_sdf_targets(problem, s_z, center=center, h_hole=h_hole)
        X_b, _, _ = harness.denoise_binary_observations(problem)
        drill = harness._base()._drill_observation(problem)
        d_pts = np.asarray(drill["points"], dtype=float)
        d_lab = np.asarray(drill["labels"], dtype=float)
        X_c = d_pts[d_lab == 0.5]
        X_phys = np.concatenate([X_b, X_c], axis=0) if len(X_c) else X_b
        cap = max(2.0 * float(h_hole), 1e-12)
        return X_phys, np.clip(v_raw / cap, -1.0, 1.0), w_all
    if target_kind == "sign":
        d = harness.extract_public_data(problem)
        X_b, y_b, w_b = d["X_b"], d["y_b"], d["w_b"]
        X_c, w_c = d["X_c"], d["w_c"]
        v_b = np.where(y_b > 0.5, 1.0, -1.0)
        if len(X_c):
            return (
                np.concatenate([X_b, X_c], axis=0),
                np.concatenate([v_b, np.zeros(len(X_c), dtype=float)], axis=0),
                np.concatenate([w_b, w_c], axis=0),
            )
        return X_b, v_b, w_b
    raise ValueError(f"unknown FDI target kind: {target_kind!r}")


def fdi_smooth(problem, budget=None, seed=0, *, decision_rule=None, diagnostics=None):
    """M5 Discrete smooth implicit interpolation (finite-difference interpolator after Grose et al. 2021)."""
    del seed
    started = time.monotonic()
    data = harness.extract_public_data(problem)
    if not (data["y_b"] > 0.5).any():
        return harness.no_ore_fallback(problem, data, diagnostics=diagnostics, started=started)

    rules = harness._allowed_rules(decision_rule)
    deadline = harness._wall_deadline(budget, started)
    folds = harness.make_hole_folds(data, problem)

    best_idx = None
    best_rule = rules[0] if len(rules) == 1 else "share"
    best_score = -1.0
    cv_evaluated = 0
    deadline_stopped = False

    if folds:
        c_shape, c_min, c_step = _grid_geometry(data["origin"], data["spacing"], data["shape"], coarse=True)
        need_S = "share" in rules
        A_S = trilinear_matrix(data["S"], c_min, c_step, c_shape) if need_S else None
        rtr_cache = {}
        fold_cache = []
        for fold in folds:
            A_val = trilinear_matrix(fold["X_val"], c_min, c_step, c_shape)
            fold_cache.append({
                "fold": fold,
                "A_val": A_val,
                "data_mats": {},
                "targets": {},
                "warm": {},
            })

        for idx, cfg in enumerate(FDI_CONFIGS):
            if time.monotonic() >= deadline:
                deadline_stopped = True
                break
            s_z = cfg["s_z"]
            lam = cfg["lam"]
            t_kind = cfg["target"]
            if s_z not in rtr_cache:
                rtr_cache[s_z] = regulariser_gram(c_shape, c_step, s_z)
            RtR = rtr_cache[s_z]

            counts = {r: [0, 0, 0] for r in rules}
            failed = False
            for fc in fold_cache:
                if time.monotonic() >= deadline:
                    deadline_stopped = True
                    failed = True
                    break
                fold = fc["fold"]
                t_key = (t_kind, s_z) if t_kind == "sdf" else ("sign", 0)
                if t_key not in fc["targets"]:
                    try:
                        fc["targets"][t_key] = _targets_for_problem(
                            fold["problem_tr"], t_kind, s_z, data["center"], data["h_hole"]
                        )
                    except Exception:
                        fc["targets"][t_key] = None
                tgt = fc["targets"][t_key]
                if tgt is None:
                    failed = True
                    break
                X_tr, v_tr, w_tr = tgt
                m_key = t_kind
                if m_key not in fc["data_mats"]:
                    n_tr = len(X_tr)
                    A_tr = trilinear_matrix(X_tr, c_min, c_step, c_shape)
                    W_tr = sparse.diags(w_tr / float(n_tr), 0, shape=(n_tr, n_tr), format="csr")
                    AtW = (A_tr.T @ W_tr).tocsr()
                    fc["data_mats"][m_key] = (AtW, (AtW @ A_tr).tocsr())
                AtW, AtWA = fc["data_mats"][m_key]
                AtWv = np.asarray(AtW @ v_tr, dtype=float).reshape(-1)
                w_key = (s_z, t_kind)
                try:
                    u_c = _solve_cg(AtWA, AtWv, RtR, lam, x0=fc["warm"].get(w_key), rtol=1e-8)
                except Exception:
                    failed = True
                    break
                fc["warm"][w_key] = u_c
                f_val = np.asarray(fc["A_val"] @ u_c, dtype=float).reshape(-1)
                f_S = np.asarray(A_S @ u_c, dtype=float).reshape(-1) if need_S else None
                y_true = fold["y_val"]
                for r in rules:
                    if r == "native":
                        pred = f_val > 0.0
                    else:
                        q = float(np.quantile(f_S, 1.0 - fold["p_train"]))
                        if np.any(f_S > q) or float(f_S.max()) <= float(f_S.min()):
                            pred = f_val > q
                        else:
                            pred = f_val >= q
                    counts[r][0] += int(np.count_nonzero(pred & y_true))
                    counts[r][1] += int(np.count_nonzero(pred & ~y_true))
                    counts[r][2] += int(np.count_nonzero(~pred & y_true))

            if failed:
                if deadline_stopped:
                    break
                continue
            cv_evaluated += 1
            for r in rules:
                sc = harness.pooled_jaccard(*counts[r])
                if sc > best_score + harness.TIE_TOL:
                    best_score = sc
                    best_idx = idx
                    best_rule = r

    if best_idx is None:
        order = list(range(len(FDI_CONFIGS)))
        best_score = None
        best_rule = rules[0] if len(rules) == 1 else "share"
    else:
        order = [best_idx] + [i for i in range(len(FDI_CONFIGS)) if i != best_idx]

    for idx in order:
        cfg = FDI_CONFIGS[idx]
        try:
            X_full, v_full, w_full = _targets_for_problem(
                problem, cfg["target"], cfg["s_z"], data["center"], data["h_hole"]
            )
            u_full, _, _, _ = fit_fdi_grid(
                X_full, v_full, w_full, data["origin"], data["spacing"], data["shape"],
                s_z=cfg["s_z"], lam=cfg["lam"], coarse=False, rtol=1e-7,
            )
            f = u_full.reshape(data["shape"])[data["feasible"]]
            return harness.finalize_field(
                problem,
                data,
                f,
                0.0,
                best_rule,
                selected_config=cfg,
                cv_score=best_score if idx == best_idx else None,
                cv_evaluated=cv_evaluated,
                deadline_stopped=deadline_stopped,
                diagnostics=diagnostics,
                started=started,
            )
        except Exception:
            continue

    return harness.stretched_fallback(
        problem, data, diagnostics=diagnostics, started=started,
        cv_evaluated=cv_evaluated, deadline_stopped=deadline_stopped,
    )


def fdi_smooth_native(problem, budget=None, seed=0, *, diagnostics=None):
    """M5 fdi_smooth with decision rule forced to native."""
    return fdi_smooth(problem, budget, seed, decision_rule="native", diagnostics=diagnostics)


FDI_DEFAULT_CONFIG = FDI_CONFIGS[0]
FDI_SIGN_FALLBACK_CONFIG = {"s_z": 4, "lam": 0.1, "target": "sign"}


def fdi_smooth_default(problem, budget=None, seed=0, *, diagnostics=None):
    """M5 default: full-grid discrete smooth implicit interpolation (s_z=4, lam=0.1, sdf), native u > 0, no CV."""
    del budget, seed
    started = time.monotonic()
    data = harness.extract_public_data(problem)
    if not (data["y_b"] > 0.5).any():
        return harness.no_ore_fallback(problem, data, diagnostics=diagnostics, started=started)

    for cfg in (FDI_DEFAULT_CONFIG, FDI_SIGN_FALLBACK_CONFIG):
        try:
            X_full, v_full, w_full = _targets_for_problem(
                problem, cfg["target"], cfg["s_z"], data["center"], data["h_hole"]
            )
            u_full, _, _, _ = fit_fdi_grid(
                X_full, v_full, w_full, data["origin"], data["spacing"], data["shape"],
                s_z=cfg["s_z"], lam=cfg["lam"], coarse=False, rtol=1e-7,
            )
            f = u_full.reshape(data["shape"])[data["feasible"]]
            return harness.finalize_field(
                problem,
                data,
                f,
                0.0,
                "native",
                selected_config=cfg,
                cv_score=None,
                cv_evaluated=0,
                deadline_stopped=False,
                diagnostics=diagnostics,
                started=started,
            )
        except Exception:
            continue

    return harness.stretched_fallback(
        problem, data, diagnostics=diagnostics, started=started,
        cv_evaluated=0, deadline_stopped=False, default_mode=True,
    )
