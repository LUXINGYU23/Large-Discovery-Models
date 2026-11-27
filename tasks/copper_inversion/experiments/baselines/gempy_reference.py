"""M4 Stage-A GemPy potential-field co-kriging runner (generator-matched).

Must run under $R/env/gempy/bin/python, which provides gempy and numpy ONLY
(no scipy). Reads ONLY public problem.{json,npz} inputs, performs grouped K-fold
CV over the 12 configurations of baselines-lit.md §3 M4, and writes ore.npy,
optional ore_native.npy, and choice.json per case.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
import numpy as np

from tasks.copper_inversion.v2.contract import validate
from tasks.copper_inversion.v2.data import read_problem

TIE_TOL = 1e-12
SHARE_CLIP = (0.02, 0.65)


def _build_gempy_configs():
    default = {"form": "L", "orient": "plane", "range": 1.7}
    configs = [default]
    for form in ("L", "C"):
        for krange in (1.7, 3.4, 0.85):
            for orient in ("plane", "flat"):
                cfg = {"form": form, "orient": orient, "range": krange}
                if cfg != default:
                    configs.append(cfg)
    return tuple(configs)


GEMPY_CONFIGS = _build_gempy_configs()


def _drill_obs(problem):
    for obs in problem["observations"]:
        if obs["name"] == "drill":
            return obs
    return problem["observations"][0]


def _grid_centers_np(problem):
    origin = np.asarray(problem["origin"], dtype=float)
    spacing = np.asarray(problem["spacing"], dtype=float)
    axes = [origin[i] + (np.arange(n, dtype=float) + 0.5) * spacing[i] for i, n in enumerate(problem["shape"])]
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)


def _symmetric_subsample_mask(feasible):
    shape = feasible.shape
    slices = tuple(slice((1 - n) % 3, n, 3) if n >= 3 else slice(None) for n in shape)
    sub = np.zeros(shape, dtype=bool)
    sub[slices] = True
    sub &= feasible
    return sub if sub.any() else feasible


def _extract_public_np(problem):
    validate(problem)
    shape = tuple(int(n) for n in problem["shape"])
    origin = np.asarray(problem["origin"], dtype=float)
    spacing = np.asarray(problem["spacing"], dtype=float)
    extent = np.asarray(shape, dtype=float) * spacing
    feasible = np.asarray(problem["domain_mask"], dtype=bool) & ~np.asarray(problem["forbidden_mask"], dtype=bool)
    xyz = _grid_centers_np(problem)
    sub_mask = _symmetric_subsample_mask(feasible)
    S = xyz[sub_mask]

    drill = _drill_obs(problem)
    d_pts = np.asarray(drill["points"], dtype=float)
    d_lab = np.asarray(drill["labels"], dtype=float)
    d_grp = np.asarray(drill["groups"], dtype=np.int64)
    d_bin = d_lab != 0.5
    hole_ids = np.unique(d_grp) if len(d_grp) else np.zeros(0, dtype=np.int64)
    bin_hole_ids = np.unique(d_grp[d_bin]) if d_bin.any() else np.zeros(0, dtype=np.int64)

    any_ore = bool((d_lab[d_bin] > 0.5).any()) if d_bin.any() else False
    if not any_ore:
        for obs in problem["observations"]:
            oy = np.asarray(obs["labels"], dtype=float)
            ob = oy != 0.5
            if ob.any() and (oy[ob] > 0.5).any():
                any_ore = True
                break

    return {
        "shape": shape,
        "origin": origin,
        "spacing": spacing,
        "extent": extent,
        "xyz": xyz,
        "S": S,
        "d_pts": d_pts,
        "d_lab": d_lab,
        "d_grp": d_grp,
        "d_bin": d_bin,
        "hole_ids": hole_ids,
        "bin_hole_ids": bin_hole_ids,
        "any_ore": any_ore,
    }


def _filtered_problem_np(problem, keep_holes):
    drill = _drill_obs(problem)
    keep = np.isin(np.asarray(drill["groups"]), keep_holes)
    new_obs = []
    for obs in problem["observations"]:
        if obs is drill:
            new_obs.append({
                **obs,
                "points": np.asarray(obs["points"], dtype=float)[keep],
                "labels": np.asarray(obs["labels"], dtype=float)[keep],
                "groups": np.asarray(obs["groups"])[keep],
            })
        else:
            new_obs.append(obs)
    return {**problem, "observations": new_obs}


def _make_folds_np(info, problem):
    d_bin = info["d_bin"]
    if not d_bin.any():
        return []
    yb = info["d_lab"][d_bin] > 0.5
    bin_holes = info["bin_hole_ids"]
    if len(bin_holes) < 2 or not yb.any() or yb.all():
        return []
    hole_ids = info["hole_ids"]
    K = min(len(bin_holes), 6)
    hole_to_fold = {int(h): int(r % K) for r, h in enumerate(bin_holes)}
    for r, h in enumerate(hole_ids):
        if int(h) not in hole_to_fold:
            hole_to_fold[int(h)] = int(r % K)

    d_pts, d_lab, d_grp = info["d_pts"], info["d_lab"], info["d_grp"]
    folds = []
    for k in range(K):
        val_holes = np.array([h for h in hole_ids if hole_to_fold[int(h)] == k], dtype=np.int64)
        tr_holes = np.array([h for h in hole_ids if hole_to_fold[int(h)] != k], dtype=np.int64)
        val_mask = d_bin & np.isin(d_grp, val_holes)
        if not val_mask.any():
            continue
        tr_bin_mask = d_bin & np.isin(d_grp, tr_holes)
        tr_bin_holes = np.unique(d_grp[tr_bin_mask])
        if len(tr_bin_holes):
            p_raw = float(np.mean([
                (d_lab[tr_bin_mask & (d_grp == h)] > 0.5).mean() for h in tr_bin_holes
            ]))
        else:
            p_raw = SHARE_CLIP[0]
        folds.append({
            "fold": k,
            "problem_tr": _filtered_problem_np(problem, tr_holes),
            "X_val": d_pts[val_mask],
            "y_val": d_lab[val_mask] > 0.5,
            "p_train": float(np.clip(p_raw, SHARE_CLIP[0], SHARE_CLIP[1])),
        })
    return folds


def classify_contacts(problem, form):
    """Classify drill contacts into top (barren above, ore below) and bottom (ore above, barren below).

    In form 'L', if a hole has multiple ore runs, keep only contacts bounding its longest run.
    Returns (top_xyz, top_gid, bot_xyz, bot_gid).
    """
    drill = _drill_obs(problem)
    pts = np.asarray(drill["points"], dtype=float)
    lab = np.asarray(drill["labels"], dtype=float)
    grp = np.asarray(drill["groups"], dtype=np.int64)

    top_pts, top_gid, bot_pts, bot_gid = [], [], [], []
    for gid in np.unique(grp):
        h_sel = grp == gid
        h_pts = pts[h_sel]
        h_lab = lab[h_sel]
        b_sel = h_lab != 0.5
        c_sel = ~b_sel
        if int(b_sel.sum()) < 2 or not c_sel.any():
            continue
        b_pts = h_pts[b_sel]
        b_y = h_lab[b_sel] > 0.5
        order = np.argsort(b_pts[:, 2], kind="mergesort")
        zb = b_pts[order, 2]
        yb = b_y[order]

        c_pts = h_pts[c_sel]
        c_order = np.argsort(c_pts[:, 2], kind="mergesort")
        c_pts = c_pts[c_order]

        if form == "L":
            runs = []
            start = None
            for i, is_ore in enumerate(yb):
                if is_ore and start is None:
                    start = i
                elif not is_ore and start is not None:
                    runs.append((start, i - 1))
                    start = None
            if start is not None:
                runs.append((start, len(yb) - 1))
            if not runs:
                continue
            if len(runs) > 1:
                a, b = max(runs, key=lambda ab: (float(zb[ab[1]] - zb[ab[0]]), ab[1] - ab[0] + 1))
                # Keep at most the bottom contact below zb[a] and top contact above zb[b].
                if a > 0 and not yb[a - 1]:
                    z_lo, z_hi = zb[a - 1], zb[a]
                    in_gap = [cp for cp in c_pts if z_lo < cp[2] < z_hi]
                    if in_gap:
                        mid = 0.5 * (z_lo + z_hi)
                        best_bot = min(in_gap, key=lambda cp: abs(cp[2] - mid))
                        bot_pts.append(best_bot)
                        bot_gid.append(int(gid))
                if b < len(zb) - 1 and not yb[b + 1]:
                    z_lo, z_hi = zb[b], zb[b + 1]
                    in_gap = [cp for cp in c_pts if z_lo < cp[2] < z_hi]
                    if in_gap:
                        mid = 0.5 * (z_lo + z_hi)
                        best_top = min(in_gap, key=lambda cp: abs(cp[2] - mid))
                        top_pts.append(best_top)
                        top_gid.append(int(gid))
                continue

        for cp in c_pts:
            zc = cp[2]
            below_idx = np.where(zb < zc)[0]
            above_idx = np.where(zb > zc)[0]
            if len(below_idx) == 0 or len(above_idx) == 0:
                continue
            y_below = bool(yb[below_idx[-1]])
            y_above = bool(yb[above_idx[0]])
            if y_above == y_below:
                continue
            if (not y_above) and y_below:
                top_pts.append(cp)
                top_gid.append(int(gid))
            elif y_above and (not y_below):
                bot_pts.append(cp)
                bot_gid.append(int(gid))

    t_arr = np.asarray(top_pts, dtype=float).reshape((-1, 3)) if top_pts else np.zeros((0, 3), dtype=float)
    tg_arr = np.asarray(top_gid, dtype=np.int64) if top_gid else np.zeros(0, dtype=np.int64)
    b_arr = np.asarray(bot_pts, dtype=float).reshape((-1, 3)) if bot_pts else np.zeros((0, 3), dtype=float)
    bg_arr = np.asarray(bot_gid, dtype=np.int64) if bot_gid else np.zeros(0, dtype=np.int64)
    return t_arr, tg_arr, b_arr, bg_arr


def estimate_poles(contacts_xyz, contacts_gid, orient):
    """Estimate upward unit poles (0, 0, 1) or local OLS plane poles (-b, -c, 1)/norm."""
    n = len(contacts_xyz)
    if n == 0:
        return np.zeros((0, 3), dtype=float)
    poles = np.tile(np.array([0.0, 0.0, 1.0], dtype=float), (n, 1))
    if orient == "flat":
        return poles
    if orient != "plane":
        raise ValueError(f"unknown orientation mode: {orient!r}")

    for i in range(n):
        p0 = contacts_xyz[i]
        other = contacts_gid != contacts_gid[i]
        other_pts = contacts_xyz[other]
        if len(other_pts) < 2:
            continue
        dh = np.hypot(other_pts[:, 0] - p0[0], other_pts[:, 1] - p0[1])
        take = np.argsort(dh, kind="mergesort")[:5]
        pts = np.vstack([p0[None, :], other_pts[take]])
        if len(pts) < 3:
            continue
        dx = pts[:, 0] - p0[0]
        dy = pts[:, 1] - p0[1]
        dz = pts[:, 2] - p0[2]
        design_xy = np.column_stack([dx, dy])
        svals = np.linalg.svd(design_xy, compute_uv=False)
        if len(svals) < 2 or svals[1] <= 1e-6 * max(1.0, float(svals[0])):
            continue
        A = np.column_stack([np.ones(len(pts)), dx, dy])
        coef, _, _, _ = np.linalg.lstsq(A, dz, rcond=None)
        b, c = float(coef[1]), float(coef[2])
        if not (np.isfinite(b) and np.isfinite(c)):
            continue
        vec = np.array([-b, -c, 1.0], dtype=float)
        poles[i] = vec / np.linalg.norm(vec)
    return poles


def evaluate_gempy(problem, cfg, query_points):
    """Build GemPy potential-field model for cfg and evaluate discrete lithology at query_points."""
    priv_tmp = f"/tmp/gempy_tmp_{os.geteuid()}_{os.getpid()}"
    os.makedirs(priv_tmp, exist_ok=True)
    os.environ["TMPDIR"] = priv_tmp
    tempfile.tempdir = priv_tmp
    with contextlib.redirect_stdout(io.StringIO()):
        import gempy as gp
        from gempy_engine.modules.weights_cache.weights_cache_interface import WeightCache
        WeightCache.initialize_cache_dir(os.path.join(priv_tmp, "gempy_cache"))
        form = cfg["form"]
        orient = cfg["orient"]
        krange = float(cfg["range"])
        shape = tuple(int(n) for n in problem["shape"])
        origin = np.asarray(problem["origin"], dtype=float)
        spacing = np.asarray(problem["spacing"], dtype=float)
        extent = np.asarray(shape, dtype=float) * spacing

        top_pts, top_gid, bot_pts, bot_gid = classify_contacts(problem, form)
        frame = gp.data.StructuralFrame.initialize_default_structure()
        model = gp.create_geomodel(
            project_name=f"m4-{form.lower()}",
            extent=np.column_stack([origin, origin + extent]).ravel().tolist(),
            resolution=list(shape),
            structural_frame=frame,
        )

        if form == "L":
            if len(top_pts) < 2 or len(bot_pts) < 2:
                raise ValueError("form L requires >=2 top and >=2 bottom contacts")
            poles_top = estimate_poles(top_pts, top_gid, orient)
            gp.add_surface_points(
                geo_model=model, x=top_pts[:, 0], y=top_pts[:, 1], z=top_pts[:, 2], elements_names="surface1"
            )
            gp.add_orientations(
                geo_model=model, x=top_pts[:, 0], y=top_pts[:, 1], z=top_pts[:, 2],
                elements_names=["surface1"] * len(top_pts), pole_vector=poles_top,
            )
            model.structural_frame.structural_groups[0].append_element(gp.data.StructuralElement(
                name="surface2",
                color="#884422",
                surface_points=gp.data.SurfacePointsTable.from_arrays(
                    x=bot_pts[:, 0], y=bot_pts[:, 1], z=bot_pts[:, 2], names="surface2"
                ),
                orientations=gp.data.OrientationsTable.initialize_empty(),
            ))
            allowed_ids = {1, 2, 3}
        elif form == "C":
            all_pts = np.vstack([top_pts, bot_pts]) if (len(top_pts) or len(bot_pts)) else np.zeros((0, 3), dtype=float)
            if len(all_pts) < 4:
                raise ValueError("form C requires >=4 classified contacts")
            poles_top = estimate_poles(top_pts, top_gid, orient)
            poles_bot = -estimate_poles(bot_pts, bot_gid, orient)
            all_poles = np.vstack([poles_top, poles_bot])
            gp.add_surface_points(
                geo_model=model, x=all_pts[:, 0], y=all_pts[:, 1], z=all_pts[:, 2], elements_names="surface1"
            )
            gp.add_orientations(
                geo_model=model, x=all_pts[:, 0], y=all_pts[:, 1], z=all_pts[:, 2],
                elements_names=["surface1"] * len(all_pts), pole_vector=all_poles,
            )
            allowed_ids = {1, 2}
        else:
            raise ValueError(f"unknown GemPy form: {form!r}")

        model.update_transform(gp.data.GlobalAnisotropy.NONE)
        model.interpolation_options.mesh_extraction = False
        model.interpolation_options.cache_mode = model.interpolation_options.CacheMode.IN_MEMORY_CACHE
        model.interpolation_options.kernel_options.range = krange

        q = np.asarray(query_points, dtype=float)
        gp.set_custom_grid(model.grid, q, reset=True)
        try:
            gp.compute_model(model)
        finally:
            WeightCache.memory_cache.clear()
    raw = np.asarray(model.solutions.raw_arrays.custom, dtype=float)
    if raw.shape != (len(q),) or not np.isfinite(raw).all():
        raise ValueError("GemPy returned malformed or non-finite custom-grid values")
    ids = np.rint(raw).astype(np.int64)
    uniq = set(int(v) for v in np.unique(ids))
    if not uniq.issubset(allowed_ids):
        raise ValueError(f"unexpected GemPy lithology ids {sorted(uniq)} for form {form}")
    return (ids == 2), ids


def _approx_signed_distance(query_pts, query_ore, S_pts, S_ore):
    """Pure-NumPy signed distance proxy from custom-grid predictions for CV share-rule quantiles."""
    if S_ore.any() and (~S_ore).any():
        pos = S_pts[S_ore]
        neg = S_pts[~S_ore]
        out = np.empty(len(query_pts), dtype=float)
        q_pos = query_ore
        if q_pos.any():
            d2 = np.sum((query_pts[q_pos, None, :] - neg[None, :, :]) ** 2, axis=-1)
            out[q_pos] = np.sqrt(np.min(d2, axis=1))
        if (~q_pos).any():
            d2 = np.sum((query_pts[~q_pos, None, :] - pos[None, :, :]) ** 2, axis=-1)
            out[~q_pos] = -np.sqrt(np.min(d2, axis=1))
        return out
    return np.where(query_ore, 1.0, -1.0)


def _pooled_jaccard(tp, fp, fn):
    return float(tp) / float(tp + fp + fn) if tp > 0 else 0.0


DEFAULT_GEMPY_CONFIG = GEMPY_CONFIGS[0]
DEFAULT_GEMPY_FALLBACK_CONFIG = {"form": "C", "orient": "plane", "range": 1.7}


def run_case(public_dir, out_dir, *, wall_budget_seconds=120.0, mode="cv"):
    """Run M4 Stage-A on one public problem directory and save ore.npy + choice.json."""
    if mode not in ("cv", "default"):
        raise ValueError(f"unknown GemPy Stage-A mode: {mode!r}")
    t0 = time.monotonic()
    out_dir = Path(out_dir)
    if (out_dir / "choice.json").is_file() and (
        (out_dir / "ore.npy").is_file() or json.loads((out_dir / "choice.json").read_text()).get("fallback")
        or json.loads((out_dir / "choice.json").read_text()).get("no_ore")
    ):
        return json.loads((out_dir / "choice.json").read_text())

    problem = read_problem(public_dir)
    info = _extract_public_np(problem)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not info["any_ore"]:
        choice = {
            "method": "potential_field",
            "mode": mode,
            "generator_matched": True,
            "case_id": problem["case_id"],
            "no_ore": True,
            "fallback": False,
            "wall_seconds": round(float(time.monotonic() - t0), 4),
        }
        (out_dir / "choice.json").write_text(json.dumps(choice, indent=2, sort_keys=True))
        return choice

    if mode == "default":
        full_pts = info["xyz"].reshape(-1, 3)
        used_cfg = None
        ore_default = None
        for cfg in (DEFAULT_GEMPY_CONFIG, DEFAULT_GEMPY_FALLBACK_CONFIG):
            try:
                ore_flat, _ = evaluate_gempy(problem, cfg, full_pts)
                ore_default = ore_flat.reshape(info["shape"])
                used_cfg = cfg
                break
            except Exception:
                continue
        if ore_default is None:
            choice = {
                "method": "potential_field",
                "mode": "default",
                "generator_matched": True,
                "case_id": problem["case_id"],
                "no_ore": False,
                "fallback": True,
                "cv_evaluated": 0,
                "deadline_stopped": False,
                "wall_seconds": round(float(time.monotonic() - t0), 4),
            }
            (out_dir / "choice.json").write_text(json.dumps(choice, indent=2, sort_keys=True))
            return choice
        np.save(out_dir / "ore.npy", ore_default, allow_pickle=False)
        block = {
            "selected_rule": "native",
            "selected_config": used_cfg,
            "cv_score": None,
            "cv_evaluated": 0,
            "same_as_cv": True,
        }
        choice = {
            "method": "potential_field",
            "mode": "default",
            "generator_matched": True,
            "case_id": problem["case_id"],
            "no_ore": False,
            "fallback": False,
            "cv_evaluated": 0,
            "deadline_stopped": False,
            "default": block,
            "native": block,
            "cv": block,
            "wall_seconds": round(float(time.monotonic() - t0), 4),
        }
        (out_dir / "choice.json").write_text(json.dumps(choice, indent=2, sort_keys=True))
        return choice

    folds = _make_folds_np(info, problem)
    deadline = t0 + 0.5 * float(wall_budget_seconds) if wall_budget_seconds > 0 else None
    S = info["S"]

    best_cv_idx, best_cv_rule, best_cv_score = None, "share", -1.0
    best_nat_idx, best_nat_score = None, -1.0
    cv_evaluated = 0
    deadline_stopped = False

    if folds:
        for idx, cfg in enumerate(GEMPY_CONFIGS):
            if deadline is not None and time.monotonic() >= deadline:
                deadline_stopped = True
                break
            counts = {"native": [0, 0, 0], "share": [0, 0, 0]}
            failed = False
            for fold in folds:
                if deadline is not None and time.monotonic() >= deadline:
                    deadline_stopped = True
                    failed = True
                    break
                n_val = len(fold["X_val"])
                q_pts = np.vstack([fold["X_val"], S])
                try:
                    q_ore, _ = evaluate_gempy(fold["problem_tr"], cfg, q_pts)
                except Exception:
                    failed = True
                    break
                ore_val = q_ore[:n_val]
                ore_S = q_ore[n_val:]
                f_all = _approx_signed_distance(q_pts, q_ore, S, ore_S)
                f_val = f_all[:n_val]
                f_S = f_all[n_val:]
                y_true = fold["y_val"]
                for r in ("native", "share"):
                    if r == "native":
                        pred = ore_val
                    else:
                        q_cut = float(np.quantile(f_S, 1.0 - fold["p_train"]))
                        if np.any(f_S > q_cut) or float(f_S.max()) <= float(f_S.min()):
                            pred = f_val > q_cut
                        else:
                            pred = f_val >= q_cut
                    counts[r][0] += int(np.count_nonzero(pred & y_true))
                    counts[r][1] += int(np.count_nonzero(pred & ~y_true))
                    counts[r][2] += int(np.count_nonzero(~pred & y_true))
            if failed:
                if deadline_stopped:
                    break
                continue
            cv_evaluated += 1
            sc_nat = _pooled_jaccard(*counts["native"])
            sc_shr = _pooled_jaccard(*counts["share"])
            if sc_nat > best_nat_score + TIE_TOL:
                best_nat_score = sc_nat
                best_nat_idx = idx
            for r, sc in (("native", sc_nat), ("share", sc_shr)):
                if sc > best_cv_score + TIE_TOL:
                    best_cv_score = sc
                    best_cv_idx = idx
                    best_cv_rule = r

    full_pts = info["xyz"].reshape(-1, 3)

    def _fit_ordered(preferred_idx):
        order = (
            list(range(len(GEMPY_CONFIGS)))
            if preferred_idx is None
            else [preferred_idx] + [i for i in range(len(GEMPY_CONFIGS)) if i != preferred_idx]
        )
        for i in order:
            try:
                ore_flat, _ = evaluate_gempy(problem, GEMPY_CONFIGS[i], full_pts)
                return i, ore_flat.reshape(info["shape"])
            except Exception:
                continue
        return None, None

    fit_cv_idx, ore_cv = _fit_ordered(best_cv_idx)
    if ore_cv is None:
        choice = {
            "method": "potential_field",
            "generator_matched": True,
            "case_id": problem["case_id"],
            "no_ore": False,
            "fallback": True,
            "cv_evaluated": cv_evaluated,
            "deadline_stopped": deadline_stopped,
            "wall_seconds": round(float(time.monotonic() - t0), 4),
        }
        (out_dir / "choice.json").write_text(json.dumps(choice, indent=2, sort_keys=True))
        return choice

    np.save(out_dir / "ore.npy", ore_cv, allow_pickle=False)

    target_nat_idx = best_nat_idx if best_nat_idx is not None else fit_cv_idx
    same_as_cv = target_nat_idx == fit_cv_idx
    fit_nat_idx = fit_cv_idx
    if not same_as_cv:
        cand_idx, ore_nat = _fit_ordered(target_nat_idx)
        if ore_nat is not None and cand_idx != fit_cv_idx:
            fit_nat_idx = cand_idx
            np.save(out_dir / "ore_native.npy", ore_nat, allow_pickle=False)
        else:
            same_as_cv = True

    choice = {
        "method": "potential_field",
        "generator_matched": True,
        "case_id": problem["case_id"],
        "no_ore": False,
        "fallback": False,
        "cv_evaluated": cv_evaluated,
        "deadline_stopped": deadline_stopped,
        "cv": {
            "selected_rule": best_cv_rule if best_cv_idx is not None else "share",
            "selected_config": GEMPY_CONFIGS[fit_cv_idx],
            "cv_score": best_cv_score if fit_cv_idx == best_cv_idx and best_cv_idx is not None else None,
            "cv_evaluated": cv_evaluated,
        },
        "native": {
            "selected_rule": "native",
            "selected_config": GEMPY_CONFIGS[fit_nat_idx],
            "cv_score": best_nat_score if fit_nat_idx == best_nat_idx and best_nat_idx is not None else None,
            "cv_evaluated": cv_evaluated,
            "same_as_cv": same_as_cv,
        },
        "wall_seconds": round(float(time.monotonic() - t0), 4),
    }
    (out_dir / "choice.json").write_text(json.dumps(choice, indent=2, sort_keys=True))
    return choice


def _worker(task):
    pub_dir, out_dir, budget_s, mode = task
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    try:
        return run_case(pub_dir, out_dir, wall_budget_seconds=budget_s, mode=mode)
    finally:
        shutil.rmtree(f"/tmp/gempy_tmp_{os.geteuid()}_{os.getpid()}", ignore_errors=True)


def run_split(release_dir, split, output_root, *, workers=4, wall_budget_seconds=240.0, mode="cv"):
    release_dir = Path(release_dir).resolve()
    manifest_path = release_dir / split / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    tasks = [
        (str(manifest_path.parent / c["public"]), str(Path(output_root) / split / c["case_id"]),
         wall_budget_seconds, mode)
        for c in manifest["cases"]
    ]
    if workers <= 1:
        results = [_worker(t) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=min(int(workers), 16)) as pool:
            results = list(pool.map(_worker, tasks))
    return {
        "release": str(release_dir),
        "split": split,
        "mode": mode,
        "cases": len(results),
        "fallbacks": sum(1 for r in results if r.get("fallback")),
        "median_seconds": float(np.median([r["wall_seconds"] for r in results])) if results else 0.0,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run M4 GemPy Stage-A on public split inputs.")
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument(
        "--split", required=True,
        choices=("calibration", "train", "train_s", "train_m", "validation", "test_id", "test_ood"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--wall-budget-seconds", type=float, default=240.0)
    parser.add_argument("--mode", default="cv", choices=("cv", "default"))
    args = parser.parse_args(argv)
    summary = run_split(
        args.release, args.split, args.output,
        workers=args.workers, wall_budget_seconds=args.wall_budget_seconds, mode=args.mode,
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
