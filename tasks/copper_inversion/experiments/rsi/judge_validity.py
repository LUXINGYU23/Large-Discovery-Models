"""Pre-registered judge validity battery and criteria C1-C7 (bench-design.md §3.3).

Strictly restricted to `calibration` or `train` splits; never opens `validation`
or sealed test splits. Emits suite- and family-level aggregates only.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import glob
import heapq
import importlib.util
import json
import os
for _blas_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_blas_var, "1")
from pathlib import Path
import numpy as np
from scipy import ndimage
from scipy.interpolate import RBFInterpolator
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

from tasks.copper_inversion.v2 import baselines
from tasks.copper_inversion.v2.baselines import constrained_initial, initialize
from tasks.copper_inversion.v2.contract import LEGACY_OBJECTIVE, OBJECTIVE, OBJECTIVE_VERSIONS
from tasks.copper_inversion.v2.data import read_problem
from tasks.copper_inversion.v2.geometry import canonicalize, diagnostics
from tasks.copper_inversion.v2.objective import K_TOL, score
from tasks.copper_inversion.v2.suite import confirmation_utility, file_hash, load_manifest

ALLOWED_SPLITS = ("calibration", "train")
FORBIDDEN_PATH_PARTS = frozenset({
    "validation", "validation_q25", "test_id", "test_ood", "sealed", "calibration_ood",
})
DEFAULT_V6_RUNS = Path("/Aidata/shared/xingyulu/rsi4geo/runs/rsi-r1-85dad8f92cab7b37/M1")
DEFAULT_V6_SERVICES = Path("/Aidata/shared/xingyulu/rsi4geo/private/rsi-r1-85dad8f92cab7b37/services/M1")
DEFAULT_V6_RELEASE = Path("/Aidata/shared/xingyulu/rsi4geo/private/rsi-r1/release")
V6_SPONGE_SPECS = (
    ("v6_sponge_s102", "ldm4geo-s102", "4d4bc197c1"),
    ("v6_sponge_s104", "ldm4geo-s104", "a142cfc3b2"),
)
MAX_WORKERS = 16


def _check_allowed_path(path, release, split):
    resolved = Path(path).resolve()
    release = Path(release).resolve()
    if not resolved.is_relative_to(release):
        raise ValueError(f"path escapes release root: {resolved}")
    parts = set(resolved.relative_to(release).parts)
    if parts & FORBIDDEN_PATH_PARTS:
        raise ValueError(f"judge_validity refuses forbidden split path: {resolved}")
    if split not in parts:
        raise ValueError(f"judge_validity requires every opened path under {split}/: {resolved}")
    return resolved


def resolve_split_manifest(release, split):
    if split not in ALLOWED_SPLITS:
        raise ValueError(f"judge_validity allows splits {ALLOWED_SPLITS} only, got {split!r}")
    release = Path(release).resolve()
    candidates = [release / split / "manifest.json", release / "dev" / split / "manifest.json"]
    manifest_path = next((p for p in candidates if p.is_file()), None)
    if manifest_path is None:
        raise FileNotFoundError(f"no {split} manifest found under {release}")
    _check_allowed_path(manifest_path, release, split)
    raw = json.loads(manifest_path.read_text())
    if raw.get("split") != split:
        raise ValueError(f"manifest split {raw.get('split')!r} does not match requested {split!r}")
    for row in raw.get("cases", ()):
        _check_allowed_path(manifest_path.parent / row["public"], release, split)
        _check_allowed_path(manifest_path.parent / row["truth"], release, split)
    manifest = load_manifest(manifest_path)
    return manifest_path, manifest


def _load_calibration_manifest(manifest_path, split):
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "copper-suite-v2" or manifest.get("split") != split or not manifest.get("cases"):
        raise ValueError("invalid calibration suite manifest")
    for row in manifest["cases"]:
        for filename, expected in row["sha256"].items():
            folder = row["truth"] if filename == "truth.npy" or filename.startswith("confirmation/") else row["public"]
            if file_hash(manifest_path.parent / folder / filename) != expected:
                raise ValueError("dataset digest mismatch")
    return manifest


def _resolve_families(release, split, cases):
    release = Path(release).resolve()
    by_case, by_parent = {}, {}
    for row in cases:
        fam = row.get("family") or (row.get("strata") or {}).get("family")
        if fam:
            by_case[row["case_id"]] = fam
            by_parent[row["parent_id"]] = fam
    strata_csv = release / "strata.csv"
    if strata_csv.is_file():
        with strata_csv.open(newline="") as handle:
            for item in csv.DictReader(handle):
                fam = item.get("family")
                if fam:
                    if item.get("case_id"):
                        by_case[item["case_id"]] = fam
                    if item.get("parent_id"):
                        by_parent[item["parent_id"]] = fam
    layout_path = release / "evidence" / "layout.json"
    positional = []
    if layout_path.is_file():
        assignment = json.loads(layout_path.read_text()).get("assignment", [])
        positional = [a for a in assignment if a.get("split") == split]
    for dataset_cand in (release.parent / "dataset", release.parent.parent / "datasets" / release.name):
        plan_file = dataset_cand / "id" / "plan.json"
        if plan_file.is_file():
            for entry in json.loads(plan_file.read_text()).get("parents", []):
                if entry.get("split") == split:
                    pman = dataset_cand / "id" / "packs" / f"{entry['split']}-{entry['index']:02}" / "private-manifest.json"
                    if pman.is_file():
                        by_parent[json.loads(pman.read_text())["parent_digest"]] = entry["family"]
    out = {}
    for idx, row in enumerate(cases):
        cid, pid = row["case_id"], row["parent_id"]
        if cid in by_case:
            out[cid] = by_case[cid]
        elif pid in by_parent:
            out[cid] = by_parent[pid]
        elif len(positional) == len(cases) and positional[idx].get("selected_modality") == row["modality"]:
            out[cid] = positional[idx]["family"]
        else:
            out[cid] = "unknown"
    return out


def _adapt_confirmation_for_objective(conf, objective_version):
    if conf["objective_version"] == objective_version:
        return conf
    adapted = dict(conf, objective_version=objective_version)
    if conf["objective_version"] == LEGACY_OBJECTIVE and objective_version == OBJECTIVE:
        obs_list = []
        for obs in conf["observations"]:
            if obs["name"] == "outcrop" and float(obs["weight"]) == 1.:
                obs_list.append({**obs, "weight": .25})
            else:
                obs_list.append(obs)
        adapted["observations"] = obs_list
    return adapted


def _to_volume(score_field, feasible, target_cells):
    n_feas = int(feasible.sum())
    k = int(np.clip(round(target_cells), 1, n_feas))
    tie = np.arange(score_field.size, dtype=float).reshape(score_field.shape) * 1e-12
    vals = np.where(feasible, score_field + tie, -np.inf)
    thr = np.partition(vals.ravel(), -k)[-k]
    out = (vals >= thr) & feasible
    if int(out.sum()) > k:
        keep_idx = np.argsort(vals[out])[::-1][:k]
        coords = np.argwhere(out)[keep_idx]
        out = np.zeros_like(feasible)
        out[tuple(coords.T)] = True
    return out


def make_legal(problem, raw_mask, rank=None):
    feasible = problem["domain_mask"] & ~problem["forbidden_mask"]
    mask = np.asarray(raw_mask, dtype=bool) & feasible
    struct6 = ndimage.generate_binary_structure(3, 1)
    max_comp = problem["priors"]["max_components"]
    cell = float(np.prod(problem["spacing"]))
    bounds = problem["priors"]["hard_volume_bounds"]
    lo = 0 if bounds is None else int(np.ceil(bounds[0] / cell - 1e-9))
    hi = int(feasible.sum()) if bounds is None else int(np.floor(bounds[1] / cell + 1e-9))

    if max_comp is not None and mask.any():
        lab, count = ndimage.label(mask, structure=struct6)
        if count > max_comp:
            sizes = np.bincount(lab.ravel())[1:]
            order = np.argsort(sizes)[::-1][:max_comp] + 1
            mask = np.isin(lab, order)

    n = int(mask.sum())
    if n > hi and hi > 0:
        lab, count = ndimage.label(mask, structure=struct6)
        sizes = np.bincount(lab.ravel())[1:]
        quotas = np.maximum(1, np.floor(sizes / sizes.sum() * hi).astype(int))
        while int(quotas.sum()) > hi:
            quotas[int(np.argmax(quotas))] -= 1
        while int(quotas.sum()) < hi:
            room = sizes - quotas
            if not np.any(room > 0):
                break
            quotas[int(np.argmax(room))] += 1
        trimmed = np.zeros_like(mask)
        base_rank = (rank if rank is not None
                     else ndimage.distance_transform_edt(mask, sampling=problem["spacing"]))
        shape = mask.shape
        for comp_idx in range(1, count + 1):
            quota = int(quotas[comp_idx - 1])
            comp_mask = lab == comp_idx
            pts = np.argwhere(comp_mask)
            start = tuple(pts[int(np.argmax(base_rank[comp_mask]))])
            visited = {start}
            heap = [(-float(base_rank[start]), start)]
            picked = 0
            while heap and picked < quota:
                _, pos = heapq.heappop(heap)
                trimmed[pos] = True
                picked += 1
                px, py, pz = pos
                for dx, dy, dz in ((-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)):
                    nx, ny, nz = px + dx, py + dy, pz + dz
                    if 0 <= nx < shape[0] and 0 <= ny < shape[1] and 0 <= nz < shape[2]:
                        npos = (nx, ny, nz)
                        if npos not in visited and comp_mask[npos]:
                            visited.add(npos)
                            heapq.heappush(heap, (-float(base_rank[npos]), npos))
        mask = trimmed
        n = int(mask.sum())

    if n < lo:
        if n > 0:
            flab, _ = ndimage.label(feasible, structure=struct6)
            touched = np.unique(flab[mask])
            touched = touched[touched > 0]
            cap = int(np.isin(flab, touched).sum())
            if cap >= lo:
                grow_rank = (rank if rank is not None
                             else -ndimage.distance_transform_edt(~mask, sampling=problem["spacing"]))
                grown = mask.copy()
                visited = mask.copy()
                shape = mask.shape
                heap = []
                for pos in map(tuple, np.argwhere(mask)):
                    px, py, pz = pos
                    for dx, dy, dz in ((-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)):
                        nx, ny, nz = px + dx, py + dy, pz + dz
                        if 0 <= nx < shape[0] and 0 <= ny < shape[1] and 0 <= nz < shape[2]:
                            npos = (nx, ny, nz)
                            if feasible[npos] and not visited[npos]:
                                visited[npos] = True
                                heapq.heappush(heap, (-float(grow_rank[npos]), npos))
                filled = n
                while heap and filled < lo:
                    _, pos = heapq.heappop(heap)
                    grown[pos] = True
                    filled += 1
                    px, py, pz = pos
                    for dx, dy, dz in ((-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)):
                        nx, ny, nz = px + dx, py + dy, pz + dz
                        if 0 <= nx < shape[0] and 0 <= ny < shape[1] and 0 <= nz < shape[2]:
                            npos = (nx, ny, nz)
                            if feasible[npos] and not visited[npos]:
                                visited[npos] = True
                                heapq.heappush(heap, (-float(grow_rank[npos]), npos))
                mask = grown
        if int(mask.sum()) < lo:
            fallback_rank = (rank if rank is not None
                             else -np.linalg.norm(np.indices(feasible.shape, dtype=float)
                                                  - np.asarray(feasible.shape)[:, None, None, None] / 2., axis=0))
            return constrained_initial(problem, mask, fallback_rank) > 0

    try:
        canonicalize(problem, np.where(mask, 1., -1.))
        return mask
    except ValueError:
        fallback_rank = rank if rank is not None else np.where(mask, 1., 0.)
        return constrained_initial(problem, mask, fallback_rank) > 0


# Reference solvers (uses v2.baselines when Track A3 is merged, else built-in deterministic equivalents).

def _grid_xyz(problem):
    return np.stack(np.meshgrid(*[
        problem["origin"][i] + (np.arange(n) + .5) * problem["spacing"][i]
        for i, n in enumerate(problem["shape"])], indexing="ij"), axis=-1)


def _drill_ore_share(problem):
    drill = next((o for o in problem["observations"] if o["name"] == "drill"), problem["observations"][0])
    y, g = np.asarray(drill["labels"]), np.asarray(drill["groups"])
    bi = y != .5
    if not bi.any():
        return .08
    yb, gb = (y[bi] > .5).astype(float), g[bi]
    return float(np.mean([yb[gb == u].mean() for u in np.unique(gb)]))


def _fallback_stretched_nearest_class(problem):
    feasible = problem["domain_mask"] & ~problem["forbidden_mask"]
    xyz = _grid_xyz(problem)
    drill = next((o for o in problem["observations"] if o["name"] == "drill"), problem["observations"][0])
    pts_all = np.concatenate([o["points"][o["labels"] != .5] for o in problem["observations"]])
    lab_all = np.concatenate([o["labels"][o["labels"] != .5] > .5 for o in problem["observations"]])
    n_feas = int(feasible.sum())
    if not lab_all.any() or (~lab_all).sum() == 0:
        f = initialize(problem)
        return f > 0

    yd = np.asarray(drill["labels"])
    bi = yd != .5
    pd, ld, gd = np.asarray(drill["points"])[bi], yd[bi] > .5, np.asarray(drill["groups"])[bi]
    holes = np.unique(gd)
    best_zs, best_key = 4., (-1., -np.inf)
    if len(holes) >= 2 and ld.any() and (~ld).any():
        for zs in (.5, 1., 1.5, 2., 3., 4., 6., 8.):
            sc = np.array([1., 1., zs])
            tp = fp = fn = 0
            margins = []
            for h_id in holes:
                hold = gd == h_id
                tr_p, tr_l = pd[~hold] * sc, ld[~hold]
                if not tr_l.any() or (~tr_l).sum() == 0:
                    continue
                te_p, te_l = pd[hold] * sc, ld[hold]
                dp = cKDTree(tr_p[tr_l]).query(te_p)[0]
                dn = cKDTree(tr_p[~tr_l]).query(te_p)[0]
                pred = dp < dn
                tp += int(np.sum(pred & te_l))
                fp += int(np.sum(pred & ~te_l))
                fn += int(np.sum(~pred & te_l))
                margins.append(float(np.mean((2. * te_l - 1.) * (dn - dp) / (dn + dp + 1e-9))))
            if tp > 0 and margins:
                iou_cv = tp / max(1, tp + fp + fn)
                key = (iou_cv, float(np.mean(margins)))
                if key > best_key:
                    best_key, best_zs = key, zs

    sc = np.array([1., 1., best_zs])
    g_pts = xyz[feasible] * sc
    x_pts = pts_all * sc
    dp = cKDTree(x_pts[lab_all]).query(g_pts)[0]
    dn = cKDTree(x_pts[~lab_all]).query(g_pts)[0]
    contrast = (dn - dp) / (dn + dp + 1e-9)
    rank = np.full(feasible.shape, -1., dtype=float)
    rank[feasible] = contrast
    share = np.clip(_drill_ore_share(problem), .025, .60)
    raw = _to_volume(rank, feasible, share * n_feas)
    return make_legal(problem, raw, rank=rank)


def _fallback_layered_horizon(problem):
    feasible = problem["domain_mask"] & ~problem["forbidden_mask"]
    xyz = _grid_xyz(problem)
    drill = next((o for o in problem["observations"] if o["name"] == "drill"), problem["observations"][0])
    pts, y, g = np.asarray(drill["points"]), np.asarray(drill["labels"]), np.asarray(drill["groups"])
    hz = float(problem["spacing"][2])
    ore_xy, z_bots, z_tops, barren_xy = [], [], [], []
    for h_id in np.unique(g):
        m = g == h_id
        hp, hy = pts[m], y[m]
        bi = hy != .5
        if not bi.any():
            continue
        bp, by = hp[bi], (hy[bi] > .5).astype(int)
        order = np.argsort(bp[:, 2])
        bp, by = bp[order], by[order]
        if len(by) >= 5:
            by = ndimage.median_filter(by, size=3, mode="nearest")
        if by.any():
            idx = np.flatnonzero(by)
            ore_xy.append(bp[idx, :2].mean(axis=0))
            z_bots.append(float(bp[idx[0], 2] - .5 * hz))
            z_tops.append(float(bp[idx[-1], 2] + .5 * hz))
        else:
            barren_xy.append(bp[:, :2].mean(axis=0))
    if not ore_xy:
        return initialize(problem) > 0
    ore_xy = np.asarray(ore_xy, dtype=float)
    z_bots = np.asarray(z_bots, dtype=float)
    z_tops = np.asarray(z_tops, dtype=float)
    grid_xy = xyz[..., :2].reshape(-1, 2)
    span = np.maximum(np.asarray(problem["shape"][:2], dtype=float) * np.asarray(problem["spacing"][:2]), 1e-6)
    origin_xy = np.asarray(problem["origin"][:2], dtype=float)
    norm_ore = (ore_xy - origin_xy) / span
    norm_grid = (grid_xy - origin_xy) / span
    if len(ore_xy) >= 3 and np.linalg.matrix_rank(norm_ore - norm_ore.mean(0)) >= 2:
        zb = RBFInterpolator(norm_ore, z_bots, kernel="thin_plate_spline", smoothing=1e-3)(norm_grid).reshape(xyz.shape[:3])
        zt = RBFInterpolator(norm_ore, z_tops, kernel="thin_plate_spline", smoothing=1e-3)(norm_grid).reshape(xyz.shape[:3])
    elif len(ore_xy) >= 2:
        d = cKDTree(norm_ore).query(norm_grid, k=min(2, len(ore_xy)))
        w = 1. / np.maximum(d[0], 1e-6)
        w /= w.sum(axis=1, keepdims=True)
        zb = np.sum(w * z_bots[d[1]], axis=1).reshape(xyz.shape[:3])
        zt = np.sum(w * z_tops[d[1]], axis=1).reshape(xyz.shape[:3])
    else:
        zb = np.full(xyz.shape[:3], z_bots[0])
        zt = np.full(xyz.shape[:3], z_tops[0])
    thick = np.maximum(zt - zb, hz)
    z_mid = .5 * (zb + zt)
    vert = 1. - np.abs(xyz[..., 2] - z_mid) / (.5 * thick)
    if barren_xy:
        b_xy = np.asarray(barren_xy, dtype=float)
        d_ore = cKDTree(ore_xy).query(grid_xy)[0].reshape(xyz.shape[:3])
        d_bar = cKDTree(b_xy).query(grid_xy)[0].reshape(xyz.shape[:3])
        lat = (d_bar - d_ore) / (d_bar + d_ore + 1e-6)
        rank = np.minimum(vert, 2. * lat)
    else:
        rank = vert
    raw = (rank > 0.) & feasible
    return make_legal(problem, raw, rank=rank)


def _fallback_ellipsoid_fit(problem):
    feasible = problem["domain_mask"] & ~problem["forbidden_mask"]
    xyz = _grid_xyz(problem)
    drill = next((o for o in problem["observations"] if o["name"] == "drill"), problem["observations"][0])
    pts, y = np.asarray(drill["points"]), np.asarray(drill["labels"])
    ore_pts = pts[y == 1.]
    if not len(ore_pts):
        all_ore = [o["points"][o["labels"] == 1.] for o in problem["observations"] if np.any(o["labels"] == 1.)]
        if not all_ore:
            return initialize(problem) > 0
        ore_pts = np.concatenate(all_ore, axis=0)
    mu = ore_pts.mean(axis=0)
    h = np.asarray(problem["spacing"], dtype=float)
    n_feas = int(feasible.sum())
    share = float(np.clip(_drill_ore_share(problem), .025, .60))
    target_vol = share * n_feas * float(np.prod(h))
    r_eq = float((3. * target_vol / (4. * np.pi)) ** (1. / 3.))
    n_holes = max(1, len(np.unique(drill["groups"])))
    area_xy = float(problem["shape"][0] * h[0] * problem["shape"][1] * h[1])
    h_hole = float(np.sqrt(area_xy / n_holes))
    sigma_h = max(.5 * float(np.hypot(h[0], h[1])), min(.35 * h_hole, .5 * r_eq))
    sigma_z = .5 * float(h[2])
    cov = np.cov(ore_pts.T) if len(ore_pts) >= 4 else np.zeros((3, 3), dtype=float)
    cov = np.atleast_2d(cov).astype(float)
    reg = np.diag([sigma_h ** 2, sigma_h ** 2, sigma_z ** 2])
    inv_cov = np.linalg.inv(cov + reg)
    diff = xyz - mu
    d2 = np.einsum("...i,ij,...j->...", diff, inv_cov, diff)
    rank = -d2
    raw = _to_volume(rank, feasible, share * n_feas)
    return make_legal(problem, raw, rank=rank)


def compute_reference_masks(problem):
    refs = {"ref_initial": initialize(problem) > 0}
    for name, attr, fallback in (
        ("ref_stretched_nearest_class", "stretched_nearest_class", _fallback_stretched_nearest_class),
        ("ref_layered_horizon", "layered_horizon", _fallback_layered_horizon),
        ("ref_ellipsoid_fit", "ellipsoid_fit", _fallback_ellipsoid_fit),
    ):
        fn = getattr(baselines, attr, None)
        if callable(fn):
            field = fn(problem, dict(wall_time_seconds=30, cpu_count=1, memory_mb=2048), 47)
            refs[name] = np.asarray(field) > 0
        else:
            refs[name] = fallback(problem)
    return refs


def _discover_sponge_solvers(sponge_args, v6_runs_root, auto_v6):
    found = []
    for item in sponge_args or ():
        if "=" not in item:
            raise ValueError(f"--sponge-solver requires NAME=PATH, got {item!r}")
        name, raw_path = item.split("=", 1)
        p = Path(raw_path).resolve()
        if not p.is_file():
            raise FileNotFoundError(f"sponge solver not found: {p}")
        found.append((name, str(p)))
    if auto_v6 and not found and v6_runs_root is not None and Path(v6_runs_root).is_dir():
        base = Path(v6_runs_root)
        for label, run_name, sha_prefix in V6_SPONGE_SPECS:
            matches = sorted((base / run_name / "candidates").glob(f"{sha_prefix}*/solver.py"))
            if matches:
                found.append((label, str(matches[0].resolve())))
    return found


def _run_external_solver(solver_path, problem):
    spec = importlib.util.spec_from_file_location("archived_sponge_" + file_hash(solver_path)[:12], solver_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    prob_copy = read_problem_copy(problem)
    field = module.solve(prob_copy, dict(wall_time_seconds=30, cpu_count=1, memory_mb=2048), 47)
    try:
        mask, _ = canonicalize(problem, np.asarray(field, dtype=float))
        return mask
    except Exception:
        arr = np.asarray(field, dtype=float)
        if arr.shape == tuple(problem["shape"]) and np.isfinite(arr).all():
            return make_legal(problem, arr > 0, rank=arr)
        return initialize(problem) > 0


def read_problem_copy(problem):
    out = {
        **problem,
        "shape": list(problem["shape"]),
        "origin": list(problem["origin"]),
        "spacing": list(problem["spacing"]),
        "domain_mask": np.asarray(problem["domain_mask"], dtype=bool).copy(),
        "forbidden_mask": np.asarray(problem["forbidden_mask"], dtype=bool).copy(),
        "observations": [
            {**o, "points": np.asarray(o["points"]).copy(),
             "labels": np.asarray(o["labels"]).copy(),
             "groups": np.asarray(o["groups"]).copy()}
            for o in problem["observations"]
        ],
        "priors": dict(problem["priors"]),
    }
    for arr in [out["domain_mask"], out["forbidden_mask"],
                *[o[k] for o in out["observations"] for k in ("points", "labels", "groups")]]:
        arr.flags.writeable = False
    return out


def _apply_overlay(base_mask, overlay_type, period, phase, indices):
    i, j, k = indices
    pillars = (i % 6 == 0) & (j % 6 == 0)
    if overlay_type == "none":
        return base_mask
    if overlay_type == "sheets_h":
        return base_mask | (((k + phase) % period) == 0) | pillars
    if overlay_type == "sheets_v":
        return base_mask | (((i + phase) % period) == 0) | ((j % (2 * period)) == 0)
    if overlay_type == "lattice":
        return base_mask | (((i + phase) % period) == 0) | (((j + phase) % period) == 0) | (((k + phase) % period) == 0)
    if overlay_type == "xor_sheets":
        return (base_mask ^ (((k + phase) % period) == 0)) | pillars
    raise ValueError(f"unknown overlay_type {overlay_type!r}")


def _sample_adaptive_recipes(n_recipes, seed):
    rng = np.random.default_rng(seed)
    ref_names = ("ref_initial", "ref_stretched_nearest_class", "ref_layered_horizon", "ref_ellipsoid_fit")
    overlays = ("none", "sheets_h", "sheets_v", "lattice", "xor_sheets")
    periods = (2, 3, 4, 5, 6)
    recipes = []
    # Seed the search with exact clean reference baselines and canonical overlays, then random draws.
    for rname in ref_names:
        recipes.append(dict(ref=rname, overlay="none", period=0, phase=0, vol_mult=1.))
    while len(recipes) < n_recipes:
        rname = str(ref_names[int(rng.integers(len(ref_names)))])
        ov = str(overlays[int(rng.integers(len(overlays)))])
        per = 0 if ov == "none" else int(periods[int(rng.integers(len(periods)))])
        phase = 0 if ov == "none" else int(rng.integers(per))
        vm = float(np.round(np.exp(rng.uniform(np.log(.5), np.log(2.5))), 3))
        recipes.append(dict(ref=rname, overlay=ov, period=per, phase=phase, vol_mult=vm))
    return recipes[:n_recipes]


def _score_mask(conf, mask, truth, objective_version, k_tol):
    field = np.where(mask, 1., -1.)
    res = score(conf, field, objective_version=objective_version, k_tol=k_tol)
    loss = float(res["loss_total"])
    risk = float(1. - confirmation_utility(loss, objective_version))
    union = int(np.count_nonzero(mask | truth))
    iou = float(np.count_nonzero(mask & truth) / union) if union else 1.
    return dict(iou=iou, loss=loss, risk=risk, reg_area=float(res["regularizer_area"]))


def _evaluate_one_body(task):
    (body_idx, pub_dir, truth_dir, other_truth_path, family, objective_version,
     k_tol, check_alt_ktol, sponge_solvers, recipes, body_seed) = task
    pub = read_problem(pub_dir)
    raw_conf = read_problem(Path(truth_dir) / "confirmation")
    conf = _adapt_confirmation_for_objective(raw_conf, objective_version)
    truth = np.load(Path(truth_dir) / "truth.npy", allow_pickle=False)
    other_truth = np.load(other_truth_path, allow_pickle=False)
    feasible = pub["domain_mask"] & ~pub["forbidden_mask"]
    spacing = np.asarray(pub["spacing"], dtype=float)
    n_feas = int(feasible.sum())
    n_truth = int(truth.sum())
    cell = float(np.prod(spacing))
    lo_cells = int(np.ceil(pub["priors"]["hard_volume_bounds"][0] / cell - 1e-9))
    rng = np.random.default_rng(body_seed)
    indices = np.indices(truth.shape)

    sd_truth = (ndimage.distance_transform_edt(~truth, sampling=spacing)
                - ndimage.distance_transform_edt(truth, sampling=spacing))
    refs = compute_reference_masks(pub)

    items = []  # (name, kind, base_name, is_overlay, mask)
    items.append(("truth", "honest", None, False, make_legal(pub, truth)))
    for iters in (1, 2):
        items.append((f"dilate_{iters}", "honest", None, False,
                      make_legal(pub, ndimage.binary_dilation(truth, iterations=iters), rank=-sd_truth)))
        items.append((f"erode_{iters}", "honest", None, False,
                      make_legal(pub, ndimage.binary_erosion(truth, iterations=iters), rank=-sd_truth)))
    for shift in (1, 2, 3, 5):
        s_mask = np.roll(truth, shift, axis=0)
        items.append((f"shift_{shift}", "honest", None, False, make_legal(pub, s_mask)))
    for mult in (.5, .75, 1.5, 2., 3.):
        items.append((f"vol_x{mult}", "honest", None, False,
                      make_legal(pub, _to_volume(-sd_truth, feasible, mult * n_truth), rank=-sd_truth)))
    for sigma in (2, 4):
        blurred = ndimage.gaussian_filter(truth.astype(float), float(sigma))
        items.append((f"blur_{sigma}", "honest", None, False,
                      make_legal(pub, _to_volume(blurred, feasible, n_truth), rank=blurred)))
    for rname, rmask in refs.items():
        items.append((rname, "honest", None, False, make_legal(pub, rmask)))

    bases = [("truth", truth), *list(refs.items())]
    for bname, bmask in bases:
        sd_b = (ndimage.distance_transform_edt(~bmask, sampling=spacing)
                - ndimage.distance_transform_edt(bmask, sampling=spacing))
        for p in (2, 3, 4, 6):
            ov = _apply_overlay(bmask, "sheets_h", p, 0, indices)
            items.append((f"{bname}+sheets_h_p{p}", "adversary", bname, True, make_legal(pub, ov)))
        items.append((f"{bname}+sheets_v_p4", "adversary", bname, True,
                      make_legal(pub, _apply_overlay(bmask, "sheets_v", 4, 0, indices))))
        items.append((f"{bname}+lattice_p4", "adversary", bname, True,
                      make_legal(pub, _apply_overlay(bmask, "lattice", 4, 0, indices))))
        items.append((f"{bname}^sheets_p2", "adversary", bname, True,
                      make_legal(pub, _apply_overlay(bmask, "xor_sheets", 2, 0, indices))))
        for frac, tag in ((.25, "25"), (.45, "45"), (.65, "65")):
            items.append((f"{bname}+inflate_{tag}", "adversary", bname, False,
                          make_legal(pub, _to_volume(-sd_b, feasible, frac * n_feas), rank=-sd_b)))
        items.append((f"{bname}+deflate_min", "adversary", bname, False,
                      make_legal(pub, _to_volume(-sd_b, feasible, lo_cells), rank=-sd_b)))

    # Standalone adversaries: random-place same-volume blob, resampled other body truth, public ore tubes, v6 sponges.
    feas_coords = np.argwhere(feasible)
    center = feas_coords[int(rng.integers(len(feas_coords)))]
    d2_blob = sum(((indices[a] - center[a]) * spacing[a]) ** 2 for a in range(3))
    items.append(("blob_same_volume", "adversary", None, False,
                  make_legal(pub, _to_volume(-d2_blob, feasible, n_truth), rank=-d2_blob)))

    zoom_factors = tuple(truth.shape[a] / other_truth.shape[a] for a in range(3))
    resampled_other = ndimage.zoom(other_truth.astype(np.uint8), zoom_factors, order=0) > 0
    if resampled_other.shape != truth.shape:
        fixed = np.zeros_like(truth)
        sl = tuple(slice(0, min(truth.shape[a], resampled_other.shape[a])) for a in range(3))
        fixed[sl] = resampled_other[sl]
        resampled_other = fixed
    items.append(("other_body_truth", "adversary", None, False, make_legal(pub, resampled_other)))

    ore_pts = [o["points"][o["labels"] == 1.] for o in pub["observations"] if np.any(o["labels"] == 1.)]
    if ore_pts:
        ore_idx = (np.concatenate(ore_pts, axis=0) - np.asarray(pub["origin"])) / spacing - .5
        grid_idx = np.stack(indices, axis=-1).reshape(-1, 3)
        dist_cells = cKDTree(ore_idx).query(grid_idx)[0].reshape(truth.shape)
        tubes = (dist_cells <= 1.5) & feasible
        items.append(("public_ore_tubes", "adversary", None, False, make_legal(pub, tubes, rank=-dist_cells)))
    else:
        items.append(("public_ore_tubes", "adversary", None, False, make_legal(pub, np.zeros_like(truth))))

    for sname, spath in sponge_solvers:
        smask = _run_external_solver(spath, pub)
        items.append((sname, "adversary", None, False, smask))

    rows = []
    alt_ktol = 0. if abs(float(k_tol) - .5) < 1e-9 else .5
    for name, kind, bname, is_ov, mask in items:
        scored = _score_mask(conf, mask, truth, objective_version, k_tol)
        alt_risk = None
        if check_alt_ktol and objective_version == OBJECTIVE:
            alt_loss = float(score(conf, np.where(mask, 1., -1.),
                                   objective_version=OBJECTIVE, k_tol=alt_ktol)["loss_total"])
            alt_risk = float(1. - confirmation_utility(alt_loss, OBJECTIVE))
        rows.append(dict(
            body_idx=body_idx, family=family, name=name, kind=kind,
            base=bname, is_overlay=is_ov, alt_risk=alt_risk, **scored,
        ))

    # Adaptive attack candidate recipes on this body.
    sd_refs = {
        rname: (ndimage.distance_transform_edt(~rm, sampling=spacing)
                - ndimage.distance_transform_edt(rm, sampling=spacing))
        for rname, rm in refs.items()
    }
    adaptive_ious = np.zeros(len(recipes), dtype=float)
    adaptive_risks = np.zeros(len(recipes), dtype=float)
    for r_idx, rec in enumerate(recipes):
        rm = refs[rec["ref"]]
        if abs(rec["vol_mult"] - 1.) > 1e-6:
            scaled = _to_volume(-sd_refs[rec["ref"]], feasible, rec["vol_mult"] * max(1, int(rm.sum())))
        else:
            scaled = rm
        raw = _apply_overlay(scaled, rec["overlay"], rec["period"], rec["phase"], indices)
        lmask = make_legal(pub, raw)
        sc = _score_mask(conf, lmask, truth, objective_version, k_tol)
        adaptive_ious[r_idx] = sc["iou"]
        adaptive_risks[r_idx] = sc["risk"]

    truth_diag = diagnostics(pub, truth)
    return dict(
        body_idx=body_idx,
        family=family,
        rows=rows,
        adaptive_ious=adaptive_ious,
        adaptive_risks=adaptive_risks,
        truth_reg_area=float(truth_diag["regularizer_area"]),
    )


# Replay on archived v6 training evaluations (Criterion C7).

def _prep_fast_v3(conf, k_tol):
    origin = np.asarray(conf["origin"], dtype=float)
    spacing = np.asarray(conf["spacing"], dtype=float)
    shape = np.asarray(conf["shape"], dtype=int)
    disc = .5 * float(np.linalg.norm(spacing))
    prepped = []
    for obs in conf["observations"]:
        pts = np.asarray(obs["points"], dtype=float)
        y = np.asarray(obs["labels"], dtype=float)
        groups = np.asarray(obs["groups"])
        idx = np.clip(np.rint((pts - origin) / spacing - .5).astype(int), 0, shape - 1)
        binary = y != .5
        contact = ~binary
        sigma_meas = float(np.sqrt(max(0., float(obs["transition_scale"]) ** 2 - disc ** 2)))
        delta = float(np.hypot(float(k_tol) * spacing[2], sigma_meas))
        keep = binary.copy()
        if contact.any() and delta > 0.:
            for grp in np.unique(groups[contact]):
                c_pts = pts[contact & (groups == grp)]
                m = np.flatnonzero(binary & (groups == grp))
                if len(m):
                    d = np.min(np.linalg.norm(pts[m][:, None, :] - c_pts[None, :, :], axis=2), axis=1)
                    keep[m[d < delta]] = False
        prepped.append((
            tuple(idx[keep].T),
            y[keep] > .5,
            float(obs["label_error"]),
            float(obs["weight"]),
        ))
    prior_weights = tuple(float(conf["priors"][f"{n}_weight"]) for n in ("area", "volume", "fragment"))
    return prepped, prior_weights


def _fast_v3_risk(mask, prepped_conf, reg_vals=(0., 0., 0.)):
    obs_list, prior_weights = prepped_conf
    tp = fp = fn = 0.
    for idx_tup, y_keep, eps, w in obs_list:
        p_keep = mask[idx_tup]
        n_pos = float(np.count_nonzero(p_keep))
        n_neg = float(np.count_nonzero(~p_keep))
        a = float(np.count_nonzero(p_keep & y_keep))
        c = float(np.count_nonzero(~p_keep & y_keep))
        if eps > 0.:
            denom = 1. - 2. * eps
            a = float(np.clip((a - eps * n_pos) / denom, 0., n_pos))
            c = float(np.clip((c - eps * n_neg) / denom, 0., n_neg))
        tp += w * a
        fp += w * (n_pos - a)
        fn += w * c
    j = 1. - (tp + 1.) / (tp + fp + fn + 1.)
    loss = j + sum(pw * rv for pw, rv in zip(prior_weights, reg_vals))
    return float(1. - confirmation_utility(loss, OBJECTIVE))


def _replay_one_run(task):
    run_dir, service_evals_dir, prepped_cases, objective_version = task
    run_dir = Path(run_dir)
    ledger_path = run_dir / "ledger.jsonl"
    if not ledger_path.is_file() or not Path(service_evals_dir).is_dir():
        return None
    valid_ids = {
        json.loads(line)["program_id"]
        for line in ledger_path.read_text().splitlines()
        if line.strip() and json.loads(line).get("status") == "valid"
    }
    if len(valid_ids) < 4:
        return None
    programs = []
    for res_file in sorted(glob.glob(str(Path(service_evals_dir) / "*/result.json"))):
        res = json.loads(Path(res_file).read_text())
        pid = res.get("candidate_id")
        if res.get("status") != "succeeded" or pid not in valid_ids:
            continue
        eval_dir = Path(res_file).parent
        pc_file = eval_dir / "private-case-results.json"
        if not pc_file.is_file():
            continue
        pc = json.loads(pc_file.read_text())
        areas = [float(c["metrics"]["regularizer_area"]) for c in pc
                 if c.get("status") == "succeeded" and "regularizer_area" in c.get("metrics", {})]
        mean_area = float(np.mean(areas)) if areas else 0.
        mean_iou = float(res["metrics"]["mean_iou"])
        if objective_version == LEGACY_OBJECTIVE:
            risk = float(res["metrics"]["loss_total"])
        else:
            case_risks = []
            for c in pc:
                cid = c["case_id"]
                seed = c.get("seed", 47)
                if c.get("status") != "succeeded" or cid not in prepped_cases:
                    case_risks.append(1.)
                    continue
                model_path = eval_dir / cid / str(seed) / "model" / "model.npy"
                if not model_path.is_file():
                    case_risks.append(1.)
                    continue
                mask = np.load(model_path, allow_pickle=False) > 0
                m = c.get("metrics", {})
                regs = (float(m.get("regularizer_area", 0.)),
                        float(m.get("regularizer_volume", 0.)),
                        float(m.get("regularizer_fragment", 0.)))
                case_risks.append(_fast_v3_risk(mask, prepped_cases[cid], regs))
            risk = float(np.mean(case_risks))
        programs.append((pid, risk, mean_iou, mean_area))
    if len(programs) < 4:
        return None
    programs.sort(key=lambda x: x[0])
    risks = np.array([p[1] for p in programs], dtype=float)
    ious = np.array([p[2] for p in programs], dtype=float)
    areas = np.array([p[3] for p in programs], dtype=float)
    rho = float(spearmanr(risks, 1. - ious).statistic)
    q25 = risks <= np.quantile(risks, .25)
    rho_top = float(spearmanr(risks[q25], 1. - ious[q25]).statistic) if int(q25.sum()) >= 3 and np.ptp(risks[q25]) > 0 and np.ptp(ious[q25]) > 0 else float("nan")
    pick_idx = int(np.argmin(risks))
    pick_iou = float(ious[pick_idx])
    max_iou = float(np.max(ious))
    sponges = areas > 6.
    sponge_pct = float(np.mean([np.mean(risks < r) for r in risks[sponges]])) if np.any(sponges) else None
    return dict(
        n_programs=len(programs),
        rho=rho,
        rho_best_quartile=None if np.isnan(rho_top) else rho_top,
        pick_iou=pick_iou,
        max_iou=max_iou,
        pick_iou_gap=float(max_iou - pick_iou),
        sponge_count=int(np.sum(sponges)),
        sponge_pct=sponge_pct,
    )


def evaluate_archived_replay(runs_root, services_root, replay_release, objective_version, k_tol, workers):
    runs_root, services_root, replay_release = Path(runs_root), Path(services_root), Path(replay_release)
    train_manifest = replay_release / "train_s" / "manifest.json"
    if not train_manifest.is_file():
        train_manifest = replay_release / "train" / "manifest.json"
    if not (runs_root.is_dir() and services_root.is_dir() and train_manifest.is_file()):
        return dict(evaluated=False, passed=None, reason="archived replay directories not available")

    manifest = json.loads(train_manifest.read_text())
    prepped_cases = {}
    if objective_version == OBJECTIVE:
        for c in manifest["cases"]:
            conf = _adapt_confirmation_for_objective(
                read_problem(train_manifest.parent / c["truth"] / "confirmation"), OBJECTIVE)
            prepped_cases[c["case_id"]] = _prep_fast_v3(conf, k_tol)

    tasks = []
    for run_dir in sorted(runs_root.glob("*-s*")):
        if not run_dir.is_dir():
            continue
        res_path = run_dir / "result.json"
        if res_path.is_file() and json.loads(res_path.read_text()).get("status") != "completed":
            continue
        evals_dir = services_root / run_dir.name / "evaluations"
        if evals_dir.is_dir():
            tasks.append((str(run_dir), str(evals_dir), prepped_cases, objective_version))
    if not tasks:
        return dict(evaluated=False, passed=None, reason="no completed archived runs found")

    if workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=min(workers, MAX_WORKERS, len(tasks))) as pool:
            results = [r for r in pool.map(_replay_one_run, tasks) if r is not None]
    else:
        results = [r for r in map(_replay_one_run, tasks) if r is not None]
    if not results:
        return dict(evaluated=False, passed=None, reason="no valid archived evaluation records")

    mean_rho = float(np.mean([r["rho"] for r in results]))
    top_rhos = [r["rho_best_quartile"] for r in results if r["rho_best_quartile"] is not None]
    mean_top_rho = float(np.mean(top_rhos)) if top_rhos else None
    sp_vals = [r["sponge_pct"] for r in results if r["sponge_pct"] is not None]
    mean_sponge_pct = float(np.mean(sp_vals)) if sp_vals else 1.
    mean_gap = float(np.mean([r["pick_iou_gap"] for r in results]))
    passed = bool(mean_rho >= .80 and mean_sponge_pct >= .80 and mean_gap <= .03)
    return dict(
        evaluated=True,
        passed=passed,
        runs=len(results),
        programs=int(sum(r["n_programs"] for r in results)),
        within_run_spearman=mean_rho,
        best_quartile_spearman=mean_top_rho,
        sponge_rank_percentile=mean_sponge_pct if sp_vals else None,
        runs_with_sponges=len(sp_vals),
        pick_iou_gap=mean_gap,
        pick_mean_iou=float(np.mean([r["pick_iou"] for r in results])),
        pick_min_iou=float(min(r["pick_iou"] for r in results)),
        max_train_iou=float(np.mean([r["max_iou"] for r in results])),
        thresholds=dict(within_run_spearman=.80, sponge_percentile=.80, pick_iou_gap=.03),
    )


def _bootstrap_ci(body_matrix_risk, body_matrix_iou, n_boot, seed):
    # body_matrix_risk, body_matrix_iou: shape (n_bodies, n_items)
    n_bodies = body_matrix_risk.shape[0]
    if n_bodies < 2 or n_boot < 1:
        return dict(c1=[None, None], c3_pooled=[None, None], c3_within=[None, None], truth_risk=[None, None])
    rng = np.random.default_rng(seed)
    within_per_body = np.array([
        float(spearmanr(body_matrix_risk[b], 1. - body_matrix_iou[b]).statistic)
        for b in range(n_bodies)
    ])
    c1_s, c3p_s, c3w_s, tr_s = [], [], [], []
    for _ in range(n_boot):
        pick = rng.integers(0, n_bodies, size=n_bodies)
        r_sub = body_matrix_risk[pick]
        i_sub = body_matrix_iou[pick]
        c1_s.append(float(spearmanr(r_sub.mean(axis=0), 1. - i_sub.mean(axis=0)).statistic))
        c3p_s.append(float(spearmanr(r_sub.ravel(), 1. - i_sub.ravel()).statistic))
        c3w_s.append(float(within_per_body[pick].mean()))
        tr_s.append(float(r_sub[:, 0].mean()))
    ci = lambda arr: [float(np.nanpercentile(arr, 2.5)), float(np.nanpercentile(arr, 97.5))]
    return dict(c1=ci(c1_s), c3_pooled=ci(c3p_s), c3_within=ci(c3w_s), truth_risk=ci(tr_s))


def run_battery(release, split="calibration", objective=OBJECTIVE, *, k_tol=K_TOL,
                workers=8, seed=20261124, adaptive_masks=2000, bootstrap=500,
                sponge_solvers=(), v6_runs_root=DEFAULT_V6_RUNS, v6_services_root=DEFAULT_V6_SERVICES,
                v6_release=DEFAULT_V6_RELEASE, auto_v6=True, run_replay=True):
    if split not in ALLOWED_SPLITS:
        raise ValueError(f"judge_validity allows splits {ALLOWED_SPLITS} only, got {split!r}")
    if objective not in OBJECTIVE_VERSIONS:
        raise ValueError(f"unsupported objective {objective!r}; expected one of {OBJECTIVE_VERSIONS}")
    workers = max(1, min(int(workers), MAX_WORKERS))

    manifest_path, manifest = resolve_split_manifest(release, split)
    cases = manifest["cases"]
    n_bodies = len(cases)
    if n_bodies < 2:
        raise ValueError("judge_validity requires at least 2 bodies")
    families_by_case = _resolve_families(release, split, cases)
    sponges = _discover_sponge_solvers(sponge_solvers, v6_runs_root, auto_v6)

    n_recipes = max(24, int(round(adaptive_masks / n_bodies)))
    recipes = _sample_adaptive_recipes(n_recipes, seed)

    # Pair each body with a template truth from another body (preferring a different family).
    truth_paths = [str((manifest_path.parent / c["truth"] / "truth.npy").resolve()) for c in cases]
    fams = [families_by_case[c["case_id"]] for c in cases]
    other_paths = []
    for idx in range(n_bodies):
        offset = n_bodies // 2
        chosen = (idx + offset) % n_bodies
        for step in range(1, n_bodies):
            cand = (idx + offset + step - 1) % n_bodies
            if cand != idx and fams[cand] != fams[idx]:
                chosen = cand
                break
        other_paths.append(truth_paths[chosen])

    tasks = [
        (idx,
         str((manifest_path.parent / c["public"]).resolve()),
         str((manifest_path.parent / c["truth"]).resolve()),
         other_paths[idx],
         fams[idx],
         objective,
         float(k_tol),
         objective == OBJECTIVE,
         sponges,
         recipes,
         int(seed + 1009 * (idx + 1)))
        for idx, c in enumerate(cases)
    ]

    if workers > 1 and n_bodies > 1:
        with ProcessPoolExecutor(max_workers=min(workers, n_bodies)) as pool:
            body_results = list(pool.map(_evaluate_one_body, tasks))
    else:
        body_results = list(map(_evaluate_one_body, tasks))
    body_results.sort(key=lambda r: r["body_idx"])

    item_names = [r["name"] for r in body_results[0]["rows"]]
    item_meta = {r["name"]: (r["kind"], r["base"], r["is_overlay"]) for r in body_results[0]["rows"]}
    n_items = len(item_names)
    name_to_col = {name: col for col, name in enumerate(item_names)}

    risk_mat = np.zeros((n_bodies, n_items), dtype=float)
    loss_mat = np.zeros((n_bodies, n_items), dtype=float)
    iou_mat = np.zeros((n_bodies, n_items), dtype=float)
    area_mat = np.zeros((n_bodies, n_items), dtype=float)
    alt_risk_mat = np.zeros((n_bodies, n_items), dtype=float) if objective == OBJECTIVE else None

    for b_idx, bres in enumerate(body_results):
        for row in bres["rows"]:
            col = name_to_col[row["name"]]
            risk_mat[b_idx, col] = row["risk"]
            loss_mat[b_idx, col] = row["loss"]
            iou_mat[b_idx, col] = row["iou"]
            area_mat[b_idx, col] = row["reg_area"]
            if alt_risk_mat is not None and row["alt_risk"] is not None:
                alt_risk_mat[b_idx, col] = row["alt_risk"]

    mean_risks = risk_mat.mean(axis=0)
    mean_losses = loss_mat.mean(axis=0)
    mean_ious = iou_mat.mean(axis=0)
    mean_areas = area_mat.mean(axis=0)

    # C1: Spearman across item types (suite means) >= 0.90
    c1_rho = float(spearmanr(mean_risks, 1. - mean_ious).statistic)
    c1_pass = bool(c1_rho >= .90)

    # C2: No adversary scores better (<=) than any honest item whose suite-mean IoU is higher by >= 0.10
    honest_cols = [c for c, n in enumerate(item_names) if item_meta[n][0] == "honest"]
    adv_cols = [c for c, n in enumerate(item_names) if item_meta[n][0] == "adversary"]
    violations = []
    for ac in adv_cols:
        for hc in honest_cols:
            if mean_ious[hc] >= mean_ious[ac] + .10 and mean_risks[ac] <= mean_risks[hc]:
                violations.append(dict(
                    adversary=item_names[ac],
                    honest=item_names[hc],
                    adversary_iou=float(mean_ious[ac]),
                    honest_iou=float(mean_ious[hc]),
                    adversary_risk=float(mean_risks[ac]),
                    honest_risk=float(mean_risks[hc]),
                ))
    c2_pass = len(violations) == 0

    # C3: Pooled Spearman >= 0.75 and within-body mean >= 0.85
    c3_pooled = float(spearmanr(risk_mat.ravel(), 1. - iou_mat.ravel()).statistic)
    within_rhos = [float(spearmanr(risk_mat[b], 1. - iou_mat[b]).statistic) for b in range(n_bodies)]
    c3_within = float(np.mean(within_rhos))
    c3_pass = bool(c3_pooled >= .75 and c3_within >= .85)

    k_tol_comparison = None
    if alt_risk_mat is not None:
        alt_ktol = 0. if abs(float(k_tol) - .5) < 1e-9 else .5
        alt_pooled = float(spearmanr(alt_risk_mat.ravel(), 1. - iou_mat.ravel()).statistic)
        alt_within = float(np.mean([
            float(spearmanr(alt_risk_mat[b], 1. - iou_mat[b]).statistic) for b in range(n_bodies)
        ]))
        by_k = {
            f"{float(k_tol):.1f}": dict(pooled_spearman=c3_pooled, within_body_spearman=c3_within,
                                        mean_c3=.5 * (c3_pooled + c3_within)),
            f"{alt_ktol:.1f}": dict(pooled_spearman=alt_pooled, within_body_spearman=alt_within,
                                    mean_c3=.5 * (alt_pooled + alt_within)),
        }
        chosen_k = max(by_k.items(), key=lambda kv: (kv[1]["within_body_spearman"] + kv[1]["pooled_spearman"], kv[0]))[0]
        k_tol_comparison = dict(candidates=by_k, recommended_k_tol=float(chosen_k))

    # C4: No overlay beats its own base output (without improving true IoU) in more than 10% of bodies
    overlay_shares = []
    beats_base_by_col = {}
    for col, name in enumerate(item_names):
        kind, bname, is_ov = item_meta[name]
        if bname is not None:
            bcol = name_to_col[bname]
            raw_share = float(np.mean(risk_mat[:, col] < risk_mat[:, bcol]))
            spurious_share = float(np.mean((risk_mat[:, col] < risk_mat[:, bcol]) & (iou_mat[:, col] <= iou_mat[:, bcol])))
            beats_base_by_col[col] = spurious_share
            if is_ov:
                overlay_shares.append(dict(overlay=name, base=bname, win_share=spurious_share,
                                           unconditional_win_share=raw_share,
                                           mean_iou=float(mean_ious[col]), mean_risk=float(mean_risks[col])))
    overlay_shares.sort(key=lambda r: (-r["win_share"], -r["unconditional_win_share"], r["overlay"]))
    max_ov_share = float(overlay_shares[0]["win_share"]) if overlay_shares else 0.
    max_raw_ov_share = float(max((r["unconditional_win_share"] for r in overlay_shares), default=0.))
    c4_pass = bool(max_ov_share <= .10)

    # C5: Truth risk <= 0.15, and <= 0.30 in every family
    truth_col = name_to_col["truth"]
    truth_risks = risk_mat[:, truth_col]
    truth_mean_risk = float(truth_risks.mean())
    fam_names = sorted(set(fams))
    truth_by_family = {
        fam: float(np.mean([truth_risks[i] for i in range(n_bodies) if fams[i] == fam]))
        for fam in fam_names
    }
    max_fam_truth_risk = float(max(truth_by_family.values()))
    c5_pass = bool(truth_mean_risk <= .15 and max_fam_truth_risk <= .30)

    # C6: Adaptive attack — stratified half-split selection on half A, IoU gap on half B <= 0.03
    half_a, half_b = [], []
    for fam in fam_names:
        members = [i for i in range(n_bodies) if fams[i] == fam]
        for pos, bi in enumerate(members):
            (half_a if pos % 2 == 0 else half_b).append(bi)
    if not half_b:
        half_a = list(range(n_bodies // 2))
        half_b = list(range(n_bodies // 2, n_bodies))
    adap_risk_mat = np.stack([b["adaptive_risks"] for b in body_results], axis=0)
    adap_iou_mat = np.stack([b["adaptive_ious"] for b in body_results], axis=0)
    half_a_risk = adap_risk_mat[half_a].mean(axis=0)
    half_b_iou = adap_iou_mat[half_b].mean(axis=0)
    pick_recipe_idx = int(np.argmin(half_a_risk))
    pick_half_b_iou = float(half_b_iou[pick_recipe_idx])
    best_half_b_iou = float(np.max(half_b_iou))
    c6_gap = float(best_half_b_iou - pick_half_b_iou)
    c6_pass = bool(c6_gap <= .03)

    # C7: Replay on archived v6 training programs
    if run_replay and v6_runs_root is not None and v6_services_root is not None and v6_release is not None:
        c7 = evaluate_archived_replay(v6_runs_root, v6_services_root, v6_release, objective, k_tol, workers)
    else:
        c7 = dict(evaluated=False, passed=None, reason="replay disabled")

    cis = _bootstrap_ci(risk_mat, iou_mat, int(bootstrap), seed + 77)
    truth_areas = np.array([b["truth_reg_area"] for b in body_results], dtype=float)

    item_summary = [
        dict(
            name=name,
            kind=item_meta[name][0],
            base=item_meta[name][1],
            is_overlay=item_meta[name][2],
            legal_rate=1.,
            mean_iou=float(mean_ious[col]),
            mean_risk=float(mean_risks[col]),
            mean_loss=float(mean_losses[col]),
            mean_reg_area=float(mean_areas[col]),
            beats_base_share=beats_base_by_col.get(col),
        )
        for col, name in enumerate(item_names)
    ]

    criteria = dict(
        C1=dict(
            passed=c1_pass,
            spearman_suite_means=c1_rho,
            ci95=cis["c1"],
            threshold=.90,
            item_type_count=n_items,
        ),
        C2=dict(
            passed=c2_pass,
            violation_count=len(violations),
            violations=violations[:12],
            iou_margin=.10,
        ),
        C3=dict(
            passed=c3_pass,
            pooled_spearman=c3_pooled,
            pooled_ci95=cis["c3_pooled"],
            within_body_spearman=c3_within,
            within_body_ci95=cis["c3_within"],
            thresholds=dict(pooled=.75, within_body=.85),
        ),
        C4=dict(
            passed=c4_pass,
            max_overlay_win_share=max_ov_share,
            max_unconditional_overlay_win_share=max_raw_ov_share,
            threshold=.10,
            worst_overlays=overlay_shares[:8],
        ),
        C5=dict(
            passed=c5_pass,
            truth_mean_risk=truth_mean_risk,
            truth_mean_loss=float(mean_losses[truth_col]),
            truth_risk_ci95=cis["truth_risk"],
            max_family_truth_risk=max_fam_truth_risk,
            truth_risk_by_family=truth_by_family,
            thresholds=dict(overall=.15, per_family=.30),
        ),
        C6=dict(
            passed=c6_pass,
            pick_iou_gap=c6_gap,
            threshold=.03,
            selected_recipe=recipes[pick_recipe_idx],
            selected_half_a_risk=float(half_a_risk[pick_recipe_idx]),
            pick_half_b_iou=pick_half_b_iou,
            best_half_b_iou=best_half_b_iou,
            half_a_bodies=len(half_a),
            half_b_bodies=len(half_b),
            recipes_evaluated=len(recipes),
            total_masks_evaluated=int(len(recipes) * n_bodies),
        ),
        C7=c7,
    )

    blocking_passed = bool(c2_pass and c4_pass and c6_pass)
    evaluated_passes = [c1_pass, c2_pass, c3_pass, c4_pass, c5_pass, c6_pass]
    if c7.get("evaluated"):
        evaluated_passes.append(bool(c7["passed"]))
    all_passed = bool(blocking_passed and all(evaluated_passes))

    return dict(
        schema="copper-judge-validity-v1",
        status="passed" if all_passed else "failed",
        blocking_passed=blocking_passed,
        all_criteria_passed=all_passed,
        release=Path(release).resolve().name,
        split=split,
        scope="calibration" if split == "calibration" else "dry_run_train",
        manifest_sha256=file_hash(manifest_path),
        objective_version=objective,
        k_tol=float(k_tol) if objective == OBJECTIVE else None,
        k_tol_comparison=k_tol_comparison,
        body_count=n_bodies,
        family_counts={fam: int(sum(x == fam for x in fams)) for fam in fam_names},
        archived_sponge_solvers=[name for name, _ in sponges],
        truth_regularizer_area=dict(
            mean=float(truth_areas.mean()),
            median=float(np.median(truth_areas)),
            p95=float(np.percentile(truth_areas, 95)),
            max=float(truth_areas.max()),
            fallback_hinge_1_1x_max=float(1.1 * truth_areas.max()),
        ),
        criteria=criteria,
        item_summary=item_summary,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True, help="release root directory")
    parser.add_argument("--split", choices=ALLOWED_SPLITS, default="calibration",
                        help="calibration (formal) or train (dry run)")
    parser.add_argument("--objective", choices=OBJECTIVE_VERSIONS, default=OBJECTIVE,
                        help="objective version to evaluate")
    parser.add_argument("--k-tol", type=float, default=K_TOL,
                        help="contact tolerance band factor in cells for objective_v3")
    parser.add_argument("--output", type=Path, help="optional path to write JSON report")
    parser.add_argument("--workers", type=int, default=8, help="worker processes (capped at 16)")
    parser.add_argument("--seed", type=int, default=20261124)
    parser.add_argument("--adaptive-masks", type=int, default=2000,
                        help="target mask count for C6 adaptive attack search")
    parser.add_argument("--bootstrap", type=int, default=500,
                        help="bootstrap draws over bodies for 95%% intervals")
    parser.add_argument("--sponge-solver", action="append", default=[], metavar="NAME=PATH",
                        help="optional archived sponge solver source(s)")
    parser.add_argument("--v6-runs-root", type=Path, default=DEFAULT_V6_RUNS)
    parser.add_argument("--v6-services-root", type=Path, default=DEFAULT_V6_SERVICES)
    parser.add_argument("--v6-release", type=Path, default=DEFAULT_V6_RELEASE)
    parser.add_argument("--no-auto-v6", action="store_true",
                        help="disable automatic discovery of v6 archived sponge solvers")
    parser.add_argument("--no-replay", action="store_true",
                        help="skip C7 archived training program replay")
    args = parser.parse_args(argv)

    report = run_battery(
        args.release,
        split=args.split,
        objective=args.objective,
        k_tol=args.k_tol,
        workers=args.workers,
        seed=args.seed,
        adaptive_masks=args.adaptive_masks,
        bootstrap=args.bootstrap,
        sponge_solvers=args.sponge_solver,
        v6_runs_root=args.v6_runs_root,
        v6_services_root=args.v6_services_root,
        v6_release=args.v6_release,
        auto_v6=not args.no_auto_v6,
        run_replay=not args.no_replay,
    )
    encoded = json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
