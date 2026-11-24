"""Unit and acceptance tests for Engineer B baselines (M5 fdi_smooth, Stage-B M4/M7/RF, GemPy helpers)."""

import ast
import json
import time
from pathlib import Path
import numpy as np
import pytest

from tasks.copper_inversion.core.runner import input_digest
from tasks.copper_inversion.experiments.baselines import gempy_reference
from tasks.copper_inversion.v2 import baseline_harness as harness
from tasks.copper_inversion.v2 import baselines, fdi_baseline, offline_baselines
from tasks.copper_inversion.v2.contract import validate
from tasks.copper_inversion.v2.data import canonical_problem, invert_orientation, save_problem
from tasks.copper_inversion.v2.fixtures import fixtures
from tasks.copper_inversion.v2.geometry import canonicalize
from tasks.copper_inversion.v2.metrics import compare
from tasks.copper_inversion.tests.test_v2_baselines import (
    _make_ellipsoid_lens,
    _make_flat_slab,
    _make_folded_layer,
    _synthetic_problem,
)
from tasks.copper_inversion.tests.test_v2_lit_baselines import _irregular_lens_problem


@pytest.mark.parametrize("solver", (fdi_baseline.fdi_smooth, fdi_baseline.fdi_smooth_native))
def test_fdi_smooth_interface_legality_determinism_and_5s_budget(solver):
    budget = {"wall_time_seconds": 5.0}
    for prob in fixtures():
        before = input_digest(prob)
        t0 = time.monotonic()
        out1 = solver(prob, budget, 0)
        elapsed = time.monotonic() - t0
        out2 = solver(prob, budget, 999)
        assert elapsed < 5.0
        assert input_digest(prob) == before
        assert isinstance(out1, np.ndarray) and not np.ma.isMaskedArray(out1)
        assert out1.shape == tuple(prob["shape"])
        assert out1.dtype.kind == "f" and np.isfinite(out1).all()
        assert np.array_equal(out1, out2)
        canonicalize(prob, out1)


def test_fdi_smooth_degenerate_inputs():
    shape = (12, 12, 10)
    holes = [(2, 2), (6, 6), (9, 9)]
    budget = {"wall_time_seconds": 10.0}

    p_empty, _ = _synthetic_problem(np.zeros(shape, dtype=bool), holes, shape=shape, case_id="no-ore")
    canonicalize(p_empty, fdi_baseline.fdi_smooth(p_empty, budget, 0))

    p_full, _ = _synthetic_problem(np.ones(shape, dtype=bool), holes, shape=shape, case_id="all-ore")
    canonicalize(p_full, fdi_baseline.fdi_smooth(p_full, budget, 0))

    slab = np.zeros(shape, dtype=bool)
    slab[:, :, 3:6] = True
    p_one, _ = _synthetic_problem(slab, [(5, 5)], shape=shape, case_id="one-hole")
    canonicalize(p_one, fdi_baseline.fdi_smooth(p_one, budget, 0))

    origin = np.asarray(p_empty["origin"], dtype=float)
    spacing = np.asarray(p_empty["spacing"], dtype=float)
    outcrop_pts = origin + (np.array([[4, 4, 8], [5, 5, 8], [6, 6, 8], [2, 8, 8]], dtype=float) + 0.5) * spacing
    p_out = validate({
        **p_empty,
        "case_id": "outcrop-only-ore",
        "observations": list(p_empty["observations"]) + [{
            "name": "outcrop",
            "points": outcrop_pts,
            "labels": np.array([1.0, 1.0, 1.0, 0.0]),
            "groups": np.array([0, 0, 1, 2], dtype=np.int64),
            "weight": 0.25,
            "transition_scale": float(np.linalg.norm(spacing)),
            "contact_sigma": float(spacing[2]),
            "label_error": 0.0,
        }],
    })
    canonicalize(p_out, fdi_baseline.fdi_smooth(p_out, budget, 0))


def test_fdi_smooth_mirror_equivariance():
    prob, _ = _irregular_lens_problem()
    rng = np.random.default_rng(19)
    drill = prob["observations"][0]
    pts = drill["points"].copy()
    for gid in np.unique(drill["groups"]):
        sel = drill["groups"] == gid
        pts[sel, :2] += rng.uniform(-0.3, 0.3, size=2)
        pts[sel, 2] += rng.uniform(0.05, 0.35, size=int(sel.sum()))
    prob = validate({**prob, "observations": [{**drill, "points": pts}]})
    budget = {"wall_time_seconds": 30.0}
    out0 = fdi_baseline.fdi_smooth(prob, budget, 0)
    for ori in (4, 6):
        p_flip = canonical_problem(prob, canonical_origin=False, orientation=ori)
        out_flip = fdi_baseline.fdi_smooth(p_flip, budget, 0)
        assert np.array_equal(out0, invert_orientation(out_flip, ori))


def test_fdi_recovers_linear_field_exactly_as_lambda_tends_to_zero():
    """T8 acceptance test: FDI recovers a 3D affine field across the full grid."""
    shape = (10, 9, 8)
    origin = np.array([-20.0, 15.0, -50.0])
    spacing = np.array([4.0, 5.0, 3.0])
    rng = np.random.default_rng(42)
    node_min = origin + 0.5 * spacing
    node_max = origin + (np.asarray(shape, dtype=float) - 0.5) * spacing
    pts = rng.uniform(node_min, node_max, size=(60, 3))

    def linear_fn(xyz):
        return 1.75 + 0.22 * xyz[:, 0] - 0.15 * xyz[:, 1] + 0.41 * xyz[:, 2]

    vals = linear_fn(pts)
    weights = np.ones(len(pts), dtype=float)
    axes = [origin[i] + (np.arange(n, dtype=float) + 0.5) * spacing[i] for i, n in enumerate(shape)]
    grid_pts = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    expected = linear_fn(grid_pts)
    for lam in (0.1, 1e-3, 1e-4):
        u, _, _, _ = fdi_baseline.fit_fdi_grid(
            pts, vals, weights, origin, spacing, shape, s_z=3.0, lam=lam, rtol=1e-12
        )
        assert np.max(np.abs(u - expected)) < 5e-5


def test_fdi_smooth_recovers_folded_and_lens_bodies():
    for maker, floor in ((_make_folded_layer, 0.70), (_make_ellipsoid_lens, 0.55)):
        prob, truth = maker()
        diag = {}
        out = fdi_baseline.fdi_smooth(prob, {"wall_time_seconds": 30.0}, 0, diagnostics=diag)
        canonicalize(prob, out)
        assert diag["fallback"] is False
        assert compare(prob, out, truth)["iou"] >= floor


def test_fdi_in_container_imports_allowed_only():
    allowed = {"time", "numpy", "scipy", "skfmm", "tasks"}
    mod_path = Path(fdi_baseline.__file__)
    tree = ast.parse(mod_path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] in allowed, f"disallowed import {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            if node.level > 0:
                continue
            assert node.module.split(".")[0] in allowed, f"disallowed import {node.module}"


def test_gempy_contact_classification_and_plane_poles():
    prob, _ = _make_folded_layer()
    top_l, tg_l, bot_l, bg_l = gempy_reference.classify_contacts(prob, "L")
    assert len(top_l) >= 8 and len(bot_l) >= 8
    # Top contacts lie strictly above bottom contacts in every shared hole.
    for gid in np.intersect1d(tg_l, bg_l):
        assert float(top_l[tg_l == gid, 2].mean()) > float(bot_l[bg_l == gid, 2].mean())
    poles = gempy_reference.estimate_poles(top_l, tg_l, "plane")
    assert poles.shape == (len(top_l), 3)
    norms = np.linalg.norm(poles, axis=1)
    assert np.allclose(norms, 1.0)
    assert np.all(poles[:, 2] > 0.0)


def test_offline_stage_b_loaders_and_cli(tmp_path, capsys):
    p_slab, t_slab = _make_flat_slab()
    data = harness.extract_public_data(p_slab)
    release = tmp_path / "release"
    save_problem(p_slab, release / "calibration" / "public" / p_slab["case_id"])
    tdir = release / "calibration" / "private" / p_slab["case_id"]
    tdir.mkdir(parents=True, exist_ok=True)
    np.save(tdir / "truth.npy", t_slab, allow_pickle=False)
    (release / "calibration" / "manifest.json").write_text(json.dumps({
        "schema": "copper-suite-v2",
        "stage": "development",
        "split": "calibration",
        "seeds": [0],
        "cases": [{
            "case_id": p_slab["case_id"],
            "parent_id": "p-slab",
            "family": "folded",
            "modality": "drill",
            "public": f"public/{p_slab['case_id']}",
            "truth": f"private/{p_slab['case_id']}",
        }],
    }))

    # Missing fields_dir must raise clearly.
    with pytest.raises(ValueError, match="fields_dir"):
        offline_baselines.potential_field(p_slab, {"wall_time_seconds": 10.0}, 0)

    # M4 Stage-A mock output (including native variant).
    m4_dir = tmp_path / "m4_fields" / "calibration" / p_slab["case_id"]
    m4_dir.mkdir(parents=True)
    np.save(m4_dir / "ore.npy", t_slab, allow_pickle=False)
    (m4_dir / "choice.json").write_text(json.dumps({
        "method": "potential_field",
        "generator_matched": True,
        "case_id": p_slab["case_id"],
        "no_ore": False,
        "fallback": False,
        "cv": {"selected_rule": "native", "selected_config": {"form": "L", "orient": "plane", "range": 1.7}, "cv_score": 0.9, "cv_evaluated": 12},
        "native": {"selected_rule": "native", "selected_config": {"form": "L", "orient": "plane", "range": 1.7}, "cv_score": 0.9, "cv_evaluated": 12, "same_as_cv": True},
        "wall_seconds": 3.25,
    }))
    assert baselines.main([
        "--release", str(release), "--split", "calibration", "--solver", "potential_field",
        "--fields-dir", str(tmp_path / "m4_fields"), "--json",
    ]) == 0
    rep_m4 = json.loads(capsys.readouterr().out)
    assert rep_m4["generator_matched"] is True
    assert rep_m4["overall"]["mean_iou"] > 0.95
    assert rep_m4["median_case_seconds"] >= 3.25

    # M7 & RF Stage-A mock per-seed outputs.
    for method, tau_shift in (("inr_mlp", 0.0), ("rf_xyz", 0.5)):
        sdir = tmp_path / f"{method}_fields" / "calibration" / p_slab["case_id"] / "2"
        sdir.mkdir(parents=True)
        f_feasible = np.where(t_slab[data["feasible"]], 0.8, -0.8) + tau_shift
        np.save(sdir / "field.npy", f_feasible.astype(np.float32), allow_pickle=False)
        (sdir / "choice.json").write_text(json.dumps({
            "method": method,
            "case_id": p_slab["case_id"],
            "seed": 2,
            "no_ore": False,
            "fallback": False,
            "cv": {"selected_rule": "share", "selected_config": {"s_z": 4}, "cv_score": 0.85, "cv_evaluated": 6},
            "native": {"selected_rule": "native", "selected_config": {"s_z": 4}, "cv_score": 0.82, "cv_evaluated": 6, "same_as_cv": True},
            "wall_seconds": 1.5,
        }))
        assert baselines.main([
            "--release", str(release), "--split", "calibration", "--solver", method,
            "--fields-dir", str(tmp_path / f"{method}_fields"), "--seed", "2", "--json",
        ]) == 0
        rep = json.loads(capsys.readouterr().out)
        assert rep["seed"] == 2
        assert rep["overall"]["mean_iou"] > 0.78
