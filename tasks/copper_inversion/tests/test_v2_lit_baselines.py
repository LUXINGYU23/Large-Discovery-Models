"""Unit and acceptance tests for Engineer A's published baselines (M1, M2, M3, M6, M8-kNN)."""

import ast
import json
import time
from pathlib import Path
import numpy as np
import pytest
from scipy.spatial.distance import cdist
from scipy.special import expit

from tasks.copper_inversion.core.runner import input_digest
from tasks.copper_inversion.v2 import baseline_harness as harness
from tasks.copper_inversion.v2 import baselines, lit_baselines
from tasks.copper_inversion.v2.contract import validate
from tasks.copper_inversion.v2.data import canonical_problem, invert_orientation, save_problem
from tasks.copper_inversion.v2.fixtures import fixtures
from tasks.copper_inversion.v2.geological_policy import public_priors
from tasks.copper_inversion.v2.geometry import canonicalize
from tasks.copper_inversion.v2.metrics import compare
from tasks.copper_inversion.tests.test_v2_baselines import (
    _make_ellipsoid_lens,
    _make_flat_slab,
    _make_folded_layer,
    _synthetic_problem,
)

LIT_SOLVERS = tuple(lit_baselines.LIT_REFERENCE_SOLVERS[k] for k in lit_baselines.LIT_SOLVER_NAMES)
LIT_ALL = tuple(lit_baselines.LIT_REFERENCE_SOLVERS.values())


def _irregular_lens_problem(shape=(18, 18, 14)):
    ix, iy, iz = np.indices(shape, dtype=float)
    truth = ((ix - 8.5) / 5.5) ** 2 + ((iy - 9.0) / 5.0) ** 2 + ((iz - 6.5) / 2.5) ** 2 <= 1.0
    holes = [
        (2, 3), (3, 12), (5, 7), (7, 14), (8, 4), (9, 9),
        (11, 6), (12, 13), (14, 3), (15, 10), (6, 11), (13, 8),
    ]
    return _synthetic_problem(truth, holes, shape=shape, case_id="irr-lens")


@pytest.mark.parametrize("solver", LIT_ALL, ids=[f.__name__ for f in LIT_ALL])
def test_lit_solvers_interface_legality_determinism_and_5s_budget(solver):
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


@pytest.mark.parametrize("solver", LIT_SOLVERS, ids=[f.__name__ for f in LIT_SOLVERS])
def test_lit_solvers_degenerate_inputs(solver):
    shape = (12, 12, 10)
    holes = [(2, 2), (6, 6), (9, 9)]
    budget = {"wall_time_seconds": 10.0}

    # 1. No ore sample
    p_empty, _ = _synthetic_problem(np.zeros(shape, dtype=bool), holes, shape=shape, case_id="no-ore")
    canonicalize(p_empty, solver(p_empty, budget, 0))

    # 2. All holes one class (all ore)
    p_full, _ = _synthetic_problem(np.ones(shape, dtype=bool), holes, shape=shape, case_id="all-ore")
    canonicalize(p_full, solver(p_full, budget, 0))

    # 3. Single hole
    slab = np.zeros(shape, dtype=bool)
    slab[:, :, 3:6] = True
    p_one, _ = _synthetic_problem(slab, [(5, 5)], shape=shape, case_id="one-hole")
    canonicalize(p_one, solver(p_one, budget, 0))

    # 4. Outcrop-only ore (drill holes all barren, outcrop has ore)
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
    canonicalize(p_out, solver(p_out, budget, 0))


@pytest.mark.parametrize("solver", LIT_SOLVERS, ids=[f.__name__ for f in LIT_SOLVERS])
def test_lit_solvers_mirror_equivariance(solver):
    prob, _ = _irregular_lens_problem()
    # Add tiny irrational jitter to hole coordinates so kNN/IDW has no artificial lattice distance ties.
    rng = np.random.default_rng(7)
    drill = prob["observations"][0]
    pts = drill["points"].copy()
    for gid in np.unique(drill["groups"]):
        sel = drill["groups"] == gid
        pts[sel, :2] += rng.uniform(-0.3, 0.3, size=2)
        pts[sel, 2] += rng.uniform(0.05, 0.35, size=int(sel.sum()))
    prob = validate({
        **prob,
        "observations": [{**drill, "points": pts}],
    })
    budget = {"wall_time_seconds": 30.0}
    out0 = solver(prob, budget, 0)
    for ori in (4, 6):  # x-flip and y-flip
        p_flip = canonical_problem(prob, canonical_origin=False, orientation=ori)
        out_flip = solver(p_flip, budget, 0)
        assert np.array_equal(out0, invert_orientation(out_flip, ori)), f"{solver.__name__} failed ori={ori}"


@pytest.mark.parametrize("solver", LIT_SOLVERS, ids=[f.__name__ for f in LIT_SOLVERS])
def test_lit_solvers_recover_synthetic_lens(solver):
    prob, truth = _make_ellipsoid_lens()
    diag = {}
    out = solver(prob, {"wall_time_seconds": 30.0}, 0, diagnostics=diag)
    canonicalize(prob, out)
    stats = compare(prob, out, truth)
    assert diag["fallback"] is False
    assert stats["iou"] >= 0.55


def test_indicator_kriging_reproduces_data_when_nugget_zero():
    prob, _ = _irregular_lens_problem()
    data = harness.extract_public_data(prob)
    cfg = {"s_z": 4, "rho": "exponential", "nu": 0.0}
    pred = lit_baselines.fit_indicator_kriging(
        cfg, data["X_b"], data["y_b"], data["w_b"], data["X_c"], prob, eps=0.0
    )
    f_b = pred(data["X_b"])
    assert np.max(np.abs(f_b - data["y_b"])) < 1e-5
    if len(data["X_c"]):
        f_c = pred(data["X_c"])
        assert np.max(np.abs(f_c - 0.5)) < 1e-5


def test_svm_kkt_residual_on_synthetic_problems():
    for prob, _ in (_make_flat_slab(), _make_folded_layer(), _make_ellipsoid_lens()):
        data = harness.extract_public_data(prob)
        X_s = harness.stretch_points(data["X_b"], data["center"], 4)
        for C in (1.0, 100.0):
            alpha, Q, upper, _ = lit_baselines.solve_svm_dual(X_s, data["y_b"], data["w_b"], data["h_hole"], C)
            assert lit_baselines.svm_kkt_residual(Q, alpha, upper) < 1e-3


def test_gp_evidence_prefers_generating_length_scale():
    rng = np.random.default_rng(1234)
    shape = (18, 18, 14)
    spacing = np.array([10.0, 10.0, 5.0])
    origin = np.zeros(3)
    E = np.asarray(shape, dtype=float) * spacing
    center = origin + 0.5 * E
    # Build 16 drill holes with 10 samples each = 160 points, sampled from Matérn-3/2 with ell_true = h_hole.
    xs = rng.uniform(10.0, E[0] - 10.0, size=16)
    ys = rng.uniform(10.0, E[1] - 10.0, size=16)
    zs = np.linspace(5.0, E[2] - 10.0, 10)
    pts, grps = [], []
    for gid, (x, y) in enumerate(zip(xs, ys)):
        for z in zs:
            pts.append([x, y, z])
            grps.append(gid)
    pts = np.asarray(pts, dtype=float)
    grps = np.asarray(grps, dtype=np.int64)
    h_hole = float(np.sqrt(E[0] * E[1] / 16.0))
    s_z_true = 4.0
    ell_true = 1.0 * h_hole
    X_s = harness.stretch_points(pts, center, s_z_true)
    K_true = lit_baselines._matern32(cdist(X_s, X_s), ell_true, 4.0)
    K_true[np.diag_indices(len(X_s))] += 1e-6
    g_true = rng.multivariate_normal(np.full(len(X_s), -0.5), K_true)
    y_bin = (rng.random(len(X_s)) < expit(g_true)).astype(float)

    domain = np.ones(shape, dtype=bool)
    forbidden = np.zeros(shape, dtype=bool)
    forbidden[:, :, -1] = True
    prob = validate({
        "schema_version": "problem_schema_v2",
        "geometry_version": "occupancy_v2",
        "objective_version": "objective_v2",
        "case_id": "gp-synth",
        "shape": list(shape),
        "origin": origin.tolist(),
        "spacing": spacing.tolist(),
        "domain_mask": domain,
        "forbidden_mask": forbidden,
        "open_boundary_faces": ["x-", "x+", "y-", "y+", "z-"],
        "observations": [{
            "name": "drill",
            "points": pts,
            "labels": y_bin,
            "groups": grps,
            "weight": 1.0,
            "transition_scale": float(np.linalg.norm(spacing)),
            "contact_sigma": float(spacing[2]),
            "label_error": 0.0,
        }],
        "priors": public_priors(domain & ~forbidden, spacing),
    })
    w_b = np.ones(len(y_bin), dtype=float)
    X_c = np.zeros((0, 3), dtype=float)
    ev = {}
    for mult in (0.1, 1.0, 10.0):
        _, log_q = lit_baselines.fit_gp_laplace(
            {"s_z": 4, "ell_mult": mult, "sigma_f2": 4.0}, pts, y_bin, w_b, X_c, prob,
            center=center, h_hole=h_hole,
        )
        ev[mult] = log_q
    assert ev[1.0] > ev[0.1] and ev[1.0] > ev[10.0]


def test_in_container_imports_allowed_only():
    allowed = {"time", "numpy", "scipy", "skfmm", "tasks"}
    for mod_path in (Path(harness.__file__), Path(lit_baselines.__file__)):
        tree = ast.parse(mod_path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split(".")[0] in allowed, f"disallowed import {alias.name}"
            elif isinstance(node, ast.ImportFrom):
                if node.level > 0:
                    continue
                assert node.module.split(".")[0] in allowed, f"disallowed import {node.module}"


def test_baselines_cli_runs_lit_solvers_and_reports_repair_rate(tmp_path, capsys):
    release = tmp_path / "release"
    p_lens, t_lens = _make_ellipsoid_lens()
    save_problem(p_lens, release / "calibration" / "public" / "c-lens")
    tdir = release / "calibration" / "private" / "c-lens"
    tdir.mkdir(parents=True, exist_ok=True)
    np.save(tdir / "truth.npy", t_lens, allow_pickle=False)
    (release / "calibration" / "manifest.json").write_text(json.dumps({
        "schema": "copper-suite-v2",
        "stage": "development",
        "split": "calibration",
        "seeds": [0],
        "cases": [{
            "case_id": "c-lens",
            "parent_id": "p-lens",
            "family": "lens",
            "modality": "drill",
            "public": "public/c-lens",
            "truth": "private/c-lens",
        }],
    }))
    assert baselines.main(["--release", str(release), "--split", "calibration", "--solver", "rbf_sdf", "--json"]) == 0
    rep = json.loads(capsys.readouterr().out)
    assert rep["solver"] == "rbf_sdf"
    assert rep["repair_rate"] is not None
    assert rep["fallback_count"] == 0
    assert rep["overall"]["mean_iou"] >= 0.55

    assert baselines.main(["--release", str(release), "--split", "calibration", "--solver", "knn_idw_native", "--json"]) == 0
    rep_nat = json.loads(capsys.readouterr().out)
    assert rep_nat["rule_counts"] == {"native": 1}
