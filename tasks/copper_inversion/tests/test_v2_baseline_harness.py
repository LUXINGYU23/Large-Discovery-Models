"""Unit tests for the shared published-baseline harness (v2/baseline_harness.py)."""

import time
import numpy as np

from tasks.copper_inversion.v2 import baseline_harness as harness
from tasks.copper_inversion.v2.data import canonical_problem, invert_orientation
from tasks.copper_inversion.v2.fixtures import fixtures
from tasks.copper_inversion.v2.geometry import canonicalize
from tasks.copper_inversion.tests.test_v2_baselines import _make_flat_slab, _synthetic_problem


def test_fold_assignment_outcrop_in_train_and_contacts_follow_hole():
    problem, _ = _make_flat_slab()
    # Add an outcrop modality that must stay in training for every fold.
    origin = np.asarray(problem["origin"], dtype=float)
    spacing = np.asarray(problem["spacing"], dtype=float)
    outcrop_pts = origin + (np.array([[2, 2, 18], [8, 8, 18], [13, 13, 18]], dtype=float) + 0.5) * spacing
    problem = {
        **problem,
        "observations": list(problem["observations"]) + [{
            "name": "outcrop",
            "points": outcrop_pts,
            "labels": np.array([0.0, 1.0, 0.0]),
            "groups": np.array([0, 1, 2], dtype=np.int64),
            "weight": 0.25,
            "transition_scale": 10.0,
            "contact_sigma": 5.0,
            "label_error": 0.0,
        }],
    }
    data = harness.extract_public_data(problem)
    folds = harness.make_hole_folds(data, problem)
    assert len(folds) == 6
    seen_val_holes = set()
    for f in folds:
        vh = set(int(h) for h in f["val_holes"])
        th = set(int(h) for h in f["tr_holes"])
        assert vh.isdisjoint(th)
        seen_val_holes |= vh
        # Outcrop weight 0.25 samples (3 of them) must all appear in every training fold.
        assert int(np.count_nonzero(f["w_b_tr"] == 0.25)) == 3
        # Held-out drill hole IDs must not appear in problem_tr drill observations (binary or contact).
        tr_drill_groups = set(int(g) for g in f["problem_tr"]["observations"][0]["groups"])
        assert tr_drill_groups.isdisjoint(vh)
        assert 0.02 <= f["p_train"] <= 0.65
    assert seen_val_holes == set(int(h) for h in data["hole_ids"])


def test_symmetric_subsample_and_rule_selection_on_slab():
    problem, truth = _make_flat_slab()
    data = harness.extract_public_data(problem)

    # Subsample S must be invariant under horizontal mirror flips.
    for ori in (4, 6):
        p_flip = canonical_problem(problem, canonical_origin=False, orientation=ori)
        d_flip = harness.extract_public_data(p_flip)
        assert len(d_flip["S"]) == len(data["S"])

    # Synthetic predictor: shifted by a constant bias (-10) so native (f > 0) predicts all barren
    # while share rule recovers the slab via quantile cut.
    configs = [{"bias": -10.0}, {"bias": 0.0}]
    z_lo = problem["origin"][2] + 7.0 * problem["spacing"][2]
    z_hi = problem["origin"][2] + 12.0 * problem["spacing"][2]
    x_lo = problem["origin"][0] + 4.0 * problem["spacing"][0]
    x_hi = problem["origin"][0] + 20.0 * problem["spacing"][0]
    y_lo = problem["origin"][1] + 4.0 * problem["spacing"][1]
    y_hi = problem["origin"][1] + 20.0 * problem["spacing"][1]

    def fit(cfg, X_b, y_b, w_b, X_c, prob):
        b = cfg["bias"]
        def predict(pts):
            inside = (
                (pts[:, 2] >= z_lo) & (pts[:, 2] <= z_hi)
                & (pts[:, 0] >= x_lo) & (pts[:, 0] <= x_hi)
                & (pts[:, 1] >= y_lo) & (pts[:, 1] <= y_hi)
            )
            return np.where(inside, 1.0, -1.0) + b
        return predict

    diag = {}
    out = harness.tune_and_solve(problem, {"wall_time_seconds": 10.0}, configs, fit, tau0=0.0, diagnostics=diag)
    canonicalize(problem, out)
    # Earlier config (bias=-10) with share ties/beats bias=0; either way share works on biased field.
    diag_bias = {}
    out_bias = harness.tune_and_solve(
        problem, {"wall_time_seconds": 10.0}, [{"bias": -10.0}], fit, tau0=0.0, diagnostics=diag_bias
    )
    canonicalize(problem, out_bias)
    assert diag_bias["selected_rule"] == "share"
    assert diag_bias["cv_score"] > 0.7

    # Forcing native rule on unbiased predictor selects native and needs no repair.
    diag_nat = {}
    out_nat = harness.tune_and_solve(
        problem, {"wall_time_seconds": 10.0}, [{"bias": 0.0}], fit, tau0=0.0,
        decision_rule="native", diagnostics=diag_nat,
    )
    canonicalize(problem, out_nat)
    assert diag_nat["selected_rule"] == "native"
    assert diag_nat["repaired"] is False


def test_deadline_honoured_and_fallback_counted():
    problem, _ = _make_flat_slab()
    configs = [{"id": i} for i in range(24)]

    def slow_fit(cfg, X_b, y_b, w_b, X_c, prob):
        time.sleep(0.08)
        return lambda pts: np.zeros(len(pts), dtype=float)

    diag = {}
    t0 = time.monotonic()
    out = harness.tune_and_solve(problem, {"wall_time_seconds": 0.5}, configs, slow_fit, tau0=0.0, diagnostics=diag)
    elapsed = time.monotonic() - t0
    canonicalize(problem, out)
    assert diag["deadline_stopped"] is True
    assert diag["cv_evaluated"] < len(configs)
    assert elapsed < 2.0

    # All configs fail -> fallback to stretched_nearest_class and fallback flag set.
    def broken_fit(cfg, X_b, y_b, w_b, X_c, prob):
        raise np.linalg.LinAlgError("singular")

    diag_fb = {}
    out_fb = harness.tune_and_solve(problem, {"wall_time_seconds": 5.0}, configs[:2], broken_fit, tau0=0.0, diagnostics=diag_fb)
    canonicalize(problem, out_fb)
    assert diag_fb["fallback"] is True


def test_harness_legal_on_all_interface_fixtures_and_equivariant():
    configs = [{"s_z": 2}]

    def simple_fit(cfg, X_b, y_b, w_b, X_c, prob):
        center = np.asarray(prob["origin"], dtype=float) + 0.5 * np.asarray(prob["shape"], dtype=float) * np.asarray(prob["spacing"], dtype=float)
        ore = X_b[y_b > 0.5]
        Xs = harness.stretch_points(ore, center, cfg["s_z"])
        def predict(pts):
            ps = harness.stretch_points(pts, center, cfg["s_z"])
            d2 = np.sum((ps[:, None, :] - Xs[None, :, :]) ** 2, axis=-1)
            return np.sum(np.exp(-d2 / (2.0 * 40.0 ** 2)), axis=1)
        return predict

    for prob in fixtures():
        t0 = time.monotonic()
        out = harness.tune_and_solve(prob, {"wall_time_seconds": 5.0}, configs, simple_fit, tau0=0.5)
        assert time.monotonic() - t0 < 5.0
        canonicalize(prob, out)

    # Irregular hole layout (like real drill holes) with smooth field gives exact D4 equivariance.
    shape = (20, 20, 16)
    ix, iy, iz = np.indices(shape, dtype=float)
    truth = (iz >= 5) & (iz <= 9) & (ix >= 3) & (ix <= 15) & (iy >= 4) & (iy <= 16)
    holes = [(2, 3), (4, 14), (7, 6), (9, 11), (11, 4), (13, 15), (16, 8), (17, 13), (5, 10), (14, 5)]
    p_irr, _ = _synthetic_problem(truth, holes, shape=shape, case_id="irr-equiv")
    out0 = harness.tune_and_solve(p_irr, {"wall_time_seconds": 10.0}, configs, simple_fit, tau0=0.5)
    for ori in (4, 6):
        p_flip = canonical_problem(p_irr, canonical_origin=False, orientation=ori)
        out_flip = harness.tune_and_solve(p_flip, {"wall_time_seconds": 10.0}, configs, simple_fit, tau0=0.5)
        assert np.array_equal(out0, invert_orientation(out_flip, ori))


def test_no_ore_and_one_hole_edge_cases():
    shape = (12, 12, 10)
    p_empty, _ = _synthetic_problem(np.zeros(shape, dtype=bool), [(2, 2), (6, 6)], shape=shape, case_id="no-ore")
    diag = {}
    out = harness.tune_and_solve(p_empty, {"wall_time_seconds": 5.0}, [{"s_z": 1}], lambda *a: None, diagnostics=diag)
    canonicalize(p_empty, out)
    assert diag["no_ore"] is True and diag["repaired"] is True

    # Single hole skips CV without error.
    truth = np.zeros(shape, dtype=bool)
    truth[:, :, 3:6] = True
    p_one, _ = _synthetic_problem(truth, [(5, 5)], shape=shape, case_id="one-hole")
    data = harness.extract_public_data(p_one)
    assert harness.make_hole_folds(data, p_one) == []
