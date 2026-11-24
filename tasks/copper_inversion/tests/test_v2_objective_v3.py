"""Unit tests for objective_v3 (noise-corrected line Jaccard) and legacy v2 compatibility."""

import copy
import json
from pathlib import Path
import numpy as np
import pytest

from tasks.copper_inversion.rsi.audit import IOU
from tasks.copper_inversion.rsi.exposure import risk_by_case
from tasks.copper_inversion.v2.baselines import initialize
from tasks.copper_inversion.v2.contract import (
    LEGACY_OBJECTIVE, OBJECTIVE, OBJECTIVE_VERSIONS, validate,
)
from tasks.copper_inversion.v2.data import read_problem, save_problem
from tasks.copper_inversion.v2.fixtures import fixtures
from tasks.copper_inversion.v2.geological_policy import public_priors
from tasks.copper_inversion.v2.objective import K_TOL, score
from tasks.copper_inversion.v2.research import PublicTrainingExport
from tasks.copper_inversion.v2.scorer import score_case
from tasks.copper_inversion.v2.suite import (
    AGGREGATIONS, LEGACY_CONFIRMATION_AGGREGATION, aggregate, confirmation_utility,
)
from tasks.copper_inversion.tests.test_v2_science import fixture as legacy_fixture, model as legacy_model


def make_v3_problem(shape=(16, 16, 16), *, noisy=False, with_outcrop=True, hole_slice=(0, 36)):
    origin = np.array([0., 0., 0.])
    spacing = np.array([2., 2., 2.])
    domain = np.ones(shape, dtype=bool)
    forbidden = np.zeros(shape, dtype=bool)
    forbidden[:, :, -1] = True
    feasible = domain & ~forbidden

    truth = np.zeros(shape, dtype=bool)
    truth[4:12, 4:12, 4:10] = True
    truth &= feasible

    rng = np.random.default_rng(17)
    all_cells = [(ix, iy) for ix in range(1, shape[0] - 1) for iy in range(1, shape[1] - 1)]
    rng.shuffle(all_cells)
    holes_xy = [(ix * 2. + 1., iy * 2. + 1.) for ix, iy in all_cells[hole_slice[0]:hole_slice[1]]]
    pts, labels, groups = [], [], []
    for g, (hx, hy) in enumerate(holes_xy):
        in_xy = 8. <= hx <= 24. and 8. <= hy <= 24.
        for kz in range(shape[2] - 1):
            zc = kz * 2. + 1.
            in_ore = in_xy and (8. <= zc <= 20.)
            pts.append([hx, hy, zc])
            labels.append(1. if in_ore else 0.)
            groups.append(g)
        if in_xy:
            pts.append([hx, hy, 8.])
            labels.append(.5)
            groups.append(g)
            pts.append([hx, hy, 20.])
            labels.append(.5)
            groups.append(g)

    disc = .5 * float(np.linalg.norm(spacing))
    meas = .25 * float(spacing.min()) if noisy else 0.
    eff = float(np.hypot(meas, disc))
    eps = .02 if noisy else 0.
    drill = dict(
        name="drill",
        points=np.asarray(pts, dtype=float),
        labels=np.asarray(labels, dtype=float),
        groups=np.asarray(groups, dtype=int),
        weight=1.,
        transition_scale=eff,
        contact_sigma=eff,
        label_error=eps,
    )
    observations = [drill]
    if with_outcrop:
        out_pts = np.array([[x * 2. + 1., y * 2. + 1., (shape[2] - 2) * 2. + 1.]
                            for x in range(1, shape[0] - 1, 2) for y in range(1, shape[1] - 1, 2)], dtype=float)
        out_labels = np.zeros(len(out_pts), dtype=float)
        out_groups = np.arange(len(out_pts), dtype=int) % 16
        observations.append(dict(
            name="outcrop", points=out_pts, labels=out_labels, groups=out_groups,
            weight=.25, transition_scale=eff, contact_sigma=eff, label_error=eps,
        ))
    problem = dict(
        schema_version="problem_schema_v2",
        geometry_version="occupancy_v2",
        objective_version=OBJECTIVE,
        case_id="synthetic-v3",
        shape=list(shape),
        origin=origin.tolist(),
        spacing=spacing.tolist(),
        domain_mask=domain,
        forbidden_mask=forbidden,
        open_boundary_faces=["x-", "x+", "y-", "y+", "z-"],
        observations=observations,
        priors=public_priors(feasible, spacing),
    )
    return validate(problem, legacy=False), truth


def test_contract_versions_and_legacy_gate():
    assert OBJECTIVE == "objective_v3"
    assert LEGACY_OBJECTIVE == "objective_v2"
    assert OBJECTIVE_VERSIONS == ("objective_v3", "objective_v2")
    p3, _ = make_v3_problem()
    assert validate(p3, legacy=False) is p3
    p2 = copy.deepcopy(p3)
    p2["objective_version"] = LEGACY_OBJECTIVE
    assert validate(p2, legacy=True) is p2
    with pytest.raises(ValueError, match="unsupported objective_version"):
        validate(p2, legacy=False)
    p1 = copy.deepcopy(p3)
    p1["objective_version"] = "objective_v1"
    with pytest.raises(ValueError, match="unsupported objective_version"):
        validate(p1, legacy=True)
    for prob in fixtures():
        assert prob["objective_version"] == "objective_v3"
        validate(prob, legacy=False)


def test_truth_scores_best_against_perturbations():
    public_problem, _ = make_v3_problem(hole_slice=(0, 10))
    problem, truth = make_v3_problem(hole_slice=(10, 50))
    feasible = problem["domain_mask"] & ~problem["forbidden_mask"]
    truth_field = np.where(truth, 1., -1.)
    truth_res = score(problem, truth_field)
    assert truth_res["objective_version"] == "objective_v3"
    assert truth_res["loss_total"] == truth_res["loss_jaccard"] == pytest.approx(0., abs=1e-12)
    assert truth_res["fp"] == 0. and truth_res["fn"] == 0. and truth_res["tp"] > 0.

    shifted = np.roll(truth, 2, axis=0) & feasible
    dilated = truth.copy()
    dilated[2:14, 2:14, 2:12] = True
    dilated &= feasible
    eroded = np.zeros_like(truth)
    eroded[5:11, 5:11, 4:10] = True
    eroded &= feasible
    init_field = initialize(public_problem)

    for candidate in (np.where(shifted, 1., -1.), np.where(dilated, 1., -1.),
                      np.where(eroded, 1., -1.), init_field):
        res = score(problem, candidate)
        assert res["loss_total"] > truth_res["loss_total"]


def test_sponge_and_sheet_overlays_score_worse_than_honest_output():
    problem, truth = make_v3_problem()
    feasible = problem["domain_mask"] & ~problem["forbidden_mask"]
    honest_mask = np.zeros_like(truth)
    honest_mask[3:11, 5:13, 4:10] = True
    honest_mask &= feasible
    honest_field = np.where(honest_mask, 1., -1.)
    honest_v3 = score(problem, honest_field)["loss_total"]

    i, j, k = np.indices(truth.shape)
    pillars = (i % 6 == 0) & (j % 6 == 0)
    overlays = {
        "sheets_h_p3": (honest_mask | (k % 3 == 0) | pillars) & feasible,
        "sheets_h_p4": (honest_mask | (k % 4 == 0) | pillars) & feasible,
        "sheets_h_p6": (honest_mask | (k % 6 == 0) | pillars) & feasible,
        "xor_sheets_p2": ((honest_mask ^ (k % 2 == 0)) | pillars) & feasible,
        "sheets_v_p4": (honest_mask | (i % 4 == 0) | (j % 8 == 0)) & feasible,
    }
    for name, mask in overlays.items():
        overlay_loss = score(problem, np.where(mask, 1., -1.))["loss_total"]
        assert overlay_loss > honest_v3, f"{name} did not score worse than honest output"


def test_noise_correction_unbiased_on_synthetic_labels():
    shape = (10, 10, 10)
    forbidden = np.zeros(shape, dtype=bool)
    forbidden[:, :, -1] = True
    pts = np.array([[i + .5, j + .5, k + .5]
                    for i in range(10) for j in range(10) for k in range(8)], dtype=float)
    true_y = ((pts[:, 0] < 5.) & (pts[:, 1] < 6.) & (pts[:, 2] < 5.)).astype(float)
    pred_mask = np.zeros(shape, dtype=bool)
    pred_mask[:6, :5, :6] = True
    pred_field = np.where(pred_mask, 1., -1.)

    idx = np.rint(pts - .5).astype(int)
    p = pred_mask[tuple(idx.T)]
    true_a = float(np.sum(p & (true_y > .5)))
    true_c = float(np.sum((~p) & (true_y > .5)))
    true_tp, true_fp, true_fn = true_a, float(np.sum(p)) - true_a, true_c
    true_j = 1. - (true_tp + 1.) / (true_tp + true_fp + true_fn + 1.)

    eps = .08
    rng = np.random.default_rng(20261124)
    est_tp, est_fn, est_j, raw_j = [], [], [], []
    base = dict(
        schema_version="problem_schema_v2", geometry_version="occupancy_v2",
        objective_version=OBJECTIVE, case_id="noise-check", shape=list(shape),
        origin=[0., 0., 0.], spacing=[1., 1., 1.],
        domain_mask=np.ones(shape, dtype=bool), forbidden_mask=forbidden,
        open_boundary_faces=["x-", "x+", "y-", "y+", "z-"],
        priors=public_priors(~forbidden, [1., 1., 1.]),
    )
    for _ in range(400):
        noisy_y = np.where(rng.random(len(true_y)) < eps, 1. - true_y, true_y)
        prob_corr = {**base, "observations": [dict(
            name="drill", points=pts, labels=noisy_y, groups=np.zeros(len(pts), dtype=int),
            weight=1., transition_scale=.5 * np.sqrt(3.), contact_sigma=1., label_error=eps,
        )]}
        prob_raw = {**base, "observations": [dict(
            name="drill", points=pts, labels=noisy_y, groups=np.zeros(len(pts), dtype=int),
            weight=1., transition_scale=.5 * np.sqrt(3.), contact_sigma=1., label_error=0.,
        )]}
        r_corr = score(prob_corr, pred_field)
        r_raw = score(prob_raw, pred_field)
        est_tp.append(r_corr["tp"])
        est_fn.append(r_corr["fn"])
        est_j.append(r_corr["loss_jaccard"])
        raw_j.append(r_raw["loss_jaccard"])

    assert abs(float(np.mean(est_tp)) - true_tp) < 1.5
    assert abs(float(np.mean(est_fn)) - true_fn) < 1.5
    assert abs(float(np.mean(est_j)) - true_j) < .005
    assert abs(float(np.mean(raw_j)) - true_j) > .05


def test_contact_band_excludes_exact_same_hole_samples():
    shape = (8, 8, 10)
    forbidden = np.zeros(shape, dtype=bool)
    forbidden[:, :, -1] = True
    spacing = np.array([2., 2., 4.])
    disc = .5 * float(np.linalg.norm(spacing))
    sigma_meas = 1.5
    eff = float(np.hypot(sigma_meas, disc))
    # delta = hypot(0.5 * 4.0, 1.5) = hypot(2.0, 1.5) = 2.5
    expected_delta = 2.5
    assert np.hypot(K_TOL * spacing[2], sigma_meas) == pytest.approx(expected_delta)

    pts = np.array([
        [3., 3., 16.],       # contact in hole 0
        [3., 3., 16. + 2.4], # hole 0, d = 2.4 < 2.5 -> EXCLUDED (wrong label planted!)
        [3., 3., 16. - 2.6], # hole 0, d = 2.6 > 2.5 -> KEPT (true positive)
        [7., 3., 16. + 0.2], # hole 1 (no contact in hole 1!), close in z -> KEPT!
        [11., 11., 6.],      # hole 2, barren -> KEPT
    ], dtype=float)
    labels = np.array([.5, 0., 1., 1., 0.], dtype=float)
    groups = np.array([0, 0, 0, 1, 2], dtype=int)
    problem = dict(
        schema_version="problem_schema_v2", geometry_version="occupancy_v2",
        objective_version=OBJECTIVE, case_id="band-check", shape=list(shape),
        origin=[0., 0., 0.], spacing=spacing.tolist(),
        domain_mask=np.ones(shape, dtype=bool), forbidden_mask=forbidden,
        open_boundary_faces=["x-", "x+", "y-", "y+", "z-"],
        observations=[dict(
            name="drill", points=pts, labels=labels, groups=groups,
            weight=1., transition_scale=eff, contact_sigma=eff, label_error=0.,
        )],
        priors=public_priors(~forbidden, spacing),
    )
    mask = np.zeros(shape, dtype=bool)
    mask[:5, :5, 2:6] = True
    field = np.where(mask, 1., -1.)

    with_band = score(problem, field)
    assert with_band["drill_kept_samples"] == 3
    assert (with_band["tp"], with_band["fp"], with_band["fn"]) == (2., 0., 0.)
    assert with_band["loss_jaccard"] == pytest.approx(0.)

    # Without the K_TOL band AND with clean transition_scale, the wrong near-contact sample enters as FP.
    clean_problem = copy.deepcopy(problem)
    clean_problem["observations"][0]["transition_scale"] = disc
    no_band = score(clean_problem, field, k_tol=0.)
    assert no_band["drill_kept_samples"] == 4
    assert (no_band["tp"], no_band["fp"], no_band["fn"]) == (2., 1., 0.)
    assert no_band["loss_jaccard"] == pytest.approx(1. - 3. / 4.)


def test_modality_weights_smoothing_and_priors():
    shape = (8, 8, 8)
    forbidden = np.zeros(shape, dtype=bool)
    forbidden[:, :, -1] = True
    spacing = [1., 1., 1.]
    disc = .5 * float(np.linalg.norm(spacing))
    mask = np.zeros(shape, dtype=bool)
    mask[1:5, 1:5, 1:5] = True
    field = np.where(mask, 1., -1.)

    drill = dict(
        name="drill",
        points=np.array([[2.5, 2.5, 2.5], [6.5, 6.5, 2.5]]),
        labels=np.array([1., 1.]),  # 1 TP, 1 FN
        groups=np.array([0, 1]),
        weight=1., transition_scale=disc, contact_sigma=disc, label_error=0.,
    )
    outcrop = dict(
        name="outcrop",
        points=np.array([[2.5, 2.5, 3.5], [3.5, 3.5, 3.5]]),
        labels=np.array([0., 0.]),  # 2 FP at weight 0.25 -> 0.5 FP
        groups=np.array([0, 1]),
        weight=.25, transition_scale=disc, contact_sigma=disc, label_error=0.,
    )
    problem = dict(
        schema_version="problem_schema_v2", geometry_version="occupancy_v2",
        objective_version=OBJECTIVE, case_id="weights-check", shape=list(shape),
        origin=[0., 0., 0.], spacing=spacing,
        domain_mask=np.ones(shape, dtype=bool), forbidden_mask=forbidden,
        open_boundary_faces=["x-", "x+", "y-", "y+", "z-"],
        observations=[drill, outcrop],
        priors=public_priors(~forbidden, spacing),
    )
    res = score(problem, field)
    assert (res["tp"], res["fp"], res["fn"]) == pytest.approx((1., .5, 1.))
    expected = 1. - (1. + 1.) / (1. + .5 + 1. + 1.)
    assert res["loss_jaccard"] == pytest.approx(expected)
    assert res["loss_total"] == pytest.approx(expected)

    # Empty counts (0 TP, 0 FP, 0 FN) smooth to (0 + 1) / (0 + 1) = 1 -> loss 0.
    barren = copy.deepcopy(problem)
    barren["observations"] = [dict(
        name="drill", points=np.array([[6.5, 6.5, 2.5]]), labels=np.array([0.]),
        groups=np.array([0]), weight=1., transition_scale=disc, contact_sigma=disc, label_error=0.,
    )]
    assert score(barren, field)["loss_jaccard"] == pytest.approx(0.)

    # Nonzero prior weight adds regularizer on top of loss_jaccard.
    with_prior = copy.deepcopy(problem)
    with_prior["priors"] = {**problem["priors"], "area_weight": .05}
    res_prior = score(with_prior, field)
    assert res_prior["loss_total"] == pytest.approx(res["loss_jaccard"] + .05 * res["regularizer_area"])


def test_old_objective_v2_unchanged_and_selectable():
    p = legacy_fixture()
    f = legacy_model(p)
    res_auto = score(p, f)
    assert "loss_binary" in res_auto and "loss_contact" in res_auto
    assert "loss_jaccard" not in res_auto

    # Exact manual v2 recomputation matches bit-for-bit.
    p3 = copy.deepcopy(p)
    p3["objective_version"] = OBJECTIVE
    res_forced_v2 = score(p3, f, objective_version=LEGACY_OBJECTIVE)
    assert res_forced_v2 == res_auto

    res_forced_v3 = score(p, f, objective_version=OBJECTIVE)
    assert res_forced_v3["objective_version"] == OBJECTIVE
    assert "loss_jaccard" in res_forced_v3


def test_scorer_suite_aggregate_and_exposure_for_v3_and_legacy_v2(tmp_path):
    problem, truth = make_v3_problem()
    inputs, model_dir, truth_dir = tmp_path / "input", tmp_path / "model", tmp_path / "truth"
    save_problem(problem, inputs)
    save_problem(problem, truth_dir / "confirmation")
    np.save(truth_dir / "truth.npy", truth)
    model_dir.mkdir()
    np.save(model_dir / "model.npy", np.where(truth, 1., -1.))

    scored = score_case(inputs, model_dir, truth_dir)
    assert scored["status"] == "succeeded"
    assert scored["metrics"]["confirmation_objective_version"] == OBJECTIVE
    assert scored["metrics"]["confirmation_loss_jaccard"] == pytest.approx(0.)
    assert scored["metrics"]["confirmation_loss_total"] == pytest.approx(0.)

    # Illegal solve (all ones violates forbidden_mask & volume upper bound) -> invalid -> risk 1.
    bad_dir = tmp_path / "bad_model"
    bad_dir.mkdir()
    np.save(bad_dir / "model.npy", np.ones(problem["shape"], dtype=float))
    invalid = score_case(inputs, bad_dir, truth_dir)
    assert invalid["status"] == "invalid" and invalid["metrics"]["iou"] == 0.

    # Version mismatch between public problem and confirmation raises infrastructure ValueError.
    mismatch_truth = tmp_path / "mismatch_truth"
    p2 = copy.deepcopy(problem)
    p2["objective_version"] = LEGACY_OBJECTIVE
    save_problem(p2, mismatch_truth / "confirmation")
    with pytest.raises(ValueError, match="objective version differs"):
        score_case(inputs, model_dir, mismatch_truth)

    # Suite aggregate & exposure for objective_v3 vs legacy objective_v2.
    assert AGGREGATIONS["confirmation"] == "parent-macro-line-jaccard-v1"
    assert LEGACY_CONFIRMATION_AGGREGATION == "parent-macro-confirmation-utility-v1"
    assert confirmation_utility(.25, OBJECTIVE) == pytest.approx(.75)
    assert confirmation_utility(.25, LEGACY_OBJECTIVE) == pytest.approx(.8)

    v3_rows = [
        dict(parent_id="p1", case_id="c1", status="succeeded", objective_version=OBJECTIVE,
             metrics={"iou": .6, "confirmation_loss_total": .25}, solver_wall_seconds=1.),
        dict(parent_id="p2", case_id="c2", status="invalid", objective_version=OBJECTIVE,
             metrics={"iou": 0.}, solver_wall_seconds=1.),
    ]
    agg_v3 = aggregate(v3_rows)
    assert agg_v3["loss_total"] == pytest.approx((.25 + 1.) / 2.)
    assert agg_v3["research_utility"] == pytest.approx((.75 + 0.) / 2.)
    assert risk_by_case(v3_rows, ["c1", "c2"]) == {"c1": pytest.approx(.25), "c2": 1.}

    v2_rows = [
        dict(parent_id="p1", case_id="c1", status="succeeded",
             metrics={"iou": .6, "confirmation_loss_total": .25}, solver_wall_seconds=1.),
        dict(parent_id="p2", case_id="c2", status="failed",
             metrics={"iou": 0.}, solver_wall_seconds=1.),
    ]
    agg_v2 = aggregate(v2_rows)
    assert agg_v2["loss_total"] == pytest.approx((.2 + 1.) / 2.)
    assert risk_by_case(v2_rows, ["c1", "c2"]) == {"c1": pytest.approx(.2), "c2": 1.}


def test_exported_judge_files_and_task_md_have_v3_and_no_blind_leak(tmp_path):
    task_root = Path(__file__).resolve().parents[1]
    rsi_task = (task_root / "resources/rsi/TASK.md").read_text()
    v2_task = (task_root / "resources/v2/TASK.md").read_text()
    assert "L/(1+L)" not in rsi_task and "line-Jaccard" in rsi_task
    assert "L/(1+L)" not in v2_task and "line-Jaccard" in v2_task
    # Blind scan searches every exported file for case-insensitive "iou".
    for rel in ("resources/rsi/TASK.md", "v2/objective.py", "v2/contract.py",
                "v2/fixtures.py", "v2/geological_policy.py"):
        assert IOU.search((task_root / rel).read_bytes()) is None, f"unexpected 'iou' token in {rel}"

    # Verify PublicTrainingExport copies the updated judge modules.
    prob = next(fixtures())
    cat_root = tmp_path / "catalog"
    save_problem(prob, cat_root / "cases/0000")
    from tasks.copper_inversion.v2.public_catalog import SCHEMA, file_hash
    (cat_root / "catalog.json").write_text(json.dumps(dict(
        schema=SCHEMA, stage="development", split="train",
        cases=[dict(case_id=prob["case_id"], directory="cases/0000",
                    sha256={n: file_hash(cat_root / "cases/0000" / n) for n in ("problem.json", "problem.npz")})],
    )))
    export_root = tmp_path / "export"
    PublicTrainingExport(cat_root / "catalog.json")(export_root, None)
    exported_obj = (export_root / "public/tasks/copper_inversion/v2/objective.py").read_text()
    exported_contract = (export_root / "public/tasks/copper_inversion/v2/contract.py").read_text()
    assert "loss_jaccard" in exported_obj
    assert 'OBJECTIVE = "objective_v3"' in exported_contract
    exported_case = read_problem(export_root / "public/cases/0000")
    assert exported_case["objective_version"] == "objective_v3"
