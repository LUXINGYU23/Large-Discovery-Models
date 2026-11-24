"""Tests for the deterministic reference solvers and CLI in v2/baselines.py."""

import json
import numpy as np
import pytest

from tasks.copper_inversion.core.runner import input_digest
from tasks.copper_inversion.v2 import baselines
from tasks.copper_inversion.v2.contract import validate
from tasks.copper_inversion.v2.data import save_problem
from tasks.copper_inversion.v2.fixtures import fixtures
from tasks.copper_inversion.v2.geological_policy import public_priors
from tasks.copper_inversion.v2.geometry import canonicalize
from tasks.copper_inversion.v2.metrics import compare


SOLVERS = (
    baselines.deterministic,
    baselines.stretched_nearest_class,
    baselines.layered_horizon,
    baselines.ellipsoid_fit,
)


def _synthetic_problem(truth, holes_xy, *, shape, origin=(0.0, 0.0, 0.0), spacing=(10.0, 10.0, 5.0),
                       case_id="synthetic-0", include_contacts=True, flip_indices=()):
    origin = np.asarray(origin, dtype=float)
    spacing = np.asarray(spacing, dtype=float)
    domain = np.ones(shape, dtype=bool)
    forbidden = np.zeros(shape, dtype=bool)
    forbidden[:, :, -1] = True
    truth = np.asarray(truth, dtype=bool) & domain & ~forbidden
    points, labels, groups = [], [], []
    for gid, (ix, iy) in enumerate(holes_xy):
        prev = None
        for iz in range(shape[2] - 1):
            pt = origin + (np.array([ix, iy, iz], dtype=float) + 0.5) * spacing
            val = 1.0 if truth[ix, iy, iz] else 0.0
            if (gid, iz) in flip_indices:
                val = 1.0 - val
            if include_contacts and prev is not None and prev != val:
                c_pt = origin + (np.array([ix, iy, iz - 0.5], dtype=float) + 0.5) * spacing
                points.append(c_pt)
                labels.append(0.5)
                groups.append(gid)
            points.append(pt)
            labels.append(val)
            groups.append(gid)
            prev = val
    priors = public_priors(domain & ~forbidden, spacing)
    problem = dict(
        schema_version="problem_schema_v2",
        geometry_version="occupancy_v2",
        objective_version="objective_v2",
        case_id=case_id,
        shape=list(shape),
        origin=origin.tolist(),
        spacing=spacing.tolist(),
        domain_mask=domain,
        forbidden_mask=forbidden,
        open_boundary_faces=["x-", "x+", "y-", "y+", "z-"],
        observations=[dict(
            name="drill",
            points=np.asarray(points, dtype=float),
            labels=np.asarray(labels, dtype=float),
            groups=np.asarray(groups, dtype=np.int64),
            weight=1.0,
            transition_scale=float(np.linalg.norm(spacing)),
            contact_sigma=float(spacing[2]),
            label_error=0.02 if flip_indices else 0.0,
        )],
        priors=priors,
    )
    return validate(problem), truth


def _make_folded_layer(shape=(24, 24, 20)):
    ix, iy, iz = np.indices(shape, dtype=float)
    mid = 8.0 + 2.5 * np.sin(2.0 * np.pi * ix / shape[0]) + 1.5 * np.cos(2.0 * np.pi * iy / shape[1])
    truth = np.abs(iz - mid) <= 2.2
    holes = [(x, y) for x in (3, 9, 15, 21) for y in (3, 9, 15, 21)]
    return _synthetic_problem(truth, holes, shape=shape, case_id="folded-synth", flip_indices=[(0, 1), (5, 16)])


def _make_ellipsoid_lens(shape=(24, 24, 20)):
    ix, iy, iz = np.indices(shape, dtype=float)
    x = (ix - 12.0) / 7.0
    y = (iy - 12.0) / 6.5
    z = (iz - 9.0) / 3.5
    # Tilted ellipsoid in x-z, intersected by multiple rows and columns of uniform holes.
    u = 0.9 * x + 0.436 * z
    v = -0.436 * x + 0.9 * z
    truth = (u ** 2 + y ** 2 + v ** 2) <= 1.0
    holes = [(x0, y0) for x0 in (2, 7, 12, 17, 22) for y0 in (2, 7, 12, 17, 22)]
    return _synthetic_problem(truth, holes, shape=shape, case_id="lens-synth", flip_indices=[(0, 2)])


def _make_flat_slab(shape=(24, 24, 20)):
    ix, iy, iz = np.indices(shape, dtype=float)
    truth = (iz >= 7) & (iz <= 11) & (ix >= 4) & (ix <= 19) & (iy >= 4) & (iy <= 19)
    holes = [(x, y) for x in (2, 7, 12, 17, 22) for y in (2, 7, 12, 17, 22)]
    return _synthetic_problem(truth, holes, shape=shape, case_id="slab-synth")


@pytest.mark.parametrize("solver", SOLVERS)
def test_reference_solvers_interface_legality_determinism_on_fixtures(solver):
    budget = {"wall_time_seconds": 30.0}
    for problem in fixtures():
        before = input_digest(problem)
        out1 = solver(problem, budget, 0)
        out2 = solver(problem, budget, 999)
        assert input_digest(problem) == before
        assert isinstance(out1, np.ndarray) and not np.ma.isMaskedArray(out1)
        assert out1.shape == tuple(problem["shape"])
        assert out1.dtype.kind == "f" and np.isfinite(out1).all()
        assert np.array_equal(out1, out2)
        canonicalize(problem, out1)


@pytest.mark.parametrize("solver", SOLVERS)
def test_reference_solvers_handle_zero_ore_and_all_ore_public_holes(solver):
    shape = (12, 12, 10)
    holes = [(2, 2), (6, 6), (9, 9)]
    # No ore observed in any public hole
    empty_truth = np.zeros(shape, dtype=bool)
    p_empty, _ = _synthetic_problem(empty_truth, holes, shape=shape, case_id="empty-obs")
    out_empty = solver(p_empty, {"wall_time_seconds": 10.0}, 0)
    canonicalize(p_empty, out_empty)

    # All samples positive
    full_truth = np.ones(shape, dtype=bool)
    p_full, _ = _synthetic_problem(full_truth, holes, shape=shape, case_id="full-obs")
    out_full = solver(p_full, {"wall_time_seconds": 10.0}, 0)
    canonicalize(p_full, out_full)


def test_stretched_nearest_class_recovers_flat_slab_and_selects_stretch():
    problem, truth = _make_flat_slab()
    out = baselines.stretched_nearest_class(problem, {"wall_time_seconds": 30.0}, 0)
    stats = compare(problem, out, truth)
    assert stats["iou"] >= 0.70
    assert baselines._select_z_stretch_cv(problem) >= 1.0


def test_layered_horizon_recovers_folded_layer():
    problem, truth = _make_folded_layer()
    out = baselines.layered_horizon(problem, {"wall_time_seconds": 30.0}, 0)
    stats = compare(problem, out, truth)
    assert stats["iou"] >= 0.80


def test_ellipsoid_fit_recovers_tilted_ellipsoid():
    problem, truth = _make_ellipsoid_lens()
    out = baselines.ellipsoid_fit(problem, {"wall_time_seconds": 30.0}, 0)
    stats = compare(problem, out, truth)
    assert stats["iou"] >= 0.65


def test_baselines_cli_aggregate_table_and_split_guard(tmp_path, capsys):
    release = tmp_path / "release"
    p_fold, t_fold = _make_folded_layer()
    p_lens, t_lens = _make_ellipsoid_lens()
    cases = []
    for cid, fam, prob, tr in (("c-fold", "folded", p_fold, t_fold), ("c-lens", "lens", p_lens, t_lens)):
        save_problem(prob, release / "calibration" / "public" / cid)
        tdir = release / "calibration" / "private" / cid
        tdir.mkdir(parents=True, exist_ok=True)
        np.save(tdir / "truth.npy", tr, allow_pickle=False)
        cases.append({
            "case_id": cid,
            "parent_id": "p-" + cid,
            "family": fam,
            "modality": "drill",
            "public": f"public/{cid}",
            "truth": f"private/{cid}",
        })
    (release / "calibration" / "manifest.json").write_text(json.dumps({
        "schema": "copper-suite-v2", "stage": "development", "split": "calibration", "seeds": [0], "cases": cases,
    }))

    assert baselines.main(["--release", str(release), "--split", "calibration", "--solver", "layered_horizon"]) == 0
    table = capsys.readouterr().out
    assert "layered_horizon" in table and "folded" in table and "lens" in table and "OVERALL" in table

    assert baselines.main(["--release", str(release), "--split", "calibration", "--solver", "all", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert [r["solver"] for r in payload] == ["initial", "stretched_nearest_class", "layered_horizon", "ellipsoid_fit"]
    by_name = {r["solver"]: r for r in payload}
    assert by_name["layered_horizon"]["by_family"]["folded"]["mean_iou"] >= 0.80
    assert by_name["ellipsoid_fit"]["by_family"]["lens"]["mean_iou"] >= 0.65

    # Refuses validation or test splits.
    with pytest.raises(SystemExit):
        baselines.main(["--release", str(release), "--split", "validation", "--solver", "initial"])
    with pytest.raises(ValueError, match="unsupported split"):
        baselines.evaluate_release(release, "test_id", "initial")
