import numpy as np
import pytest
from tasks.copper_inversion.core.data import load_problem, CASES
from tasks.copper_inversion.core.geometry import signed_distance, read_model
from tasks.copper_inversion.core.objective import score, binary_loss


def test_binary_contacts():
    phi = np.array([-1.0, 2.0, 0.25])
    y = np.array([0.0, 1.0, 0.5])
    expected = (
        np.logaddexp(0, -1) + np.logaddexp(0, -2) + np.logaddexp(0, -0.25) + 2 * 0.25**2
    )
    assert binary_loss(phi, y, 1.0) == pytest.approx(expected)
    assert binary_loss(phi, y * 0, 1.0) == pytest.approx(np.logaddexp(0, phi).sum())


@pytest.mark.parametrize("case", CASES)
def test_actual_data(case):
    p = load_problem(case)
    v = score(p, p["initial_level_set"])
    expected = {
        "drillholes_sketch_2d": 293.2197607509265,
        "drillholes_3d": 405.0795810830953,
        "drillholes_outcrops_3d": 1248.8385761871953,
    }
    assert v["loss_total"] == pytest.approx(expected[case], rel=1e-10)
    assert sum(np.isfinite(o["values"]).sum() for o in p["observations"]) == (
        35 if case.endswith("2d") else 3600 if "outcrops" in case else 70
    )
    assert v["loss_total"] == pytest.approx(-v["research_utility"])
    if not case.endswith("2d"):
        assert p["shape"] == (40, 50, 30)
        assert p["origin"] == [0.0, 0.0, -25.0]
        assert p["axis_names"] == ["x", "y", "z"]
        assert int(p["active_mask"].sum()) == 55517
        assert int((~p["active_mask"]).sum()) == 4483
        assert int((p["observations"][0]["values"] == 0.5).sum()) == 4


def test_handwritten_ubc_coordinate_fixture(tmp_path, monkeypatch):
    """UBC rows descend in z, then advance x, then y; no writer round-trip."""
    import hashlib
    import json
    from tasks.copper_inversion.core import data

    root = tmp_path / "cases" / "drillholes_3d"
    root.mkdir(parents=True)
    (root / "mesh.txt").write_text("2 3 4\n10 20 100\n2*2\n3*3\n4*5\n")
    labels = np.full(24, np.nan)
    labels[0], labels[7], labels[19] = 1.0, 0.5, 0.0
    initial = np.zeros(24)
    initial[0] = 1
    active = np.ones(24)
    active[7] = 0
    for name, values in [("drillholes.txt", labels), ("initial_model.txt", initial),
                         ("ind_active.txt", active)]:
        np.savetxt(root / name, values)
    files = {str(p.relative_to(tmp_path)): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in root.glob("*.txt")}
    (tmp_path / "provenance.json").write_text(json.dumps({"files": files}))
    monkeypatch.setattr(data, "RESOURCES", tmp_path)
    problem = data.load_problem("drillholes_3d")
    assert problem["shape"] == (2, 3, 4)
    assert problem["origin"] == [10.0, 20.0, 80.0]
    assert problem["spacing"] == [2.0, 3.0, 5.0]
    observed = problem["observations"][0]
    assert observed["mask"].sum() == 3
    for index, label, coordinate in [((0, 0, 3), 1, [11, 21.5, 97.5]),
                                     ((1, 0, 0), 0.5, [13, 21.5, 82.5]),
                                     ((0, 2, 0), 0, [11, 27.5, 82.5])]:
        assert observed["values"][index] == label
        np.testing.assert_array_equal(np.array(problem["origin"]) +
                                      (np.array(index) + 0.5) * problem["spacing"], coordinate)
    assert problem["initial_level_set"][0, 0, 3] == 0.5
    assert (problem["initial_level_set"] > 0).sum() == 1
    assert not problem["active_mask"][1, 0, 0]
    assert problem["active_mask"].sum() == 23


@pytest.mark.parametrize("case", CASES)
def test_scale_invariance(case):
    p = load_problem(case, fixture=True)
    f = p["initial_level_set"]
    a = signed_distance(f, p["shape"], p["spacing"])
    b = signed_distance(20 * f, p["shape"], p["spacing"])
    np.testing.assert_allclose(a, b, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize(
    "value", [np.full((4, 3), np.nan), np.ones((4, 3)), np.zeros((4, 3))]
)
def test_invalid_geometry(value):
    with pytest.raises(ValueError):
        signed_distance(value, (4, 3), (1, 1))


def test_bounded_npy(tmp_path):
    p = tmp_path / "model.npy"
    np.save(p, np.ones((4, 3)))
    assert read_model(p, (4, 3)).shape == (4, 3)
    with pytest.raises(ValueError):
        read_model(p, (3, 4))
    link = tmp_path / "link.npy"
    link.symlink_to(p)
    with pytest.raises(ValueError):
        read_model(link, (4, 3))
    np.save(p, np.array([{}], dtype=object))
    with pytest.raises(ValueError):
        read_model(p, (1,))


@pytest.mark.parametrize("shape", [(9, 8), (9, 8, 7)])
def test_distance_storage_order_and_analytic_plane(shape):
    expected = np.indices(shape, dtype=float)[0] - 3.25
    for dtype in (np.float32, np.float64):
        for order in ("C", "F"):
            field = np.array(expected, dtype=dtype, order=order)
            actual = signed_distance(field, shape, (1.0,) * len(shape))
            np.testing.assert_allclose(actual, expected, atol=1e-12)
            np.testing.assert_array_equal(actual > 0, field > 0)


@pytest.mark.parametrize("case", CASES)
def test_actual_data_storage_order_invariance(case):
    problem = load_problem(case)
    field = problem["initial_level_set"]
    assert score(problem, np.array(field, order="C")) == score(
        problem, np.array(field, order="F")
    )


@pytest.mark.parametrize("case", CASES)
def test_physical_measure_and_air_diagnostics_do_not_change_loss(case):
    p = load_problem(case, fixture=True)
    p["spacing"] = [2.0, 3.0] if len(p["shape"]) == 2 else [2.0, 3.0, 4.0]
    f = p["initial_level_set"]
    before = score(p, f)
    measure = "area" if f.ndim == 2 else "volume"
    assert before[f"positive_{measure}"] == np.count_nonzero(f > 0) * np.prod(p["spacing"])
    assert "air_positive_voxels" not in before  # No mask means unknown, not zero.
    p["active_mask"] = np.ones(p["shape"], dtype=bool)
    p["active_mask"][0:p["shape"][0] // 2] = False
    after = score(p, f)
    air_count = np.count_nonzero((f > 0) & ~p["active_mask"])
    assert 0 < air_count < np.count_nonzero(f > 0)
    assert after["air_positive_voxels"] == air_count
    assert after[f"air_positive_{measure}"] == air_count * np.prod(p["spacing"])
    assert after["air_fraction_of_positive"] == air_count / np.count_nonzero(f > 0)
    assert after["loss_total"] == before["loss_total"]
    assert after["research_utility"] == before["research_utility"]
