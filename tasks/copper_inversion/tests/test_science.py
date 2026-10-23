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
        "drillholes_3d": 514.0113798896817,
        "drillholes_outcrops_3d": 1060.7747438220217,
    }
    assert v["loss_total"] == pytest.approx(expected[case], rel=1e-10)
    assert sum(np.isfinite(o["values"]).sum() for o in p["observations"]) == (
        35 if case.endswith("2d") else 3600 if "outcrops" in case else 70
    )
    assert v["loss_total"] == pytest.approx(-v["research_utility"])


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
