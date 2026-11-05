"""Development contract checks; run only on the authorized remote host."""

import copy
import tempfile
import unittest
import numpy as np
from tasks.copper_inversion.v2.contract import validate
from tasks.copper_inversion.v2.geometry import canonicalize
from tasks.copper_inversion.v2.objective import score
from tasks.copper_inversion.v2.metrics import compare
from tasks.copper_inversion.v2.data import save_problem, read_problem


def fixture():
    shape = (9, 8, 7)
    forbidden = np.zeros(shape, bool)
    forbidden[:, :, -1] = True
    return dict(schema_version="problem_schema_v2", geometry_version="occupancy_v2",
        objective_version="objective_v2", case_id="fixture", shape=shape,
        origin=[0., 0., 0.], spacing=[1., 2., 3.], domain_mask=np.ones(shape, bool),
        forbidden_mask=forbidden, open_boundary_faces=["x-", "x+", "y-", "y+", "z-"],
        observations=[dict(name="drill", points=np.array([[2.5, 5., 7.5], [5.5, 5., 7.5], [4., 5., 7.5]]),
            labels=np.array([1., 0., .5]), groups=np.array([0, 0, 0]), weight=1.,
            transition_scale=1., contact_sigma=1., label_error=.01)],
        priors=dict(area_weight=.02, volume_weight=0., volume_bounds=None,
            fragment_weight=0., fragment_volume=0., hard_volume_bounds=None, max_components=None))


def model(p):
    f = np.full(p["shape"], -1.)
    f[:4, :, :-1] = 1.
    return f


class ScienceV2(unittest.TestCase):
    def test_forbidden_not_clipped(self):
        p = fixture(); f = model(p); f[0, 0, -1] = 1
        with self.assertRaisesRegex(ValueError, "feasible"):
            score(p, f)

    def test_canonical_amplitude_invariance(self):
        p = fixture(); f = model(p)
        np.testing.assert_array_equal(canonicalize(p, f)[1], canonicalize(p, f * 200)[1])
        self.assertEqual(score(p, f), score(p, f * 200))

    def test_empty_and_open(self):
        p = fixture()
        self.assertTrue(np.isfinite(score(p, np.full(p["shape"], -1.))["loss_total"]))
        self.assertEqual(compare(p, model(p), model(p) > 0)["iou"], 1.)
        p["open_boundary_faces"].remove("x-")
        with self.assertRaisesRegex(ValueError, "shell"):
            validate(p)

    def test_contact_not_positive(self):
        p = fixture(); o = p["observations"][0]
        o.update(points=o["points"][2:], labels=o["labels"][2:], groups=o["groups"][2:])
        self.assertEqual(score(p, model(p))["loss_binary"], 0.)

    def test_physical_unit_scaling(self):
        p = fixture(); q = copy.deepcopy(p)
        q["spacing"] = (np.asarray(q["spacing"]) * 1000).tolist()
        for o in q["observations"]:
            o["points"] *= 1000; o["contact_sigma"] *= 1000; o["transition_scale"] *= 1000
        self.assertAlmostEqual(score(p, model(p))["loss_total"], score(q, model(q))["loss_total"], places=10)

    def test_group_duplication(self):
        p = fixture(); q = copy.deepcopy(p); o = q["observations"][0]
        for key in ("points", "labels", "groups"):
            o[key] = np.repeat(o[key], 3, axis=0)
        self.assertAlmostEqual(score(p, model(p))["loss_total"], score(q, model(q))["loss_total"])

    def test_readonly_and_hidden_fields(self):
        p = fixture()
        with tempfile.TemporaryDirectory() as d:
            save_problem(p, d); q = read_problem(d)
            self.assertEqual(score(p, model(p)), score(q, model(q)))
            with self.assertRaises(ValueError):
                q["forbidden_mask"][0, 0, 0] = True
        p["truth"] = model(p) > 0
        with self.assertRaisesRegex(ValueError, "fields"):
            validate(p)

    def test_coordinate_roundoff_not_extrapolation(self):
        p = fixture(); p["observations"][0]["points"][0, 0] = .5 - 1e-12
        validate(p)
        p["observations"][0]["points"][0, 0] = .49
        with self.assertRaisesRegex(ValueError, "interpolation"):
            validate(p)

    def test_shared_baseline_objective(self):
        from tasks.copper_inversion.v2.baselines import initialize, metropolis
        p = fixture()
        initial = initialize(p)
        field, info = metropolis(p, {"wall_time_seconds": .2}, 47)
        self.assertLessEqual(score(p, field)["loss_total"], score(p, initial)["loss_total"])
        self.assertEqual(info["loss_total"], score(p, field)["loss_total"])
        self.assertGreater(info["proposals"], 0)

    def test_metric_empty(self):
        p = fixture(); f = np.full(p["shape"], -1.)
        self.assertEqual(compare(p, f, f > 0)["surface_f1"], 1.)
        self.assertEqual(compare(p, f, model(p) > 0)["iou"], 0.)


if __name__ == "__main__":
    unittest.main()
