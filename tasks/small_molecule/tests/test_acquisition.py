"""Tests for shared acquisition behavior used by the small-molecule task."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ldm_tts.optimization.acquisition import (  # noqa: E402
    chebyshev_scalarize,
    confidence_bound,
    dominates,
    expected_hypervolume_improvement,
    expected_improvement,
    hypervolume,
    pareto_front,
    sample_simplex_weights,
)
from tasks.small_molecule.core.rng import RNG  # noqa: E402


# ---------------------------------------------------------------------------
# Single-objective acquisitions used by the LDM-TTS loop
# ---------------------------------------------------------------------------


class ExpectedImprovementTests(unittest.TestCase):
    def test_basic_minimize(self) -> None:
        mu = np.array([0.0, -1.0, 1.0])
        sigma = np.array([0.1, 0.1, 0.1])
        ei = expected_improvement(mu, sigma, best=0.0, xi=0.0, minimize=True)
        self.assertGreater(ei[1], 0.0)  # improvement at -1
        self.assertAlmostEqual(ei[2], 0.0, places=10)  # ~0 at +1 (numerical noise)

    def test_basic_maximize(self) -> None:
        mu = np.array([0.0, 1.0, -1.0])
        sigma = np.array([0.1, 0.1, 0.1])
        ei = expected_improvement(mu, sigma, best=0.0, xi=0.0, minimize=False)
        self.assertGreater(ei[1], 0.0)  # improvement at +1
        self.assertAlmostEqual(ei[2], 0.0, places=10)  # ~0 at -1 (numerical noise)

    def test_zero_sigma_zero_ei(self) -> None:
        mu = np.array([0.0, 1.0])
        sigma = np.array([0.0, 0.0])
        ei = expected_improvement(mu, sigma, best=0.0, minimize=True)
        np.testing.assert_array_equal(ei, np.zeros(2))


class ConfidenceBoundTests(unittest.TestCase):
    def test_ucb_minimize(self) -> None:
        mu = np.array([0.0, 1.0])
        sigma = np.array([1.0, 1.0])
        cb = confidence_bound(mu, sigma, beta=2.0, minimize=False)
        np.testing.assert_array_equal(cb, mu + 2.0 * sigma)

    def test_lcb_minimize(self) -> None:
        mu = np.array([0.0, 1.0])
        sigma = np.array([1.0, 1.0])
        cb = confidence_bound(mu, sigma, beta=2.0, minimize=True)
        np.testing.assert_array_equal(cb, 2.0 * sigma - mu)


# ---------------------------------------------------------------------------
# Dominance / Pareto front
# ---------------------------------------------------------------------------


class DominatesTests(unittest.TestCase):
    def test_basic_minimize(self) -> None:
        self.assertTrue(dominates((0.0, 0.0), (1.0, 1.0), (True, True)))
        self.assertFalse(dominates((0.0, 1.0), (1.0, 0.0), (True, True)))
        self.assertFalse(dominates((1.0, 1.0), (1.0, 1.0), (True, True)))

    def test_mixed_minimize_maximize(self) -> None:
        self.assertTrue(dominates((0.0, 2.0), (1.0, 1.0), (True, False)))
        self.assertFalse(dominates((0.0, 0.0), (1.0, 1.0), (True, False)))

    def test_3d(self) -> None:
        minimize = (True, True, True)
        self.assertTrue(dominates((1, 2, 3), (2, 3, 4), minimize))
        self.assertFalse(dominates((1, 2, 5), (2, 3, 4), minimize))

    def test_length_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError):
            dominates((1.0, 2.0), (1.0,), (True, True))


class ParetoFrontTests(unittest.TestCase):
    def test_2d_simple(self) -> None:
        points = [(1.0, 4.0), (2.0, 3.0), (4.0, 1.0), (3.0, 2.0)]
        front = pareto_front(points, (True, True))
        self.assertEqual(set(front), {(1.0, 4.0), (2.0, 3.0), (3.0, 2.0), (4.0, 1.0)})

    def test_2d_with_dominated(self) -> None:
        points = [(1.0, 5.0), (2.0, 4.0), (3.0, 3.0), (5.0, 1.0), (4.0, 4.0)]
        front = pareto_front(points, (True, True))
        # (4,4) is dominated by (3,3); others are non-dominated.
        self.assertNotIn((4.0, 4.0), front)
        self.assertEqual(len(front), 4)

    def test_3d_mixed(self) -> None:
        points = [(1.0, 5.0, 3.0), (2.0, 4.0, 2.0), (3.0, 3.0, 1.0), (5.0, 1.0, 4.0)]
        front = pareto_front(points, (True, True, True))
        # (3,3,1) is dominated by (2,4,2) along obj0/obj1, but wins on obj2.
        # Verify all are in the front.
        self.assertEqual(len(front), 4)

    def test_empty_input(self) -> None:
        self.assertEqual(pareto_front([], (True, True)), [])

    def test_preserves_first_seen_order(self) -> None:
        points = [(4.0, 1.0), (1.0, 4.0), (2.0, 3.0), (3.0, 2.0)]
        front = pareto_front(points, (True, True))
        self.assertEqual(front[0], (4.0, 1.0))  # first-seen


# ---------------------------------------------------------------------------
# Hypervolume (exact)
# ---------------------------------------------------------------------------


class Hypervolume1DTests(unittest.TestCase):
    def test_empty(self) -> None:
        self.assertEqual(hypervolume([], [0.0]), 0.0)

    def test_minimize(self) -> None:
        self.assertEqual(hypervolume([[1.0]], [3.0]), 2.0)
        self.assertEqual(hypervolume([[0.0], [1.0]], [3.0]), 3.0)

    def test_maximize(self) -> None:
        # For a maximised objective with point=4, ref=2: in flipped
        # space the point is -4 and ref is -2; HV = -2 - (-4) = 2.
        self.assertEqual(hypervolume([[4.0]], [2.0], minimize=[False]), 2.0)

    def test_point_at_ref_zero_volume(self) -> None:
        self.assertEqual(hypervolume([[3.0]], [3.0]), 0.0)


class Hypervolume2DTests(unittest.TestCase):
    def test_empty(self) -> None:
        self.assertEqual(hypervolume([], [0.0, 0.0]), 0.0)

    def test_single_point(self) -> None:
        # One point (1, 1) with ref (3, 3): box is 2 x 2 = 4.
        self.assertAlmostEqual(hypervolume([[1.0, 1.0]], [3.0, 3.0]), 4.0)

    def test_two_points(self) -> None:
        # (1, 2) and (2, 1) with ref (3, 3).
        # Box for (1,2): [1,3]x[2,3] = 2*1 = 2.
        # Box for (2,1): [2,3]x[1,3] = 1*2 = 2.
        # Overlap: [2,3]x[2,3] = 1.
        # Union: 2 + 2 - 1 = 3.
        self.assertAlmostEqual(
            hypervolume([[1.0, 2.0], [2.0, 1.0]], [3.0, 3.0]),
            3.0,
        )

    def test_known_2d_reference(self) -> None:
        # Five non-dominated 2D points: (1,5), (2,4), (3,3), (4,2), (5,1)
        # ref (6, 6). The dominated region is a stair-step pyramid
        # with a triangle on top. Volume:
        #   base 5x5 = 25, minus the 5x5/2 = 12.5 triangle above the
        #   diagonal step (a right isoceles from (1,5) to (5,1)) -> 12.5.
        #   Wait: the Pareto front runs from (1,5) to (5,1) -- these
        #   five points ARE the front. Sweep gives:
        #     strip 0: x in [1, 6], height (6-5) = 1
        #     strip 1: x in [2, 6], height (6-4) = 2
        #     ... sum = 1*(2-1) + 2*(3-2) + 3*(4-3) + 4*(5-4) + 5*(6-5)
        #            = 1+2+3+4+5 = 15
        self.assertAlmostEqual(
            hypervolume(
                [[1.0, 5.0], [2.0, 4.0], [3.0, 3.0], [4.0, 2.0], [5.0, 1.0]],
                [6.0, 6.0],
            ),
            15.0,
        )

    def test_dominated_point_internal_filter(self) -> None:
        # (1,1) Pareto-DOMINATES (2,2) (better on both axes).
        # Union = box for (1,1) only: [1,4]x[1,4] = 3*3 = 9.
        # (2,2)'s box [2,4]x[2,4] is fully inside (1,1)'s box, so it
        # contributes no extra volume.
        self.assertAlmostEqual(
            hypervolume([[1.0, 1.0], [2.0, 2.0]], [4.0, 4.0]),
            9.0,
        )
        # Equivalently, dropping the dominated point should not change HV.
        self.assertAlmostEqual(
            hypervolume([[1.0, 1.0], [2.0, 2.0]], [4.0, 4.0]),
            hypervolume([[1.0, 1.0]], [4.0, 4.0]),
        )

    def test_mixed_minimize_maximize(self) -> None:
        # obj0 minimised, obj1 maximised. Point (1, 5) with ref (3, 2).
        # In flipped obj1 space: point (1, -5), ref (3, -2).
        # Box: (3-1) * (-2 - -5) = 2 * 3 = 6.
        self.assertAlmostEqual(
            hypervolume([[1.0, 5.0]], [3.0, 2.0], minimize=[True, False]),
            6.0,
        )

    def test_point_not_dominated_by_ref_excluded(self) -> None:
        # (4, 4) with ref (3, 3) is not dominated (both coords worse).
        self.assertEqual(hypervolume([[4.0, 4.0]], [3.0, 3.0]), 0.0)


class HypervolumeDispatchTests(unittest.TestCase):
    def test_n_obj_3_raises(self) -> None:
        with self.assertRaises(NotImplementedError):
            hypervolume([[1.0, 1.0, 1.0]], [3.0, 3.0, 3.0])

    def test_n_obj_4_raises(self) -> None:
        with self.assertRaises(NotImplementedError):
            hypervolume([[1.0, 1.0, 1.0, 1.0]], [3.0, 3.0, 3.0, 3.0])

    def test_length_mismatch_raises(self) -> None:
        """Point and ref have different dimensionalities → ValueError."""
        with self.assertRaises(ValueError):
            hypervolume([[1.0, 2.0, 3.0]], [3.0, 3.0])


# ---------------------------------------------------------------------------
# Expected Hypervolume Improvement (2D, MC)
# ---------------------------------------------------------------------------


class EHVITests(unittest.TestCase):
    def test_empty_candidates(self) -> None:
        out = expected_hypervolume_improvement(
            mu_per_obj=[np.array([]), np.array([])],
            sigma_per_obj=[np.array([]), np.array([])],
            pareto_points=[],
            ref=[0.0, 0.0],
            minimize=(True, True),
            n_samples=10,
            rng=RNG(seed=0),
        )
        self.assertEqual(out.shape, (0,))

    def test_sigma_zero_is_hv_increment(self) -> None:
        # With sigma=0, the candidate's predicted point is exactly mu.
        # EHVI = HV(pareto + mu) - HV(pareto).
        pareto = [(1.0, 5.0), (5.0, 1.0)]
        mu0, mu1 = 2.0, 2.0
        ref = (6.0, 6.0)
        out = expected_hypervolume_improvement(
            mu_per_obj=[np.array([mu0]), np.array([mu1])],
            sigma_per_obj=[np.array([0.0]), np.array([0.0])],
            pareto_points=pareto,
            ref=list(ref),
            minimize=(True, True),
            n_samples=20,
            rng=RNG(seed=0),
        )
        expected_hv_inc = hypervolume(
            pareto + [(mu0, mu1)], list(ref)
        ) - hypervolume(pareto, list(ref))
        self.assertAlmostEqual(out[0], expected_hv_inc, places=6)

    def test_n_obj_3_raises(self) -> None:
        with self.assertRaises(NotImplementedError):
            expected_hypervolume_improvement(
                mu_per_obj=[np.array([0.0])] * 3,
                sigma_per_obj=[np.array([1.0])] * 3,
                pareto_points=[],
                ref=[0.0, 0.0, 0.0],
                minimize=(True, True, True),
                rng=RNG(seed=0),
            )

    def test_ehvi_monotonic_in_mu_for_minimize(self) -> None:
        """For minimize, smaller mu is better; EHVI should be larger
        for a smaller mu than for a larger one (with sigma held fixed)."""
        pareto = [(2.0, 2.0)]
        ref = [5.0, 5.0]
        rng = RNG(seed=42)
        ehvi_small = expected_hypervolume_improvement(
            mu_per_obj=[np.array([1.0]), np.array([2.0])],
            sigma_per_obj=[np.array([0.5]), np.array([0.5])],
            pareto_points=pareto,
            ref=ref,
            minimize=(True, True),
            n_samples=512,
            rng=rng,
        )
        rng = RNG(seed=42)
        ehvi_large = expected_hypervolume_improvement(
            mu_per_obj=[np.array([4.0]), np.array([2.0])],
            sigma_per_obj=[np.array([0.5]), np.array([0.5])],
            pareto_points=pareto,
            ref=ref,
            minimize=(True, True),
            n_samples=512,
            rng=rng,
        )
        self.assertGreater(float(ehvi_small[0]), float(ehvi_large[0]))

    def test_ehvi_reproducible_with_seed(self) -> None:
        kwargs = dict(
            mu_per_obj=[np.array([1.0, 2.0]), np.array([2.0, 1.0])],
            sigma_per_obj=[np.array([0.5, 0.5]), np.array([0.5, 0.5])],
            pareto_points=[(1.0, 1.0)],
            ref=[5.0, 5.0],
            minimize=(True, True),
            n_samples=64,
        )
        out1 = expected_hypervolume_improvement(**kwargs, rng=RNG(seed=7))
        out2 = expected_hypervolume_improvement(**kwargs, rng=RNG(seed=7))
        np.testing.assert_array_almost_equal(out1, out2)

    def test_random_rng_promoted(self) -> None:
        """Old-style random.Random is auto-promoted."""
        import random
        out1 = expected_hypervolume_improvement(
            mu_per_obj=[np.array([1.0]), np.array([2.0])],
            sigma_per_obj=[np.array([0.5]), np.array([0.5])],
            pareto_points=[(1.0, 1.0)],
            ref=[5.0, 5.0],
            minimize=(True, True),
            n_samples=32,
            rng=random.Random(3),
        )
        self.assertEqual(out1.shape, (1,))

    def test_ehvi_maximize_axis(self) -> None:
        """For a maximised obj1, the EHVI should be larger when mu1
        is larger (with mu0 fixed)."""
        pareto = [(1.0, 5.0), (5.0, 1.0)]
        ref = [6.0, 6.0]
        # Same mu0; mu1 small (bad for maximise).
        rng = RNG(seed=11)
        ehvi_low = expected_hypervolume_improvement(
            mu_per_obj=[np.array([2.5]), np.array([2.0])],
            sigma_per_obj=[np.array([0.3]), np.array([0.3])],
            pareto_points=pareto,
            ref=ref,
            minimize=(True, False),
            n_samples=256,
            rng=rng,
        )
        rng = RNG(seed=11)
        ehvi_high = expected_hypervolume_improvement(
            mu_per_obj=[np.array([2.5]), np.array([5.5])],
            sigma_per_obj=[np.array([0.3]), np.array([0.3])],
            pareto_points=pareto,
            ref=ref,
            minimize=(True, False),
            n_samples=256,
            rng=rng,
        )
        # mu1=5.5 dominates the front on obj1 (closer to ref via negative);
        # the inverted-space point is more negative and dominates more.
        self.assertGreater(float(ehvi_high[0]), float(ehvi_low[0]))


# ---------------------------------------------------------------------------
# Chebyshev scalarization (any N)
# ---------------------------------------------------------------------------


class ChebyshevScalarizeTests(unittest.TestCase):
    def test_n_1_minimize(self) -> None:
        v = chebyshev_scalarize(
            point=[3.0],
            weights=[1.0],
            ideal=[0.0],
            minimize=[True],
        )
        self.assertEqual(v, 3.0)

    def test_n_1_maximize(self) -> None:
        v = chebyshev_scalarize(
            point=[3.0],
            weights=[1.0],
            ideal=[5.0],
            minimize=[False],
        )
        # gap = 1 * (5 - 3) = 2
        self.assertEqual(v, 2.0)

    def test_n_2_minimize(self) -> None:
        v = chebyshev_scalarize(
            point=[3.0, 4.0],
            weights=[0.5, 0.5],
            ideal=[0.0, 0.0],
            minimize=[True, True],
        )
        # gap0 = 0.5*3 = 1.5; gap1 = 0.5*4 = 2.0; max = 2.0
        self.assertEqual(v, 2.0)

    def test_n_2_mixed(self) -> None:
        v = chebyshev_scalarize(
            point=[3.0, 2.0],
            weights=[0.5, 0.5],
            ideal=[0.0, 5.0],
            minimize=[True, False],
        )
        # gap0 = 0.5 * (3 - 0) = 1.5
        # gap1 = 0.5 * (5 - 2) = 1.5
        # max = 1.5
        self.assertEqual(v, 1.5)

    def test_n_3_works(self) -> None:
        v = chebyshev_scalarize(
            point=[1.0, 2.0, 3.0],
            weights=[1 / 3, 1 / 3, 1 / 3],
            ideal=[0.0, 0.0, 0.0],
            minimize=[True, True, True],
        )
        # max weighted gap = (1/3) * 3 = 1.0
        self.assertAlmostEqual(v, 1.0)

    def test_n_5_works(self) -> None:
        v = chebyshev_scalarize(
            point=[1.0, 2.0, 3.0, 4.0, 5.0],
            weights=[0.2, 0.2, 0.2, 0.2, 0.2],
            ideal=[0.0, 0.0, 0.0, 0.0, 0.0],
            minimize=[True, True, True, True, True],
        )
        self.assertAlmostEqual(v, 1.0)  # 0.2 * 5

    def test_gap_clamped_non_negative(self) -> None:
        # If point < ideal on a minimise axis, the gap should be 0
        # (point is at or beyond the ideal — not worse).
        v = chebyshev_scalarize(
            point=[-1.0, 5.0],
            weights=[0.5, 0.5],
            ideal=[0.0, 0.0],
            minimize=[True, True],
        )
        # gap0 = 0.5 * (-1 - 0) clamped to 0
        # gap1 = 0.5 * 5 = 2.5
        self.assertEqual(v, 2.5)

    def test_length_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError):
            chebyshev_scalarize(
                point=[1.0, 2.0],
                weights=[1.0],
                ideal=[0.0, 0.0],
                minimize=[True, True],
            )

    def test_negative_weight_raises(self) -> None:
        with self.assertRaises(ValueError):
            chebyshev_scalarize(
                point=[1.0],
                weights=[-0.5],
                ideal=[0.0],
                minimize=[True],
            )


class SimplexWeightsTests(unittest.TestCase):
    def test_shape_and_sum(self) -> None:
        w = sample_simplex_weights(RNG(seed=0), n=5, alpha=1.0)
        self.assertEqual(w.shape, (5,))
        self.assertAlmostEqual(float(w.sum()), 1.0, places=8)
        self.assertTrue(np.all(w >= 0.0))

    def test_alpha_one_uniform(self) -> None:
        """With alpha=1, each coordinate is uniform on the simplex in
        expectation (mean weight is 1/n)."""
        rng = RNG(seed=0)
        n = 4
        samples = np.stack([sample_simplex_weights(rng, n, alpha=1.0) for _ in range(2000)])
        mean = samples.mean(axis=0)
        np.testing.assert_allclose(mean, np.full(n, 1.0 / n), atol=0.03)

    def test_reproducible(self) -> None:
        w1 = sample_simplex_weights(RNG(seed=42), n=3, alpha=2.0)
        w2 = sample_simplex_weights(RNG(seed=42), n=3, alpha=2.0)
        np.testing.assert_array_equal(w1, w2)

    def test_random_promoted(self) -> None:
        import random
        w = sample_simplex_weights(random.Random(1), n=3, alpha=1.0)
        self.assertEqual(w.shape, (3,))
        self.assertAlmostEqual(float(w.sum()), 1.0, places=8)

    def test_invalid_n(self) -> None:
        with self.assertRaises(ValueError):
            sample_simplex_weights(RNG(seed=0), n=0)
        with self.assertRaises(ValueError):
            sample_simplex_weights(RNG(seed=0), n=-1)

    def test_invalid_alpha(self) -> None:
        with self.assertRaises(ValueError):
            sample_simplex_weights(RNG(seed=0), n=3, alpha=0.0)
        with self.assertRaises(ValueError):
            sample_simplex_weights(RNG(seed=0), n=3, alpha=-1.0)


if __name__ == "__main__":
    unittest.main()
