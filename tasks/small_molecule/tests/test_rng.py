"""Tests for :mod:`tasks.small_molecule.core.rng`."""

from __future__ import annotations

import os
import random
import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tasks.small_molecule.core.rng import RNG, as_rng  # noqa: E402


class RNGConstructionTests(unittest.TestCase):
    def test_deterministic_seed_round_trips(self) -> None:
        rng1 = RNG(seed=42)
        rng2 = RNG(seed=42)
        self.assertEqual(rng1.seed, 42)
        self.assertEqual(rng2.seed, 42)
        self.assertTrue(rng1.is_deterministic)
        self.assertEqual(rng1.python.random(), rng2.python.random())

    def test_none_seed_marks_nondeterministic(self) -> None:
        rng = RNG(seed=None)
        self.assertFalse(rng.is_deterministic)
        self.assertIsInstance(rng.seed, int)
        self.assertGreaterEqual(rng.seed, 0)

    def test_invalid_seed_type_raises(self) -> None:
        with self.assertRaises(TypeError):
            RNG(seed=3.14)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            RNG(seed="abc")  # type: ignore[arg-type]

    def test_python_numpy_seeds_independent(self) -> None:
        """Python and NumPy are *separate* sources; the same seed gives
        two different sequences (because the underlying PRNGs differ)."""
        rng = RNG(seed=0)
        py_first = rng.python.random()
        np_first = rng.numpy.random()
        self.assertNotAlmostEqual(py_first, np_first, places=6)


class RNGStreamTests(unittest.TestCase):
    def test_python_stream_deterministic(self) -> None:
        a, b = RNG(seed=7), RNG(seed=7)
        seq_a = [a.python.random() for _ in range(20)]
        seq_b = [b.python.random() for _ in range(20)]
        self.assertEqual(seq_a, seq_b)

    def test_numpy_stream_deterministic(self) -> None:
        a, b = RNG(seed=11), RNG(seed=11)
        seq_a = a.numpy.random(size=50)
        seq_b = b.numpy.random(size=50)
        np.testing.assert_array_equal(seq_a, seq_b)

    def test_different_seeds_diverge(self) -> None:
        a, b = RNG(seed=1), RNG(seed=2)
        self.assertNotEqual(a.python.random(), b.python.random())
        np.testing.assert_raises(AssertionError, np.testing.assert_array_equal,
                                 a.numpy.random(size=10), b.numpy.random(size=10))

    def test_beta_helper_shape_and_range(self) -> None:
        rng = RNG(seed=3)
        out = rng.beta(2.0, size=100)
        self.assertEqual(out.shape, (100,))
        self.assertTrue(np.all(out >= 0.0))
        self.assertTrue(np.all(out <= 1.0))

    def test_normal_helper_shape(self) -> None:
        rng = RNG(seed=4)
        out = rng.normal(mu=0.0, sigma=1.0, size=64)
        self.assertEqual(out.shape, (64,))
        # mean ~0, std ~1 within 3 sigmas at n=64
        self.assertAlmostEqual(float(out.mean()), 0.0, delta=0.5)
        self.assertAlmostEqual(float(out.std()), 1.0, delta=0.3)

    def test_derive_child_different_salts_diverge(self) -> None:
        parent = RNG(seed=99)
        child_a = parent.derive_child(salt="acq")
        child_b = parent.derive_child(salt="expansion")
        # Different salts → different child seeds → different streams.
        self.assertNotEqual(child_a.python.random(), child_b.python.random())
        # Parent stream is untouched by derive_child (no draws on parent).
        seq_parent = [parent.python.random() for _ in range(5)]
        replay = RNG(seed=99)
        seq_replay = [replay.python.random() for _ in range(5)]
        self.assertEqual(seq_parent, seq_replay)

    def test_derive_child_same_salt_deterministic(self) -> None:
        """Two children with the same salt must be deterministic (same seed)."""
        parent = RNG(seed=99)
        child_a = parent.derive_child(salt="acq")
        child_b = parent.derive_child(salt="acq")
        self.assertEqual(child_a.python.random(), child_b.python.random())
        self.assertEqual(child_a.numpy.random(size=10).tolist(),
                         child_b.numpy.random(size=10).tolist())


class RNGTorchTests(unittest.TestCase):
    def test_torch_seed_helper(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch not available")
        rng = RNG(seed=13)
        rng.torch()
        # Two consecutive draws after the same seed should match two
        # fresh RNGs with the same seed.
        a = torch.rand(8)
        rng.torch()
        b = torch.rand(8)
        rng2 = RNG(seed=13)
        rng2.torch()
        c = torch.rand(8)
        rng2.torch()
        d = torch.rand(8)
        torch.testing.assert_close(a, c)
        torch.testing.assert_close(b, d)

    def test_torch_seed_property(self) -> None:
        rng = RNG(seed=17)
        self.assertEqual(rng.torch_seed, 17)


class AsRNGPromotionTests(unittest.TestCase):
    def test_none_yields_fresh_rng(self) -> None:
        rng = as_rng(None)
        self.assertIsInstance(rng, RNG)
        self.assertFalse(rng.is_deterministic)

    def test_rng_passthrough(self) -> None:
        rng = RNG(seed=5)
        self.assertIs(as_rng(rng), rng)

    def test_random_random_promoted(self) -> None:
        py = random.Random(8)
        rng = as_rng(py)
        self.assertIsInstance(rng, RNG)
        # After promotion, rng.python must be the *same instance* the
        # caller already holds. Verify by comparing the next-N draws of
        # rng.python against an independent random.Random constructed
        # with the same seed — they must match from this point forward.
        twin = random.Random(8)
        for _ in range(5):
            self.assertEqual(rng.python.random(), twin.random())
        # And the original `py` (which rng.python wraps) must have been
        # advanced by exactly the same number of draws.
        self.assertEqual(py.random(), twin.random())

    def test_promotion_reproducible(self) -> None:
        """Two ``random.Random`` instances constructed with the same
        seed must derive the same :class:`RNG` numpy stream."""
        py1 = random.Random(0)
        py2 = random.Random(0)
        rng1 = as_rng(py1)
        rng2 = as_rng(py2)
        np.testing.assert_array_equal(
            rng1.numpy.random(size=20), rng2.numpy.random(size=20)
        )

    def test_invalid_type_raises(self) -> None:
        with self.assertRaises(TypeError):
            as_rng(42)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            as_rng("rng")  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
