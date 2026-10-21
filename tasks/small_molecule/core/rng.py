"""Unified RNG helper: one seed, three deterministic sources.

A :class:`RNG` instance exposes a :class:`random.Random` (Python's
``random``), a :class:`numpy.random.Generator` (``numpy.random``), and a
:func:`torch.manual_seed` (PyTorch's global RNG) — all derived from a
single integer seed. This guarantees that the multi-objective BO loop's
candidate selection (numpy-based MC for EHVI / Chebyshev), the loop's
analogue pick order (``random.sample`` for the pool subsample), and
the GP fit (torch) all use a single reproducible stream.

Why this exists
---------------
The search entry points accept ``rng: Optional[random.Random]``.
Multi-objective EHVI is a Monte-Carlo
estimate of the expected hypervolume improvement, which calls
``numpy.random`` — using the global ``numpy.random`` would break
reproducibility. ``RNG.numpy`` exposes a seeded
:class:`numpy.random.Generator` so the BO loop stays reproducible.
GPSurrogate's torch seeding was historically driven by
``GPConfig.seed`` (an int defaulting to 0); the ``RNG`` class also
exposes a :meth:`torch` helper for callers that want to re-seed torch
before fitting a GP. This is intentionally *not* automatic — the
caller (the search loop) controls when torch gets re-seeded, to avoid
surprising side effects on the GP.

Backwards-compatibility
-----------------------
A :class:`random.Random` instance is auto-promoted to :class:`RNG` at
all search entry points. The promotion
derives a deterministic ``SeedSequence`` from the random instance's
internal state, so pre-existing single-objective code that passes a
plain ``random.Random(seed)`` keeps working unchanged.
"""

from __future__ import annotations

import random
from typing import Optional, Union

import numpy as np


__all__ = ["RNG", "as_rng"]


def _state_to_seed(rng: random.Random) -> int:
    """Derive a deterministic int seed from a ``random.Random`` instance.

    :class:`random.Random` stores its state in ``self._randbelow`` /
    ``self._state``; both are not part of the public API and may
    change across Python versions. We use the public ``getstate()``
    tuple and hash it to a stable int.
    """
    state = rng.getstate()
    version = state[0]
    internal = state[1]
    gauss_next = state[2] if len(state) > 2 else 0.0
    h = hash((version, internal, gauss_next))
    return h & 0x7FFFFFFF


class RNG:
    """A single seed → Python / NumPy / PyTorch deterministic sources.

    Args:
        seed: An integer seed, ``None`` for non-deterministic (os.urandom).
            When ``None``, ``seed`` is set to an os.urandom-derived int
            and ``is_deterministic`` is ``False``.

    Attributes:
        seed (int): The seed used to derive the three sources. May be
            the user-supplied seed or the auto-derived one.
        is_deterministic (bool): ``False`` iff the user passed ``None``
            (in which case the underlying sources are un-seeded).
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        if seed is None:
            import os
            seed = int.from_bytes(os.urandom(8), "big") & 0x7FFFFFFF
            self.is_deterministic = False
        else:
            if not isinstance(seed, int):
                raise TypeError(
                    f"seed must be int or None, got {type(seed).__name__}"
                )
            self.is_deterministic = True
        self.seed: int = int(seed)

        self._python: random.Random = random.Random(self.seed)
        # numpy SeedSequence gives us a structured way to spawn child
        # generators later if we need them. For now we use one Generator.
        seq = np.random.SeedSequence(self.seed)
        self._numpy: np.random.Generator = np.random.default_rng(seq)
        # Store a torch_seed to be applied via .torch(); we don't touch
        # the global torch RNG here.
        self._torch_seed: int = self.seed

    @property
    def python(self) -> random.Random:
        """Python's :class:`random.Random` (for ``rng.sample``, etc.)."""
        return self._python

    @property
    def numpy(self) -> np.random.Generator:
        """NumPy :class:`numpy.random.Generator` (for MC sampling)."""
        return self._numpy

    @property
    def torch_seed(self) -> int:
        """The int to pass to ``torch.manual_seed`` for reproducibility."""
        return self._torch_seed

    def torch(self) -> None:
        """Re-seed PyTorch's global RNG (and CUDA, if available).

        Callers (the search loop) call this *before* fitting a GP so
        the GP hyperparameters are reproducible. The CUDA seed is set
        only when CUDA is available, so non-CUDA environments do not
        emit spurious warnings.
        """
        try:
            import torch
        except ImportError:
            return
        torch.manual_seed(self._torch_seed)
        if torch.cuda.is_available():
            try:
                torch.cuda.manual_seed_all(self._torch_seed)
            except Exception:
                pass

    def beta(self, alpha: float, size: int) -> np.ndarray:
        """Draw ``size`` i.i.d. samples from ``Beta(alpha, 1)``.

        Helper for ``sample_simplex_weights``: caller normalizes the
        result to lie on the simplex.
        """
        if size < 1:
            raise ValueError(f"size must be >= 1, got {size}")
        return self._numpy.beta(alpha, 1.0, size=size)

    def normal(
        self,
        mu: Union[float, np.ndarray],
        sigma: Union[float, np.ndarray],
        size: int,
    ) -> np.ndarray:
        """Draw ``size`` i.i.d. samples from ``N(mu, sigma**2)``.

        Helper for EHVI Monte Carlo: caller passes ``mu`` and ``sigma``
        scalars (one per objective) and we draw ``size`` joint samples
        from a per-objective normal. Returns a ``(size,)`` array.
        """
        if size < 1:
            raise ValueError(f"size must be >= 1, got {size}")
        return self._numpy.normal(loc=float(mu), scale=float(sigma), size=size)

    def derive_child(self, salt: Union[int, str]) -> "RNG":
        """Spawn a child RNG that is independent but deterministic.

        Uses ``numpy.random.SeedSequence.spawn`` so the child's seed
        is a deterministic function of the parent's seed and ``salt``.
        Useful when one wants to scope separate RNG streams (e.g.,
        acquisition sampling vs analogue picking) without polluting
        the parent's stream.
        """
        seq = np.random.SeedSequence(self.seed)
        child_seq = seq.spawn(1)[0]
        # Mix the salt into the child sequence entropy.
        if isinstance(salt, str):
            salt_int = int.from_bytes(salt.encode("utf-8")[:8].ljust(8, b"\0"), "big")
        else:
            salt_int = int(salt)
        mixed = np.random.default_rng(child_seq).integers(0, 2**31 - 1)
        return RNG(int(mixed) ^ salt_int)


def as_rng(rng: Optional[Union["RNG", random.Random]]) -> RNG:
    """Auto-promote ``random.Random`` (or ``None``) to :class:`RNG`.

    Accepts:
        * ``None`` → fresh non-deterministic :class:`RNG`.
        * :class:`RNG` → returned unchanged.
        * :class:`random.Random` → wrapped: a deterministic seed is
          derived from its ``getstate()``; the new :class:`RNG` shares
          the same Python stream (so existing ``rng.sample`` /
          ``rng.shuffle`` calls keep working) and exposes the same
          numpy stream for new MC samplers.

    Raises:
        TypeError: if ``rng`` is none of the above.
    """
    if rng is None:
        return RNG(seed=None)
    if isinstance(rng, RNG):
        return rng
    if isinstance(rng, random.Random):
        derived = _state_to_seed(rng)
        out = RNG(seed=derived)
        # Reuse the caller's Python stream so they keep observing the
        # exact sequence they would have with the original random.Random.
        out._python = rng
        return out
    raise TypeError(
        f"rng must be RNG, random.Random, or None; got {type(rng).__name__}"
    )
