"""Seeded Python/NumPy streams and explicit PyTorch seeding for molecule search."""


from __future__ import annotations

import random
from typing import Optional, Union

import numpy as np


__all__ = ["RNG", "as_rng"]


def _state_to_seed(rng: random.Random) -> int:
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
