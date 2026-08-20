"""Task-local Nyström/FITC representation for SynthonBench tuple candidates."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ldm_tts.contracts import Candidate, SurrogateSpaceSpec
from ldm_tts.optimization import SurrogateVector
from tasks.synthonbench.core.product_proxy import DEFAULT_MORGAN_RADIUS, SynthonProductProxy
from tasks.synthonbench.core.tanimoto_kernel import (
    composite_count_tanimoto_matrix,
    composite_count_tanimoto_to_many,
)


DEFAULT_KERNEL_JITTER = 1.0e-8
DEFAULT_LANDMARK_COUNT = 256
LANDMARK_ENUMERATION_LIMIT = 10_000
LANDMARK_ATTEMPTS_PER_SAMPLE = 128
MIN_RESIDUAL_TOLERANCE = 1.0e-8


@dataclass(frozen=True)
class Landmark:
    """One reproducibly sampled public tuple used in the Nyström basis."""

    reaction_id: str
    synthon_ids: tuple[int, ...]


class SynthonNystromEncoder:
    """Encode a tuple as Nyström features and its FITC diagonal residual."""

    def __init__(
        self,
        space: Any,
        allowed_reactions: Sequence[str],
        *,
        landmark_count: int = DEFAULT_LANDMARK_COUNT,
        seed: int,
        fingerprint_bits: int,
        radius: int = DEFAULT_MORGAN_RADIUS,
        include_chirality: bool = False,
        kernel_jitter: float = DEFAULT_KERNEL_JITTER,
        reaction_weight: float = 0.0,
    ) -> None:
        if landmark_count < 1:
            raise ValueError("landmark_count must be positive")
        if seed < 0:
            raise ValueError("landmark seed must be non-negative")
        if not math.isfinite(kernel_jitter) or kernel_jitter <= 0.0:
            raise ValueError("kernel_jitter must be finite and positive")
        if not math.isfinite(reaction_weight) or reaction_weight < 0.0:
            raise ValueError("reaction_weight must be finite and non-negative")
        self.space = space
        self.reactions = tuple(str(item) for item in allowed_reactions)
        if not self.reactions:
            raise ValueError("Nyström encoder requires at least one allowed reaction")
        self.landmark_count = int(landmark_count)
        self.seed = int(seed)
        self.kernel_jitter = float(kernel_jitter)
        self.reaction_weight = float(reaction_weight)
        self.product_proxy = SynthonProductProxy(
            space,
            self.reactions,
            fingerprint_bits=fingerprint_bits,
            radius=radius,
            include_chirality=include_chirality,
        )
        self.landmarks = self._sample_landmarks()
        self._landmark_counts = np.asarray(
            [self.product_proxy.tuple_counts(item.reaction_id, item.synthon_ids) for item in self.landmarks],
            dtype=float,
        )
        self._landmark_reactions = np.asarray([item.reaction_id for item in self.landmarks], dtype=str)
        kernel = composite_count_tanimoto_matrix(
            self._landmark_counts,
            self._landmark_reactions,
            reaction_weight=self.reaction_weight,
        )
        self._cholesky = _factor_kernel(kernel, self.kernel_jitter)
        self._inverse_cholesky = np.linalg.solve(
            self._cholesky,
            np.eye(self.landmark_count, dtype=float),
        )
        self.landmark_digest = _landmark_digest(self.landmarks)
        self.dimension = self.landmark_count + 1
        self.version = (
            f"synthon_nystrom_fitc_{self.product_proxy.version}_m{self.landmark_count}_"
            f"rw{self.reaction_weight:g}_{self.landmark_digest[:12]}_v1"
        )

    def describe(self) -> SurrogateSpaceSpec:
        """Describe the fixed vector shared with the LDMEngine checkpoint path."""

        return SurrogateSpaceSpec(
            kind="vector",
            representation=(
                "Nyström coordinates of a raw-connector Count-Morgan tuple proxy "
                "followed by its FITC diagonal residual"
            ),
            dimension_policy="fixed",
            dimension=self.dimension,
            encoder="tasks.synthonbench.core.nystrom_encoder:SynthonNystromEncoder",
            version=self.version,
            metadata={
                "kernel": "count_tanimoto",
                "product_proxy": self.product_proxy.version,
                "landmark_count": self.landmark_count,
                "landmark_digest": self.landmark_digest,
                "landmark_sampling": "fixed_seed_reaction_balanced_public_tuples",
                "kernel_jitter": self.kernel_jitter,
                "reaction_weight": self.reaction_weight,
                "fitc_residual_index": self.dimension - 1,
            },
        )

    def encode(self, candidate: Candidate) -> SurrogateVector:
        """Encode a source-valid tuple without assembling or scoring a product."""

        payload = candidate.payload
        if not isinstance(payload, dict):
            raise TypeError("SynthonBench candidate payload must be a mapping")
        reaction_id = payload.get("reaction_id")
        raw_ids = payload.get("synthon_ids")
        if not isinstance(reaction_id, str) or not isinstance(raw_ids, list):
            raise ValueError("SynthonBench candidate requires reaction_id and synthon_ids")
        counts = self.product_proxy.tuple_counts(reaction_id, raw_ids)
        cross_kernel = composite_count_tanimoto_to_many(
            counts,
            reaction_id,
            self._landmark_counts,
            self._landmark_reactions,
            reaction_weight=self.reaction_weight,
        )
        phi = self._inverse_cholesky @ cross_kernel
        residual = _fitc_residual(float(phi @ phi))
        values = tuple(float(item) for item in np.concatenate((phi, (residual,))))
        return SurrogateVector(values, self.version, candidate.candidate_id)

    def _sample_landmarks(self) -> tuple[Landmark, ...]:
        capacities = {reaction: _reaction_capacity(self.space, reaction) for reaction in self.reactions}
        total_capacity = sum(capacities.values())
        if self.landmark_count > total_capacity:
            raise ValueError(
                f"landmark_count={self.landmark_count} exceeds {total_capacity} valid public tuples"
            )
        quotas = _reaction_balanced_quotas(self.reactions, capacities, self.landmark_count)
        landmarks: list[Landmark] = []
        for reaction in self.reactions:
            landmarks.extend(_sample_reaction_tuples(
                self.space,
                reaction,
                quotas[reaction],
                seed=_reaction_seed(self.seed, reaction),
            ))
        if len(landmarks) != self.landmark_count:
            raise RuntimeError("Nyström landmark sampling did not produce the requested basis size")
        return tuple(landmarks)


def _reaction_capacity(space: Any, reaction_id: str) -> int:
    capacity = int(space.product_count_estimate(reaction_id))
    if capacity < 1:
        raise ValueError(f"reaction {reaction_id!r} has no public tuples")
    return capacity


def _reaction_balanced_quotas(
    reactions: Sequence[str], capacities: dict[str, int], landmark_count: int
) -> dict[str, int]:
    quotas = {reaction: 0 for reaction in reactions}
    remaining = landmark_count
    while remaining:
        progressed = False
        for reaction in reactions:
            if quotas[reaction] >= capacities[reaction]:
                continue
            quotas[reaction] += 1
            remaining -= 1
            progressed = True
            if not remaining:
                break
        if not progressed:
            raise RuntimeError("Nyström landmark quota allocation exhausted the public tuple space")
    return quotas


def _sample_reaction_tuples(space: Any, reaction_id: str, count: int, *, seed: int) -> tuple[Landmark, ...]:
    if count == 0:
        return ()
    slot_ids = tuple(
        tuple(int(item) for item in space.synthon_ids(reaction_id, int(position)))
        for position in space.positions(reaction_id)
    )
    if not slot_ids or any(not ids for ids in slot_ids):
        raise ValueError(f"reaction {reaction_id!r} has an empty public synthon slot")
    capacity = math.prod(len(ids) for ids in slot_ids)
    if count > capacity:
        raise ValueError(f"reaction {reaction_id!r} cannot supply {count} unique public landmarks")
    rng = np.random.default_rng(seed)
    if capacity <= LANDMARK_ENUMERATION_LIMIT:
        tuples = tuple(itertools.product(*slot_ids))
        indices = rng.choice(len(tuples), size=count, replace=False)
        return tuple(Landmark(reaction_id, tuple(int(item) for item in tuples[int(index)])) for index in indices)
    sampled: set[tuple[int, ...]] = set()
    max_attempts = max(LANDMARK_ATTEMPTS_PER_SAMPLE, count * LANDMARK_ATTEMPTS_PER_SAMPLE)
    for _ in range(max_attempts):
        sampled.add(tuple(int(rng.choice(ids)) for ids in slot_ids))
        if len(sampled) == count:
            return tuple(Landmark(reaction_id, item) for item in sorted(sampled))
    raise RuntimeError(f"could not sample {count} unique public landmarks for reaction {reaction_id!r}")


def _factor_kernel(kernel: np.ndarray, jitter: float) -> np.ndarray:
    try:
        return np.linalg.cholesky(kernel + jitter * np.eye(len(kernel), dtype=float))
    except np.linalg.LinAlgError as exc:
        raise ValueError("Nyström landmark kernel is not positive definite at the configured jitter") from exc


def _fitc_residual(projected_diagonal: float) -> float:
    residual = 1.0 - projected_diagonal
    if residual < -MIN_RESIDUAL_TOLERANCE:
        raise ValueError("Nyström projected diagonal exceeds the unit Tanimoto prior")
    return max(0.0, residual)


def _landmark_digest(landmarks: Sequence[Landmark]) -> str:
    payload = [[item.reaction_id, list(item.synthon_ids)] for item in landmarks]
    encoded = json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _reaction_seed(seed: int, reaction_id: str) -> int:
    payload = f"{seed}|{reaction_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


__all__ = [
    "DEFAULT_KERNEL_JITTER",
    "DEFAULT_LANDMARK_COUNT",
    "Landmark",
    "SynthonNystromEncoder",
]
