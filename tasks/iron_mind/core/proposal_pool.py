"""Iron Mind empirical proposal masses and finite BO-pool maintenance."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from ldm_tts.contracts import Candidate

from tasks.iron_mind.core.candidate import IRON_MIND_Q0_METADATA_KEY
from tasks.iron_mind.core.ldm_policy import AcquisitionTiltConfig, gumbel_top_k


SEED_BYTES = 8


@dataclass(frozen=True)
class EmpiricalPool:
    """The admitted proposal reservoir and its maintained BO subset."""

    proposal_reservoir: tuple[Candidate, ...]
    candidates: tuple[Candidate, ...]
    valid_proposal_occurrences: int
    maintenance_seed: int | None
    maintenance_method: str


def maintain_empirical_pool(
    candidates: tuple[Candidate, ...],
    config: AcquisitionTiltConfig,
    history_size: int,
) -> EmpiricalPool:
    """Reduce an oversized proposal reservoir by q0-weighted sampling."""

    valid_occurrences = sum(_candidate_occurrence_count(item) for item in candidates)
    reported_totals = {_candidate_valid_occurrence_count(item) for item in candidates}
    if reported_totals != {valid_occurrences}:
        raise ValueError("candidate empirical q0 occurrence totals are inconsistent")
    if config.proposal_sample_count is not None and valid_occurrences > config.proposal_sample_count:
        raise ValueError("valid proposal occurrences exceed configured proposal samples")
    limit = config.pool_size or len(candidates)
    if len(candidates) <= limit:
        return EmpiricalPool(
            candidates,
            candidates,
            valid_occurrences,
            None,
            "all_unique_candidates",
        )
    pool_seed = candidate_set_seed(
        config.seed,
        history_size,
        candidates,
        phase="bo_pool_maintenance",
    )
    indices = gumbel_top_k(
        empirical_base_masses(candidates),
        limit,
        np.random.default_rng(pool_seed),
    )
    maintained = tuple(
        sorted(
            (candidates[index] for index in indices),
            key=lambda item: item.candidate_id,
        )
    )
    return EmpiricalPool(
        candidates,
        maintained,
        valid_occurrences,
        pool_seed,
        "q0_gumbel_top_k_without_replacement",
    )


def empirical_base_masses(candidates: Sequence[Candidate]) -> np.ndarray:
    masses = np.asarray([_candidate_base_mass(candidate) for candidate in candidates], dtype=float)
    total = float(masses.sum())
    if masses.size and total <= 0:
        raise ValueError("candidate empirical base masses must contain positive mass")
    return masses if not masses.size else masses / total


def proposal_base_measure_records(candidates: Sequence[Candidate]) -> list[dict[str, object]]:
    """Return the complete admitted proposal distribution for run artifacts."""

    return [
        {
            "candidate_id": candidate.candidate_id,
            "occurrence_count": _candidate_occurrence_count(candidate),
            "proposal_q0_base_mass": _candidate_base_mass(candidate),
        }
        for candidate in candidates
    ]


def candidate_set_seed(
    seed: int,
    history_size: int,
    candidates: Sequence[Candidate],
    *,
    phase: str,
) -> int:
    payload = json.dumps(
        {
            "seed": seed,
            "history_size": history_size,
            "candidate_ids": [item.candidate_id for item in candidates],
            "phase": phase,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:SEED_BYTES], "big")


def _candidate_base_mass(candidate: Candidate) -> float:
    record = _base_measure_record(candidate)
    if "probability" not in record:
        raise ValueError(f"candidate {candidate.candidate_id!r} has no empirical probability")
    value = float(record["probability"])
    if not math.isfinite(value) or value <= 0:
        raise ValueError("candidate empirical base mass must be finite and positive")
    return value


def _candidate_occurrence_count(candidate: Candidate) -> int:
    value = _base_measure_record(candidate).get("occurrence_count")
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError("candidate proposal occurrence count must be a positive integer")
    return value


def _candidate_valid_occurrence_count(candidate: Candidate) -> int:
    value = _base_measure_record(candidate).get("valid_occurrence_count")
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError("valid proposal occurrence count must be a positive integer")
    return value


def _base_measure_record(candidate: Candidate) -> Mapping[str, object]:
    record = candidate.metadata.get(IRON_MIND_Q0_METADATA_KEY)
    if not isinstance(record, Mapping):
        raise ValueError(f"candidate {candidate.candidate_id!r} has no empirical base measure")
    return record


__all__ = [
    "EmpiricalPool",
    "candidate_set_seed",
    "empirical_base_masses",
    "maintain_empirical_pool",
    "proposal_base_measure_records",
]
