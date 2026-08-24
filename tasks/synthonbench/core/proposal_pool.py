"""q0-preserving BO-pool maintenance for SynthonBench proposal reservoirs."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from ldm_tts.contracts import Candidate
from tasks.synthonbench.core.constants import Q0_METADATA_KEY
from tasks.synthonbench.core.ldm_policy import AcquisitionTiltConfig, gumbel_top_k


@dataclass(frozen=True)
class EmpiricalPool:
    """The full unique proposal reservoir and its q0-maintained BO subset."""

    proposal_reservoir: tuple[Candidate, ...]
    candidates: tuple[Candidate, ...]
    valid_proposal_occurrences: int
    maintenance_seed: int | None
    maintenance_method: str


def maintain_empirical_pool(candidates: tuple[Candidate, ...], config: AcquisitionTiltConfig,
                            history_size: int) -> EmpiricalPool:
    """Maintain K candidates by q0-weighted Gumbel sampling when M_unique>K."""

    occurrences = sum(_occurrence_count(item) for item in candidates)
    _validate_totals(candidates, occurrences, config)
    limit = config.pool_size or len(candidates)
    if len(candidates) <= limit:
        return EmpiricalPool(candidates, candidates, occurrences, None, "all_unique_candidates")
    seed = candidate_set_seed(config.seed, history_size, candidates, phase="bo_pool_maintenance")
    indices = gumbel_top_k(empirical_base_masses(candidates), limit, np.random.default_rng(seed))
    maintained = tuple(sorted((candidates[index] for index in indices), key=lambda item: item.candidate_id))
    return EmpiricalPool(candidates, maintained, occurrences, seed, "q0_gumbel_top_k_without_replacement")


def empirical_base_masses(candidates: Sequence[Candidate]) -> np.ndarray:
    masses = np.asarray([_base_mass(item) for item in candidates], dtype=float)
    return masses if not len(masses) else masses / float(masses.sum())


def proposal_base_measure_records(candidates: Sequence[Candidate]) -> list[dict[str, object]]:
    return [
        {"candidate_id": item.candidate_id, "occurrence_count": _occurrence_count(item),
         "proposal_q0_base_mass": _base_mass(item)}
        for item in candidates
    ]


def candidate_set_seed(seed: int, history_size: int, candidates: Sequence[Candidate], *, phase: str) -> int:
    payload = json.dumps(
        {"seed": seed, "history_size": history_size, "candidate_ids": [item.candidate_id for item in candidates], "phase": phase},
        sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _validate_totals(candidates: Sequence[Candidate], occurrences: int,
                     config: AcquisitionTiltConfig) -> None:
    totals = {_valid_occurrence_count(item) for item in candidates}
    if totals != {occurrences}:
        raise ValueError("candidate q0 occurrence totals are inconsistent")
    if config.proposal_sample_count is not None and occurrences > config.proposal_sample_count:
        raise ValueError("valid proposal occurrences exceed configured proposal samples")


def _base_mass(candidate: Candidate) -> float:
    value = float(_q0_record(candidate)["probability"])
    if not math.isfinite(value) or value <= 0:
        raise ValueError("candidate q0 base mass must be finite and positive")
    return value


def _occurrence_count(candidate: Candidate) -> int:
    return _positive_integer(_q0_record(candidate).get("occurrence_count"), "occurrence_count")


def _valid_occurrence_count(candidate: Candidate) -> int:
    return _positive_integer(_q0_record(candidate).get("valid_occurrence_count"), "valid_occurrence_count")


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"q0 {name} must be a positive integer")
    return value


def _q0_record(candidate: Candidate) -> Mapping[str, object]:
    record = candidate.metadata.get(Q0_METADATA_KEY)
    if not isinstance(record, Mapping):
        raise TypeError(f"candidate {candidate.candidate_id!r} has no empirical q0 record")
    return record


__all__ = [
    "EmpiricalPool",
    "candidate_set_seed",
    "empirical_base_masses",
    "maintain_empirical_pool",
    "proposal_base_measure_records",
]
