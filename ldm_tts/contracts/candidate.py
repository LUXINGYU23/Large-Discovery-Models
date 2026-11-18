"""Task-neutral candidate admission and reservoir construction."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol, Union, runtime_checkable


@dataclass(frozen=True)
class RawProposal:
    """One untrusted proposal emitted by a reservoir expansion adapter."""

    payload: Any
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Candidate:
    """One admitted task-valid candidate.

    ``canonical_key`` is the task-owned identity used for reservoir and history
    deduplication. ``candidate_id`` is the run-facing identifier used by
    trajectories, observations, and evaluator artifacts.
    """

    candidate_id: str
    payload: Any
    canonical_key: str
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.candidate_id.strip():
            raise ValueError("candidate_id must not be empty")
        if not self.canonical_key.strip():
            raise ValueError("candidate canonical_key must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CandidateRejection:
    """A proposal rejected before expensive external evaluation."""

    reason: str
    message: str = ""
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.reason.strip():
            raise ValueError("candidate rejection reason must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


CandidateAdmission = Union[Candidate, CandidateRejection]


@runtime_checkable
class CandidateDomainAdapter(Protocol):
    """Task-owned scientific normalization and validation seam."""

    def admit(self, proposal: RawProposal) -> CandidateAdmission:
        """Return an admitted candidate or an explicit rejection."""


@dataclass(frozen=True)
class ReservoirBuildResult:
    """Accepted candidates and complete rejection accounting for one build."""

    candidates: tuple[Candidate, ...]
    rejections: tuple[CandidateRejection, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def drop_counts(self) -> dict[str, int]:
        return dict(Counter(item.reason for item in self.rejections))

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidates": [item.to_dict() for item in self.candidates],
            "rejections": [item.to_dict() for item in self.rejections],
            "drop_counts": self.drop_counts,
            "metadata": dict(self.metadata),
        }


class ReservoirBuilder:
    """Build a finite reservoir around one task's admission adapter.

    The adapter decides scientific validity and canonical identity. The builder
    owns run-level policy: history exclusion, within-reservoir deduplication,
    capacity enforcement, and rejection aggregation. ``pending_keys`` (optional)
    are canonical keys with an in-flight experiment; they are rejected as
    ``already_pending`` after the ``already_evaluated`` check.
    """

    def __init__(self, adapter: CandidateDomainAdapter) -> None:
        self.adapter = adapter

    def build(
        self,
        proposals: Iterable[RawProposal],
        *,
        evaluated_keys: Iterable[str] = (),
        pending_keys: Iterable[str] | None = None,
        max_size: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ReservoirBuildResult:
        if max_size is not None and max_size < 1:
            raise ValueError("reservoir max_size must be positive when provided")

        evaluated = {str(key) for key in evaluated_keys}
        pending = {str(key) for key in pending_keys or ()}
        admitted: list[Candidate] = []
        rejections: list[CandidateRejection] = []
        seen: dict[str, Candidate] = {}

        for proposal in proposals:
            result = self.adapter.admit(proposal)
            if isinstance(result, CandidateRejection):
                rejections.append(result)
                continue
            if not isinstance(result, Candidate):
                raise TypeError(
                    "CandidateDomainAdapter.admit() must return Candidate or CandidateRejection"
                )

            key = result.canonical_key
            if key in evaluated:
                rejections.append(
                    CandidateRejection(
                        reason="already_evaluated",
                        message="candidate identity is already present in observation history",
                        source=result.source or proposal.source,
                        metadata={"canonical_key": key, "candidate_id": result.candidate_id},
                    )
                )
                continue
            if key in pending:
                rejections.append(
                    CandidateRejection(
                        reason="already_pending",
                        message="candidate identity has a non-terminal in-flight experiment",
                        source=result.source or proposal.source,
                        metadata={"canonical_key": key, "candidate_id": result.candidate_id},
                    )
                )
                continue
            if key in seen:
                rejections.append(
                    CandidateRejection(
                        reason="duplicate",
                        message="candidate identity is duplicated within this reservoir build",
                        source=result.source or proposal.source,
                        metadata={
                            "canonical_key": key,
                            "candidate_id": result.candidate_id,
                            "duplicate_of": seen[key].candidate_id,
                        },
                    )
                )
                continue
            if max_size is not None and len(admitted) >= max_size:
                rejections.append(
                    CandidateRejection(
                        reason="reservoir_capacity",
                        message=f"candidate reservoir is limited to {max_size} candidates",
                        source=result.source or proposal.source,
                        metadata={"canonical_key": key, "candidate_id": result.candidate_id},
                    )
                )
                continue

            seen[key] = result
            admitted.append(result)

        return ReservoirBuildResult(
            candidates=tuple(admitted),
            rejections=tuple(rejections),
            metadata=dict(metadata or {}),
        )


__all__ = [
    "Candidate",
    "CandidateAdmission",
    "CandidateDomainAdapter",
    "CandidateRejection",
    "RawProposal",
    "ReservoirBuildResult",
    "ReservoirBuilder",
]
