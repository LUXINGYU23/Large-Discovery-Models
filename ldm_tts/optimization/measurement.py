"""Pure measurement policies and bounded reservations over a ranked candidate set."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Protocol

from ldm_tts.contracts import Candidate
from ldm_tts.optimization.records import BOObservation, SurrogateVector


@dataclass(frozen=True)
class MeasurementContext:
    candidates: tuple[Candidate, ...]
    representations: Mapping[str, SurrogateVector]
    history: tuple[BOObservation, ...]
    round_index: int
    pool_size: int
    selection_count: int


@dataclass(frozen=True)
class EvaluationGroup:
    group_id: str
    candidate_ids: tuple[str, ...]
    min_pool: int
    min_attempts: int

    def __post_init__(self) -> None:
        if not self.group_id or not self.candidate_ids:
            raise ValueError("evaluation groups require an ID and candidate IDs")
        if len(set(self.candidate_ids)) != len(self.candidate_ids):
            raise ValueError("evaluation group candidate IDs must be unique")
        if (type(self.min_pool) is not int or type(self.min_attempts) is not int
                or not 1 <= self.min_attempts <= self.min_pool):
            raise ValueError("evaluation groups require min_pool >= min_attempts >= 1")


@dataclass(frozen=True)
class SelectionRequirements:
    groups: tuple[EvaluationGroup, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self, context: MeasurementContext) -> None:
        candidates = {candidate.candidate_id for candidate in context.candidates}
        seen: set[str] = set()
        group_ids: set[str] = set()
        for group in self.groups:
            members = set(group.candidate_ids)
            if group.group_id in group_ids or members & seen:
                raise ValueError("evaluation groups must have distinct IDs and disjoint candidates")
            if not members <= candidates:
                raise ValueError("evaluation group references candidates outside the reservoir")
            seen.update(members)
            group_ids.add(group.group_id)
        if sum(group.min_pool for group in self.groups) > context.pool_size:
            raise ValueError("evaluation group reservations exceed the pool budget")
        if sum(group.min_attempts for group in self.groups) > context.selection_count:
            raise ValueError("evaluation group reservations exceed the selection budget")

    def to_dict(self) -> dict:
        return asdict(self)

    def select(
        self,
        ranked_ids: Sequence[str],
        count: int,
        *,
        stage: Literal["pool", "evaluation"],
    ) -> tuple[str, ...]:
        """Reserve from one shared ranking, then fill globally without replacement."""
        if stage not in {"pool", "evaluation"}:
            raise ValueError("unknown reservation stage")
        if len(set(ranked_ids)) != len(ranked_ids):
            raise ValueError("candidate ranking must contain unique IDs")
        selected: list[str] = []
        for group in self.groups:
            members = set(group.candidate_ids)
            minimum = group.min_pool if stage == "pool" else group.min_attempts
            selected.extend([key for key in ranked_ids if key in members][:minimum])
        if len(selected) > count:
            raise ValueError("reserved candidates exceed the selection count")
        reserved = set(selected)
        selected.extend(key for key in ranked_ids if key not in reserved)
        return tuple(selected[:count])


class MeasurementPolicy(Protocol):
    def identity(self) -> dict: ...

    def plan(self, context: MeasurementContext) -> SelectionRequirements: ...
