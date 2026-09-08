"""Authoritative task-neutral evaluation and observation records."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Literal, Protocol, TypeVar, runtime_checkable

from ldm_tts.contracts.candidate import Candidate
from ldm_tts.contracts.task import ObjectiveSpec


EvaluationStatus = Literal["succeeded", "failed", "timed_out", "invalid"]
EVALUATION_STATUSES = frozenset({"succeeded", "failed", "timed_out", "invalid"})


@dataclass(frozen=True)
class EvaluationResult:
    """Outcome of one candidate's external evaluation."""

    candidate_id: str
    status: EvaluationStatus
    metrics: dict[str, float] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    resource_usage: dict[str, float] = field(default_factory=dict)
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.candidate_id.strip():
            raise ValueError("evaluation candidate_id must not be empty")
        if self.status not in EVALUATION_STATUSES:
            raise ValueError(
                f"unknown evaluation status {self.status!r}; "
                f"expected one of {sorted(EVALUATION_STATUSES)}"
            )
        normalized_metrics: dict[str, float] = {}
        for name, value in self.metrics.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"evaluation metric {name!r} must be numeric")
            normalized_metrics[str(name)] = float(value)
        normalized_usage: dict[str, float] = {}
        for name, value in self.resource_usage.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value < 0
            ):
                raise ValueError(
                    f"evaluation resource usage {name!r} must be finite and non-negative"
                )
            normalized_usage[str(name)] = float(value)
        object.__setattr__(self, "metrics", normalized_metrics)
        object.__setattr__(self, "artifacts", {str(k): str(v) for k, v in self.artifacts.items()})
        object.__setattr__(self, "resource_usage", normalized_usage)

    @property
    def succeeded(self) -> bool:
        return self.status == "succeeded"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Observation:
    """An admitted candidate paired with its authoritative evaluation."""

    candidate: Candidate
    evaluation: EvaluationResult
    surrogate: Any = None
    round_idx: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.candidate.candidate_id != self.evaluation.candidate_id:
            raise ValueError(
                "observation candidate and evaluation candidate_id values must match"
            )
        if self.round_idx is not None and self.round_idx < 0:
            raise ValueError("observation round_idx must be non-negative when provided")

    @property
    def candidate_id(self) -> str:
        return self.candidate.candidate_id

    @property
    def canonical_key(self) -> str:
        return self.candidate.canonical_key

    @property
    def metrics(self) -> Mapping[str, float]:
        return self.evaluation.metrics

    def to_dict(self) -> dict[str, Any]:
        surrogate = self.surrogate
        if hasattr(surrogate, "to_dict"):
            surrogate = surrogate.to_dict()
        elif surrogate is not None and is_dataclass(surrogate):
            surrogate = asdict(surrogate)
        return {
            "candidate": self.candidate.to_dict(),
            "evaluation": self.evaluation.to_dict(),
            "surrogate": surrogate,
            "round_idx": self.round_idx,
            "metadata": dict(self.metadata),
        }


@runtime_checkable
class CandidateEvaluator(Protocol):
    """Task-owned external evaluator seam."""

    def evaluate(self, candidate: Candidate) -> EvaluationResult:
        """Evaluate one admitted candidate and return a classified outcome."""


@runtime_checkable
class BatchCandidateEvaluator(CandidateEvaluator, Protocol):
    """Evaluator that scores a selected minibatch in one authoritative call."""

    def evaluate_batch(
        self, candidates: Sequence[Candidate]
    ) -> Sequence[EvaluationResult]:
        """Evaluate candidates in order and return one result per candidate."""


class CallableCandidateEvaluator:
    """Local adapter around a deterministic or task-owned evaluation callable."""

    def __init__(
        self,
        operation: Callable[[Candidate], EvaluationResult | Mapping[str, float]],
    ) -> None:
        self.operation = operation

    def evaluate(self, candidate: Candidate) -> EvaluationResult:
        result = self.operation(candidate)
        if isinstance(result, EvaluationResult):
            return result
        return EvaluationResult(
            candidate_id=candidate.candidate_id,
            status="succeeded",
            metrics={str(name): float(value) for name, value in result.items()},
        )


@dataclass(frozen=True)
class ObjectiveSet:
    """Validate and compare results using one ordered objective declaration."""

    specs: tuple[ObjectiveSpec, ...]

    def __post_init__(self) -> None:
        if not self.specs:
            raise ValueError("objective set must declare at least one objective")
        names = [item.name for item in self.specs]
        if any(not name.strip() for name in names):
            raise ValueError("objective names must not be empty")
        if len(names) != len(set(names)):
            raise ValueError("objective names must be unique")
        invalid = [
            item.direction
            for item in self.specs
            if item.direction not in {"minimize", "maximize"}
        ]
        if invalid:
            raise ValueError(
                f"objective directions must be 'minimize' or 'maximize', got {invalid!r}"
            )

    @classmethod
    def from_specs(cls, specs: Sequence[ObjectiveSpec]) -> "ObjectiveSet":
        return cls(tuple(specs))

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(item.name for item in self.specs)

    @property
    def minimize(self) -> tuple[bool, ...]:
        return tuple(item.direction == "minimize" for item in self.specs)

    def validate_metrics(
        self,
        metrics: Mapping[str, Any],
        *,
        require_all: bool = True,
    ) -> dict[str, float]:
        missing = [name for name in self.names if name not in metrics]
        if missing and require_all:
            raise ValueError("evaluation is missing objective metric(s): " + ", ".join(missing))
        normalized: dict[str, float] = {}
        for name in self.names:
            if name not in metrics:
                continue
            value = metrics[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"objective metric {name!r} must be numeric")
            numeric = float(value)
            if not math.isfinite(numeric):
                raise ValueError(f"objective metric {name!r} must be finite")
            normalized[name] = numeric
        return normalized

    def validate_result(self, result: EvaluationResult) -> EvaluationResult:
        if result.succeeded:
            self.validate_metrics(result.metrics)
        return result

    def to_vector(self, metrics: Mapping[str, Any]) -> tuple[float, ...]:
        normalized = self.validate_metrics(metrics)
        return tuple(normalized[name] for name in self.names)

    def orient_for_maximization(
        self, metrics_or_vector: Mapping[str, Any] | Sequence[float]
    ) -> tuple[float, ...]:
        if isinstance(metrics_or_vector, Mapping):
            values = self.to_vector(metrics_or_vector)
        else:
            values = tuple(float(value) for value in metrics_or_vector)
            if len(values) != len(self.specs):
                raise ValueError("objective vector length does not match objective declaration")
            if any(not math.isfinite(value) for value in values):
                raise ValueError("objective vector values must be finite")
        return tuple(
            -value if minimize else value
            for value, minimize in zip(values, self.minimize)
        )

    def is_better(self, left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
        if len(self.specs) != 1:
            raise ValueError("is_better requires exactly one objective; use pareto_front otherwise")
        return self.orient_for_maximization(left)[0] > self.orient_for_maximization(right)[0]

    def incumbent(self, observations: Sequence[Observation]) -> Observation | None:
        if len(self.specs) != 1:
            raise ValueError("incumbent requires exactly one objective; use pareto_front otherwise")
        valid = self._successful(observations)
        if not valid:
            return None
        return max(valid, key=lambda item: self.orient_for_maximization(item.metrics)[0])

    def pareto_front(self, observations: Sequence[Observation]) -> tuple[Observation, ...]:
        valid = self._successful(observations)
        oriented = [self.orient_for_maximization(item.metrics) for item in valid]
        keep: list[Observation] = []
        for index, point in enumerate(oriented):
            dominated = any(
                other_index != index
                and all(other >= value for other, value in zip(other_point, point))
                and any(other > value for other, value in zip(other_point, point))
                for other_index, other_point in enumerate(oriented)
            )
            if not dominated:
                keep.append(valid[index])
        return tuple(keep)

    def _successful(self, observations: Sequence[Observation]) -> list[Observation]:
        valid: list[Observation] = []
        for observation in observations:
            if not observation.evaluation.succeeded:
                continue
            self.validate_metrics(observation.metrics)
            valid.append(observation)
        return valid


ScoreItem = TypeVar("ScoreItem")


def is_finite_number(value: Any) -> bool:
    """Return whether ``value`` can be interpreted as a finite float."""

    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def as_float(value: Any) -> float | None:
    """Convert a finite numeric value to float, otherwise return ``None``."""

    return float(value) if is_finite_number(value) else None


finite_or_none = as_float


def ranked_items(
    items: Iterable[ScoreItem],
    score: Callable[[ScoreItem], Any],
    *,
    minimize: bool = True,
) -> list[ScoreItem]:
    """Return finite-scored items ordered from best to worst."""

    scored: list[tuple[ScoreItem, float]] = []
    for item in items:
        value = score(item)
        if is_finite_number(value):
            scored.append((item, float(value)))
    scored.sort(key=lambda pair: pair[1], reverse=not minimize)
    return [item for item, _ in scored]


def best_item(
    items: Iterable[ScoreItem],
    score: Callable[[ScoreItem], Any],
    *,
    minimize: bool = True,
) -> ScoreItem | None:
    """Return the best finite-scored item, or ``None`` when none are usable."""

    ranked = ranked_items(items, score, minimize=minimize)
    return ranked[0] if ranked else None


__all__ = [
    "BatchCandidateEvaluator",
    "CandidateEvaluator",
    "CallableCandidateEvaluator",
    "EVALUATION_STATUSES",
    "EvaluationResult",
    "EvaluationStatus",
    "ObjectiveSet",
    "Observation",
    "as_float",
    "best_item",
    "finite_or_none",
    "is_finite_number",
    "ranked_items",
]
