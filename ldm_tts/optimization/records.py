"""Lightweight BO interface records for LDM task adapters."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from collections.abc import Mapping
from typing import Any, Protocol, Sequence

from ldm_tts.contracts import AcquisitionSpec, Candidate, Observation, SurrogateSpaceSpec


@dataclass(frozen=True)
class SurrogateVector:
    """Encoded candidate representation consumed by a surrogate model."""

    values: tuple[float, ...]
    version: str
    source_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


FeatureVector = SurrogateVector


@dataclass(frozen=True)
class BOObservation:
    """One evaluated candidate and its objective values."""

    candidate_id: str
    objectives: tuple[float | None, ...]
    feature: SurrogateVector | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.candidate_id.strip():
            raise ValueError("BO observation candidate_id must not be empty")
        if not self.objectives:
            raise ValueError("BO observation must contain at least one objective")
        if self.feature is not None and self.feature.source_id not in {"", self.candidate_id}:
            raise ValueError("BO observation feature source_id must match candidate_id")

    @classmethod
    def from_observation(
        cls,
        observation: Observation,
        *,
        objective_names: Sequence[str],
        feature: SurrogateVector | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "BOObservation":
        if not observation.evaluation.succeeded:
            raise ValueError("only successful evaluations can become BO observations")
        missing = [name for name in objective_names if name not in observation.metrics]
        if missing:
            raise ValueError("observation is missing BO objective(s): " + ", ".join(missing))
        return cls(
            candidate_id=observation.candidate_id,
            objectives=tuple(float(observation.metrics[name]) for name in objective_names),
            feature=feature,
            metadata=dict(metadata or {}),
        )

    @classmethod
    def scalar(
        cls,
        candidate_id: str,
        score: float,
        feature_values: Sequence[float],
        *,
        feature_version: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> "BOObservation":
        return cls(
            candidate_id=candidate_id,
            objectives=(float(score),),
            feature=SurrogateVector(
                values=tuple(float(value) for value in feature_values),
                version=feature_version,
                source_id=candidate_id,
            ),
            metadata=dict(metadata or {}),
        )

    @property
    def scalar_score(self) -> float:
        if len(self.objectives) != 1:
            raise ValueError("scalar_score requires exactly one objective")
        value = self.objectives[0]
        if value is None:
            raise ValueError("scalar_score is unavailable for a missing objective")
        return float(value)

    @property
    def feature_vector(self) -> tuple[float, ...]:
        if self.feature is None:
            raise ValueError("BO observation does not contain a surrogate representation")
        return self.feature.values

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BOPrediction:
    """Surrogate/acquisition readout for one candidate."""

    candidate_id: str
    mean: tuple[float, ...] = ()
    std: tuple[float, ...] = ()
    acquisition_score: float | None = None
    score_direction: str = "maximize"
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def scalar(
        cls,
        candidate_id: str,
        *,
        mean: float,
        std: float,
        acquisition_score: float,
        score_direction: str = "maximize",
        metadata: dict[str, Any] | None = None,
    ) -> "BOPrediction":
        return cls(
            candidate_id=candidate_id,
            mean=(float(mean),),
            std=(float(std),),
            acquisition_score=float(acquisition_score),
            score_direction=score_direction,
            metadata=dict(metadata or {}),
        )

    @property
    def scalar_mean(self) -> float:
        if len(self.mean) != 1:
            raise ValueError("scalar_mean requires exactly one predicted objective")
        return float(self.mean[0])

    @property
    def scalar_std(self) -> float:
        if len(self.std) != 1:
            raise ValueError("scalar_std requires exactly one predicted objective")
        return float(self.std[0])

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BOSelectionResult:
    """Selected candidate ids plus the prediction pool used to select them."""

    selected_candidate_ids: tuple[str, ...]
    predictions: tuple[BOPrediction, ...] = ()
    fallback_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SurrogateEncoder(Protocol):
    """Protocol for task-owned surrogate representations."""

    def describe(self) -> SurrogateSpaceSpec:
        ...

    def encode(self, candidate: Any) -> SurrogateVector:
        ...


FeatureEncoder = SurrogateEncoder


class AcquisitionSelector(Protocol):
    """Protocol for task adapters that fit surrogates and select candidates.

    Acquisition math belongs to :mod:`ldm_tts.optimization.acquisition`. Candidate encoding
    is performed once by the engine and supplied explicitly to the selector.
    The engine supplies round_idx independently of successful training observations.
    """

    def describe(self) -> AcquisitionSpec:
        ...

    def fit(self, history: Sequence[BOObservation]) -> None:
        ...

    def select(
        self,
        candidates: Sequence[Candidate],
        representations: Mapping[str, SurrogateVector],
        *,
        count: int = 1,
        round_idx: int = 0,
    ) -> BOSelectionResult:
        ...
