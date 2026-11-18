"""Lightweight BO interface records for LDM task adapters."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from collections.abc import Mapping
from typing import Any, Protocol, Sequence, runtime_checkable

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


@dataclass(frozen=True)
class PendingExperiment:
    """One in-flight experiment as seen by a selector.

    PLANNED, QUEUED, RUNNING and cancel-requested experiments are conditioned on
    identically; ``status`` is informational only.
    """

    candidate: Candidate
    feature: SurrogateVector
    status: str = "pending"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SelectionSeed:
    """Seed material; step j draws from a stream derived from (campaign_seed, decision_id, j)."""

    campaign_seed: int
    decision_id: str

    def __post_init__(self) -> None:
        if isinstance(self.campaign_seed, bool) or not isinstance(self.campaign_seed, int):
            raise ValueError("campaign_seed must be an integer")
        if not str(self.decision_id).strip():
            raise ValueError("decision_id must not be empty")


@dataclass(frozen=True)
class SelectionStep:
    """Complete probability table of one conditional selection step.

    Arrays are aligned with ``candidate_ids`` (the eligible set E_j); ``probability``
    is pi_j and ``selected_probability`` is the step propensity pi_j(x_j).
    """

    index: int
    candidate_ids: tuple[str, ...]
    canonical_keys: tuple[str, ...]
    pending_keys: tuple[str, ...]
    mean: tuple[float, ...]
    std: tuple[float, ...]
    acquisition: tuple[float, ...]
    normalized_acquisition: tuple[float, ...]
    q0: tuple[float, ...]
    probability: tuple[float, ...]
    selected_candidate_id: str
    selected_probability: float
    uniform: float
    beta: float
    effective_sample_size: float
    max_rho: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PendingAwareSelection:
    """Ordered selections with per-step conditional propensities (no marginal claim)."""

    selected_candidate_ids: tuple[str, ...]
    steps: tuple[SelectionStep, ...]
    parameters: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def log_propensity(self) -> float:
        return float(sum(math.log(step.selected_probability) for step in self.steps))

    @property
    def propensity(self) -> float:
        return math.exp(self.log_propensity)

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "log_propensity": self.log_propensity}


@runtime_checkable
class PendingAwareSelector(Protocol):
    """Optional selector capability: condition on in-flight experiments.

    ``describe().parameters["pending_handling"]`` declares how pending experiments are
    treated. ``prior`` is a raw-unit prior table indexed by canonical key (one policy
    application); ``q0`` is indexed by canonical key. Selectors lacking this protocol
    keep working through :class:`AcquisitionSelector` with framework-level dedup only.
    """

    def describe(self) -> AcquisitionSpec:
        ...

    def fit(
        self,
        history: Sequence[BOObservation],
        *,
        history_keys: Sequence[str] | None = None,
        prior: Mapping[str, float] | None = None,
    ) -> None:
        ...

    def select_with_pending(
        self,
        candidates: Sequence[Candidate],
        representations: Mapping[str, SurrogateVector],
        *,
        pending: Sequence[PendingExperiment],
        seed: SelectionSeed,
        count: int = 1,
        q0: Mapping[str, float] | None = None,
    ) -> PendingAwareSelection:
        ...
