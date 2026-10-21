"""Engine lifecycle configuration, state and durable checkpoint representation."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from ldm_tts.contracts import Candidate, EvaluationResult, Observation
from ldm_tts.optimization.records import SurrogateVector


@dataclass(frozen=True)
class LDMEngineConfig:
    """Task-neutral lifecycle policy for one engine campaign."""

    iterations: int
    reservoir_size: int
    evaluations_per_round: int = 1
    max_empty_reservoir_rounds: int = 3
    target_observations: int | None = None
    target_successful_evaluations: int | None = None
    max_evaluation_attempts: int | None = None
    max_evaluation_attempts_per_round: int | None = None
    replace_failed_evaluations: bool = False

    def __post_init__(self) -> None:
        if self.iterations < 0:
            raise ValueError("engine iterations must be non-negative")
        if self.reservoir_size < 1:
            raise ValueError("engine reservoir_size must be positive")
        if self.evaluations_per_round < 1:
            raise ValueError("engine evaluations_per_round must be positive")
        if self.max_empty_reservoir_rounds < 1:
            raise ValueError("engine max_empty_reservoir_rounds must be positive")
        if self.target_observations is not None and self.target_observations < 0:
            raise ValueError("engine target_observations must be non-negative")
        if (
            self.target_successful_evaluations is not None
            and self.target_successful_evaluations < 0
        ):
            raise ValueError(
                "engine target_successful_evaluations must be non-negative"
            )
        if (
            self.target_observations is not None
            and self.target_successful_evaluations is not None
        ):
            raise ValueError(
                "engine must target observations or successful evaluations, not both"
            )
        if self.max_evaluation_attempts is not None and self.max_evaluation_attempts < 0:
            raise ValueError("engine max_evaluation_attempts must be non-negative")
        if (
            self.max_evaluation_attempts_per_round is not None
            and self.max_evaluation_attempts_per_round < 1
        ):
            raise ValueError(
                "engine max_evaluation_attempts_per_round must be positive"
            )
        if self.replace_failed_evaluations and self.target_successful_evaluations is None:
            raise ValueError(
                "replace_failed_evaluations requires target_successful_evaluations"
            )


@dataclass
class LDMEngineState:
    """Caller-resumable in-memory engine state."""

    observations: list[Observation] = field(default_factory=list)
    expansion_schema: dict[str, Any] = field(default_factory=dict)
    next_round: int = 0
    empty_reservoir_rounds: int = 0

    def __post_init__(self) -> None:
        if self.next_round < 0:
            raise ValueError("engine next_round must be non-negative")
        if self.empty_reservoir_rounds < 0:
            raise ValueError("engine empty_reservoir_rounds must be non-negative")

    def to_checkpoint(self) -> dict[str, Any]:
        return {
            "next_round": self.next_round,
            "empty_reservoir_rounds": self.empty_reservoir_rounds,
            "expansion_schema": jsonable(self.expansion_schema),
            "observations": [jsonable(item.to_dict()) for item in self.observations],
        }

    @classmethod
    def from_checkpoint(cls, payload: Mapping[str, Any]) -> "LDMEngineState":
        observations: list[Observation] = []
        raw_observations = payload.get("observations", [])
        if not isinstance(raw_observations, list):
            raise ValueError("engine checkpoint observations must be a list")
        for raw in raw_observations:
            if not isinstance(raw, Mapping):
                raise ValueError("engine checkpoint observation must be an object")
            candidate_payload = raw.get("candidate")
            evaluation_payload = raw.get("evaluation")
            if not isinstance(candidate_payload, Mapping) or not isinstance(
                evaluation_payload, Mapping
            ):
                raise ValueError("engine checkpoint observation records are malformed")
            candidate = Candidate(
                candidate_id=str(candidate_payload.get("candidate_id", "")),
                payload=candidate_payload.get("payload"),
                canonical_key=str(candidate_payload.get("canonical_key", "")),
                source=str(candidate_payload.get("source", "")),
                metadata=dict(candidate_payload.get("metadata", {})),
            )
            evaluation = EvaluationResult(
                candidate_id=str(evaluation_payload.get("candidate_id", "")),
                status=str(evaluation_payload.get("status", "failed")),  # type: ignore[arg-type]
                metrics=dict(evaluation_payload.get("metrics", {})),
                artifacts=dict(evaluation_payload.get("artifacts", {})),
                resource_usage=dict(evaluation_payload.get("resource_usage", {})),
                error=str(evaluation_payload.get("error", "")),
                metadata=dict(evaluation_payload.get("metadata", {})),
            )
            surrogate_payload = raw.get("surrogate")
            surrogate = None
            if surrogate_payload is not None:
                if not isinstance(surrogate_payload, Mapping):
                    raise ValueError("engine checkpoint surrogate must be an object")
                surrogate = SurrogateVector(
                    values=tuple(surrogate_payload["values"]),
                    version=surrogate_payload["version"],
                    source_id=surrogate_payload.get("source_id", ""),
                    metadata=dict(surrogate_payload.get("metadata", {})),
                )
            round_idx = raw.get("round_idx")
            observations.append(
                Observation(
                    candidate=candidate,
                    evaluation=evaluation,
                    surrogate=surrogate,
                    round_idx=None if round_idx is None else int(round_idx),
                    metadata=dict(raw.get("metadata", {})),
                )
            )
        expansion_schema = payload.get("expansion_schema", {})
        if not isinstance(expansion_schema, Mapping):
            raise ValueError("engine checkpoint expansion_schema must be an object")
        return cls(
            observations=observations,
            expansion_schema=dict(expansion_schema),
            next_round=int(payload.get("next_round", 0)),
            empty_reservoir_rounds=int(payload.get("empty_reservoir_rounds", 0)),
        )


@dataclass(frozen=True)
class LDMEngineResult:
    """Final engine state and publication-facing summary."""

    state: LDMEngineState
    rounds_run: int
    stop_reason: str
    summary: dict[str, Any]


def _successful_evaluation_count(state: LDMEngineState) -> int:
    return sum(item.evaluation.succeeded for item in state.observations)


def completion_reason(
    state: LDMEngineState,
    config: LDMEngineConfig,
) -> str | None:
    if (
        config.target_observations is not None
        and len(state.observations) >= config.target_observations
    ):
        return "observation_target"
    if (
        config.target_successful_evaluations is not None
        and _successful_evaluation_count(state)
        >= config.target_successful_evaluations
    ):
        return "successful_evaluation_target"
    return None


def desired_round_results(
    state: LDMEngineState,
    config: LDMEngineConfig,
) -> int:
    if config.target_observations is not None:
        remaining = max(0, config.target_observations - len(state.observations))
        return min(config.evaluations_per_round, remaining)
    if config.target_successful_evaluations is not None:
        remaining = max(
            0,
            config.target_successful_evaluations
            - _successful_evaluation_count(state),
        )
        return min(config.evaluations_per_round, remaining)
    return config.evaluations_per_round


def jsonable(value: Any) -> Any:
    """Normalize task metadata for durable engine artifacts."""

    return json.loads(json.dumps(value, default=str))
