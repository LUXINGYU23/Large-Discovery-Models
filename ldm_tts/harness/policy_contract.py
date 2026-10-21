"""Task-facing contracts for compiled optimization policies."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

import numpy as np

from ldm_tts.harness.policy_execution import PolicyExecutionResult
from ldm_tts.harness.protocol import (
    HarnessArtifactRule,
    HarnessSubmissionContract,
    HarnessSubmissionError,
    canonical_sha256,
)

POLICY_API_VERSION = 1
POLICY_CAPABILITIES = frozenset({"prior_mean@1", "ldm_weights@1"})
POLICY_ARTIFACT_NAME = "optimization_policy.py"
POLICY_ARTIFACT_MAX_BYTES = 128 * 1024


@dataclass(frozen=True)
class PolicyCapabilityContract:
    task_id: str
    api_version: int
    enabled_capabilities: tuple[str, ...]
    feature_names: tuple[str, ...]
    feature_groups: Mapping[str, tuple[int, int]]
    mean_clip: float
    default_alpha: float
    default_eta: float

    def __post_init__(self) -> None:
        if not self.task_id.strip():
            raise ValueError("policy task_id must not be empty")
        if self.api_version != POLICY_API_VERSION:
            raise ValueError(f"policy api_version must equal {POLICY_API_VERSION}")
        capabilities = tuple(sorted(self.enabled_capabilities))
        if (
            not capabilities
            or len(set(capabilities)) != len(capabilities)
            or any(value not in POLICY_CAPABILITIES for value in capabilities)
        ):
            raise ValueError("policy enabled_capabilities are invalid")
        names = tuple(self.feature_names)
        if not names or len(set(names)) != len(names) or any(not name for name in names):
            raise ValueError("policy feature_names must be non-empty and unique")
        groups: dict[str, tuple[int, int]] = {}
        for name, raw_range in self.feature_groups.items():
            if not name or len(raw_range) != 2:
                raise ValueError("policy feature_groups require named half-open ranges")
            start, end = raw_range
            if (
                isinstance(start, bool)
                or isinstance(end, bool)
                or not isinstance(start, int)
                or not isinstance(end, int)
                or start < 0
                or end <= start
                or end > len(names)
            ):
                raise ValueError("policy feature_groups contain an invalid range")
            groups[name] = (start, end)
        numeric: dict[str, float] = {}
        for value, label, positive in (
            (self.mean_clip, "mean_clip", True),
            (self.default_alpha, "default_alpha", False),
            (self.default_eta, "default_eta", False),
        ):
            try:
                number = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"policy {label} is out of range") from exc
            if isinstance(value, bool) or not np.isfinite(number) or (
                number <= 0 if positive else number < 0
            ):
                raise ValueError(f"policy {label} is out of range")
            numeric[label] = number
        object.__setattr__(self, "enabled_capabilities", capabilities)
        object.__setattr__(self, "feature_names", names)
        object.__setattr__(self, "feature_groups", dict(sorted(groups.items())))
        for label, number in numeric.items():
            object.__setattr__(self, label, number)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "api_version": self.api_version,
            "enabled_capabilities": list(self.enabled_capabilities),
            "feature_names": list(self.feature_names),
            "feature_groups": {
                name: list(value) for name, value in self.feature_groups.items()
            },
            "mean_clip": float(self.mean_clip),
            "default_alpha": float(self.default_alpha),
            "default_eta": float(self.default_eta),
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())


@dataclass(frozen=True)
class PolicyRoundInput:
    round_index: int
    history_features: np.ndarray
    history_utilities: np.ndarray
    query_features: np.ndarray
    history_candidate_ids: tuple[str, ...]
    history_rounds: tuple[int, ...]
    measured_observations: tuple[Mapping[str, Any], ...]
    research_snapshot: Mapping[str, Any]
    execution_context: Mapping[str, Any]
    diagnostic_arrays: Mapping[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.round_index, bool) or not isinstance(self.round_index, int) or self.round_index < 0:
            raise ValueError("policy round_index must be non-negative")
        history = _readonly_array(self.history_features, dimensions=2, name="history_features")
        utilities = _readonly_array(
            self.history_utilities,
            dimensions=(1, 2),
            name="history_utilities",
        )
        query = _readonly_array(self.query_features, dimensions=2, name="query_features")
        if len(history) != len(utilities):
            raise ValueError("policy history features and utilities must have equal length")
        if utilities.ndim == 2 and utilities.shape[1] == 0:
            raise ValueError("policy history must contain at least one objective")
        if history.shape[1] != query.shape[1]:
            raise ValueError("policy history and query features must have equal width")
        if len(query) == 0:
            raise ValueError("policy query_features must contain at least one row")
        ids = tuple(self.history_candidate_ids)
        rounds = tuple(self.history_rounds)
        measured = tuple(json_mapping(row, "measured observation") for row in self.measured_observations)
        if not (len(ids) == len(rounds) == len(measured) == len(history)):
            raise ValueError("policy history identifiers, rounds and measurements must align with numeric history")
        if any(not isinstance(value, str) or not value.strip() for value in ids):
            raise ValueError("policy history candidate IDs must be non-empty strings")
        if any(isinstance(value, bool) or not isinstance(value, int) or not -1 <= value < self.round_index for value in rounds):
            raise ValueError("policy history rounds must be initialization or preceding search rounds")
        if any(left > right for left, right in zip(rounds, rounds[1:])):
            raise ValueError("policy history must be chronological")
        research = json_mapping(self.research_snapshot, "research_snapshot")
        if {"history_candidate_ids", "history_rounds", "measured_observations"} & research.keys():
            raise ValueError("policy history belongs in explicit input fields, not research_snapshot")
        context = json_mapping(self.execution_context, "execution_context")
        if not isinstance(context.get("mean_context"), dict) or not isinstance(
            context.get("weight_context"), dict
        ):
            raise ValueError("policy execution_context requires object mean/weight contexts")
        object.__setattr__(self, "history_features", history)
        object.__setattr__(self, "history_utilities", utilities)
        object.__setattr__(self, "query_features", query)
        object.__setattr__(self, "history_candidate_ids", ids)
        object.__setattr__(self, "history_rounds", rounds)
        object.__setattr__(self, "measured_observations", measured)
        object.__setattr__(self, "research_snapshot", research)
        object.__setattr__(self, "execution_context", context)
        diagnostic_arrays = {}
        for name, value in self.diagnostic_arrays.items():
            if not name.startswith("diagnostic_") or not name.isidentifier():
                raise ValueError("invalid policy diagnostic array name")
            array = np.asarray(value, dtype=float).copy()
            if array.ndim not in (1, 2) or not np.isfinite(array).all():
                raise ValueError("policy diagnostic arrays must be finite vectors or matrices")
            array.setflags(write=False)
            diagnostic_arrays[name] = array
        object.__setattr__(self, "diagnostic_arrays", diagnostic_arrays)


@dataclass(frozen=True)
class CompiledOptimizationPolicy:
    epoch_id: str | None
    artifact_digest: str | None
    history_prior_mean: np.ndarray
    query_prior_mean: np.ndarray
    stage: str
    alpha: float
    eta: float
    source: Literal["artifact", "previous", "default"]
    degraded: bool
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.source not in {"artifact", "previous", "default"}:
            raise ValueError("unsupported compiled policy source")
        if (self.epoch_id is None) != (self.artifact_digest is None):
            raise ValueError("compiled policy epoch and artifact digest must be paired")
        if not self.stage.strip() or len(self.stage) > 64:
            raise ValueError("compiled policy stage is invalid")
        if any(
            isinstance(value, bool) or not np.isfinite(value) or value < 0
            for value in (self.alpha, self.eta)
        ):
            raise ValueError("compiled policy weights must be finite and non-negative")
        object.__setattr__(
            self,
            "history_prior_mean",
            _readonly_array(self.history_prior_mean, dimensions=(1, 2), name="history_prior_mean"),
        )
        object.__setattr__(
            self,
            "query_prior_mean",
            _readonly_array(self.query_prior_mean, dimensions=(1, 2), name="query_prior_mean"),
        )
        if self.history_prior_mean.shape[1:] != self.query_prior_mean.shape[1:]:
            raise ValueError("compiled policy prior means must have matching objective dimensions")
        if self.query_prior_mean.ndim == 2 and self.query_prior_mean.shape[1] == 0:
            raise ValueError("compiled policy prior means must contain at least one objective")
        object.__setattr__(self, "metadata", json_mapping(self.metadata, "metadata"))


class OptimizationPolicyAdapter(Protocol):
    def capability_contract(self) -> PolicyCapabilityContract: ...

    def with_feedback(
        self, round_input: PolicyRoundInput, records: Sequence[Mapping[str, Any]],
    ) -> PolicyRoundInput: ...

    def validate_task_execution(
        self,
        execution: PolicyExecutionResult,
        round_input: PolicyRoundInput,
    ) -> Sequence[HarnessSubmissionError]: ...


def policy_submission_contract(
    max_validation_attempts: int = 3,
) -> HarnessSubmissionContract:
    return HarnessSubmissionContract(
        contract_id="optimization_policy",
        tool_name="submit_optimization_policy",
        payload_schema={
            "type": "object",
            "description": (
                "Use exactly one action shape: replace with artifact_path, "
                "or keep/disable with action only."
            ),
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["replace", "keep", "disable"],
                },
                "artifact_path": {
                    "type": "string",
                    "enum": [POLICY_ARTIFACT_NAME],
                    "description": "Include only when action is replace.",
                },
            },
            "required": ["action"],
            "additionalProperties": False,
            "allOf": [{
                "if": {
                    "properties": {"action": {"const": "replace"}},
                    "required": ["action"],
                },
                "then": {"required": ["artifact_path"]},
                "else": {"not": {"required": ["artifact_path"]}},
            }],
        },
        artifact_rules=(HarnessArtifactRule(
            path_pointer="/artifact_path",
            allowed_suffixes=(".py",),
            max_bytes=POLICY_ARTIFACT_MAX_BYTES,
        ),),
        max_validation_attempts=max_validation_attempts,
    )


def _readonly_array(value: Any, *, dimensions: int | tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).copy()
    allowed = (dimensions,) if isinstance(dimensions, int) else dimensions
    if array.ndim not in allowed or not np.isfinite(array).all():
        raise ValueError(f"policy {name} must be a finite {dimensions}-dimensional array")
    array.setflags(write=False)
    return array


def json_mapping(value: Mapping[str, Any], name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"policy {name} must be a mapping")
    try:
        copied = json.loads(json.dumps(dict(value), allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"policy {name} must be JSON serializable") from exc
    return copied
