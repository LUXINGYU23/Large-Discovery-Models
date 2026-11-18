"""Experiment codec plugin (design 2.3, 3.9, 9.1): candidate -> request; terminal resource -> result.

Both directions are pure and deterministic. The framework fixes the status mapping of 9.1 and
checks the codec against it; ``CANCELED`` never reaches ``decode`` (it produces no observation).
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from ldm_tts.contracts import Candidate, EvaluationResult

from .wire import canonical


@dataclass(frozen=True)
class ExperimentRequest:
    kind: str
    kind_version: str
    spec: Mapping[str, Any]
    limits: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        for name in ("kind", "kind_version"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"experiment request {name} must be a non-empty string")
        if not isinstance(self.spec, Mapping) or not isinstance(self.limits, Mapping):
            raise ValueError("experiment request spec and limits must be mappings")
        object.__setattr__(self, "spec", dict(self.spec))
        object.__setattr__(self, "limits", dict(self.limits))
        canonical([self.spec, self.limits])  # raises if not representable as canonical JSON


@runtime_checkable
class ExperimentCodec(Protocol):
    name: str
    version: str

    def encode(self, candidate: Candidate) -> ExperimentRequest: ...

    def decode(self, candidate: Candidate, resource: Mapping[str, Any]) -> EvaluationResult: ...


def expected_status(resource: Mapping[str, Any]) -> str | None:
    """Design 9.1: SUCCEEDED -> succeeded; FAILED by class -> invalid / timed_out / failed; CANCELED -> None."""
    state = resource.get("state")
    if state == "SUCCEEDED":
        return "succeeded"
    if state == "FAILED":
        failure_class = (resource.get("failure") or {}).get("class")
        return {"infeasible": "invalid", "timeout": "timed_out"}.get(failure_class, "failed")
    return None


def plugin_identity(plugin, default_name: str) -> str:
    name = getattr(plugin, "name", None) or type(plugin).__name__ or default_name
    return f"{name}@{getattr(plugin, 'version', '1')}"


class MappedCodec:
    """Convenience base: subclasses give ``to_request(candidate)`` and ``metrics(candidate, result)``.

    ``decode`` applies the 9.1 status mapping, keeps the failure message as ``error`` and copies
    numeric ES usage (``runtime_seconds``, ``unit_seconds``) into ``resource_usage``.
    """

    name = "mapped"
    version = "1"

    def to_request(self, candidate: Candidate) -> ExperimentRequest:
        raise NotImplementedError

    def metrics(self, candidate: Candidate, result: Mapping[str, Any]) -> Mapping[str, float]:
        raise NotImplementedError

    def encode(self, candidate):
        return self.to_request(candidate)

    def decode(self, candidate, resource):
        status = expected_status(resource)
        if status is None:
            raise ValueError("only SUCCEEDED and FAILED resources decode to evaluation results")
        usage = resource.get("usage") or {}
        resource_usage = {k: float(usage[k]) for k in ("runtime_seconds", "unit_seconds")
                          if isinstance(usage.get(k), (int, float)) and not isinstance(usage.get(k), bool)}
        failure = resource.get("failure") or {}
        metrics = dict(self.metrics(candidate, resource.get("result") or {})) if status == "succeeded" else {}
        return EvaluationResult(candidate_id=candidate.candidate_id, status=status, metrics=metrics,
                                resource_usage=resource_usage, error=str(failure.get("message") or ""),
                                metadata={"failure_class": failure.get("class")} if failure else {})
