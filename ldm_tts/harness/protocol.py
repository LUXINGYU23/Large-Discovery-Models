"""Task-neutral protocol values for persistent research harnesses."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class HarnessProfile:
    profile_id: str
    agents_path: Path
    candidates_per_turn: int
    skill_dirs: tuple[Path, ...] = ()

    def __post_init__(self) -> None:
        if re.fullmatch(r"[a-z][a-z0-9_]*", self.profile_id) is None:
            raise ValueError("harness profile_id must be a lowercase identifier")
        if self.candidates_per_turn < 1:
            raise ValueError("harness candidates_per_turn must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "profileId": self.profile_id,
            "agentsPath": str(self.agents_path),
            "skillDirs": [str(path) for path in self.skill_dirs],
            "candidatesPerTurn": self.candidates_per_turn,
        }


@dataclass(frozen=True)
class HarnessLimits:
    wall_time_seconds: int = 900
    provider_calls: int = 24
    web_calls: int = 12
    context7_calls: int = 4
    artifact_bytes: int = 256 * 1024 * 1024

    def __post_init__(self) -> None:
        if min(
            self.wall_time_seconds,
            self.provider_calls,
            self.web_calls,
            self.context7_calls,
            self.artifact_bytes,
        ) < 1:
            raise ValueError("harness limits must be positive")

    def to_dict(self) -> dict[str, int]:
        return {
            "wallTimeSeconds": self.wall_time_seconds,
            "providerCalls": self.provider_calls,
            "webCalls": self.web_calls,
            "context7Calls": self.context7_calls,
            "artifactBytes": self.artifact_bytes,
        }


@dataclass(frozen=True)
class HarnessNetworkPolicy:
    allowed_hosts: tuple[str, ...] = ()
    denied_hosts: tuple[str, ...] = ()
    forbidden_query_patterns: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, list[str]]:
        return {
            "allowedHosts": list(self.allowed_hosts),
            "deniedHosts": list(self.denied_hosts),
            "forbiddenQueryPatterns": list(self.forbidden_query_patterns),
        }


@dataclass(frozen=True)
class HarnessPoolConfig:
    artifact_root: Path
    base_url: str
    model: str
    profiles: tuple[HarnessProfile, ...]
    thinking: str = "off"
    limits: HarnessLimits = field(default_factory=HarnessLimits)
    network_policy: HarnessNetworkPolicy = field(default_factory=HarnessNetworkPolicy)
    context7_enabled: bool = True

    def __post_init__(self) -> None:
        if not self.base_url.strip() or not self.model.strip():
            raise ValueError("harness base_url and model are required")
        if not self.profiles:
            raise ValueError("harness requires at least one profile")
        if len({profile.profile_id for profile in self.profiles}) != len(self.profiles):
            raise ValueError("harness profile_id values must be unique")
        if self.thinking not in {"off", "minimal", "low", "medium", "high", "xhigh", "max"}:
            raise ValueError("unsupported harness thinking level")

    def initialize_frame(self, request_id: str) -> dict[str, Any]:
        return {
            "type": "initialize",
            "requestId": request_id,
            "protocolVersion": 1,
            "artifactRoot": str(self.artifact_root),
            "baseUrl": self.base_url,
            "wireApi": "responses",
            "model": self.model,
            "thinking": self.thinking,
            "profiles": [profile.to_dict() for profile in self.profiles],
            "networkPolicy": self.network_policy.to_dict(),
            "limits": self.limits.to_dict(),
            "webProvider": "anysearch",
            "context7Enabled": self.context7_enabled,
        }


@dataclass(frozen=True)
class HarnessTurn:
    profile_id: str
    turn_id: str
    message: str
    forbidden_query_terms: tuple[str, ...] = ()

    @property
    def input_digest(self) -> str:
        value = {
            "profileId": self.profile_id,
            "turnId": self.turn_id,
            "message": self.message,
            "forbiddenQueryTerms": list(self.forbidden_query_terms),
        }
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "profileId": self.profile_id,
            "turnId": self.turn_id,
            "inputDigest": self.input_digest,
            "message": self.message,
            "forbiddenQueryTerms": list(self.forbidden_query_terms),
        }


@dataclass(frozen=True)
class HarnessTurnResult:
    profile_id: str
    session_id: str
    turn_id: str
    input_digest: str
    submission_id: str
    candidates: tuple[dict[str, Any], ...]
    usage: dict[str, int | float]
    artifacts: dict[str, str]


__all__ = [
    "HarnessLimits",
    "HarnessNetworkPolicy",
    "HarnessPoolConfig",
    "HarnessProfile",
    "HarnessTurn",
    "HarnessTurnResult",
]
