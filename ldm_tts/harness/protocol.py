"""Task-neutral protocol values for persistent research harnesses."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


PROTOCOL_VERSION = 2
_SHA256_PATTERN = re.compile(r"[a-f0-9]{64}")


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class HarnessProfile:
    profile_id: str
    agents_path: Path
    candidates_per_turn: int
    skill_dirs: tuple[Path, ...] = ()
    agents_sha256: str = ""
    skill_dir_sha256: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if re.fullmatch(r"[a-z][a-z0-9_]*", self.profile_id) is None:
            raise ValueError("harness profile_id must be a lowercase identifier")
        if self.candidates_per_turn < 1:
            raise ValueError("harness candidates_per_turn must be positive")
        if _SHA256_PATTERN.fullmatch(self.agents_sha256) is None:
            raise ValueError("harness agents_sha256 must be a lowercase SHA-256 digest")
        if len(self.skill_dirs) != len(self.skill_dir_sha256):
            raise ValueError("harness skill directories and digests must have equal length")
        if any(_SHA256_PATTERN.fullmatch(value) is None for value in self.skill_dir_sha256):
            raise ValueError("harness skill directory digests must be lowercase SHA-256 values")

    def identity(self) -> dict[str, Any]:
        return {
            "agentsSha256": self.agents_sha256,
            "candidatesPerTurn": self.candidates_per_turn,
            "profileId": self.profile_id,
            "skillDirSha256": list(self.skill_dir_sha256),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "profileId": self.profile_id,
            "agentsPath": str(self.agents_path),
            "agentsSha256": self.agents_sha256,
            "skillDirs": [str(path) for path in self.skill_dirs],
            "skillDirSha256": list(self.skill_dir_sha256),
            "candidatesPerTurn": self.candidates_per_turn,
        }


def profile_set_sha256(profiles: tuple[HarnessProfile, ...]) -> str:
    return canonical_sha256([profile.identity() for profile in profiles])


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
    campaign_id: str
    task_id: str
    case_id: str
    seed: int
    candidate_schema_sha256: str
    thinking: str = "off"
    limits: HarnessLimits = field(default_factory=HarnessLimits)
    network_policy: HarnessNetworkPolicy = field(default_factory=HarnessNetworkPolicy)
    context7_enabled: bool = True

    def __post_init__(self) -> None:
        if not self.base_url.strip() or not self.model.strip():
            raise ValueError("harness base_url and model are required")
        if not self.campaign_id.strip() or not self.task_id.strip() or not self.case_id.strip():
            raise ValueError("harness campaign, task, and case identities are required")
        if self.seed < 0:
            raise ValueError("harness seed must be non-negative")
        if _SHA256_PATTERN.fullmatch(self.candidate_schema_sha256) is None:
            raise ValueError("candidate_schema_sha256 must be a lowercase SHA-256 digest")
        if not self.profiles:
            raise ValueError("harness requires at least one profile")
        if len({profile.profile_id for profile in self.profiles}) != len(self.profiles):
            raise ValueError("harness profile_id values must be unique")
        if self.thinking not in {"off", "minimal", "low", "medium", "high", "xhigh", "max"}:
            raise ValueError("unsupported harness thinking level")

    @property
    def profile_set_sha256(self) -> str:
        return profile_set_sha256(self.profiles)

    def common_frame(self, request_id: str, frame_type: str) -> dict[str, Any]:
        return {
            "type": frame_type,
            "requestId": request_id,
            "protocolVersion": PROTOCOL_VERSION,
            "campaignId": self.campaign_id,
        }

    def initialize_frame(self, request_id: str) -> dict[str, Any]:
        return {
            **self.common_frame(request_id, "initialize"),
            "artifactRoot": str(self.artifact_root),
            "baseUrl": self.base_url,
            "wireApi": "responses",
            "model": self.model,
            "thinking": self.thinking,
            "taskId": self.task_id,
            "caseId": self.case_id,
            "seed": self.seed,
            "candidateSchemaSha256": self.candidate_schema_sha256,
            "profileSetSha256": self.profile_set_sha256,
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
    round_index: int
    history_from_seq: int
    history_to_seq: int
    history_digest: str
    message: str
    forbidden_query_terms: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if min(self.round_index, self.history_from_seq, self.history_to_seq) < 0:
            raise ValueError("harness turn indices must be non-negative")
        if self.history_to_seq < self.history_from_seq:
            raise ValueError("harness history_to_seq must not precede history_from_seq")
        if _SHA256_PATTERN.fullmatch(self.history_digest) is None:
            raise ValueError("harness history_digest must be a lowercase SHA-256 digest")

    @property
    def input_digest(self) -> str:
        return canonical_sha256({
            "forbiddenQueryTerms": list(self.forbidden_query_terms),
            "historyDigest": self.history_digest,
            "historyFromSeq": self.history_from_seq,
            "historyToSeq": self.history_to_seq,
            "message": self.message,
            "profileId": self.profile_id,
            "roundIndex": self.round_index,
            "turnId": self.turn_id,
        })

    def to_dict(self) -> dict[str, Any]:
        return {
            "profileId": self.profile_id,
            "turnId": self.turn_id,
            "roundIndex": self.round_index,
            "historyFromSeq": self.history_from_seq,
            "historyToSeq": self.history_to_seq,
            "historyDigest": self.history_digest,
            "inputDigest": self.input_digest,
            "message": self.message,
            "forbiddenQueryTerms": list(self.forbidden_query_terms),
        }


@dataclass(frozen=True)
class HarnessTurnResult:
    profile_id: str
    session_id: str
    turn_id: str
    round_index: int
    history_from_seq: int
    history_to_seq: int
    history_digest: str
    input_digest: str
    submission_id: str
    candidates: tuple[dict[str, Any], ...]
    usage: dict[str, int | float]
    artifacts: dict[str, str]


__all__ = [
    "PROTOCOL_VERSION",
    "HarnessLimits",
    "HarnessNetworkPolicy",
    "HarnessPoolConfig",
    "HarnessProfile",
    "HarnessTurn",
    "HarnessTurnResult",
    "canonical_sha256",
    "file_sha256",
    "profile_set_sha256",
]
