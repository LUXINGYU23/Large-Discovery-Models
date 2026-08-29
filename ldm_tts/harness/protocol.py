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


@dataclass(frozen=True)
class HarnessToolExtension:
    path: Path
    sha256: str
    tool_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if _SHA256_PATTERN.fullmatch(self.sha256) is None:
            raise ValueError("harness tool extension sha256 must be a lowercase SHA-256 digest")
        if not self.tool_names or any(
            re.fullmatch(r"[a-z][a-z0-9_]*", name) is None for name in self.tool_names
        ):
            raise ValueError("harness tool extension names must be lowercase identifiers")
        if len(set(self.tool_names)) != len(self.tool_names):
            raise ValueError("harness tool extension names must be unique")

    def to_dict(self) -> dict[str, Any]:
        return {"path": str(self.path), "sha256": self.sha256, "toolNames": list(self.tool_names)}


def profile_set_sha256(profiles: tuple[HarnessProfile, ...]) -> str:
    return canonical_sha256([profile.identity() for profile in profiles])


@dataclass(frozen=True)
class HarnessLimits:
    wall_time_seconds: int = 1800

    def __post_init__(self) -> None:
        if self.wall_time_seconds < 1:
            raise ValueError("harness wall-time limit must be positive")

    def to_dict(self) -> dict[str, int]:
        return {"wallTimeSeconds": self.wall_time_seconds}


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
    candidate_schema: dict[str, Any]
    tool_extensions: tuple[HarnessToolExtension, ...] = ()
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
        if (
            not isinstance(self.candidate_schema, dict)
            or self.candidate_schema.get("type") != "object"
            or self.candidate_schema.get("additionalProperties") is not False
        ):
            raise ValueError(
                "harness candidate_schema must be a strict JSON object schema"
            )
        try:
            canonical_sha256(self.candidate_schema)
        except (TypeError, ValueError) as exc:
            raise ValueError("harness candidate_schema must be JSON serializable") from exc
        if not self.profiles:
            raise ValueError("harness requires at least one profile")
        if len({profile.profile_id for profile in self.profiles}) != len(self.profiles):
            raise ValueError("harness profile_id values must be unique")
        tool_names = [name for extension in self.tool_extensions for name in extension.tool_names]
        if len(set(tool_names)) != len(tool_names):
            raise ValueError("harness tool names must be unique across extensions")
        if self.thinking not in {"off", "minimal", "low", "medium", "high", "xhigh", "max"}:
            raise ValueError("unsupported harness thinking level")

    @property
    def profile_set_sha256(self) -> str:
        return profile_set_sha256(self.profiles)

    @property
    def candidate_schema_sha256(self) -> str:
        return canonical_sha256(self.candidate_schema)

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
            "candidateSchema": self.candidate_schema,
            "candidateSchemaSha256": self.candidate_schema_sha256,
            "profileSetSha256": self.profile_set_sha256,
            "profiles": [profile.to_dict() for profile in self.profiles],
            "toolExtensions": [extension.to_dict() for extension in self.tool_extensions],
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


@dataclass(frozen=True)
class HarnessSubmissionRequest:
    profile_id: str
    turn_id: str
    attempt_index: int
    candidates: tuple[dict[str, Any], ...]

    def __post_init__(self) -> None:
        if not self.profile_id or not self.turn_id:
            raise ValueError("harness submission identity must not be empty")
        if self.attempt_index < 1:
            raise ValueError("harness submission attempt_index must be positive")
        if not self.candidates:
            raise ValueError("harness submission candidates must not be empty")


@dataclass(frozen=True)
class HarnessSubmissionRejection:
    index: int
    code: str
    message: str

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("harness submission rejection index must be non-negative")
        if re.fullmatch(r"[a-z][a-z0-9_]*", self.code) is None:
            raise ValueError("harness submission rejection code must be a lowercase identifier")
        if not self.message.strip():
            raise ValueError("harness submission rejection message must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return {"index": self.index, "code": self.code, "message": self.message}


@dataclass(frozen=True)
class HarnessSubmissionValidation:
    rejections: tuple[HarnessSubmissionRejection, ...] = ()

    def __post_init__(self) -> None:
        indices = [rejection.index for rejection in self.rejections]
        if len(set(indices)) != len(indices):
            raise ValueError("harness submission rejection indices must be unique")

    @property
    def accepted(self) -> bool:
        return not self.rejections


__all__ = [
    "PROTOCOL_VERSION",
    "HarnessLimits",
    "HarnessNetworkPolicy",
    "HarnessPoolConfig",
    "HarnessProfile",
    "HarnessToolExtension",
    "HarnessTurn",
    "HarnessTurnResult",
    "HarnessSubmissionRequest",
    "HarnessSubmissionRejection",
    "HarnessSubmissionValidation",
    "canonical_sha256",
    "file_sha256",
    "profile_set_sha256",
]
