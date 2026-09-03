"""Task-neutral protocol values for persistent research harnesses."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from ldm_tts.harness.guest_runtime import HarnessGuestRuntime

_SHA256_PATTERN = re.compile(r"[a-f0-9]{64}")
_SEARCH_FALLBACK_KINDS = frozenset(
    {"transient", "quota", "network", "invalid-response", "unsupported"}
)
DEFAULT_NETWORK_TOOL_BUDGETS = (
    "web_search=8",
    "fetch_content=16",
    "get_search_content=16",
    "resolve-library-id=4",
    "query-docs=8",
)


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_sha256(path: Path) -> str:
    root = Path(path).resolve()
    files: list[dict[str, str]] = []

    def visit(directory: Path) -> None:
        with os.scandir(directory) as entries:
            for entry in sorted(entries, key=lambda item: os.fsencode(item.name)):
                entry_path = Path(entry.path)
                if entry.is_dir(follow_symlinks=False):
                    visit(entry_path)
                elif entry.is_file(follow_symlinks=False):
                    files.append({
                        "path": entry_path.relative_to(root).as_posix(),
                        "sha256": file_sha256(entry_path),
                    })

    visit(root)
    return canonical_sha256(files)


@dataclass(frozen=True)
class HarnessArtifactRule:
    path_pointer: str
    allowed_suffixes: tuple[str, ...]
    max_bytes: int

    def __post_init__(self) -> None:
        if not self.path_pointer.startswith("/"):
            raise ValueError("harness artifact path_pointer must be a non-root JSON Pointer")
        if not self.allowed_suffixes or any(
            not suffix.startswith(".") or "/" in suffix or "\\" in suffix
            for suffix in self.allowed_suffixes
        ):
            raise ValueError("harness artifact suffixes must be non-empty file suffixes")
        if len(set(self.allowed_suffixes)) != len(self.allowed_suffixes):
            raise ValueError("harness artifact suffixes must be unique")
        if isinstance(self.max_bytes, bool) or self.max_bytes < 1:
            raise ValueError("harness artifact max_bytes must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "pathPointer": self.path_pointer,
            "allowedSuffixes": list(self.allowed_suffixes),
            "maxBytes": self.max_bytes,
        }


@dataclass(frozen=True)
class HarnessSubmissionContract:
    contract_id: str
    tool_name: str
    payload_schema: Mapping[str, Any]
    artifact_rules: tuple[HarnessArtifactRule, ...] = ()
    max_validation_attempts: int | None = None

    def __post_init__(self) -> None:
        if re.fullmatch(r"[a-z][a-z0-9_]*", self.contract_id) is None:
            raise ValueError("harness contract_id must be a lowercase identifier")
        if re.fullmatch(r"[A-Za-z0-9_-]+", self.tool_name) is None:
            raise ValueError("harness terminal tool_name must be a function identifier")
        try:
            schema = json.loads(json.dumps(dict(self.payload_schema)))
        except (TypeError, ValueError) as exc:
            raise ValueError("harness payload_schema must be JSON serializable") from exc
        if schema.get("type") != "object" or schema.get("additionalProperties") is not False:
            raise ValueError("harness payload_schema must be a strict JSON object schema")
        if len({rule.path_pointer for rule in self.artifact_rules}) != len(self.artifact_rules):
            raise ValueError("harness artifact path pointers must be unique")
        if self.max_validation_attempts is not None and (
            isinstance(self.max_validation_attempts, bool)
            or self.max_validation_attempts < 1
        ):
            raise ValueError("harness max_validation_attempts must be positive")
        object.__setattr__(self, "payload_schema", schema)

    def to_dict(self) -> dict[str, Any]:
        return {
            "contractId": self.contract_id,
            "toolName": self.tool_name,
            "payloadSchema": dict(self.payload_schema),
            "artifactRules": [rule.to_dict() for rule in self.artifact_rules],
            "maxValidationAttempts": self.max_validation_attempts,
        }

    @property
    def canonical_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class HarnessProfile:
    profile_id: str
    agents_path: Path
    skill_dirs: tuple[Path, ...] = ()
    agents_sha256: str = ""
    skill_dir_sha256: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if re.fullmatch(r"[a-z][a-z0-9_]*", self.profile_id) is None:
            raise ValueError("harness profile_id must be a lowercase identifier")
        if _SHA256_PATTERN.fullmatch(self.agents_sha256) is None:
            raise ValueError("harness agents_sha256 must be a lowercase SHA-256 digest")
        if len(self.skill_dirs) != len(self.skill_dir_sha256):
            raise ValueError("harness skill directories and digests must have equal length")
        if any(_SHA256_PATTERN.fullmatch(value) is None for value in self.skill_dir_sha256):
            raise ValueError("harness skill directory digests must be lowercase SHA-256 values")

    def to_dict(self) -> dict[str, Any]:
        return {
            "profileId": self.profile_id,
            "agentsPath": str(self.agents_path),
            "agentsSha256": self.agents_sha256,
            "skillDirs": [str(path) for path in self.skill_dirs],
            "skillDirSha256": list(self.skill_dir_sha256),
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


@dataclass(frozen=True)
class HarnessMcpValue:
    value: str | None = None
    secret_name: str | None = None
    secret_source: str | None = None
    prefix: str = ""

    def __post_init__(self) -> None:
        if (self.value is None) == (self.secret_name is None):
            raise ValueError("MCP values require exactly one literal or secret")
        if self.secret_name is not None and not self.secret_source:
            raise ValueError("MCP secret values require a source description")
        if self.value is not None and (self.secret_source is not None or self.prefix):
            raise ValueError("MCP literal values cannot declare secret metadata")

    def to_dict(self) -> dict[str, str]:
        if self.value is not None:
            return {"value": self.value}
        assert self.secret_name is not None and self.secret_source is not None
        return {
            "secretName": self.secret_name,
            "secretSource": self.secret_source,
            "prefix": self.prefix,
        }


@dataclass(frozen=True)
class HarnessMcpServer:
    server_id: str
    transport: str
    tools: tuple[str, ...]
    config_sha256: str
    command: str | None = None
    args: tuple[str, ...] = ()
    env: tuple[tuple[str, HarnessMcpValue], ...] = ()
    url: str | None = None
    headers: tuple[tuple[str, HarnessMcpValue], ...] = ()

    def __post_init__(self) -> None:
        if re.fullmatch(r"[a-z][a-z0-9_]*", self.server_id) is None:
            raise ValueError("MCP server_id must be a lowercase identifier")
        if self.transport not in {"stdio", "streamable_http"}:
            raise ValueError("unsupported MCP transport")
        if not self.tools or len(set(self.tools)) != len(self.tools):
            raise ValueError("MCP tools must be a non-empty unique allowlist")
        if any(re.fullmatch(r"[A-Za-z0-9_-]+", name) is None for name in self.tools):
            raise ValueError("MCP tool names must be valid function identifiers")
        if any(len(f"mcp__{self.server_id}__{name}") > 64 for name in self.tools):
            raise ValueError("namespaced MCP tool names must not exceed 64 characters")
        if _SHA256_PATTERN.fullmatch(self.config_sha256) is None:
            raise ValueError("MCP config_sha256 must be a lowercase SHA-256 digest")
        if self.transport == "stdio" and (not self.command or self.url is not None or self.headers):
            raise ValueError("stdio MCP servers require command and prohibit HTTP fields")
        if self.transport == "streamable_http" and (
            not self.url or self.command is not None or self.args or self.env
        ):
            raise ValueError("HTTP MCP servers require url and prohibit stdio fields")

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "serverId": self.server_id,
            "transport": self.transport,
            "tools": list(self.tools),
            "configSha256": self.config_sha256,
        }
        if self.transport == "stdio":
            result.update(
                command=self.command,
                args=list(self.args),
                env={name: value.to_dict() for name, value in self.env},
            )
        else:
            result.update(
                url=self.url,
                headers={name: value.to_dict() for name, value in self.headers},
            )
        return result


def profile_set_sha256(profiles: tuple[HarnessProfile, ...]) -> str:
    return canonical_sha256([
        {
            "agentsSha256": profile.agents_sha256,
            "profileId": profile.profile_id,
            "skillDirSha256": list(profile.skill_dir_sha256),
        }
        for profile in profiles
    ])


@dataclass(frozen=True)
class HarnessLimits:
    wall_time_seconds: int = 1800
    tool_call_budgets: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.wall_time_seconds < 1:
            raise ValueError("harness wall-time limit must be positive")
        budgets = dict(self.tool_call_budgets)
        if any(
            re.fullmatch(r"[A-Za-z0-9_-]+", name) is None
            or isinstance(limit, bool)
            or not isinstance(limit, int)
            or limit < 0
            for name, limit in budgets.items()
        ):
            raise ValueError("harness tool budgets require valid names and non-negative integers")
        object.__setattr__(self, "tool_call_budgets", dict(sorted(budgets.items())))

    def to_dict(self) -> dict[str, Any]:
        return {
            "wallTimeSeconds": self.wall_time_seconds,
            "toolCallBudgets": dict(self.tool_call_budgets),
        }


def parse_tool_call_budgets(
    values: Sequence[str],
    *,
    excluded_tools: Sequence[str] = (),
) -> dict[str, int]:
    budgets: dict[str, int] = {}
    excluded = set(excluded_tools)
    for value in values:
        name, separator, raw_limit = value.partition("=")
        if not separator or re.fullmatch(r"[A-Za-z0-9_-]+", name) is None:
            raise ValueError("harness tool budgets must use NAME=COUNT")
        if name in excluded:
            raise ValueError(f"harness terminal tool cannot have a call budget: {name}")
        if name in budgets:
            raise ValueError(f"duplicate harness tool budget: {name}")
        if not raw_limit.isdigit():
            raise ValueError(f"harness tool budget for {name} must be a non-negative integer")
        budgets[name] = int(raw_limit)
    return dict(HarnessLimits(tool_call_budgets=budgets).tool_call_budgets)


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
class HarnessWebSearch:
    providers: tuple[str, ...] = ("parallel-mcp", "exa", "duckduckgo")
    fallback_on: tuple[str, ...] = (
        "transient",
        "quota",
        "network",
        "invalid-response",
        "unsupported",
    )

    def __post_init__(self) -> None:
        if not self.providers:
            raise ValueError("harness web search requires at least one provider")
        if len(set(self.providers)) != len(self.providers):
            raise ValueError("harness web search providers must be unique")
        if any(
            re.fullmatch(r"[a-z][a-z0-9-]*", provider) is None
            or provider in {"auto", "all"}
            for provider in self.providers
        ):
            raise ValueError(
                "harness web search providers must be resolved lowercase provider names"
            )
        if not self.fallback_on:
            raise ValueError("harness web search fallback_on must not be empty")
        if len(set(self.fallback_on)) != len(self.fallback_on):
            raise ValueError("harness web search fallback kinds must be unique")
        if any(kind not in _SEARCH_FALLBACK_KINDS for kind in self.fallback_on):
            raise ValueError("unsupported harness web search fallback kind")

    def to_dict(self) -> dict[str, Any]:
        return {
            "providers": list(self.providers),
            "fallbackOn": list(self.fallback_on),
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
    submission_contract: HarnessSubmissionContract
    guest_runtime: HarnessGuestRuntime
    tool_extensions: tuple[HarnessToolExtension, ...] = ()
    mcp_servers: tuple[HarnessMcpServer, ...] = ()
    thinking: str = "off"
    limits: HarnessLimits = field(default_factory=HarnessLimits)
    network_policy: HarnessNetworkPolicy = field(default_factory=HarnessNetworkPolicy)
    web_search: HarnessWebSearch = field(default_factory=HarnessWebSearch)
    context7_enabled: bool = True

    def __post_init__(self) -> None:
        if not self.base_url.strip() or not self.model.strip():
            raise ValueError("harness base_url and model are required")
        if not self.campaign_id.strip() or not self.task_id.strip() or not self.case_id.strip():
            raise ValueError("harness campaign, task, and case identities are required")
        if self.seed < 0:
            raise ValueError("harness seed must be non-negative")
        if not self.profiles:
            raise ValueError("harness requires at least one profile")
        if len({profile.profile_id for profile in self.profiles}) != len(self.profiles):
            raise ValueError("harness profile_id values must be unique")
        tool_names = [name for extension in self.tool_extensions for name in extension.tool_names]
        if len(set(tool_names)) != len(tool_names):
            raise ValueError("harness tool names must be unique across extensions")
        server_ids = [server.server_id for server in self.mcp_servers]
        if len(set(server_ids)) != len(server_ids):
            raise ValueError("harness MCP server IDs must be unique")
        mcp_tool_names = [
            f"mcp__{server.server_id}__{name}"
            for server in self.mcp_servers
            for name in server.tools
        ]
        if len(set(mcp_tool_names)) != len(mcp_tool_names):
            raise ValueError("harness MCP tool names must be unique")
        if set(tool_names) & set(mcp_tool_names):
            raise ValueError("harness task and MCP tool names must not conflict")
        ordinary_tools = {
            "read", "write", "bash", "web_search", "fetch_content",
            "get_search_content", *tool_names, *mcp_tool_names,
        }
        if self.context7_enabled:
            ordinary_tools.update(("resolve-library-id", "query-docs"))
        if self.submission_contract.tool_name in ordinary_tools:
            raise ValueError("harness terminal tool conflicts with another available tool")
        if self.submission_contract.tool_name in self.limits.tool_call_budgets:
            raise ValueError(
                "harness terminal tool cannot have a call budget: "
                + self.submission_contract.tool_name
            )
        unknown_budgets = set(self.limits.tool_call_budgets) - ordinary_tools
        if unknown_budgets:
            raise ValueError(
                "harness tool budgets reference unavailable tools: "
                + ", ".join(sorted(unknown_budgets))
            )
        if self.thinking not in {"off", "minimal", "low", "medium", "high", "xhigh", "max"}:
            raise ValueError("unsupported harness thinking level")

    @property
    def profile_set_sha256(self) -> str:
        return profile_set_sha256(self.profiles)

    def initialize_payload(self) -> dict[str, Any]:
        return {
            "artifactRoot": str(self.artifact_root),
            "baseUrl": self.base_url,
            "wireApi": "responses",
            "model": self.model,
            "thinking": self.thinking,
            "taskId": self.task_id,
            "caseId": self.case_id,
            "seed": self.seed,
            "submissionContractJson": self.submission_contract.canonical_json,
            "submissionContractSha256": self.submission_contract.sha256,
            "guestRuntime": self.guest_runtime.to_dict(),
            "profileSetSha256": self.profile_set_sha256,
            "profiles": [profile.to_dict() for profile in self.profiles],
            "toolExtensions": [extension.to_dict() for extension in self.tool_extensions],
            "mcpServers": [server.to_dict() for server in self.mcp_servers],
            "networkPolicy": self.network_policy.to_dict(),
            "limits": self.limits.to_dict(),
            "webSearch": self.web_search.to_dict(),
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
class HarnessSubmittedArtifact:
    path_pointer: str
    relative_path: str
    snapshot_path: str
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        if not self.path_pointer.startswith("/"):
            raise ValueError("harness artifact path_pointer must be a non-root JSON Pointer")
        for name, value in (
            ("relative_path", self.relative_path),
            ("snapshot_path", self.snapshot_path),
        ):
            path = PurePosixPath(value)
            if (
                not value
                or "\\" in value
                or ":" in value
                or path.is_absolute()
                or any(part in {"", ".", ".."} for part in path.parts)
            ):
                raise ValueError(f"harness artifact {name} must be a safe POSIX relative path")
        if _SHA256_PATTERN.fullmatch(self.sha256) is None:
            raise ValueError("harness artifact sha256 must be a lowercase SHA-256 digest")
        if isinstance(self.size_bytes, bool) or self.size_bytes < 0:
            raise ValueError("harness artifact size_bytes must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "pathPointer": self.path_pointer,
            "relativePath": self.relative_path,
            "snapshotPath": self.snapshot_path,
            "sha256": self.sha256,
            "sizeBytes": self.size_bytes,
        }


@dataclass(frozen=True)
class HarnessSubmissionError:
    path: str
    code: str
    message: str
    hint: str = ""

    def __post_init__(self) -> None:
        if self.path and not self.path.startswith("/"):
            raise ValueError("harness submission error path must be a JSON Pointer")
        if re.fullmatch(r"[a-z][a-z0-9_]*", self.code) is None:
            raise ValueError("harness submission error code must be a lowercase identifier")
        if not self.message.strip():
            raise ValueError("harness submission error message must not be empty")

    def to_dict(self) -> dict[str, str]:
        return {
            "path": self.path,
            "code": self.code,
            "message": self.message,
            "hint": self.hint,
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
    replayed: bool
    submission_status: Literal["accepted", "rejected"]
    submission_id: str
    submission_digest: str
    submission: Mapping[str, Any]
    submitted_artifacts: tuple[HarnessSubmittedArtifact, ...]
    validation_errors: tuple[HarnessSubmissionError, ...]
    usage: dict[str, Any]
    tool_budget: dict[str, dict[str, int]]
    artifacts: dict[str, str]


@dataclass(frozen=True)
class HarnessSubmissionRequest:
    profile_id: str
    turn_id: str
    attempt_index: int
    submission: Mapping[str, Any]
    artifacts: tuple[HarnessSubmittedArtifact, ...] = ()

    def __post_init__(self) -> None:
        if not self.profile_id or not self.turn_id:
            raise ValueError("harness submission identity must not be empty")
        if self.attempt_index < 1:
            raise ValueError("harness submission attempt_index must be positive")
        if not isinstance(self.submission, Mapping):
            raise ValueError("harness submission payload must be a mapping")
        object.__setattr__(self, "submission", dict(self.submission))

    @property
    def digest(self) -> str:
        return canonical_sha256({
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "submission": dict(self.submission),
        })


@dataclass(frozen=True)
class HarnessSubmissionValidation:
    decision: Literal["accept", "retry", "reject_turn"] = "accept"
    errors: tuple[HarnessSubmissionError, ...] = ()

    def __post_init__(self) -> None:
        if self.decision not in {"accept", "retry", "reject_turn"}:
            raise ValueError("unsupported harness submission decision")
        if self.decision == "accept" and self.errors:
            raise ValueError("accepted harness submissions cannot contain errors")
        if self.decision != "accept" and not self.errors:
            raise ValueError("rejected harness submissions require errors")

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "errors": [error.to_dict() for error in self.errors],
        }


__all__ = [
    "DEFAULT_NETWORK_TOOL_BUDGETS",
    "HarnessGuestRuntime",
    "HarnessArtifactRule",
    "HarnessLimits",
    "HarnessMcpServer",
    "HarnessMcpValue",
    "HarnessNetworkPolicy",
    "HarnessPoolConfig",
    "HarnessProfile",
    "HarnessSubmissionContract",
    "HarnessSubmissionError",
    "HarnessSubmissionRequest",
    "HarnessSubmissionValidation",
    "HarnessSubmittedArtifact",
    "HarnessToolExtension",
    "HarnessTurn",
    "HarnessTurnResult",
    "HarnessWebSearch",
    "canonical_sha256",
    "directory_sha256",
    "file_sha256",
    "parse_tool_call_budgets",
    "profile_set_sha256",
]
