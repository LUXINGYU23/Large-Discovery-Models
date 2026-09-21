"""Pi-specific provider, tools, guest runtime, and policy MCP configuration."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any

from ldm_tts.harness.pi_guest import PiGuestRuntime, load_pi_guest_runtime
from ldm_tts.harness.protocol import (
    HarnessMcpServer, HarnessMcpValue, HarnessNetworkPolicy,
    HarnessPoolConfig, HarnessToolExtension, canonical_sha256,
)

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


@dataclass(frozen=True)
class PiWebSearch:
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
class PiHarnessConfig(HarnessPoolConfig):
    base_url: str
    model: str
    guest_runtime: PiGuestRuntime
    tool_extensions: tuple[HarnessToolExtension, ...] = ()
    mcp_servers: tuple[HarnessMcpServer, ...] = ()
    thinking: str = "off"
    network_policy: HarnessNetworkPolicy = field(default_factory=HarnessNetworkPolicy)
    web_search: PiWebSearch = field(default_factory=PiWebSearch)
    context7_enabled: bool = True
    provider_request_body: dict[str, Any] = field(default_factory=dict)
    sol_pi: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.base_url.strip() or not self.model.strip():
            raise ValueError("harness base_url and model are required")
        super().__post_init__()
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
        if self.sol_pi is not None:
            if self.sol_pi.get("actionFusion"):
                ordinary_tools.add("edit")
            if self.sol_pi.get("observationPack"):
                ordinary_tools.add("obs_recall")
            if self.sol_pi.get("onlineContextCompact"):
                ordinary_tools.add("update_plan")
        if self.submission_contract.tool_name in ordinary_tools:
            raise ValueError("harness terminal tool conflicts with another available tool")
        unknown_budgets = set(self.limits.tool_call_budgets) - ordinary_tools
        if unknown_budgets:
            raise ValueError(
                "harness tool budgets reference unavailable tools: "
                + ", ".join(sorted(unknown_budgets))
            )
        if self.thinking not in {"off", "minimal", "low", "medium", "high", "xhigh", "max"}:
            raise ValueError("unsupported harness thinking level")

    def initialize_payload(self) -> dict[str, Any]:
        return {
            **super().initialize_payload(),
            "baseUrl": self.base_url,
            "wireApi": "responses",
            "model": self.model,
            "thinking": self.thinking,
            "guestRuntime": self.guest_runtime.to_dict(),
            "toolExtensions": [extension.to_dict() for extension in self.tool_extensions],
            "mcpServers": [server.to_dict() for server in self.mcp_servers],
            "networkPolicy": self.network_policy.to_dict(),
            "webSearch": self.web_search.to_dict(),
            "context7Enabled": self.context7_enabled,
            **({"providerRequestBody": self.provider_request_body} if self.provider_request_body else {}),
            **({"solPi": self.sol_pi} if self.sol_pi is not None else {}),
        }


def policy_mcp_server(
    artifact_root: str = "/artifacts",
    profile_id: str = "policy_architect",
    *,
    diagnostics_path: str | None = None,
    diagnostics_sha256: str | None = None,
) -> HarnessMcpServer:
    root = PurePosixPath(artifact_root)
    if not root.is_absolute():
        raise ValueError("built-in policy MCP artifact root must be absolute")
    workspace = root / "sessions" / profile_id / "workspace"
    fields = {
        "server_id": "ldm_policy",
        "transport": "stdio",
        "tools": [
            "inspect_policy_contract",
            "validate_policy_draft",
            "evaluate_policy_draft",
        ],
        "command": "node",
        "args": ["/app/dist/policy-mcp.js", "stdio"],
        "env": {
            "LDM_POLICY_ROOT": str(root),
            "LDM_POLICY_WORKSPACE": str(workspace),
        },
    }
    if (diagnostics_path is None) != (diagnostics_sha256 is None):
        raise ValueError("task diagnostics require a path and SHA-256 digest")
    if diagnostics_path is not None:
        if not PurePosixPath(diagnostics_path).is_absolute() or (
            len(diagnostics_sha256) != 64
            or any(char not in "0123456789abcdef" for char in diagnostics_sha256)
        ):
            raise ValueError("task diagnostics require an absolute path and SHA-256 digest")
        fields["env"]["LDM_POLICY_DIAGNOSTICS"] = diagnostics_path
        fields["env"]["LDM_POLICY_DIAGNOSTICS_SHA256"] = diagnostics_sha256
    return HarnessMcpServer(
        server_id=fields["server_id"],
        transport=fields["transport"],
        tools=tuple(fields["tools"]),
        config_sha256=canonical_sha256(fields),
        command=fields["command"],
        args=tuple(fields["args"]),
        env=tuple(
            (name, HarnessMcpValue(value=value))
            for name, value in sorted(fields["env"].items())
        ),
    )
