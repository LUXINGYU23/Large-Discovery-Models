"""Strict, secret-safe MCP configuration loading for Harness sidecars."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse

import yaml

from ldm_tts.harness.protocol import (
    HarnessMcpServer,
    HarnessMcpValue,
    canonical_sha256,
)


@dataclass(frozen=True)
class ResolvedHarnessMcpConfig:
    servers: tuple[HarnessMcpServer, ...] = ()
    named_secrets: Mapping[str, str] = field(default_factory=dict)


def load_harness_mcp_config(path: Path | None) -> ResolvedHarnessMcpConfig:
    if path is None:
        return ResolvedHarnessMcpConfig()
    resolved_path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
    root = _mapping(raw, "MCP config")
    _exact_keys(root, {"servers"}, "MCP config")
    raw_servers = _mapping(root["servers"], "servers")
    servers: list[HarnessMcpServer] = []
    secrets: dict[str, str] = {}
    for server_id, value in raw_servers.items():
        if not isinstance(server_id, str) or re.fullmatch(r"[a-z][a-z0-9_]*", server_id) is None:
            raise ValueError("MCP server IDs must be lowercase identifiers")
        server, server_secrets = _server(server_id, value)
        servers.append(server)
        secrets.update(server_secrets)
    return ResolvedHarnessMcpConfig(tuple(servers), secrets)


def _server(server_id: str, value: Any) -> tuple[HarnessMcpServer, dict[str, str]]:
    data = _mapping(value, f"servers.{server_id}")
    transport = data.get("transport")
    if transport == "stdio":
        _allowed_keys(
            data,
            required={"transport", "command", "tools"},
            allowed={"transport", "command", "args", "env", "tools"},
            name=server_id,
        )
        command = _nonempty_string(data["command"], f"{server_id}.command")
        args = _string_list(data.get("args", []), f"{server_id}.args")
        env, secrets = _injected_values(server_id, "env", data.get("env", {}))
        redacted = {
            "transport": transport,
            "command": command,
            "args": args,
            "env": {name: item.to_dict() for name, item in env},
            "tools": data["tools"],
        }
        return HarnessMcpServer(
            server_id=server_id,
            transport=transport,
            command=command,
            args=tuple(args),
            env=tuple(env),
            tools=_tools(data["tools"], server_id),
            config_sha256=canonical_sha256(redacted),
        ), secrets
    if transport == "streamable_http":
        _allowed_keys(
            data,
            required={"transport", "url", "tools"},
            allowed={"transport", "url", "headers", "tools"},
            name=server_id,
        )
        url = _http_url(data["url"], f"{server_id}.url")
        headers, secrets = _injected_values(server_id, "headers", data.get("headers", {}))
        redacted = {
            "transport": transport,
            "url": url,
            "headers": {name: item.to_dict() for name, item in headers},
            "tools": data["tools"],
        }
        return HarnessMcpServer(
            server_id=server_id,
            transport=transport,
            url=url,
            headers=tuple(headers),
            tools=_tools(data["tools"], server_id),
            config_sha256=canonical_sha256(redacted),
        ), secrets
    raise ValueError(f"servers.{server_id}.transport must be stdio or streamable_http")


def _injected_values(
    server_id: str,
    location: str,
    value: Any,
) -> tuple[list[tuple[str, HarnessMcpValue]], dict[str, str]]:
    data = _mapping(value, f"{server_id}.{location}")
    result: list[tuple[str, HarnessMcpValue]] = []
    secrets: dict[str, str] = {}
    for name, raw in data.items():
        if not isinstance(name, str) or not name:
            raise ValueError(f"{server_id}.{location} names must be non-empty strings")
        spec = _mapping(raw, f"{server_id}.{location}.{name}")
        if set(spec) == {"value"}:
            result.append((name, HarnessMcpValue(value=_nonempty_string(spec["value"], name))))
            continue
        allowed = {"secret_env", "prefix"} if "secret_env" in spec else {"secret_file", "prefix"}
        required = allowed - {"prefix"}
        if set(spec) != required and set(spec) != allowed:
            raise ValueError(
                f"{server_id}.{location}.{name} must contain value, secret_env, or secret_file"
            )
        source_key = next(iter(required))
        source = _nonempty_string(spec[source_key], f"{server_id}.{location}.{name}.{source_key}")
        prefix = spec.get("prefix", "")
        if not isinstance(prefix, str):
            raise ValueError(f"{server_id}.{location}.{name}.prefix must be a string")
        secret = _resolve_secret(source_key, source)
        secret_name = f"mcp.{server_id}.{location}.{canonical_sha256(name)[:16]}"
        secrets[secret_name] = secret
        result.append(
            (
                name,
                HarnessMcpValue(
                    secret_name=secret_name,
                    secret_source=f"{source_key}:{source}",
                    prefix=prefix,
                ),
            )
        )
    return result, secrets


def _resolve_secret(kind: str, source: str) -> str:
    if kind == "secret_env":
        value = os.environ.get(source, "")
    else:
        value = Path(source).expanduser().read_text(encoding="utf-8").strip()
    if not value:
        raise ValueError(f"MCP secret source {kind}:{source} is empty")
    return value


def _tools(value: Any, server_id: str) -> tuple[str, ...]:
    tools = _string_list(value, f"{server_id}.tools")
    if not tools or len(set(tools)) != len(tools):
        raise ValueError(f"{server_id}.tools must be a non-empty unique allowlist")
    if any(re.fullmatch(r"[A-Za-z0-9_-]+", name) is None for name in tools):
        raise ValueError(f"{server_id}.tools contains an invalid function name")
    return tuple(tools)


def _http_url(value: Any, name: str) -> str:
    url = _nonempty_string(value, name)
    parsed = urlparse(url)
    loopback = parsed.hostname in {"127.0.0.1", "::1", "localhost"}
    if parsed.scheme != "https" and not (parsed.scheme == "http" and loopback):
        raise ValueError(f"{name} must use HTTPS; HTTP is allowed only for loopback tests")
    if not parsed.netloc or parsed.username or parsed.password:
        raise ValueError(f"{name} must be an absolute URL without embedded credentials")
    return url


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return dict(value)


def _exact_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{name} has unexpected or missing fields")


def _allowed_keys(
    value: Mapping[str, Any],
    *,
    required: set[str],
    allowed: set[str],
    name: str,
) -> None:
    if not required <= set(value) or not set(value) <= allowed:
        raise ValueError(f"{name} has unexpected or missing fields")


def _nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _string_list(value: Any, name: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{name} must be a string array")
    return list(value)


__all__ = ["ResolvedHarnessMcpConfig", "load_harness_mcp_config"]
