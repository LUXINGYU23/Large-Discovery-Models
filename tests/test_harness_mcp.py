from __future__ import annotations

import json
from pathlib import Path

import pytest

from ldm_tts.harness import load_harness_mcp_config


def test_mcp_config_resolves_secrets_without_serializing_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret_file = tmp_path / "token"
    secret_file.write_text("file-secret\n", encoding="utf-8")
    config_path = tmp_path / "mcp.yaml"
    config_path.write_text(
        f"""
servers:
  local:
    transport: stdio
    command: node
    args: [server.js]
    env:
      API_TOKEN:
        secret_env: MCP_ENV_TOKEN
      FILE_TOKEN:
        secret_file: {json.dumps(str(secret_file))}
        prefix: "Bearer "
    tools: [search, fetch]
  remote:
    transport: streamable_http
    url: https://mcp.example/mcp
    headers:
      Authorization:
        secret_env: MCP_ENV_TOKEN
        prefix: "Bearer "
    tools: [query]
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("MCP_ENV_TOKEN", "environment-secret")

    resolved = load_harness_mcp_config(config_path)

    serialized = json.dumps([server.to_dict() for server in resolved.servers])
    assert "environment-secret" not in serialized
    assert "file-secret" not in serialized
    assert set(resolved.named_secrets.values()) == {
        "environment-secret",
        "file-secret",
    }
    assert [server.server_id for server in resolved.servers] == ["local", "remote"]


def test_mcp_config_digest_does_not_depend_on_secret_value(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "mcp.yaml"
    config_path.write_text(
        """
servers:
  remote:
    transport: streamable_http
    url: https://mcp.example/mcp
    headers:
      Authorization:
        secret_env: MCP_TOKEN
    tools: [query]
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("MCP_TOKEN", "first")
    first = load_harness_mcp_config(config_path)
    monkeypatch.setenv("MCP_TOKEN", "second")
    second = load_harness_mcp_config(config_path)

    assert first.servers == second.servers
    assert first.named_secrets != second.named_secrets


@pytest.mark.parametrize(
    "body",
    (
        "servers:\n  remote:\n    transport: streamable_http\n"
        "    url: http://mcp.example/mcp\n    tools: [query]\n",
        "servers:\n  local:\n    transport: stdio\n    command: node\n"
        "    tools: [query]\n    legacy_fallback: true\n",
    ),
)
def test_mcp_config_rejects_insecure_or_unknown_configuration(
    tmp_path: Path,
    body: str,
) -> None:
    config_path = tmp_path / "mcp.yaml"
    config_path.write_text(body, encoding="utf-8")

    with pytest.raises(ValueError):
        load_harness_mcp_config(config_path)


def test_mcp_config_rejects_an_unresolved_secret(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MCP_UNDEFINED_TOKEN", raising=False)
    config_path = tmp_path / "mcp.yaml"
    config_path.write_text(
        """
servers:
  remote:
    transport: streamable_http
    url: https://mcp.example/mcp
    headers:
      Authorization:
        secret_env: MCP_UNDEFINED_TOKEN
    tools: [query]
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="secret_env:MCP_UNDEFINED_TOKEN is empty"):
        load_harness_mcp_config(config_path)
