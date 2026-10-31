"""Tests for the OpenAI / Mock LLM client abstractions."""

import sys
import types

import pytest

from tasks.small_molecule.core.llm_advisor.client import (
    LLMClient,
    MockLLMClient,
    OpenAIChatClient,
    build_default_client_from_env,
)
from tasks.small_molecule.core.llm_advisor.config import LLMClientConfig


# ---------------------------------------------------------------------------
# MockLLMClient
# ---------------------------------------------------------------------------


def test_mock_scripted_responses_serves_per_call() -> None:
    client = MockLLMClient(scripted_responses=["a", "b", "c"])
    r1 = client.chat("s", "u")
    r2 = client.chat("s", "u")
    r3 = client.chat("s", "u")
    assert (r1, r2, r3) == ("a", "b", "c")


def test_mock_exhausted_raises() -> None:
    client = MockLLMClient(scripted_responses=["a"])
    client.chat("s", "u")
    with pytest.raises(RuntimeError, match="exhausted"):
        client.chat("s", "u")


def test_mock_records_call_log() -> None:
    client = MockLLMClient(scripted_responses=["x", "y"])
    client.chat("sys1", "user1")
    client.chat("sys2", "user2")
    assert len(client.call_log) == 2
    assert client.call_log[0]["system"] == "sys1"
    assert client.call_log[1]["user"] == "user2"


def test_mock_fail_every() -> None:
    """fail_every=2 fails the 2nd, 4th, 6th, ... calls (i.e. every other call after the first)."""
    client = MockLLMClient(
        scripted_responses=["x", "y", "z"],
        fail_every=2,
    )
    r1 = client.chat("s", "u")
    r2 = client.chat("s", "u")
    r3 = client.chat("s", "u")
    assert r1 == "x"
    assert r2 == "this is not a json block"
    assert r3 == "z"


def test_mock_no_script_raises() -> None:
    client = MockLLMClient()
    with pytest.raises(RuntimeError, match="scripted_responses is not set"):
        client.chat("s", "u")


# ---------------------------------------------------------------------------
# OpenAIChatClient construction
# ---------------------------------------------------------------------------


def test_openai_client_rejects_empty_api_key() -> None:
    from tasks.small_molecule.core.llm_advisor.config import LLMClientConfig
    with pytest.raises(ValueError, match="LLM_API_KEY"):
        LLMClientConfig(api_key="", base_url="https://x", model="m")


def test_openai_client_rejects_empty_base_url() -> None:
    with pytest.raises(ValueError, match="LLM_BASE_URL"):
        LLMClientConfig(api_key="k", base_url="", model="m")


def test_openai_client_rejects_empty_model() -> None:
    with pytest.raises(ValueError, match="model name is empty"):
        LLMClientConfig(api_key="k", base_url="https://x", model="")


def test_openai_client_from_env(monkeypatch) -> None:
    """from_env reads LLM_API_KEY + LLM_BASE_URL from env; model uses
    the hardcoded default (DeepSeek-V4-Flash)."""
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://example.com/v1")
    # Wipe any LLM_MODEL that might leak from the test env to prove
    # we don't read it.
    monkeypatch.delenv("LLM_MODEL", raising=False)
    cfg = LLMClientConfig.from_env()
    assert cfg.api_key == "test-key"
    assert cfg.base_url == "https://example.com/v1"
    assert cfg.model == "DeepSeek-V4-Flash"


def test_openai_client_from_env_model_override(monkeypatch) -> None:
    """from_env(model=...) overrides the hardcoded default."""
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("LLM_BASE_URL", "https://x.com/v1")
    monkeypatch.delenv("LLM_MODEL", raising=False)
    cfg = LLMClientConfig.from_env(model="custom-model")
    assert cfg.model == "custom-model"


def test_openai_client_from_env_ignores_LLM_MODEL(monkeypatch) -> None:
    """The LLM_MODEL env var is intentionally NOT read; setting it
    has no effect on the resulting config."""
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("LLM_BASE_URL", "https://x.com/v1")
    monkeypatch.setenv("LLM_MODEL", "should-be-ignored")
    cfg = LLMClientConfig.from_env()
    assert cfg.model == "DeepSeek-V4-Flash"


def test_openai_client_strips_trailing_slash(monkeypatch) -> None:
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("LLM_BASE_URL", "https://x.com/v1/")
    monkeypatch.delenv("LLM_MODEL", raising=False)
    cfg = LLMClientConfig.from_env()
    assert cfg.base_url == "https://x.com/v1"


# ---------------------------------------------------------------------------
# LLMClient Protocol structural conformance
# ---------------------------------------------------------------------------


def test_mock_conforms_to_protocol() -> None:
    """Static check: MockLLMClient has model_name + chat()."""
    client = MockLLMClient(scripted_responses=[])
    assert hasattr(client, "model_name")
    assert hasattr(client, "chat")
    assert client.model_name == "mock-llm"


def test_build_default_client_from_env_requires_env(monkeypatch, tmp_path) -> None:
    """If .env is missing and env vars are unset, we should still try to
    use whatever's in os.environ. The end-to-end test of the real API
    endpoint is out of scope here; we just verify the construction.
    """
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("LLM_BASE_URL", "https://x.com/v1")
    monkeypatch.delenv("LLM_MODEL", raising=False)
    client = build_default_client_from_env()
    assert isinstance(client, OpenAIChatClient)
    assert client.model_name == "DeepSeek-V4-Flash"


def test_build_default_client_from_env_with_model_override(monkeypatch) -> None:
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("LLM_BASE_URL", "https://x.com/v1")
    client = build_default_client_from_env(model="custom-x")
    assert client.model_name == "custom-x"


def test_openai_client_uses_constructor_timeout_without_request_override(monkeypatch) -> None:
    captured = {}
    init_kwargs = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            message = types.SimpleNamespace(content='{"ok": true}')
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])

    class FakeOpenAI:
        def __init__(self, **kwargs):
            init_kwargs.update(kwargs)
            self.chat = types.SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeOpenAI))
    client = OpenAIChatClient(
        LLMClientConfig(api_key="key", base_url="https://example.test", model="model"),
        timeout=12.0,
    )

    assert client.chat("system", "user") == '{"ok": true}'
    assert init_kwargs["timeout"] == 12.0
    assert "timeout" not in captured


def test_openai_client_passes_optional_max_tokens(monkeypatch) -> None:
    captured = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            message = types.SimpleNamespace(content='{"ok": true}')
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = types.SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeOpenAI))
    client = OpenAIChatClient(
        LLMClientConfig(api_key="key", base_url="https://example.test", model="model"),
        max_tokens=1536,
    )

    assert client.chat("system", "user") == '{"ok": true}'
    assert captured["max_tokens"] == 1536


def test_openai_client_passes_optional_sampling_and_extra_body(monkeypatch) -> None:
    captured = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            message = types.SimpleNamespace(content='{"ok": true}')
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = types.SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeOpenAI))
    client = OpenAIChatClient(
        LLMClientConfig(api_key="key", base_url="https://example.test", model="model"),
        top_p=0.95,
        presence_penalty=1.5,
        extra_body={
            "top_k": 20,
            "min_p": 0.0,
            "repetition_penalty": 1.0,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )

    assert client.chat("system", "user") == '{"ok": true}'
    assert captured["top_p"] == 0.95
    assert captured["presence_penalty"] == 1.5
    assert captured["extra_body"]["top_k"] == 20
    assert captured["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False


def test_build_default_client_from_env_reads_optional_max_tokens(monkeypatch) -> None:
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("LLM_BASE_URL", "https://x.com/v1")
    monkeypatch.setenv("LDM_LLM_MAX_TOKENS", "1536")

    client = build_default_client_from_env()

    assert client.max_tokens == 1536


# ---------------------------------------------------------------------------
# load_env() behavior with .env and env vars
# ---------------------------------------------------------------------------


def test_load_env_silent_when_dotenv_missing(monkeypatch, tmp_path) -> None:
    """load_env() is a no-op when no .env exists; it does not raise."""
    from tasks.small_molecule.core.llm_advisor.config import _project_root, load_env
    # Point _project_root to a tempdir so the real .env is not found.
    # Easiest: clear env vars and trust the real .env is missing — but
    # in CI it might exist. So we monkey-patch _project_root.
    monkeypatch.setattr(
        "tasks.small_molecule.core.llm_advisor.config._project_root",
        lambda: tmp_path,
    )
    # No .env in tmp_path.
    assert not (tmp_path / ".env").exists()
    load_env()  # must not raise


def test_load_env_does_not_overwrite_existing_env_var(monkeypatch, tmp_path) -> None:
    """load_env() honors override=False: env vars win over .env values."""
    # Create a .env in tmp_path with a different api_key.
    env_file = tmp_path / ".env"
    env_file.write_text(
        "LLM_API_KEY=from-dotenv\n"
        "LLM_BASE_URL=https://from-dotenv.example/v1\n",
        encoding="utf-8",
    )
    # Set env vars with different values; load_env() must NOT clobber them.
    monkeypatch.setenv("LLM_API_KEY", "from-env")
    monkeypatch.setenv("LLM_BASE_URL", "https://from-env.example/v1")
    monkeypatch.setattr(
        "tasks.small_molecule.core.llm_advisor.config._project_root",
        lambda: tmp_path,
    )
    from tasks.small_molecule.core.llm_advisor.config import load_env
    load_env()
    # Env vars still win.
    assert __import__("os").environ["LLM_API_KEY"] == "from-env"
    assert __import__("os").environ["LLM_BASE_URL"] == "https://from-env.example/v1"


def test_from_env_clear_error_message_when_no_dotenv(monkeypatch, tmp_path) -> None:
    """Without .env and without env vars, from_env() raises with a clear message."""
    # Make sure no env vars leak in.
    for var in ("LLM_API_KEY", "LLM_BASE_URL"):
        monkeypatch.delenv(var, raising=False)
    # Point to an empty tmpdir (no .env there).
    monkeypatch.setattr(
        "tasks.small_molecule.core.llm_advisor.config._project_root",
        lambda: tmp_path,
    )
    from tasks.small_molecule.core.llm_advisor.config import LLMClientConfig
    with pytest.raises(ValueError, match="LLM_API_KEY is empty; set it in .env"):
        LLMClientConfig.from_env()
