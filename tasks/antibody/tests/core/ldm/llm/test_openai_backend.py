"""tests/core/ldm/llm/test_openai_backend.py"""
from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import pytest


# The OpenAIClient constructor does `from openai import OpenAI` locally,
# so we must patch the openai SDK itself, not the module attribute.
OPENAI_PATCH = "openai.OpenAI"
DOTENV_LOAD_PATCH = "dotenv.load_dotenv"


@pytest.fixture(autouse=True)
def _disable_dotenv(monkeypatch):
    """Prevent load_dotenv from re-loading the real .env in tests."""
    monkeypatch.setattr(DOTENV_LOAD_PATCH, lambda *a, **kw: None)
    monkeypatch.delenv("LLM_MAX_TOKENS", raising=False)
    monkeypatch.delenv("LDM_LLM_MAX_TOKENS", raising=False)
    fake_openai = types.ModuleType("openai")
    fake_openai.OpenAI = MagicMock()
    monkeypatch.setitem(sys.modules, "openai", fake_openai)


class TestOpenAIClientInit:
    def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("LLM_API_KEY", raising=False)
        monkeypatch.delenv("LLM_BASE_URL", raising=False)
        from tasks.antibody.core.ldm.llm.openai_backend import OpenAIClient
        with pytest.raises(RuntimeError, match="LLM_API_KEY"):
            OpenAIClient()

    def test_missing_base_url_raises(self, monkeypatch):
        monkeypatch.setenv("LLM_API_KEY", "test-key")
        monkeypatch.delenv("LLM_BASE_URL", raising=False)
        from tasks.antibody.core.ldm.llm.openai_backend import OpenAIClient
        with pytest.raises(RuntimeError, match="LLM_BASE_URL"):
            OpenAIClient()

    def test_default_model_hardcoded(self, monkeypatch):
        monkeypatch.setenv("LLM_API_KEY", "k")
        monkeypatch.setenv("LLM_BASE_URL", "https://e/v1")
        from tasks.antibody.core.ldm.llm.openai_backend import OpenAIClient
        with patch(OPENAI_PATCH):
            c = OpenAIClient()
            assert c.model == "DeepSeek-V4-Flash"

    def test_env_model_overrides_default(self, monkeypatch):
        monkeypatch.setenv("LLM_API_KEY", "k")
        monkeypatch.setenv("LLM_BASE_URL", "https://e/v1")
        monkeypatch.setenv("LLM_MODEL", "env-model")
        from tasks.antibody.core.ldm.llm.openai_backend import OpenAIClient
        with patch(OPENAI_PATCH):
            c = OpenAIClient()
            assert c.model == "env-model"

    def test_explicit_model_arg_overrides_env(self, monkeypatch):
        monkeypatch.setenv("LLM_API_KEY", "k")
        monkeypatch.setenv("LLM_BASE_URL", "https://e/v1")
        monkeypatch.setenv("LLM_MODEL", "env-model")
        from tasks.antibody.core.ldm.llm.openai_backend import OpenAIClient
        with patch(OPENAI_PATCH):
            c = OpenAIClient(model="explicit-model")
            assert c.model == "explicit-model"

    def test_constructs_openai_client_with_env(self, monkeypatch):
        monkeypatch.setenv("LLM_API_KEY", "my-key")
        monkeypatch.setenv("LLM_BASE_URL", "https://my-endpoint/v1")
        from tasks.antibody.core.ldm.llm.openai_backend import OpenAIClient
        with patch(OPENAI_PATCH) as MockOpenAI:
            OpenAIClient()
            MockOpenAI.assert_called_once_with(
                api_key="my-key",
                base_url="https://my-endpoint/v1",
            )

    def test_loads_from_dotenv(self, tmp_path, monkeypatch):
        env_file = tmp_path / ".env"
        env_file.write_text("LLM_API_KEY=dotenv-key\nLLM_BASE_URL=https://dotenv.example/v1\n")
        monkeypatch.delenv("LLM_API_KEY", raising=False)
        monkeypatch.delenv("LLM_BASE_URL", raising=False)
        from dotenv import load_dotenv
        load_dotenv(env_file)
        monkeypatch.setenv("LLM_API_KEY", "dotenv-key")
        monkeypatch.setenv("LLM_BASE_URL", "https://dotenv.example/v1")

        from tasks.antibody.core.ldm.llm.openai_backend import OpenAIClient
        with patch(OPENAI_PATCH) as MockOpenAI:
            OpenAIClient()
            MockOpenAI.assert_called_once_with(
                api_key="dotenv-key",
                base_url="https://dotenv.example/v1",
            )


class TestOpenAIClientCall:
    def test_call_many_requests_and_returns_all_choices(self, monkeypatch):
        monkeypatch.setenv("LLM_API_KEY", "k")
        monkeypatch.setenv("LLM_BASE_URL", "https://e/v1")
        from tasks.antibody.core.ldm.llm.openai_backend import OpenAIClient
        with patch(OPENAI_PATCH) as MockOpenAI:
            mock_response = MagicMock()
            mock_response.choices = [
                MagicMock(message=MagicMock(content="first")),
                MagicMock(message=MagicMock(content="second")),
            ]
            MockOpenAI.return_value.chat.completions.create.return_value = mock_response

            client = OpenAIClient()
            assert client.call_many("prompt", 0.4, 12, n=2) == ["first", "second"]
            MockOpenAI.return_value.chat.completions.create.assert_called_once_with(
                model="DeepSeek-V4-Flash",
                messages=[{"role": "user", "content": "prompt"}],
                temperature=0.4,
                timeout=12,
                n=2,
            )

    def test_call_returns_content(self, monkeypatch):
        monkeypatch.setenv("LLM_API_KEY", "k")
        monkeypatch.setenv("LLM_BASE_URL", "https://e/v1")
        monkeypatch.delenv("LLM_DISABLE_THINKING", raising=False)
        from tasks.antibody.core.ldm.llm.openai_backend import OpenAIClient
        with patch(OPENAI_PATCH) as MockOpenAI:
            mock_response = MagicMock()
            mock_response.choices = [MagicMock(message=MagicMock(content="hello"))]
            MockOpenAI.return_value.chat.completions.create.return_value = mock_response

            c = OpenAIClient()
            out = c.call("test prompt", 0.25, 30)
            assert out == "hello"
            MockOpenAI.return_value.chat.completions.create.assert_called_once_with(
                model="DeepSeek-V4-Flash",
                messages=[{"role": "user", "content": "test prompt"}],
                temperature=0.25,
                timeout=30,
            )

    def test_call_can_disable_qwen_thinking(self, monkeypatch):
        monkeypatch.setenv("LLM_API_KEY", "k")
        monkeypatch.setenv("LLM_BASE_URL", "https://e/v1")
        monkeypatch.setenv("LLM_DISABLE_THINKING", "1")
        from tasks.antibody.core.ldm.llm.openai_backend import OpenAIClient
        with patch(OPENAI_PATCH) as MockOpenAI:
            mock_response = MagicMock()
            mock_response.choices = [MagicMock(message=MagicMock(content="hello"))]
            MockOpenAI.return_value.chat.completions.create.return_value = mock_response

            c = OpenAIClient()
            out = c.call("test prompt", 0.25, 30)
            assert out == "hello"
            MockOpenAI.return_value.chat.completions.create.assert_called_once_with(
                model="DeepSeek-V4-Flash",
                messages=[{"role": "user", "content": "test prompt"}],
                temperature=0.25,
                timeout=30,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )

    def test_call_passes_env_max_tokens(self, monkeypatch):
        monkeypatch.setenv("LLM_API_KEY", "k")
        monkeypatch.setenv("LLM_BASE_URL", "https://e/v1")
        monkeypatch.setenv("LLM_MAX_TOKENS", "1536")
        from tasks.antibody.core.ldm.llm.openai_backend import OpenAIClient
        with patch(OPENAI_PATCH) as MockOpenAI:
            mock_response = MagicMock()
            mock_response.choices = [MagicMock(message=MagicMock(content="hello"))]
            MockOpenAI.return_value.chat.completions.create.return_value = mock_response

            c = OpenAIClient()
            out = c.call("test prompt", 0.25, 30)
            assert out == "hello"
            MockOpenAI.return_value.chat.completions.create.assert_called_once_with(
                model="DeepSeek-V4-Flash",
                messages=[{"role": "user", "content": "test prompt"}],
                temperature=0.25,
                timeout=30,
                max_tokens=1536,
            )

    def test_invalid_env_max_tokens_raises(self, monkeypatch):
        monkeypatch.setenv("LLM_API_KEY", "k")
        monkeypatch.setenv("LLM_BASE_URL", "https://e/v1")
        monkeypatch.setenv("LLM_MAX_TOKENS", "0")
        from tasks.antibody.core.ldm.llm.openai_backend import OpenAIClient
        with patch(OPENAI_PATCH), pytest.raises(ValueError, match="LLM_MAX_TOKENS"):
            OpenAIClient()

    def test_call_returns_empty_string_on_none_content(self, monkeypatch):
        monkeypatch.setenv("LLM_API_KEY", "k")
        monkeypatch.setenv("LLM_BASE_URL", "https://e/v1")
        from tasks.antibody.core.ldm.llm.openai_backend import OpenAIClient
        with patch(OPENAI_PATCH) as MockOpenAI:
            mock_response = MagicMock()
            mock_response.choices = [MagicMock(message=MagicMock(content=None))]
            MockOpenAI.return_value.chat.completions.create.return_value = mock_response

            c = OpenAIClient()
            assert c.call("p", 0.25, 30) == ""

    def test_call_propagates_exception(self, monkeypatch):
        monkeypatch.setenv("LLM_API_KEY", "k")
        monkeypatch.setenv("LLM_BASE_URL", "https://e/v1")
        from tasks.antibody.core.ldm.llm.openai_backend import OpenAIClient
        with patch(OPENAI_PATCH) as MockOpenAI:
            MockOpenAI.return_value.chat.completions.create.side_effect = RuntimeError("API down")

            c = OpenAIClient()
            with pytest.raises(RuntimeError, match="API down"):
                c.call("p", 0.25, 30)
