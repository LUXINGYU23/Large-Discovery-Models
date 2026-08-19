"""Tests for portable OpenAI-compatible provider resolution."""

from __future__ import annotations

from tasks.iron_mind.core.provider import resolve_openai_provider_settings


def test_primary_environment_names_precede_legacy_aliases() -> None:
    settings = resolve_openai_provider_settings(
        base_url=None,
        model=None,
        api_key=None,
        environ={
            "LLM_BASE_URL": "https://primary.example/v1",
            "LLM_MODEL_NAME": "primary-model",
            "LLM_API_KEY": "primary-key",
            "TTS_LLM_URL": "https://legacy.example/v1",
            "TTS_LLM_MODEL": "legacy-model",
            "TTS_LLM_API_KEY": "legacy-key",
        },
    )

    assert settings.base_url == "https://primary.example/v1"
    assert settings.model == "primary-model"
    assert settings.api_key == "primary-key"


def test_explicit_cli_provider_values_override_environment() -> None:
    settings = resolve_openai_provider_settings(
        base_url="https://cli.example/v1",
        model="cli-model",
        api_key="cli-key",
        environ={
            "LLM_BASE_URL": "https://environment.example/v1",
            "LLM_MODEL_NAME": "environment-model",
            "LLM_API_KEY": "environment-key",
        },
    )

    assert settings.base_url == "https://cli.example/v1"
    assert settings.model == "cli-model"
    assert settings.api_key == "cli-key"
