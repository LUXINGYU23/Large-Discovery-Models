"""Tests for portable OpenAI-compatible provider resolution."""

from __future__ import annotations

import pytest

from tasks.iron_mind.core.provider import (
    parse_openai_extra_body_json,
    resolve_openai_provider_settings,
)


def test_provider_settings_resolve_from_standard_environment_names() -> None:
    settings = resolve_openai_provider_settings(
        base_url=None,
        model=None,
        api_key=None,
        environ={
            "LLM_BASE_URL": "https://primary.example/v1",
            "LLM_MODEL_NAME": "primary-model",
            "LLM_API_KEY": "primary-key",
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


def test_provider_extra_body_requires_a_json_object() -> None:
    assert parse_openai_extra_body_json('{"thinking":{"type":"disabled"}}') == {
        "thinking": {"type": "disabled"}
    }

    with pytest.raises(ValueError, match="not valid JSON"):
        parse_openai_extra_body_json("{")
    with pytest.raises(ValueError, match="JSON object"):
        parse_openai_extra_body_json("[]")
