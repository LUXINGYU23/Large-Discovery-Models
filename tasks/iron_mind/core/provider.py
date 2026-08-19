"""OpenAI-compatible provider settings for Iron Mind campaigns."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Mapping


BASE_URL_ENV_NAMES = ("LLM_BASE_URL", "TTS_LLM_URL", "LDM_LLM_URL")
MODEL_ENV_NAMES = ("LLM_MODEL_NAME", "TTS_LLM_MODEL", "LDM_LLM_MODEL")
API_KEY_ENV_NAMES = (
    "LLM_API_KEY",
    "TTS_LLM_API_KEY",
    "LDM_LLM_API_KEY",
    "OPENAI_API_KEY",
)


@dataclass(frozen=True)
class OpenAIProviderSettings:
    """Resolved endpoint settings; the API key is never persisted by the task."""

    base_url: str
    model: str
    api_key: str


def resolve_openai_provider_settings(
    *,
    base_url: str | None,
    model: str | None,
    api_key: str | None,
    environ: Mapping[str, str] | None = None,
) -> OpenAIProviderSettings:
    """Resolve explicit CLI values before the shared provider environment names."""

    environment = os.environ if environ is None else environ
    return OpenAIProviderSettings(
        base_url=_configured_value(base_url) or _first_environment_value(environment, BASE_URL_ENV_NAMES),
        model=_configured_value(model) or _first_environment_value(environment, MODEL_ENV_NAMES),
        api_key=_configured_value(api_key) or _first_environment_value(environment, API_KEY_ENV_NAMES),
    )


def parse_openai_extra_body_json(raw: str | None) -> dict[str, Any]:
    """Parse an optional provider-specific OpenAI-compatible request object."""

    if raw is None or not str(raw).strip():
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"--llm-extra-body-json is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("--llm-extra-body-json must decode to a JSON object.")
    return dict(payload)


def _configured_value(value: str | None) -> str:
    return "" if value is None else str(value).strip()


def _first_environment_value(environment: Mapping[str, str], names: tuple[str, ...]) -> str:
    for name in names:
        value = _configured_value(environment.get(name))
        if value:
            return value
    return ""


__all__ = [
    "API_KEY_ENV_NAMES",
    "BASE_URL_ENV_NAMES",
    "MODEL_ENV_NAMES",
    "OpenAIProviderSettings",
    "parse_openai_extra_body_json",
    "resolve_openai_provider_settings",
]
