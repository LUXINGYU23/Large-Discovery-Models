"""Generic OpenAI-compatible provider configuration for SynthonBench."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

BASE_URL_ENV_NAMES = ("LLM_BASE_URL", "TTS_LLM_URL", "LDM_LLM_URL")
MODEL_ENV_NAMES = ("LLM_MODEL_NAME", "TTS_LLM_MODEL", "LDM_LLM_MODEL")
API_KEY_ENV_NAMES = ("LLM_API_KEY", "TTS_LLM_API_KEY", "LDM_LLM_API_KEY", "OPENAI_API_KEY")


@dataclass(frozen=True)
class OpenAIProviderSettings:
    """Resolved endpoint values; callers must never serialize the API key."""

    base_url: str
    model: str
    api_key: str


def resolve_openai_provider_settings(
    *, base_url: str | None, model: str | None, api_key: str | None,
    environ: Mapping[str, str] | None = None,
) -> OpenAIProviderSettings:
    environment = os.environ if environ is None else environ
    return OpenAIProviderSettings(
        base_url=_configured(base_url) or _first(environment, BASE_URL_ENV_NAMES),
        model=_configured(model) or _first(environment, MODEL_ENV_NAMES),
        api_key=_configured(api_key) or _first(environment, API_KEY_ENV_NAMES),
    )


def parse_openai_extra_body_json(raw: str | None) -> dict[str, Any]:
    if raw is None or not str(raw).strip():
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"--llm-extra-body-json is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise TypeError("--llm-extra-body-json must decode to a JSON object")
    return dict(payload)


def _configured(value: str | None) -> str:
    return "" if value is None else str(value).strip()


def _first(environment: Mapping[str, str], names: tuple[str, ...]) -> str:
    for name in names:
        value = _configured(environment.get(name))
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
