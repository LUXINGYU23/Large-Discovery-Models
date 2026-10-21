"""Resolve NucleoBench provider settings and request-body configuration."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ProviderSettings:
    base_url: str
    model: str
    api_key: str


def resolve_provider_settings(
    args: argparse.Namespace,
    *,
    environ: Mapping[str, str] | None = None,
) -> ProviderSettings:
    environment = os.environ if environ is None else environ
    api_key = ""
    if args.api_key_file is not None:
        key_path = args.api_key_file.expanduser()
        if key_path.is_file():
            api_key = key_path.read_text(encoding="utf-8").strip()
            if not api_key:
                raise ValueError("API key file is empty")
        elif not args.dry_run:
            raise FileNotFoundError(f"API key file does not exist: {key_path}")
    if not api_key:
        api_key = _first_configured(environment, "LLM_API_KEY", "OPENAI_API_KEY")
    return ProviderSettings(
        base_url=_configured(args.llm_url)
        or _first_configured(environment, "LLM_BASE_URL"),
        model=_configured(args.llm_model_name)
        or _first_configured(environment, "LLM_MODEL_NAME"),
        api_key=api_key,
    )


def parse_extra_body(raw: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"--llm-extra-body-json is not valid JSON: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise TypeError("--llm-extra-body-json must decode to a JSON object")
    return payload


def _configured(value: str | None) -> str:
    return "" if value is None else str(value).strip()


def _first_configured(environment: Mapping[str, str], *names: str) -> str:
    for name in names:
        value = _configured(environment.get(name))
        if value:
            return value
    return ""
