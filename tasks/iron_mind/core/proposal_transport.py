"""OpenAI-compatible proposal transport construction for Iron Mind."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ldm_tts.transport.openai import (
    EndpointCircuitBreaker,
    OpenAICompatibleProposalClient,
)

TRANSIENT_MAX_RETRIES = 3
TRANSIENT_RETRY_BACKOFF_SECONDS = 10.0
TRANSIENT_CIRCUIT_FAILURE_THRESHOLD = 32


def build_openai_reaction_client(
    *,
    base_url: str,
    model: str,
    api_key: str,
    timeout_seconds: float,
    max_tokens: int,
    temperature: float = 0.7,
    json_mode: bool = False,
    extra_body: Mapping[str, Any] | None = None,
) -> OpenAICompatibleProposalClient:
    """Build the shared OpenAI-compatible transport for a real campaign."""

    return OpenAICompatibleProposalClient(
        url=base_url,
        model=model,
        api_key=api_key,
        timeout_seconds=timeout_seconds,
        max_tokens=max_tokens,
        temperature=temperature,
        max_retries=TRANSIENT_MAX_RETRIES,
        retry_backoff_seconds=TRANSIENT_RETRY_BACKOFF_SECONDS,
        extra_body=_request_extra_body(json_mode, extra_body),
        require_models_preflight=False,
        breaker=EndpointCircuitBreaker(
            failure_threshold=TRANSIENT_CIRCUIT_FAILURE_THRESHOLD,
        ),
    )


def _request_extra_body(
    json_mode: bool,
    extra_body: Mapping[str, Any] | None,
) -> dict[str, Any]:
    body = dict(extra_body or {})
    if not json_mode:
        return body
    if "response_format" in body:
        raise ValueError(
            "--llm-json-mode cannot be combined with response_format in --llm-extra-body-json."
        )
    body["response_format"] = {"type": "json_object"}
    return body


__all__ = ["build_openai_reaction_client"]
