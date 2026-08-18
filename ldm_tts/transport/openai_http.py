"""OpenAI-compatible URL, HTTP, and endpoint-preflight primitives."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from typing import Any


class EndpointRequestError(RuntimeError):
    """Raised when an endpoint request fails or returns an invalid response."""


def chat_completions_url(raw: str) -> str:
    """Normalize a base URL or complete OpenAI-compatible endpoint to chat."""

    base = _normalized_url(raw, "Chat endpoint URL")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/models"):
        return base[: -len("/models")] + "/chat/completions"
    if base.endswith("/v1"):
        return base + "/chat/completions"
    return base + "/v1/chat/completions"


def models_url(raw: str) -> str:
    """Normalize a base URL or complete OpenAI-compatible endpoint to models."""

    base = _normalized_url(raw, "Models endpoint URL")
    if base.endswith("/models"):
        return base
    if base.endswith("/chat/completions"):
        return base[: -len("/chat/completions")] + "/models"
    if base.endswith("/v1"):
        return base + "/models"
    return base + "/models"


def request_openai_models(
    *,
    url: str,
    api_key: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    """Return a validated OpenAI-compatible models response."""

    result = _request_json(
        endpoint=models_url(url),
        api_key=api_key,
        timeout_seconds=timeout_seconds,
        method="GET",
        payload=None,
    )
    _model_ids(result)
    return result


def request_openai_chat(
    *,
    url: str,
    model: str,
    api_key: str,
    messages: list[dict[str, Any]],
    timeout_seconds: float,
    max_tokens: int,
    temperature: float,
) -> str:
    """Return text from one validated OpenAI-compatible chat response."""

    result = request_openai_chat_response(
        url=url,
        model=model,
        api_key=api_key,
        messages=messages,
        timeout_seconds=timeout_seconds,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    content = result["choices"][0]["message"].get("content")
    if not isinstance(content, str) or not content.strip():
        raise EndpointRequestError("Chat response content is empty or not text")
    return content


def request_openai_chat_response(
    *,
    url: str,
    model: str,
    api_key: str,
    messages: Sequence[Mapping[str, Any]],
    timeout_seconds: float,
    max_tokens: int,
    temperature: float,
    tools: Sequence[Mapping[str, Any]] = (),
    tool_choice: Any = None,
    extra_body: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return one validated raw OpenAI-compatible chat response."""

    body = _chat_body(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        tools=tools,
        tool_choice=tool_choice,
        extra_body=extra_body,
    )
    result = _request_json(
        endpoint=chat_completions_url(url),
        api_key=api_key,
        timeout_seconds=timeout_seconds,
        method="POST",
        payload=json.dumps(body).encode("utf-8"),
    )
    _chat_message(result)
    return result


def preflight_openai_chat(
    *,
    url: str,
    model: str,
    api_key: str,
    timeout_seconds: float = 30.0,
) -> dict[str, Any]:
    """Probe one minimal chat response while preserving the legacy artifact."""

    started = time.monotonic()
    content = request_openai_chat(
        url=url,
        model=model,
        api_key=api_key,
        messages=[{"role": "user", "content": "Reply with exactly OK."}],
        timeout_seconds=timeout_seconds,
        max_tokens=8,
        temperature=0.0,
    )
    return {
        "status": "ok",
        "model": model,
        "latency_seconds": round(time.monotonic() - started, 6),
        "response_nonempty": bool(content.strip()),
    }


def preflight_openai_endpoint(
    *,
    url: str,
    model: str,
    api_key: str,
    timeout_seconds: float = 30.0,
) -> dict[str, Any]:
    """Verify visible model identity and one minimal chat response."""

    started = time.monotonic()
    model_ids = _model_ids(
        request_openai_models(
            url=url,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
        )
    )
    if model not in model_ids:
        raise EndpointRequestError("Requested model is not visible from models endpoint")
    response = request_openai_chat_response(
        url=url,
        model=model,
        api_key=api_key,
        messages=[{"role": "user", "content": "Reply with exactly OK."}],
        timeout_seconds=timeout_seconds,
        max_tokens=8,
        temperature=0.0,
    )
    response_model = response.get("model")
    if not isinstance(response_model, str) or response_model != model:
        raise EndpointRequestError("Chat response model identity does not match request")
    return {
        "status": "ok",
        "request_model": model,
        "response_model": response_model,
        "model_visible": True,
        "model_count": len(model_ids),
        "latency_seconds": round(time.monotonic() - started, 6),
    }


def _normalized_url(raw: str, label: str) -> str:
    base = str(raw).strip().rstrip("/")
    if not base:
        raise EndpointRequestError(f"{label} is empty")
    return base


def _chat_body(
    *,
    model: str,
    messages: Sequence[Mapping[str, Any]],
    max_tokens: int,
    temperature: float,
    tools: Sequence[Mapping[str, Any]],
    tool_choice: Any,
    extra_body: Mapping[str, Any] | None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "messages": [dict(message) for message in messages],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if tools:
        body["tools"] = [dict(tool) for tool in tools]
    if tool_choice is not None:
        body["tool_choice"] = tool_choice
    if extra_body:
        reserved = set(body) & set(extra_body)
        if reserved:
            raise EndpointRequestError(
                "extra_body cannot override reserved chat field(s): " + ", ".join(sorted(reserved))
            )
        body.update(dict(extra_body))
    return body


def _request_json(
    *,
    endpoint: str,
    api_key: str,
    timeout_seconds: float,
    method: str,
    payload: bytes | None,
) -> Any:
    headers = {"Content-Type": "application/json"} if payload is not None else {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(endpoint, data=payload, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise EndpointRequestError(f"HTTP {exc.code} from OpenAI-compatible endpoint") from exc
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise EndpointRequestError(
            f"OpenAI-compatible endpoint request failed: {type(exc).__name__}"
        ) from exc


def _model_ids(result: Any) -> tuple[str, ...]:
    if not isinstance(result, dict):
        raise EndpointRequestError("Models response root is not an object")
    data = result.get("data")
    if not isinstance(data, list):
        raise EndpointRequestError("Models response data is not a list")
    model_ids: list[str] = []
    for item in data:
        if not isinstance(item, Mapping):
            raise EndpointRequestError("Models response data contains a non-object model")
        model_id = item.get("id")
        if not isinstance(model_id, str) or not model_id.strip():
            raise EndpointRequestError("Models response model id is missing or invalid")
        model_ids.append(model_id)
    return tuple(model_ids)


def _chat_message(result: Any) -> Mapping[str, Any]:
    if not isinstance(result, dict):
        raise EndpointRequestError("Chat response root is not an object")
    try:
        message = result["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise EndpointRequestError("Chat response has no choices[0].message") from exc
    if not isinstance(message, Mapping):
        raise EndpointRequestError("Chat response message is not an object")
    content = message.get("content")
    tool_calls = message.get("tool_calls")
    if not (isinstance(content, str) and content.strip()) and not (
        isinstance(tool_calls, list) and tool_calls
    ):
        raise EndpointRequestError("Chat response contains neither text nor tool calls")
    return message


__all__ = [
    "EndpointRequestError",
    "chat_completions_url",
    "models_url",
    "preflight_openai_chat",
    "preflight_openai_endpoint",
    "request_openai_chat",
    "request_openai_chat_response",
    "request_openai_models",
]
