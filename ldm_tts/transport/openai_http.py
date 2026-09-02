"""OpenAI-compatible URL, HTTP, and endpoint-preflight primitives."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Mapping, Sequence
from typing import Any

import aiohttp

PREFLIGHT_MAX_TOKENS = 64
HTTP_ERROR_DETAIL_MAX_CHARS = 500


class EndpointRequestError(RuntimeError):
    """Raised when an endpoint request fails or returns an invalid response."""


class EndpointRequestTimeout(EndpointRequestError):
    """Raised when an endpoint request exceeds its total deadline."""


def chat_completions_url(raw: str) -> str:
    """Normalize a base URL or complete OpenAI-compatible endpoint to chat."""

    base = _normalized_url(raw, "Chat endpoint URL")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/models"):
        return base[: -len("/models")] + "/chat/completions"
    if base.endswith("/responses"):
        return base[: -len("/responses")] + "/chat/completions"
    if base.endswith("/v1"):
        return base + "/chat/completions"
    return base + "/v1/chat/completions"


def responses_url(raw: str) -> str:
    """Normalize a base URL or complete OpenAI-compatible endpoint to Responses."""

    base = _normalized_url(raw, "Responses endpoint URL")
    if base.endswith("/responses"):
        return base
    if base.endswith("/chat/completions"):
        return base[: -len("/chat/completions")] + "/responses"
    if base.endswith("/models"):
        return base[: -len("/models")] + "/responses"
    if base.endswith("/v1"):
        return base + "/responses"
    return base + "/v1/responses"


def models_url(raw: str) -> str:
    """Normalize a base URL or complete OpenAI-compatible endpoint to models."""

    base = _normalized_url(raw, "Models endpoint URL")
    if base.endswith("/models"):
        return base
    if base.endswith("/chat/completions"):
        return base[: -len("/chat/completions")] + "/models"
    if base.endswith("/responses"):
        return base[: -len("/responses")] + "/models"
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
    extra_body: Mapping[str, Any] | None = None,
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
        extra_body=extra_body,
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


def request_openai_responses_response(
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
    """Return one validated raw OpenAI-compatible Responses result."""

    body = _responses_body(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        tools=tools,
        tool_choice=tool_choice,
        extra_body=extra_body,
    )
    result = _request_json(
        endpoint=responses_url(url),
        api_key=api_key,
        timeout_seconds=timeout_seconds,
        method="POST",
        payload=json.dumps(body).encode("utf-8"),
    )
    extract_openai_responses_content(result)
    return result


def preflight_openai_chat(
    *,
    url: str,
    model: str,
    api_key: str,
    timeout_seconds: float = 30.0,
    extra_body: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Probe one minimal chat response while preserving the legacy artifact."""

    started = time.monotonic()
    content = request_openai_chat(
        url=url,
        model=model,
        api_key=api_key,
        messages=[{"role": "user", "content": "Reply with exactly OK."}],
        timeout_seconds=timeout_seconds,
        max_tokens=PREFLIGHT_MAX_TOKENS,
        temperature=0.0,
        extra_body=extra_body,
    )
    return {
        "status": "ok",
        "model": model,
        "latency_seconds": round(time.monotonic() - started, 6),
        "response_nonempty": bool(content.strip()),
    }


def preflight_openai_responses(
    *,
    url: str,
    model: str,
    api_key: str,
    timeout_seconds: float = 30.0,
    extra_body: Mapping[str, Any] | None = None,
    require_model_visibility: bool = False,
) -> dict[str, Any]:
    """Probe one minimal Responses result without persisting provider payloads."""

    started = time.monotonic()
    model_ids = (
        _model_ids(
            request_openai_models(
                url=url,
                api_key=api_key,
                timeout_seconds=timeout_seconds,
            )
        )
        if require_model_visibility
        else ()
    )
    if require_model_visibility and model not in model_ids:
        raise EndpointRequestError(
            "Requested model is not visible from models endpoint"
        )
    prompt = (
        "Reply with one non-empty JSON object."
        if extra_body and "text" in extra_body
        else "Reply with exactly OK."
    )
    result = request_openai_responses_response(
        url=url,
        model=model,
        api_key=api_key,
        messages=[{"role": "user", "content": prompt}],
        timeout_seconds=timeout_seconds,
        max_tokens=PREFLIGHT_MAX_TOKENS,
        temperature=0.0,
        extra_body=extra_body,
    )
    response_model = result.get("model")
    if not isinstance(response_model, str) or response_model != model:
        raise EndpointRequestError("Responses model identity does not match request")
    text, tool_calls = extract_openai_responses_content(result)
    artifact = {
        "status": "ok",
        "request_model": model,
        "response_model": response_model,
        "latency_seconds": round(time.monotonic() - started, 6),
        "response_nonempty": bool(text.strip() or tool_calls),
    }
    if require_model_visibility:
        artifact.update(model_visible=True, model_count=len(model_ids))
    return artifact


def preflight_openai_endpoint(
    *,
    url: str,
    model: str,
    api_key: str,
    timeout_seconds: float = 30.0,
    extra_body: Mapping[str, Any] | None = None,
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
        raise EndpointRequestError(
            "Requested model is not visible from models endpoint"
        )
    prompt = "Reply with exactly OK."
    if extra_body and "response_format" in extra_body:
        prompt = "Reply with one non-empty JSON object."
    response = request_openai_chat_response(
        url=url,
        model=model,
        api_key=api_key,
        messages=[{"role": "user", "content": prompt}],
        timeout_seconds=timeout_seconds,
        max_tokens=PREFLIGHT_MAX_TOKENS,
        temperature=0.0,
        extra_body=extra_body,
    )
    response_model = response.get("model")
    if not isinstance(response_model, str) or response_model != model:
        raise EndpointRequestError(
            "Chat response model identity does not match request"
        )
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
                "extra_body cannot override reserved chat field(s): "
                + ", ".join(sorted(reserved))
            )
        body.update(dict(extra_body))
    return body


def _responses_body(
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
        "input": [dict(message) for message in messages],
        "temperature": temperature,
        "max_output_tokens": max_tokens,
    }
    if tools:
        body["tools"] = [_responses_tool(tool) for tool in tools]
    if tool_choice is not None:
        body["tool_choice"] = _responses_tool_choice(tool_choice)
    if extra_body:
        reserved = set(body) & set(extra_body)
        if reserved:
            raise EndpointRequestError(
                "extra_body cannot override reserved Responses field(s): "
                + ", ".join(sorted(reserved))
            )
        body.update(dict(extra_body))
    return body


def _responses_tool(tool: Mapping[str, Any]) -> dict[str, Any]:
    function = tool.get("function")
    if tool.get("type") == "function" and isinstance(function, Mapping):
        return {"type": "function", **dict(function)}
    return dict(tool)


def _responses_tool_choice(tool_choice: Any) -> Any:
    if not isinstance(tool_choice, Mapping):
        return tool_choice
    function = tool_choice.get("function")
    if tool_choice.get("type") == "function" and isinstance(function, Mapping):
        return {"type": "function", "name": function.get("name")}
    return dict(tool_choice)


def _request_json(
    *,
    endpoint: str,
    api_key: str,
    timeout_seconds: float,
    method: str,
    payload: bytes | None,
) -> Any:
    return asyncio.run(
        _request_json_async(
            endpoint=endpoint,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
            method=method,
            payload=payload,
        )
    )


async def _request_json_async(
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
    timeout = aiohttp.ClientTimeout(total=timeout_seconds)
    try:
        async with aiohttp.ClientSession(timeout=timeout, trust_env=True) as session:
            async with session.request(
                method, endpoint, data=payload, headers=headers
            ) as response:
                raw = await response.read()
                if response.status >= 400:
                    detail = _http_error_detail(raw)
                    message = f"HTTP {response.status} from OpenAI-compatible endpoint"
                    if detail:
                        message = f"{message}: {detail}"
                    raise EndpointRequestError(message)
                return json.loads(raw.decode("utf-8"))
    except asyncio.TimeoutError as exc:
        raise EndpointRequestTimeout(
            "OpenAI-compatible endpoint request timed out after "
            f"{timeout_seconds:g} seconds"
        ) from exc
    except (aiohttp.ClientError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise EndpointRequestError(
            f"OpenAI-compatible endpoint request failed: {type(exc).__name__}"
        ) from exc


def _http_error_detail(raw: bytes | str) -> str:
    """Return a bounded, human-readable excerpt from an HTTP error body."""

    body = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw)
    body = body.strip()
    if not body:
        return ""
    if len(body) <= HTTP_ERROR_DETAIL_MAX_CHARS:
        return body
    return body[:HTTP_ERROR_DETAIL_MAX_CHARS] + "..."


def _model_ids(result: Any) -> tuple[str, ...]:
    if not isinstance(result, dict):
        raise EndpointRequestError("Models response root is not an object")
    data = result.get("data")
    if not isinstance(data, list):
        raise EndpointRequestError("Models response data is not a list")
    model_ids: list[str] = []
    for item in data:
        if not isinstance(item, Mapping):
            raise EndpointRequestError(
                "Models response data contains a non-object model"
            )
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


def extract_openai_responses_content(
    result: Any,
) -> tuple[str, tuple[dict[str, Any], ...]]:
    if not isinstance(result, dict):
        raise EndpointRequestError("Responses root is not an object")
    status = result.get("status")
    if status not in (None, "completed"):
        raise EndpointRequestError(f"Responses request did not complete: {status!r}")
    output = result.get("output")
    if not isinstance(output, list):
        raise EndpointRequestError("Responses output is not a list")
    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for item in output:
        if not isinstance(item, Mapping):
            raise EndpointRequestError("Responses output contains a non-object item")
        if item.get("type") == "message":
            content = item.get("content")
            if not isinstance(content, list):
                raise EndpointRequestError("Responses message content is not a list")
            for part in content:
                if not isinstance(part, Mapping):
                    raise EndpointRequestError(
                        "Responses message content contains a non-object part"
                    )
                if part.get("type") == "output_text":
                    text = part.get("text")
                    if isinstance(text, str) and text:
                        text_parts.append(text)
        elif item.get("type") == "function_call":
            tool_calls.append(
                {
                    "id": item.get("call_id") or item.get("id"),
                    "type": "function",
                    "function": {
                        "name": item.get("name"),
                        "arguments": item.get("arguments", ""),
                    },
                }
            )
    text = "\n".join(text_parts)
    if not text.strip() and not tool_calls:
        raise EndpointRequestError(
            "Responses output contains neither text nor tool calls"
        )
    return text, tuple(tool_calls)


__all__ = [
    "EndpointRequestError",
    "EndpointRequestTimeout",
    "chat_completions_url",
    "extract_openai_responses_content",
    "models_url",
    "preflight_openai_chat",
    "preflight_openai_endpoint",
    "preflight_openai_responses",
    "request_openai_chat",
    "request_openai_chat_response",
    "request_openai_models",
    "request_openai_responses_response",
    "responses_url",
]
