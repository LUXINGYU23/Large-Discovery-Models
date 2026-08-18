from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any

import pytest

import ldm_tts
from ldm_tts import compat
from ldm_tts.transport.openai import (
    EndpointRequestError,
    OpenAICompatibleProposalClient,
    chat_completions_url,
    models_url,
    preflight_openai_endpoint,
    request_openai_chat_response,
    request_openai_models,
)
import ldm_tts.transport.openai as openai


TOKEN = "test-api-token"
MODEL = "target-model"


class Response:
    def __init__(self, payload: Any) -> None:
        self.payload = payload

    def __enter__(self) -> "Response":
        return self

    def __exit__(self, *_: object) -> bool:
        return False

    def read(self) -> bytes:
        if isinstance(self.payload, bytes):
            return self.payload
        return json.dumps(self.payload).encode("utf-8")


def models_payload(*model_ids: str) -> dict[str, object]:
    return {"object": "list", "data": [{"id": model_id} for model_id in model_ids]}


def chat_payload(*, model: str = MODEL, message: dict[str, object] | None = None) -> dict[str, object]:
    return {
        "model": model,
        "choices": [{"message": message or {"content": "OK"}}],
    }


def recording_urlopen(
    payloads: list[Any],
    calls: list[tuple[urllib.request.Request, float]],
) -> Callable[[urllib.request.Request, float], Response]:
    def fake(request: urllib.request.Request, timeout: float) -> Response:
        calls.append((request, timeout))
        return Response(payloads.pop(0))

    return fake


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("https://host", "https://host/models"),
        ("https://host/v1", "https://host/v1/models"),
        ("https://host/v1/models", "https://host/v1/models"),
    ],
)
def test_models_url_normalizes_openai_compatible_base_urls(raw: str, expected: str) -> None:
    assert models_url(raw) == expected


def test_chat_completions_url_preserves_complete_chat_endpoint() -> None:
    assert chat_completions_url("https://host/v1/chat/completions") == (
        "https://host/v1/chat/completions"
    )


def test_models_and_chat_use_expected_methods_and_authorization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[urllib.request.Request, float]] = []
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        recording_urlopen([models_payload(MODEL), chat_payload()], calls),
    )

    request_openai_models(url="https://host/v1", api_key=TOKEN, timeout_seconds=5.0)
    request_openai_chat_response(
        url="https://host/v1",
        model=MODEL,
        api_key=TOKEN,
        messages=[{"role": "user", "content": "hello"}],
        timeout_seconds=5.0,
        max_tokens=8,
        temperature=0.0,
    )

    models_request, chat_request = (item[0] for item in calls)
    assert models_request.get_method() == "GET"
    assert models_request.full_url == "https://host/v1/models"
    assert models_request.data is None
    assert chat_request.get_method() == "POST"
    assert chat_request.full_url == "https://host/v1/chat/completions"
    assert models_request.get_header("Authorization") == f"Bearer {TOKEN}"
    assert chat_request.get_header("Authorization") == f"Bearer {TOKEN}"
    assert TOKEN not in models_request.full_url
    assert TOKEN.encode("utf-8") not in (chat_request.data or b"")


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {},
        {"data": {}},
        {"data": [{}]},
        {"data": [{"id": ""}]},
    ],
)
def test_models_response_rejects_invalid_root_data_and_model_objects(
    monkeypatch: pytest.MonkeyPatch,
    payload: object,
) -> None:
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        recording_urlopen([payload], []),
    )

    with pytest.raises(EndpointRequestError):
        request_openai_models(url="https://host", api_key=TOKEN, timeout_seconds=5.0)


@pytest.mark.parametrize(
    "message",
    [
        {"content": "OK"},
        {"content": None, "tool_calls": [{"id": "call-1", "type": "function"}]},
    ],
)
def test_chat_response_accepts_text_or_tool_calls(
    monkeypatch: pytest.MonkeyPatch,
    message: dict[str, object],
) -> None:
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        recording_urlopen([chat_payload(message=message)], []),
    )

    result = request_openai_chat_response(
        url="https://host",
        model=MODEL,
        api_key=TOKEN,
        messages=[{"role": "user", "content": "hello"}],
        timeout_seconds=5.0,
        max_tokens=8,
        temperature=0.0,
    )

    assert result["choices"][0]["message"] == message


@pytest.mark.parametrize(
    ("models", "chat", "expected"),
    [
        (models_payload("other-model"), chat_payload(), "not visible"),
        (models_payload(MODEL), chat_payload(model="other-model"), "identity"),
        (models_payload(MODEL), {"model": MODEL, "choices": []}, "choices"),
    ],
)
def test_combined_preflight_rejects_invalid_model_visibility_and_chat_identity(
    monkeypatch: pytest.MonkeyPatch,
    models: dict[str, object],
    chat: dict[str, object],
    expected: str,
) -> None:
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        recording_urlopen([models, chat], []),
    )

    with pytest.raises(EndpointRequestError, match=expected) as error:
        preflight_openai_endpoint(
            url="https://host/v1",
            model=MODEL,
            api_key=TOKEN,
            timeout_seconds=5.0,
        )

    assert TOKEN not in str(error.value)


def test_combined_preflight_returns_sanitized_model_artifact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        recording_urlopen([models_payload("other-model", MODEL), chat_payload()], []),
    )

    artifact = preflight_openai_endpoint(
        url="https://host/v1",
        model=MODEL,
        api_key=TOKEN,
        timeout_seconds=5.0,
    )

    assert artifact["status"] == "ok"
    assert artifact["request_model"] == MODEL
    assert artifact["response_model"] == MODEL
    assert artifact["model_visible"] is True
    assert artifact["model_count"] == 2
    assert isinstance(artifact["latency_seconds"], float)
    assert TOKEN not in json.dumps(artifact)
    assert "choices" not in artifact
    assert compat.resolve("models_url") is models_url
    assert ldm_tts.models_url is models_url
    assert ldm_tts.preflight_openai_endpoint is preflight_openai_endpoint


def test_endpoint_preflight_json_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[urllib.request.Request, float]] = []
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        recording_urlopen([models_payload(MODEL), chat_payload()], calls),
    )
    preflight_openai_endpoint(
        url="https://host/v1",
        model=MODEL,
        api_key=TOKEN,
        timeout_seconds=5.0,
        extra_body={"thinking": {"type": "disabled"}, "response_format": {"type": "json_object"}},
    )

    body = json.loads(calls[1][0].data or b"{}")
    assert body["thinking"] == {"type": "disabled"}
    assert body["response_format"] == {"type": "json_object"}
    assert body["messages"][0]["content"] == "Reply with one non-empty JSON object."


def test_http_and_json_errors_do_not_include_authorization_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def http_error(request: urllib.request.Request, timeout: float) -> Response:
        raise urllib.error.HTTPError(request.full_url, 401, "unauthorized", {}, None)

    monkeypatch.setattr(urllib.request, "urlopen", http_error)
    with pytest.raises(EndpointRequestError) as http_error_result:
        request_openai_models(url="https://host", api_key=TOKEN, timeout_seconds=5.0)

    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        recording_urlopen([b"not-json"], []),
    )
    with pytest.raises(EndpointRequestError) as json_error_result:
        request_openai_chat_response(
            url="https://host",
            model=MODEL,
            api_key=TOKEN,
            messages=[{"role": "user", "content": "hello"}],
            timeout_seconds=5.0,
            max_tokens=8,
            temperature=0.0,
        )

    assert TOKEN not in str(http_error_result.value)
    assert TOKEN not in str(json_error_result.value)


def test_client_uses_combined_preflight_only_when_required(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chat_artifact = {"status": "chat"}
    endpoint_artifact = {"status": "endpoint"}
    endpoint_kwargs: dict[str, Any] = {}
    monkeypatch.setattr(openai, "preflight_openai_chat", lambda **_: chat_artifact)
    monkeypatch.setattr(openai, "preflight_openai_endpoint", lambda **kwargs: endpoint_kwargs.update(kwargs) or endpoint_artifact)

    default_client = OpenAICompatibleProposalClient(url="https://host", model=MODEL)
    models_client = OpenAICompatibleProposalClient(
        url="https://host",
        model=MODEL,
        require_models_preflight=True, extra_body={"thinking": {"type": "disabled"}},
    )

    assert default_client.preflight() is chat_artifact
    assert (models_client.preflight(), endpoint_kwargs["extra_body"]) == (endpoint_artifact, {"thinking": {"type": "disabled"}})
