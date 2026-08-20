"""Unit coverage for the OpenAI-compatible proposal transport."""

from __future__ import annotations

import json
import urllib.error

import pytest

from ldm_tts.transport import ProposalRequest
from ldm_tts.transport.openai import (
    EndpointCircuitBreaker,
    EndpointCircuitOpen,
    EndpointRequestError,
    OpenAICompatibleProposalClient,
    call_with_circuit_breaker,
    chat_completions_url,
    preflight_openai_chat,
    request_openai_chat,
    request_openai_chat_response,
)


class _FakeResponse:
    """Context-manager stand-in for ``urllib.request.urlopen`` success."""

    def __init__(self, payload: dict | list | str) -> None:
        self._data = (
            payload.encode("utf-8")
            if isinstance(payload, str)
            else json.dumps(payload).encode("utf-8")
        )

    def read(self) -> bytes:
        return self._data

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *args: object) -> bool:
        return False


class _FakeHTTPError(urllib.error.HTTPError):
    """Minimal HTTPError with just the surface the transport reads."""

    def __init__(self, code: int, body: bytes = b"") -> None:
        self.code = code
        self._body = body

    def read(self) -> bytes:
        return self._body


def _chat_kwargs(**overrides: object) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "url": "https://api.example.com/v1",
        "model": "m",
        "api_key": "",
        "messages": [{"role": "user", "content": "hi"}],
        "timeout_seconds": 10.0,
        "max_tokens": 32,
        "temperature": 0.0,
    }
    kwargs.update(overrides)
    return kwargs


def _raise(value: BaseException):
    def _raiser(*args: object, **kwargs: object):
        raise value

    return _raiser


# --------------------------------------------------------------------------- #
# chat_completions_url
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("raw", "expected"),
    (
        ("https://api.example.com", "https://api.example.com/v1/chat/completions"),
        ("https://api.example.com/", "https://api.example.com/v1/chat/completions"),
        ("https://api.example.com/v1", "https://api.example.com/v1/chat/completions"),
        ("https://api.example.com/v1/", "https://api.example.com/v1/chat/completions"),
        (
            "https://api.example.com/chat/completions",
            "https://api.example.com/chat/completions",
        ),
        (
            "https://api.example.com/v1/chat/completions",
            "https://api.example.com/v1/chat/completions",
        ),
    ),
)
def test_chat_completions_url(raw: str, expected: str) -> None:
    assert chat_completions_url(raw) == expected


def test_chat_completions_url_rejects_empty() -> None:
    with pytest.raises(EndpointRequestError):
        chat_completions_url("")
    with pytest.raises(EndpointRequestError):
        chat_completions_url("   ")


# --------------------------------------------------------------------------- #
# request_openai_chat_response
# --------------------------------------------------------------------------- #


def test_request_openai_chat_response_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *a, **k: _FakeResponse(
            {"choices": [{"message": {"content": "hello"}}]}
        ),
    )
    result = request_openai_chat_response(**_chat_kwargs())
    assert result["choices"][0]["message"]["content"] == "hello"


def test_request_openai_chat_response_accepts_tool_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *a, **k: _FakeResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "tool_calls": [{"id": "1", "function": {"name": "emit", "arguments": "{}"}}],
                        }
                    }
                ]
            }
        ),
    )
    result = request_openai_chat_response(**_chat_kwargs())
    assert result["choices"][0]["message"]["tool_calls"][0]["id"] == "1"


def test_request_openai_chat_response_rejects_non_object_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *a, **k: _FakeResponse([{"choices": []}]),
    )
    with pytest.raises(EndpointRequestError, match="root is not an object"):
        request_openai_chat_response(**_chat_kwargs())


def test_request_openai_chat_response_rejects_missing_choices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *a, **k: _FakeResponse({}),
    )
    with pytest.raises(EndpointRequestError, match="no choices"):
        request_openai_chat_response(**_chat_kwargs())


def test_request_openai_chat_response_rejects_non_object_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *a, **k: _FakeResponse({"choices": [{"message": "not-an-object"}]}),
    )
    with pytest.raises(EndpointRequestError, match="message is not an object"):
        request_openai_chat_response(**_chat_kwargs())


def test_request_openai_chat_response_rejects_empty_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *a, **k: _FakeResponse({"choices": [{"message": {}}]}),
    )
    with pytest.raises(EndpointRequestError, match="neither text nor tool calls"):
        request_openai_chat_response(**_chat_kwargs())


def test_request_openai_chat_response_http_error_includes_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "urllib.request.urlopen",
        _raise(_FakeHTTPError(429, b'{"error": "rate limited"}')),
    )
    with pytest.raises(EndpointRequestError) as exc_info:
        request_openai_chat_response(**_chat_kwargs())
    message = str(exc_info.value)
    assert "HTTP 429" in message
    assert "rate limited" in message


def test_request_openai_chat_response_http_error_without_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "urllib.request.urlopen",
        _raise(_FakeHTTPError(500)),
    )
    with pytest.raises(EndpointRequestError, match="HTTP 500"):
        request_openai_chat_response(**_chat_kwargs())


def test_request_openai_chat_response_http_error_truncates_long_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "urllib.request.urlopen",
        _raise(_FakeHTTPError(400, b"x" * 2000)),
    )
    with pytest.raises(EndpointRequestError) as exc_info:
        request_openai_chat_response(**_chat_kwargs())
    message = str(exc_info.value)
    assert message.endswith("...")
    assert len(message) < 600


def test_request_openai_chat_response_json_decode_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *a, **k: _FakeResponse("not json"),
    )
    with pytest.raises(EndpointRequestError, match="request failed"):
        request_openai_chat_response(**_chat_kwargs())


def test_request_openai_chat_response_rejects_reserved_extra_body() -> None:
    with pytest.raises(EndpointRequestError, match="reserved"):
        request_openai_chat_response(**_chat_kwargs(extra_body={"model": "other"}))


# --------------------------------------------------------------------------- #
# request_openai_chat
# --------------------------------------------------------------------------- #


def test_request_openai_chat_extracts_content(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *a, **k: _FakeResponse({"choices": [{"message": {"content": "OK"}}]}),
    )
    assert request_openai_chat(**_chat_kwargs()) == "OK"


def test_request_openai_chat_rejects_empty_content(monkeypatch: pytest.MonkeyPatch) -> None:
    # A tool-call response passes the lower-level response validation, but
    # request_openai_chat is text-only and must reject the empty content.
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *a, **k: _FakeResponse(
            {"choices": [{"message": {"content": "", "tool_calls": [{"id": "1"}]}}]}
        ),
    )
    with pytest.raises(EndpointRequestError, match="empty or not text"):
        request_openai_chat(**_chat_kwargs())


def test_preflight_openai_chat(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *a, **k: _FakeResponse({"choices": [{"message": {"content": "OK"}}]}),
    )
    result = preflight_openai_chat(url="https://api.example.com/v1", model="m", api_key="")
    assert result["status"] == "ok"
    assert result["response_nonempty"] is True


# --------------------------------------------------------------------------- #
# call_with_circuit_breaker
# --------------------------------------------------------------------------- #


def test_call_with_circuit_breaker_success_resets_failures() -> None:
    breaker = EndpointCircuitBreaker(failure_threshold=3, recovery_timeout_seconds=60)
    breaker.record_failure(EndpointRequestError("a"))
    breaker.record_failure(EndpointRequestError("b"))
    assert breaker.consecutive_failures == 2

    assert call_with_circuit_breaker(breaker, lambda: "ok") == "ok"
    assert breaker.consecutive_failures == 0
    assert breaker.state == "closed"


# --------------------------------------------------------------------------- #
# OpenAICompatibleProposalClient
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "overrides",
    (
        {"timeout_seconds": 0},
        {"max_tokens": 0},
        {"max_retries": -1},
        {"retry_backoff_seconds": -1},
    ),
)
def test_proposal_client_rejects_invalid_config(overrides: dict[str, object]) -> None:
    config = {"url": "https://api.example.com/v1", "model": "m", **overrides}
    with pytest.raises(ValueError):
        OpenAICompatibleProposalClient(**config)


def _proposal() -> ProposalRequest:
    return ProposalRequest(messages=({"role": "user", "content": "hi"},))


def test_propose_captures_text_tool_calls_and_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *a, **k: _FakeResponse(
            {
                "model": "served-model",
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "tool_calls": [
                                {"id": "1", "function": {"name": "emit", "arguments": "{}"}}
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            }
        ),
    )
    client = OpenAICompatibleProposalClient(url="https://api.example.com/v1", model="m")
    response = client.propose(_proposal())

    assert response.text == ""
    assert response.tool_calls[0]["id"] == "1"
    assert response.usage["total_tokens"] == 15
    assert response.metadata["finish_reason"] == "tool_calls"
    assert response.metadata["model"] == "served-model"


def test_propose_retries_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, int] = {"n": 0}

    def flaky(*args: object, **kwargs: object):
        calls["n"] += 1
        if calls["n"] == 1:
            raise _FakeHTTPError(500, b"transient")
        return _FakeResponse({"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setattr("urllib.request.urlopen", flaky)
    client = OpenAICompatibleProposalClient(
        url="https://api.example.com/v1",
        model="m",
        max_retries=1,
        retry_backoff_seconds=0,
    )
    response = client.propose(_proposal())

    assert response.text == "ok"
    assert calls["n"] == 2


def test_propose_raises_after_retries_exhausted(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, int] = {"n": 0}

    def boom(*args: object, **kwargs: object):
        calls["n"] += 1
        raise _FakeHTTPError(500, b"boom")

    monkeypatch.setattr("urllib.request.urlopen", boom)
    client = OpenAICompatibleProposalClient(
        url="https://api.example.com/v1",
        model="m",
        max_retries=2,
        retry_backoff_seconds=0,
    )
    with pytest.raises(EndpointRequestError, match="HTTP 500"):
        client.propose(_proposal())
    assert calls["n"] == 3


def test_propose_does_not_retry_when_circuit_opens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, int] = {"n": 0}

    def boom(*args: object, **kwargs: object):
        calls["n"] += 1
        raise _FakeHTTPError(503, b"unavailable")

    monkeypatch.setattr("urllib.request.urlopen", boom)
    breaker = EndpointCircuitBreaker(failure_threshold=1, recovery_timeout_seconds=60)
    client = OpenAICompatibleProposalClient(
        url="https://api.example.com/v1",
        model="m",
        max_retries=3,
        breaker=breaker,
    )
    with pytest.raises(EndpointCircuitOpen):
        client.propose(_proposal())
    assert calls["n"] == 1
