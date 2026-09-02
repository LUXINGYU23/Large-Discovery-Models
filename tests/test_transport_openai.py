from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from typing import Any

import pytest

import ldm_tts
from ldm_tts import compat
from ldm_tts.transport import ProposalRequest
from ldm_tts.transport.openai import (
    EndpointCircuitBreaker,
    EndpointCircuitOpen,
    EndpointRequestError,
    OpenAICompatibleProposalClient,
    call_with_circuit_breaker,
    chat_completions_url,
    models_url,
    preflight_openai_responses,
    request_openai_chat_response,
    request_openai_models,
    request_openai_responses_response,
    responses_url,
)


TOKEN = "test-api-token"
MODEL = "target-model"


@contextmanager
def _http_server(
    responder: Callable[[BaseHTTPRequestHandler, dict[str, Any]], None],
) -> Iterator[tuple[str, list[dict[str, Any]]]]:
    requests: list[dict[str, Any]] = []

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_: object) -> None:
            return

        def do_GET(self) -> None:
            self._respond()

        def do_POST(self) -> None:
            self._respond()

        def _respond(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            request = {
                "method": self.command,
                "path": self.path,
                "headers": dict(self.headers.items()),
                "body": self.rfile.read(length),
            }
            requests.append(request)
            responder(self, request)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        yield f"http://{host}:{port}", requests
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()


def _json_response(
    handler: BaseHTTPRequestHandler,
    status: int,
    payload: dict[str, Any] | list[Any],
) -> None:
    raw = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(raw)))
    handler.end_headers()
    handler.wfile.write(raw)


def _chat_payload(text: str = "OK") -> dict[str, Any]:
    return {"model": MODEL, "choices": [{"message": {"content": text}}]}


def _responses_payload(text: str = "OK") -> dict[str, Any]:
    return {
        "model": MODEL,
        "status": "completed",
        "output": [
            {"type": "message", "content": [{"type": "output_text", "text": text}]}
        ],
    }


@pytest.mark.parametrize(
    ("raw", "models", "chat", "responses"),
    [
        (
            "https://api.example.com",
            "https://api.example.com/models",
            "https://api.example.com/v1/chat/completions",
            "https://api.example.com/v1/responses",
        ),
        (
            "https://api.example.com/v1",
            "https://api.example.com/v1/models",
            "https://api.example.com/v1/chat/completions",
            "https://api.example.com/v1/responses",
        ),
    ],
)
def test_openai_compatible_urls_normalize(
    raw: str,
    models: str,
    chat: str,
    responses: str,
) -> None:
    assert models_url(raw) == models
    assert chat_completions_url(raw) == chat
    assert responses_url(raw) == responses


def test_requests_use_openai_contracts_and_keep_auth_out_of_payloads() -> None:
    def respond(handler: BaseHTTPRequestHandler, request: dict[str, Any]) -> None:
        if request["path"] == "/v1/models":
            _json_response(handler, 200, {"data": [{"id": MODEL}]})
        elif request["path"] == "/v1/chat/completions":
            _json_response(handler, 200, _chat_payload())
        else:
            _json_response(handler, 200, _responses_payload())

    with _http_server(respond) as (url, requests):
        request_openai_models(url=f"{url}/v1", api_key=TOKEN, timeout_seconds=2)
        request_openai_chat_response(
            url=f"{url}/v1",
            model=MODEL,
            api_key=TOKEN,
            messages=[{"role": "user", "content": "hello"}],
            timeout_seconds=2,
            max_tokens=8,
            temperature=0.0,
        )
        request_openai_responses_response(
            url=f"{url}/v1",
            model=MODEL,
            api_key=TOKEN,
            messages=[{"role": "user", "content": "hello"}],
            timeout_seconds=2,
            max_tokens=8,
            temperature=0.0,
            extra_body={"reasoning": {"effort": "max"}},
        )

    assert [(item["method"], item["path"]) for item in requests] == [
        ("GET", "/v1/models"),
        ("POST", "/v1/chat/completions"),
        ("POST", "/v1/responses"),
    ]
    assert all(item["headers"]["Authorization"] == f"Bearer {TOKEN}" for item in requests)
    assert all(TOKEN.encode("utf-8") not in item["body"] for item in requests)
    responses_body = json.loads(requests[2]["body"])
    assert responses_body == {
        "model": MODEL,
        "input": [{"role": "user", "content": "hello"}],
        "temperature": 0.0,
        "max_output_tokens": 8,
        "reasoning": {"effort": "max"},
    }


def test_http_errors_and_invalid_payloads_are_sanitized() -> None:
    def respond(handler: BaseHTTPRequestHandler, request: dict[str, Any]) -> None:
        if request["path"] == "/v1/models":
            _json_response(handler, 200, {"data": [{}]})
        else:
            _json_response(handler, 429, {"error": "rate limited"})

    with _http_server(respond) as (url, _):
        with pytest.raises(EndpointRequestError) as invalid:
            request_openai_models(url=f"{url}/v1", api_key=TOKEN, timeout_seconds=2)
        with pytest.raises(EndpointRequestError) as limited:
            request_openai_chat_response(
                url=f"{url}/v1",
                model=MODEL,
                api_key=TOKEN,
                messages=[{"role": "user", "content": "hello"}],
                timeout_seconds=2,
                max_tokens=8,
                temperature=0.0,
            )

    assert "secret" not in str(invalid.value)
    assert TOKEN not in str(limited.value)
    assert "HTTP 429" in str(limited.value)
    assert "rate limited" in str(limited.value)


def test_total_timeout_covers_a_response_that_keeps_sending_bytes() -> None:
    def heartbeat(handler: BaseHTTPRequestHandler, _: dict[str, Any]) -> None:
        handler.send_response(200)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Transfer-Encoding", "chunked")
        handler.end_headers()
        until = time.monotonic() + 1
        while time.monotonic() < until:
            try:
                handler.wfile.write(b"1\r\n \r\n")
                handler.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                return
            time.sleep(0.02)

    with _http_server(heartbeat) as (url, _):
        started = time.monotonic()
        with pytest.raises(EndpointRequestError, match="timed out after 0.15 seconds"):
            request_openai_chat_response(
                url=f"{url}/v1",
                model=MODEL,
                api_key=TOKEN,
                messages=[{"role": "user", "content": "hello"}],
                timeout_seconds=0.15,
                max_tokens=8,
                temperature=0.0,
            )

    assert time.monotonic() - started < 0.75


def test_proposal_client_retries_and_normalizes_responses_output() -> None:
    calls = 0

    def respond(handler: BaseHTTPRequestHandler, _: dict[str, Any]) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            _json_response(handler, 500, {"error": "transient"})
            return
        _json_response(
            handler,
            200,
            {
                "model": MODEL,
                "status": "completed",
                "output": [
                    {"type": "message", "content": [{"type": "output_text", "text": "hello"}]},
                    {
                        "type": "function_call",
                        "call_id": "call-1",
                        "name": "emit",
                        "arguments": "{}",
                    },
                ],
                "usage": {"total_tokens": 15},
            },
        )

    with _http_server(respond) as (url, _):
        client = OpenAICompatibleProposalClient(
            url=f"{url}/v1",
            model=MODEL,
            api_key=TOKEN,
            timeout_seconds=2,
            max_retries=1,
            retry_backoff_seconds=0,
            wire_api="responses",
        )
        response = client.propose(
            ProposalRequest(messages=({"role": "user", "content": "hello"},))
        )

    assert calls == 2
    assert response.text == "hello"
    assert response.tool_calls[0]["function"]["name"] == "emit"
    assert response.usage["total_tokens"] == 15
    assert response.metadata["attempts"] == 2


def test_proposal_client_timeout_is_total_across_retries() -> None:
    def heartbeat(handler: BaseHTTPRequestHandler, _: dict[str, Any]) -> None:
        handler.send_response(200)
        handler.send_header("Transfer-Encoding", "chunked")
        handler.end_headers()
        while True:
            try:
                handler.wfile.write(b"1\r\n \r\n")
                handler.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                return
            time.sleep(0.02)

    with _http_server(heartbeat) as (url, requests):
        started = time.monotonic()
        client = OpenAICompatibleProposalClient(
            url=f"{url}/v1",
            model=MODEL,
            api_key=TOKEN,
            timeout_seconds=0.15,
            max_retries=3,
            retry_backoff_seconds=0,
        )
        with pytest.raises(EndpointRequestError, match="timed out after 0.15 seconds"):
            client.propose(
                ProposalRequest(messages=({"role": "user", "content": "hello"},))
            )

    assert len(requests) == 1
    assert time.monotonic() - started < 0.75


def test_preflight_and_public_exports_are_sanitized() -> None:
    with _http_server(lambda handler, _: _json_response(handler, 200, _responses_payload())) as (url, _):
        artifact = preflight_openai_responses(
            url=f"{url}/v1", model=MODEL, api_key=TOKEN, timeout_seconds=2
        )

    assert artifact["status"] == "ok"
    assert TOKEN not in json.dumps(artifact)
    assert compat.resolve("models_url") is models_url
    assert ldm_tts.models_url is models_url


def test_circuit_breaker_opens_without_repeating_requests() -> None:
    breaker = EndpointCircuitBreaker(failure_threshold=1, recovery_timeout_seconds=60)

    def unavailable() -> None:
        raise EndpointRequestError("unavailable")

    with pytest.raises(EndpointRequestError, match="unavailable"):
        call_with_circuit_breaker(breaker, unavailable)
    with pytest.raises(EndpointCircuitOpen):
        call_with_circuit_breaker(breaker, unavailable)
