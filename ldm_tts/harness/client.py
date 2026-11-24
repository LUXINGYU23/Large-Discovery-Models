"""Long-lived JSONL subprocess client for research harness sidecars."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import queue
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from ldm_tts.harness import responses
from ldm_tts.harness.protocol import (
    HARNESS_PROTOCOL_VERSION,
    HarnessError,
    HarnessPoolConfig,
    HarnessProviderAuthorizationRequest,
    HarnessSubmissionRequest,
    HarnessSubmissionError,
    HarnessSubmissionValidation,
    HarnessTurn,
    HarnessTurnResult,
)

SubmissionValidator = Callable[[HarnessSubmissionRequest], HarnessSubmissionValidation]
SubmissionBatchValidator = Callable[[tuple[HarnessSubmissionRequest, ...]], Mapping[str, HarnessSubmissionValidation]]
ProviderAuthorizer = Callable[[HarnessProviderAuthorizationRequest], bool]

MAX_PROVIDER_RETRY_WAIT_SECONDS = 1800.0
MAX_RETRY_BACKOFF_SECONDS = 60.0

_PROVIDER_BALANCE_RE = re.compile(
    r"insufficient[_ ]balance|payment\s+required|"
    r"(?:api\s+error\s*\(|http\s+|status(?:\s+code)?[\s:=]+|provider\s+|error\s*\()402\b|"
    r"\b402\s+(?:payment\s+required|insufficient\s+balance)",
    re.IGNORECASE,
)
_PROVIDER_RATE_LIMIT_RE = re.compile(
    r"rate_limit_exceeded|too\s+many\s+requests|"
    r"(?:api\s+error\s*\(|http\s+|status(?:\s+code)?[\s:=]+|provider\s+|error\s*\()429\b|"
    r"\b429\s+too\s+many",
    re.IGNORECASE,
)
_PROVIDER_SERVER_ERROR_RE = re.compile(
    r"internal_server_error|overloaded_error|bad\s+gateway|service\s+unavailable|gateway\s+timeout|\bserver_error\b|"
    r"(?:api\s+error\s*\(|http\s+|status(?:\s+code)?[\s:=]+|provider\s+|error\s*\()5[0-9]{2}\b|"
    r"\b5[0-9]{2}\s+(?:internal|bad\s+gateway|service\s+unavailable|gateway\s+timeout|server\s+error)",
    re.IGNORECASE,
)


def classify_provider_error(message: str, code: str | None = None) -> str | None:
    """Classify provider HTTP errors from sidecar messages or probe failures."""
    text = f"{code or ''} {message or ''}"
    if _PROVIDER_BALANCE_RE.search(text):
        return "provider_balance"
    if _PROVIDER_RATE_LIMIT_RE.search(text):
        return "provider_rate_limit"
    if _PROVIDER_SERVER_ERROR_RE.search(text):
        return "provider_server_error"
    return None


def probe_provider(
    base_url: str,
    model: str,
    api_key: str,
    *,
    max_output_tokens: int = 1,
    timeout_seconds: float = 30.0,
) -> dict[str, Any]:
    """Send a minimal 1-token Responses API request to verify provider health and balance."""
    if not base_url.strip() or not model.strip():
        raise ValueError("provider probe requires base_url and model")
    if not api_key or not api_key.strip():
        raise ValueError("provider probe requires a non-empty API key")
    key = api_key.strip()
    url = base_url.rstrip("/") + "/responses"
    payload = json.dumps({
        "model": model,
        "input": "ping",
        "max_output_tokens": int(max_output_tokens),
        "stream": False,
        "store": False,
    }).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=payload,
        method="POST",
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "ldm-harness-probe/1",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            status = getattr(response, "status", response.getcode())
            response.read(4096)
            if status != 200:
                msg = f"provider probe failed: OpenAI API error ({status})"
                kind = classify_provider_error(msg)
                raise HarnessError(
                    msg,
                    retryable=kind in ("provider_rate_limit", "provider_server_error"),
                    provider_error=kind,
                )
            return {"status": int(status), "url": url, "model": model}
    except urllib.error.HTTPError as exc:
        try:
            body = exc.read(512).decode("utf-8", errors="replace").strip()
        except Exception:
            body = ""
        if key in body:
            body = body.replace(key, "[REDACTED]")
        msg = f"provider probe failed: OpenAI API error ({exc.code})" + (f": {body}" if body else "")
        kind = classify_provider_error(msg)
        raise HarnessError(
            msg,
            retryable=kind in ("provider_rate_limit", "provider_server_error"),
            provider_error=kind,
        ) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        detail = str(exc).replace(key, "[REDACTED]")
        raise HarnessError(f"provider probe transport failed: {detail}", retryable=True) from exc


class HarnessClient:
    def __init__(
        self,
        command: Sequence[str],
        *,
        api_key: str,
        config: HarnessPoolConfig,
        named_secrets: Mapping[str, str] | None = None,
        response_timeout_seconds: float = 2100,
    ) -> None:
        if not command:
            raise ValueError("harness command must not be empty")
        if not api_key:
            raise ValueError("harness API key must not be empty")
        if response_timeout_seconds <= 0:
            raise ValueError("harness response timeout must be positive")
        self.command = tuple(str(part) for part in command)
        self._api_key = api_key
        self._named_secrets = dict(named_secrets or {})
        if any(not name or not value for name, value in self._named_secrets.items()):
            raise ValueError("harness named secrets require non-empty names and values")
        self.config = config
        self.response_timeout_seconds = float(response_timeout_seconds)
        self._process: subprocess.Popen[str] | None = None
        self._responses: queue.Queue[str | None] = queue.Queue()
        self._stderr: deque[str] = deque(maxlen=40)
        self._request_index = 0
        self._protocol_version: str | None = None

    def start(self) -> None:
        if self._process is not None:
            raise HarnessError("harness client is already started")
        self._start_process()

    def _start_process(self) -> None:
        if not self._api_key:
            raise HarnessError("harness client is closed")
        self._named_secrets.update(self.config.bootstrap_secrets())
        self._responses = queue.Queue()
        self._stderr = deque(maxlen=40)
        secret_values = {self._api_key, *self._named_secrets.values()}
        process = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
            env={
                name: value
                for name, value in os.environ.items()
                if name not in self._named_secrets and value not in secret_values
            },
        )
        self._process = process
        assert process.stdout is not None and process.stderr is not None
        threading.Thread(target=self._read_stdout, args=(process.stdout, self._responses), daemon=True).start()
        threading.Thread(target=self._read_stderr, args=(process.stderr, self._stderr), daemon=True).start()
        try:
            self._protocol_version = self._wait_for_ready()
            self._request(
                "bootstrap_secret",
                {"apiKey": self._api_key, "namedSecrets": self._named_secrets},
                "secret_bootstrapped",
            )
            self._request(
                "initialize", self.config.initialize_payload(),
                "initialized",
            )
        except BaseException:
            self._stop_process()
            raise

    def run_turn(
        self,
        turns: Sequence[HarnessTurn],
        *,
        submission_validator: SubmissionValidator | None = None,
        submission_batch_validator: SubmissionBatchValidator | None = None,
        provider_authorizer: ProviderAuthorizer | None = None,
        recovery_timeout_seconds: float = 0,
    ) -> tuple[HarnessTurnResult, ...]:
        """Run native turns with per-submission or incremental batch validation.

        A batch validator sees every currently unanswered snapshot and returns
        decisions keyed by profile. Omitted profiles remain pending while peer
        submissions and provider authorization events continue to be handled.
        The task owns durable admission receipts for replayed peers; this client
        owns wire routing, attempt limits, and native commit verification.
        """
        if (submission_validator is None) == (submission_batch_validator is None):
            raise ValueError("provide exactly one submission validation mode")
        if not math.isfinite(recovery_timeout_seconds) or recovery_timeout_seconds < 0:
            raise ValueError("harness recovery timeout must be finite and non-negative")
        recovery_deadline = time.monotonic() + recovery_timeout_seconds
        if not turns:
            raise ValueError("harness turn batch must not be empty")
        expected = {turn.profile_id: turn for turn in turns}
        if len(expected) != len(turns):
            raise ValueError("harness turn profile IDs must be unique")
        terminal_validations: dict[str, tuple[str, str]] = {}
        validation_submissions: dict[str, int] = {}

        def check_request(request: HarnessSubmissionRequest) -> None:
            turn = expected.get(request.profile_id)
            if turn is None or request.turn_id != turn.turn_id:
                raise HarnessError(
                    "harness submission validation request does not match the turn batch"
                )
            validation_submissions[request.profile_id] = request.attempt_index

        def record_validation(request, validation):
            if not isinstance(validation, HarnessSubmissionValidation):
                raise HarnessError("submission validator returned an invalid result")
            maximum = self.config.submission_contract.max_validation_attempts
            if (
                validation.decision == "retry"
                and maximum is not None
                and request.attempt_index >= maximum
            ):
                validation = HarnessSubmissionValidation("reject_turn", validation.errors)
            if validation.decision != "retry":
                status = "accepted" if validation.decision == "accept" else "rejected"
                terminal_validations[request.profile_id] = (status, request.digest)
            return validation

        def validate(request: HarnessSubmissionRequest) -> HarnessSubmissionValidation:
            check_request(request)
            return record_validation(request, submission_validator(request))

        def validate_batch(requests):
            for request in requests:
                check_request(request)
            rejected = [p for p, (status, _) in terminal_validations.items() if status == "rejected"]
            decisions = ({} if rejected else submission_batch_validator(requests))
            pending = {request.profile_id: request for request in requests}
            if not isinstance(decisions, Mapping) or not set(decisions).issubset(pending):
                raise HarnessError("batch validator answered an unknown pending profile")
            recorded = {profile: record_validation(pending[profile], decision)
                        for profile, decision in decisions.items()}
            rejected = [p for p, (status, _) in terminal_validations.items() if status == "rejected"]
            if rejected:
                # A closed peer cannot reach the task's all-profile barrier.
                # Close unanswered snapshots too, including later arrivals.
                for profile, request in pending.items():
                    if profile not in recorded:
                        recorded[profile] = record_validation(request, HarnessSubmissionValidation(
                            "reject_turn", (HarnessSubmissionError("/", "peer_rejected",
                                "Research batch cannot complete; rejected peers: " + ", ".join(rejected)),)))
            return recorded

        def authorize(request: HarnessProviderAuthorizationRequest) -> bool:
            if (request.campaign_id != self.config.campaign_id
                    or request.profile_id not in expected
                    or request.turn_id != expected[request.profile_id].turn_id):
                raise HarnessError("harness provider authorization request does not match the turn batch")
            decision = True if provider_authorizer is None else provider_authorizer(request)
            if type(decision) is not bool:
                raise HarnessError("harness provider authorizer must return a boolean")
            return decision

        known_usage: dict[str, dict[str, Any]] = {}
        retry_delay = 1.0
        provider_waited = 0.0
        restart_required = False
        while True:
            try:
                if restart_required:
                    self._stop_process()
                    self._start_process()
                    restart_required = False
                payload = self._request(
                    "run_turn",
                    {"turns": [turn.to_dict() for turn in turns]},
                    "turn_committed",
                    submission_validator=validate if submission_validator is not None else None,
                    submission_batch_validator=validate_batch if submission_batch_validator is not None else None,
                    provider_authorizer=authorize,
                )
                break
            except HarnessError as exc:
                # Sidecar usage is cumulative per turn, including partial recovery.
                known_usage.update(exc.turn_usage)
                for profile in expected:
                    known_usage.setdefault(profile, {})["validationSubmissions"] = (
                        validation_submissions.get(profile, 0)
                    )
                exc.turn_usage = {profile: dict(usage) for profile, usage in known_usage.items()}
                provider_error = exc.provider_error or classify_provider_error(str(exc), exc.code)
                if provider_error is not None and exc.provider_error is None:
                    exc.provider_error = provider_error
                if provider_error == "provider_balance":
                    exc.retryable = False
                    raise
                if provider_error in ("provider_rate_limit", "provider_server_error"):
                    exc.retryable = True
                remaining = recovery_deadline - time.monotonic()
                if not exc.retryable or remaining <= 0:
                    raise
                restart_required = self._process is None or self._process.poll() is not None
                if provider_error in ("provider_rate_limit", "provider_server_error"):
                    wait_budget = min(remaining, MAX_PROVIDER_RETRY_WAIT_SECONDS - provider_waited)
                    if wait_budget <= 0:
                        raise
                    delay = min(retry_delay, wait_budget)
                    logging.getLogger(__name__).warning(
                        "Retrying provider error (%s) after %.1fs (waited %.1fs/%.0fs): %s",
                        provider_error, delay, provider_waited, MAX_PROVIDER_RETRY_WAIT_SECONDS, exc,
                    )
                    time.sleep(delay)
                    provider_waited += delay
                    retry_delay = min(MAX_RETRY_BACKOFF_SECONDS, retry_delay * 2)
                else:
                    logging.getLogger(__name__).warning(
                        "Continuing unfinished Harness sessions; committed turns are replayed: %s", exc,
                    )
                    time.sleep(min(retry_delay, remaining))
                    retry_delay = min(30.0, retry_delay * 2)
                if time.monotonic() >= recovery_deadline:
                    raise
        raw_turns = payload.get("turns")
        if not isinstance(raw_turns, list):
            raise HarnessError("harness response is missing committed turns")
        results = tuple(responses.parse_turn_result(item) for item in raw_turns)
        if (
            {result.profile_id for result in results} != set(expected)
            or len(results) != len(turns)
        ):
            raise HarnessError("harness response does not match the requested profiles")
        for result in results:
            turn = expected[result.profile_id]
            result.usage["validationSubmissions"] = validation_submissions.get(
                result.profile_id,
                0,
            )
            if (
                result.turn_id != turn.turn_id
                or result.round_index != turn.round_index
                or result.history_from_seq != turn.history_from_seq
                or result.history_to_seq != turn.history_to_seq
                or result.history_digest != turn.history_digest
                or result.input_digest != turn.input_digest
            ):
                raise HarnessError("harness response does not match the requested turn")
            validation = terminal_validations.get(result.profile_id)
            if not result.replayed and validation != (
                result.submission_status,
                result.submission_digest,
            ):
                raise HarnessError(
                    "harness committed a submission without matching task validation: "
                    + result.profile_id
                )
        return results

    def close(self) -> None:
        process = self._process
        try:
            if process is not None and process.poll() is None:
                if self._protocol_version is None:
                    _terminate(process)
                else:
                    try:
                        self._request("close", {}, "closed", timeout_seconds=30)
                    except HarnessError:
                        _terminate(process)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=10)
        finally:
            self._stop_process()
            self._api_key = ""
            self._named_secrets.clear()

    def _stop_process(self) -> None:
        if self._process is not None:
            _terminate(self._process)
            if self._process.stdin is not None:
                try:
                    self._process.stdin.close()
                except (BrokenPipeError, OSError):
                    pass
        self._process = None
        self._protocol_version = None

    def __enter__(self) -> "HarnessClient":
        self.start()
        return self

    def __exit__(self, _type, _value, _traceback) -> None:
        self.close()

    def _request(
        self,
        frame_type: str,
        fields: dict[str, Any],
        expected_type: str,
        *,
        timeout_seconds: float | None = None,
        submission_validator: SubmissionValidator | None = None,
        submission_batch_validator: SubmissionBatchValidator | None = None,
        provider_authorizer: ProviderAuthorizer | None = None,
    ) -> dict[str, Any]:
        frame = self._frame(self._next_request_id(), frame_type, fields)
        process = self._process
        if process is None or process.stdin is None:
            raise HarnessError("harness client is not started")
        if process.poll() is not None:
            raise self._process_error("harness sidecar exited")
        self._send_frame(frame)
        timeout = self.response_timeout_seconds if timeout_seconds is None else timeout_seconds
        deadline = time.monotonic() + timeout
        pending = {}
        deferred = deque()

        def parse_response(line):
            if line is None:
                _terminate(process)
                raise self._process_error("harness sidecar output closed")
            try:
                response = json.loads(line)
            except json.JSONDecodeError as exc:
                raise HarnessError("harness sidecar returned invalid JSON") from exc
            if not isinstance(response, dict) or response.get("requestId") != frame["requestId"]:
                raise HarnessError("harness sidecar response requestId mismatch")
            if (response.get("protocolVersion") != self._protocol_version
                    or response.get("campaignId") != self.config.campaign_id):
                raise HarnessError("harness sidecar response protocol identity mismatch")
            return response

        def authorize(response):
            if provider_authorizer is None:
                raise HarnessError("harness requested provider authorization outside run_turn")
            self._answer_provider_authorization(response, provider_authorizer)

        def validate_without_blocking_provider(callback):
            # Admission may run isolated solver jobs. Keep provider authorization
            # responsive while preserving serial submission/budget decisions.
            # Join the worker before returning/raising: no admission writer may
            # outlive this request and race a resumed turn.
            nonlocal deadline
            validation_started = time.monotonic()
            validation_deadline = validation_started + timeout
            with ThreadPoolExecutor(max_workers=1, thread_name_prefix="harness-validation") as pool:
                future = pool.submit(callback)
                while not future.done():
                    remaining = validation_deadline - time.monotonic()
                    if remaining <= 0:
                        _terminate(process)
                        raise self._process_error("harness submission validation timed out")
                    try:
                        response = parse_response(self._responses.get(timeout=min(.05, remaining)))
                    except queue.Empty:
                        continue
                    if response.get("type") == "provider_authorization_requested":
                        authorize(response)
                    else:
                        deferred.append(response)
                result = future.result()
                deadline += time.monotonic() - validation_started
                return result

        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                _terminate(process)
                raise self._process_error("harness sidecar response timed out")
            try:
                response = deferred.popleft() if deferred else parse_response(self._responses.get(timeout=remaining))
            except queue.Empty as exc:
                _terminate(process)
                raise self._process_error("harness sidecar response timed out") from exc
            if response.get("type") == "provider_authorization_requested":
                try:
                    authorize(response)
                except BaseException:
                    _terminate(process)
                    raise
                continue
            if response.get("type") != "submission_validation_requested":
                break
            if submission_validator is None and submission_batch_validator is None:
                _terminate(process)
                raise HarnessError("harness requested submission validation outside run_turn")
            try:
                if submission_batch_validator is None:
                    request = self._parse_submission_request(response)
                    decision = validate_without_blocking_provider(lambda: submission_validator(request))
                    self._send_submission_validation(response, request, decision)
                else:
                    request = self._parse_submission_request(response)
                    if request.profile_id in pending:
                        raise HarnessError("profile has two pending submissions")
                    pending[request.profile_id] = (response, request)
                    requests = tuple(item[1] for item in pending.values())
                    decisions = validate_without_blocking_provider(lambda: submission_batch_validator(requests))
                    for profile, decision in decisions.items():
                        response_frame, request = pending.pop(profile)
                        self._send_submission_validation(response_frame, request, decision)
            except BaseException:
                _terminate(process)
                raise
        if pending and response.get("type") != "error":
            raise HarnessError("harness committed with unresolved submission validations")
        if response.get("type") == "error":
            responses.assert_response_keys(response, {"type", "requestId", "protocolVersion", "campaignId", "error"})
            error = response.get("error")
            message = error.get("message") if isinstance(error, dict) else "unknown sidecar error"
            turn_usage = {}
            expected = {turn["profileId"]: turn["turnId"] for turn in frame.get("turns", [])}
            raw_usage = error.get("turnUsage", []) if isinstance(error, dict) else []
            if not isinstance(raw_usage, list):
                raise HarnessError("harness error turnUsage must be an array")
            for item in raw_usage:
                responses.assert_response_keys(item, {"profileId", "turnId", "usage"})
                profile = responses.required_string(item["profileId"], "profileId")
                if profile in turn_usage or profile not in expected or item["turnId"] != expected[profile]:
                    raise HarnessError("harness error usage does not match the requested turn")
                turn_usage[profile] = responses.parse_usage(item["usage"])
            code = error.get("code") if isinstance(error, dict) else None
            provider_error = classify_provider_error(str(message), code)
            if provider_error == "provider_balance":
                retryable = False
            elif provider_error in ("provider_rate_limit", "provider_server_error"):
                retryable = True
            else:
                retryable = isinstance(error, dict) and code == "recoverable_turn_error"
            raise HarnessError(
                str(message), turn_usage=turn_usage,
                retryable=retryable,
                code=code,
                provider_error=provider_error,
            )
        if response.get("type") != expected_type:
            raise HarnessError(f"expected harness response {expected_type!r}")
        expected_keys = {
            "secret_bootstrapped": {"type", "requestId", "protocolVersion", "campaignId"},
            "initialized": {
                "type", "requestId", "protocolVersion", "campaignId", "profiles", "manifest",
            },
            "turn_committed": {
                "type", "requestId", "protocolVersion", "campaignId", "turns",
            },
            "closed": {"type", "requestId", "protocolVersion", "campaignId"},
        }
        responses.assert_response_keys(response, expected_keys[expected_type])
        return response

    def _answer_provider_authorization(
        self,
        response: dict[str, Any],
        authorizer: ProviderAuthorizer,
    ) -> None:
        responses.assert_response_keys(response, {
            "type", "requestId", "protocolVersion", "campaignId", "authorizationId",
            "profileId", "turnId", "providerRequestId", "requestDigest",
        })
        request = HarnessProviderAuthorizationRequest(
            campaign_id=responses.required_string(response["campaignId"], "campaignId"),
            profile_id=responses.required_string(response["profileId"], "profileId"),
            turn_id=responses.required_string(response["turnId"], "turnId"),
            provider_request_id=responses.required_string(response["providerRequestId"], "providerRequestId"),
            request_digest=responses.required_digest(response["requestDigest"], "requestDigest"),
        )
        self._send_frame({
            **self._frame(response["requestId"], "provider_authorization_result"),
            "authorizationId": responses.required_string(response["authorizationId"], "authorizationId"),
            "providerRequestId": request.provider_request_id,
            "requestDigest": request.request_digest,
            "authorized": authorizer(request),
        })

    def _parse_submission_request(self, response):
        responses.assert_response_keys(response, {
            "type", "requestId", "protocolVersion", "campaignId", "validationId",
            "profileId", "turnId", "attemptIndex", "submission", "artifacts",
            "submissionJson", "submissionDigest",
        })
        submission = response["submission"]
        if not isinstance(submission, dict):
            raise HarnessError("harness submission validation request has invalid payload")
        artifacts = responses.parse_submitted_artifacts(response["artifacts"])
        submission_digest = responses.required_digest(response["submissionDigest"], "submissionDigest")
        responses.verify_submission_json(
            response["submissionJson"],
            submission_digest,
            submission,
            response["artifacts"],
        )
        request = HarnessSubmissionRequest(
            profile_id=responses.required_string(response["profileId"], "profileId"),
            turn_id=responses.required_string(response["turnId"], "turnId"),
            attempt_index=responses.required_positive_int(response["attemptIndex"], "attemptIndex"),
            submission=dict(submission),
            artifacts=artifacts,
            submission_digest=submission_digest,
        )
        return request

    def _send_submission_validation(self, response, request, validation):
        if not isinstance(validation, HarnessSubmissionValidation):
            raise HarnessError("submission validator returned an invalid result")
        self._send_frame({
            **self._frame(response["requestId"], "submission_validation_result"),
            "validationId": responses.required_string(response["validationId"], "validationId"),
            "submissionDigest": request.digest,
            **validation.to_dict(),
        })

    def _frame(
        self,
        request_id: str,
        frame_type: str,
        fields: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self._protocol_version is None:
            raise HarnessError("harness sidecar did not declare a release version")
        return {
            "type": frame_type,
            "requestId": request_id,
            "protocolVersion": self._protocol_version,
            "campaignId": self.config.campaign_id,
            **dict(fields or {}),
        }

    def _wait_for_ready(self) -> str:
        process = self._process
        if process is None:
            raise HarnessError("harness client is not started")
        timeout = min(self.response_timeout_seconds, 30)
        try:
            line = self._responses.get(timeout=timeout)
        except queue.Empty as exc:
            raise self._process_error("harness sidecar did not declare a release version") from exc
        if line is None:
            raise self._process_error("harness sidecar exited before declaring a release version")
        try:
            response = json.loads(line)
        except json.JSONDecodeError as exc:
            raise HarnessError("harness sidecar returned invalid JSON") from exc
        responses.assert_response_keys(response, {"type", "protocolVersion"})
        if response.get("type") != "ready":
            raise HarnessError("harness sidecar did not declare a release version")
        version = response.get("protocolVersion")
        if version != HARNESS_PROTOCOL_VERSION:
            raise HarnessError(
                f"harness sidecar protocol version mismatch: expected {HARNESS_PROTOCOL_VERSION}, "
                f"received {version!r}; rebuild the sidecar image"
            )
        return version

    def _send_frame(self, frame: dict[str, Any]) -> None:
        process = self._process
        if process is None or process.stdin is None or process.poll() is not None:
            raise self._process_error("harness sidecar is not running")
        try:
            process.stdin.write(json.dumps(frame, separators=(",", ":")) + "\n")
            process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise self._process_error("harness sidecar input failed") from exc

    def _next_request_id(self) -> str:
        self._request_index += 1
        return f"python-{self._request_index:06d}"

    def _process_error(self, message: str) -> HarnessError:
        detail = "".join(self._stderr).strip()
        return HarnessError(f"{message}: {detail}" if detail else message, retryable=True)

    @staticmethod
    def _read_stdout(stream, responses: queue.Queue[str | None]) -> None:
        try:
            with stream:
                for line in stream:
                    if line.strip():
                        responses.put(line)
        finally:
            responses.put(None)

    @staticmethod
    def _read_stderr(stream, stderr: deque[str]) -> None:
        with stream:
            for line in stream:
                stderr.append(line)


def _terminate(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Harness client utilities.")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("probe", help="Send a 1-token Responses API health/balance probe.")
    p.add_argument("--base-url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--api-key-file", required=True, type=Path)
    p.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args(argv)
    if args.command == "probe":
        try:
            key = args.api_key_file.read_text(encoding="utf-8").strip()
            result = probe_provider(args.base_url, args.model, key, timeout_seconds=args.timeout)
        except Exception as exc:
            kind = exc.provider_error if isinstance(exc, HarnessError) else None
            print(json.dumps({"ok": False, "provider_error": kind, "error": str(exc)}, sort_keys=True))
            return 1
        print(json.dumps({"ok": True, **result}, sort_keys=True))
        return 0
    return 2


__all__ = [
    "HarnessClient",
    "MAX_PROVIDER_RETRY_WAIT_SECONDS",
    "MAX_RETRY_BACKOFF_SECONDS",
    "classify_provider_error",
    "probe_provider",
]


if __name__ == "__main__":
    raise SystemExit(main())
