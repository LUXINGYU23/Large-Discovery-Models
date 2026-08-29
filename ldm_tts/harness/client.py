"""Long-lived JSONL subprocess client for research harness sidecars."""

from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
import time
from collections import deque
from collections.abc import Callable, Sequence
from typing import Any

from ldm_tts.harness.protocol import (
    PROTOCOL_VERSION,
    HarnessPoolConfig,
    HarnessSubmissionRequest,
    HarnessSubmissionValidation,
    HarnessTurn,
    HarnessTurnResult,
)


SubmissionValidator = Callable[[HarnessSubmissionRequest], HarnessSubmissionValidation]


class HarnessError(RuntimeError):
    pass


class HarnessClient:
    def __init__(
        self,
        command: Sequence[str],
        *,
        api_key: str,
        config: HarnessPoolConfig,
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
        self.config = config
        self.response_timeout_seconds = float(response_timeout_seconds)
        self._process: subprocess.Popen[str] | None = None
        self._responses: queue.Queue[str] = queue.Queue()
        self._stderr: deque[str] = deque(maxlen=40)
        self._request_index = 0

    def start(self) -> None:
        if self._process is not None:
            raise HarnessError("harness client is already started")
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
                if value != self._api_key
            },
        )
        self._process = process
        assert process.stdout is not None and process.stderr is not None
        threading.Thread(target=self._read_stdout, args=(process.stdout,), daemon=True).start()
        threading.Thread(target=self._read_stderr, args=(process.stderr,), daemon=True).start()
        try:
            self._request("bootstrap_secret", {"apiKey": self._api_key}, "secret_bootstrapped")
            self._api_key = ""
            request_id = self._next_request_id()
            self._exchange(self.config.initialize_frame(request_id), "initialized")
        except BaseException:
            self.close()
            raise

    def run_turn(
        self,
        turns: Sequence[HarnessTurn],
        *,
        submission_validator: SubmissionValidator,
    ) -> tuple[HarnessTurnResult, ...]:
        if not turns:
            raise ValueError("harness turn batch must not be empty")
        expected = {turn.profile_id: turn for turn in turns}
        if len(expected) != len(turns):
            raise ValueError("harness turn profile IDs must be unique")

        def validate(request: HarnessSubmissionRequest) -> HarnessSubmissionValidation:
            turn = expected.get(request.profile_id)
            if turn is None or request.turn_id != turn.turn_id:
                raise HarnessError("harness submission validation request does not match the turn batch")
            return submission_validator(request)

        payload = self._request(
            "run_turn",
            {"turns": [turn.to_dict() for turn in turns]},
            "turn_committed",
            submission_validator=validate,
        )
        raw_turns = payload.get("turns")
        if not isinstance(raw_turns, list):
            raise HarnessError("harness response is missing committed turns")
        results = tuple(_parse_turn_result(item) for item in raw_turns)
        if {result.profile_id for result in results} != set(expected) or len(results) != len(turns):
            raise HarnessError("harness response does not match the requested profiles")
        for result in results:
            turn = expected[result.profile_id]
            if (
                result.turn_id != turn.turn_id
                or result.round_index != turn.round_index
                or result.history_from_seq != turn.history_from_seq
                or result.history_to_seq != turn.history_to_seq
                or result.history_digest != turn.history_digest
                or result.input_digest != turn.input_digest
            ):
                raise HarnessError("harness response does not match the requested turn")
        return results

    def close(self) -> None:
        process = self._process
        if process is None:
            self._api_key = ""
            return
        try:
            if process.poll() is None:
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
            self._process = None
            self._api_key = ""

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
    ) -> dict[str, Any]:
        request_id = self._next_request_id()
        return self._exchange(
            {**self.config.common_frame(request_id, frame_type), **fields},
            expected_type,
            timeout_seconds=timeout_seconds,
            submission_validator=submission_validator,
        )

    def _exchange(
        self,
        frame: dict[str, Any],
        expected_type: str,
        *,
        timeout_seconds: float | None = None,
        submission_validator: SubmissionValidator | None = None,
    ) -> dict[str, Any]:
        process = self._process
        if process is None or process.stdin is None:
            raise HarnessError("harness client is not started")
        if process.poll() is not None:
            raise self._process_error("harness sidecar exited")
        self._send_frame(frame)
        timeout = self.response_timeout_seconds if timeout_seconds is None else timeout_seconds
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                _terminate(process)
                raise self._process_error("harness sidecar response timed out")
            try:
                line = self._responses.get(timeout=remaining)
            except queue.Empty as exc:
                _terminate(process)
                raise self._process_error("harness sidecar response timed out") from exc
            try:
                response = json.loads(line)
            except json.JSONDecodeError as exc:
                raise HarnessError("harness sidecar returned invalid JSON") from exc
            if not isinstance(response, dict) or response.get("requestId") != frame["requestId"]:
                raise HarnessError("harness sidecar response requestId mismatch")
            if (
                response.get("protocolVersion") != PROTOCOL_VERSION
                or response.get("campaignId") != self.config.campaign_id
            ):
                raise HarnessError("harness sidecar response protocol identity mismatch")
            if response.get("type") != "submission_validation_requested":
                break
            if submission_validator is None:
                _terminate(process)
                raise HarnessError("harness requested submission validation outside run_turn")
            try:
                self._answer_submission_validation(response, submission_validator)
            except BaseException:
                _terminate(process)
                raise
        if response.get("type") == "error":
            _assert_response_keys(response, {"type", "requestId", "protocolVersion", "campaignId", "error"})
            error = response.get("error")
            message = error.get("message") if isinstance(error, dict) else "unknown sidecar error"
            raise HarnessError(str(message))
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
        _assert_response_keys(response, expected_keys[expected_type])
        return response

    def _answer_submission_validation(
        self,
        response: dict[str, Any],
        validator: SubmissionValidator,
    ) -> None:
        _assert_response_keys(response, {
            "type", "requestId", "protocolVersion", "campaignId", "validationId",
            "profileId", "turnId", "attemptIndex", "candidates",
        })
        candidates = response["candidates"]
        if not isinstance(candidates, list) or not candidates or any(
            not isinstance(candidate, dict) for candidate in candidates
        ):
            raise HarnessError("harness submission validation request has invalid candidates")
        request = HarnessSubmissionRequest(
            profile_id=_required_string(response["profileId"], "profileId"),
            turn_id=_required_string(response["turnId"], "turnId"),
            attempt_index=_required_positive_int(response["attemptIndex"], "attemptIndex"),
            candidates=tuple(dict(candidate) for candidate in candidates),
        )
        validation = validator(request)
        if not isinstance(validation, HarnessSubmissionValidation):
            raise HarnessError("submission validator returned an invalid result")
        if any(rejection.index >= len(candidates) for rejection in validation.rejections):
            raise HarnessError("submission validator returned an out-of-range rejection index")
        self._send_frame({
            **self.config.common_frame(response["requestId"], "submission_validation_result"),
            "validationId": _required_string(response["validationId"], "validationId"),
            "accepted": validation.accepted,
            "rejected": [rejection.to_dict() for rejection in validation.rejections],
            "requiredReplacements": len(validation.rejections),
        })

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
        return HarnessError(f"{message}: {detail}" if detail else message)

    def _read_stdout(self, stream) -> None:
        for line in stream:
            if line.strip():
                self._responses.put(line)

    def _read_stderr(self, stream) -> None:
        for line in stream:
            self._stderr.append(line)


def _terminate(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def _parse_turn_result(value: Any) -> HarnessTurnResult:
    if not isinstance(value, dict):
        raise HarnessError("committed harness turn must be an object")
    _assert_response_keys(value, {
        "profileId", "sessionId", "turnId", "roundIndex", "historyFromSeq",
        "historyToSeq", "historyDigest", "inputDigest", "submission", "usage",
        "artifacts",
    })
    submission = value.get("submission")
    usage = value.get("usage")
    artifacts = value.get("artifacts")
    candidates = submission.get("candidates") if isinstance(submission, dict) else None
    if not isinstance(candidates, list) or any(not isinstance(item, dict) for item in candidates):
        raise HarnessError("committed harness turn has invalid candidates")
    if not isinstance(usage, dict) or not isinstance(artifacts, dict):
        raise HarnessError("committed harness turn has invalid metadata")
    assert isinstance(submission, dict)
    _assert_response_keys(submission, {"submissionId", "candidates"})
    _assert_response_keys(usage, {"providerCalls", "webCalls", "context7Calls", "artifactBytes"})
    _assert_response_keys(artifacts, {"turn", "session"})
    try:
        history_from_seq = _required_nonnegative_int(value["historyFromSeq"], "historyFromSeq")
        history_to_seq = _required_nonnegative_int(value["historyToSeq"], "historyToSeq")
        if history_to_seq < history_from_seq:
            raise HarnessError("committed harness turn has invalid history range")
        return HarnessTurnResult(
            profile_id=_required_string(value["profileId"], "profileId"),
            session_id=_required_string(value["sessionId"], "sessionId"),
            turn_id=_required_string(value["turnId"], "turnId"),
            round_index=_required_nonnegative_int(value["roundIndex"], "roundIndex"),
            history_from_seq=history_from_seq,
            history_to_seq=history_to_seq,
            history_digest=_required_digest(value["historyDigest"], "historyDigest"),
            input_digest=_required_digest(value["inputDigest"], "inputDigest"),
            submission_id=_required_string(submission["submissionId"], "submissionId"),
            candidates=tuple(dict(item) for item in candidates),
            usage=dict(usage),
            artifacts={str(key): str(item) for key, item in artifacts.items() if item is not None},
        )
    except KeyError as exc:
        raise HarnessError("committed harness turn is missing identity fields") from exc


def _required_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise HarnessError(f"committed harness turn has invalid {name}")
    return value


def _required_nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise HarnessError(f"committed harness turn has invalid {name}")
    return value


def _required_positive_int(value: Any, name: str) -> int:
    result = _required_nonnegative_int(value, name)
    if result < 1:
        raise HarnessError(f"harness frame has invalid {name}")
    return result


def _required_digest(value: Any, name: str) -> str:
    result = _required_string(value, name)
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise HarnessError(f"committed harness turn has invalid {name}")
    return result


def _assert_response_keys(response: dict[str, Any], expected: set[str]) -> None:
    if set(response) != expected:
        raise HarnessError("harness sidecar returned unexpected response fields")


__all__ = ["HarnessClient", "HarnessError"]
