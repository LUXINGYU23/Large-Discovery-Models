"""Decode and validate sidecar responses independently of process transport."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from ldm_tts.harness.protocol import (
    HarnessError, HarnessSubmissionError, HarnessSubmittedArtifact, HarnessTurnResult,
)


def parse_turn_result(value: Any) -> HarnessTurnResult:
    if not isinstance(value, dict):
        raise HarnessError("committed harness turn must be an object")
    assert_response_keys(value, {
        "profileId", "sessionId", "turnId", "roundIndex", "historyFromSeq",
        "historyToSeq", "historyDigest", "inputDigest", "replayed",
        "submissionStatus", "submissionId", "submissionJson", "submissionDigest", "submission",
        "submittedArtifacts", "validationErrors", "usage", "toolBudget", "artifacts",
    })
    submission = value.get("submission")
    usage = value.get("usage")
    artifacts = value.get("artifacts")
    if not isinstance(submission, dict):
        raise HarnessError("committed harness turn has invalid submission")
    if not isinstance(usage, dict) or not isinstance(artifacts, dict):
        raise HarnessError("committed harness turn has invalid metadata")
    usage = parse_usage(usage)
    tool_budget = _tool_budget(value["toolBudget"], usage["toolCalls"])
    assert_response_keys(artifacts, {"turn", "session"})
    submitted_artifacts = parse_submitted_artifacts(value["submittedArtifacts"])
    validation_errors = _parse_submission_errors(value["validationErrors"])
    submission_status = value["submissionStatus"]
    if submission_status not in {"accepted", "rejected"}:
        raise HarnessError("committed harness turn has invalid submissionStatus")
    if submission_status == "accepted" and validation_errors:
        raise HarnessError("accepted harness turn cannot contain validation errors")
    if submission_status == "rejected" and not validation_errors:
        raise HarnessError("rejected harness turn must contain validation errors")
    submission_digest = required_digest(value["submissionDigest"], "submissionDigest")
    verify_submission_json(
        value["submissionJson"],
        submission_digest,
        submission,
        value["submittedArtifacts"],
    )
    replayed = value["replayed"]
    if not isinstance(replayed, bool):
        raise HarnessError("committed harness turn has invalid replayed")
    history_from_seq = _required_nonnegative_int(value["historyFromSeq"], "historyFromSeq")
    history_to_seq = _required_nonnegative_int(value["historyToSeq"], "historyToSeq")
    if history_to_seq < history_from_seq:
        raise HarnessError("committed harness turn has invalid history range")
    return HarnessTurnResult(
        profile_id=required_string(value["profileId"], "profileId"),
        session_id=required_string(value["sessionId"], "sessionId"),
        turn_id=required_string(value["turnId"], "turnId"),
        round_index=_required_nonnegative_int(value["roundIndex"], "roundIndex"),
        history_from_seq=history_from_seq,
        history_to_seq=history_to_seq,
        history_digest=required_digest(value["historyDigest"], "historyDigest"),
        input_digest=required_digest(value["inputDigest"], "inputDigest"),
        replayed=replayed,
        submission_status=submission_status,
        submission_id=required_string(value["submissionId"], "submissionId"),
        submission_digest=submission_digest,
        submission=dict(submission),
        submitted_artifacts=submitted_artifacts,
        validation_errors=validation_errors,
        usage=usage,
        tool_budget=tool_budget,
        artifacts={str(key): str(item) for key, item in artifacts.items() if item is not None},
    )


def parse_usage(value: Any) -> dict[str, Any]:
    assert_response_keys(value, {"providerCalls", "toolCalls", "artifactBytes"})
    return {
        "providerCalls": _required_nonnegative_int(value["providerCalls"], "providerCalls"),
        "toolCalls": _nonnegative_int_mapping(value["toolCalls"], "toolCalls"),
        "artifactBytes": _required_nonnegative_int(value["artifactBytes"], "artifactBytes"),
    }


def verify_submission_json(
    value: Any,
    digest: str,
    submission: Any,
    artifacts: Any,
) -> None:
    if not isinstance(value, str):
        raise HarnessError("harness submissionJson must be a string")
    if hashlib.sha256(value.encode("utf-8")).hexdigest() != digest:
        raise HarnessError("harness submissionJson digest mismatch")
    try:
        envelope = json.loads(value)
    except json.JSONDecodeError as error:
        raise HarnessError("harness submissionJson is invalid JSON") from error
    if (
        not isinstance(envelope, dict)
        or set(envelope) != {"artifacts", "submission"}
        or envelope["submission"] != submission
        or envelope["artifacts"] != artifacts
    ):
        raise HarnessError("harness submissionJson does not match the submitted payload")


def parse_submitted_artifacts(value: Any) -> tuple[HarnessSubmittedArtifact, ...]:
    if not isinstance(value, list):
        raise HarnessError("harness submittedArtifacts must be an array")
    artifacts: list[HarnessSubmittedArtifact] = []
    for raw in value:
        if not isinstance(raw, dict):
            raise HarnessError("harness submitted artifact must be an object")
        assert_response_keys(raw, {
            "pathPointer", "relativePath", "snapshotPath", "sha256", "sizeBytes",
        })
        try:
            artifact = HarnessSubmittedArtifact(
                path_pointer=required_string(raw["pathPointer"], "pathPointer"),
                relative_path=required_string(raw["relativePath"], "relativePath"),
                snapshot_path=required_string(raw["snapshotPath"], "snapshotPath"),
                sha256=required_digest(raw["sha256"], "sha256"),
                size_bytes=_required_nonnegative_int(raw["sizeBytes"], "sizeBytes"),
            )
        except ValueError as exc:
            raise HarnessError(str(exc)) from exc
        artifacts.append(artifact)
    if len({artifact.path_pointer for artifact in artifacts}) != len(artifacts):
        raise HarnessError("harness submitted artifact pointers must be unique")
    return tuple(artifacts)


def _parse_submission_errors(value: Any) -> tuple[HarnessSubmissionError, ...]:
    if not isinstance(value, list):
        raise HarnessError("harness validationErrors must be an array")
    errors: list[HarnessSubmissionError] = []
    for raw in value:
        if not isinstance(raw, dict):
            raise HarnessError("harness validation error must be an object")
        assert_response_keys(raw, {"path", "code", "message", "hint"})
        path = raw["path"]
        hint = raw["hint"]
        if not isinstance(path, str) or not isinstance(hint, str):
            raise HarnessError("harness validation error path and hint must be strings")
        try:
            errors.append(HarnessSubmissionError(
                path=path,
                code=required_string(raw["code"], "code"),
                message=required_string(raw["message"], "message"),
                hint=hint,
            ))
        except ValueError as exc:
            raise HarnessError(str(exc)) from exc
    return tuple(errors)


def required_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise HarnessError(f"committed harness turn has invalid {name}")
    return value


def _required_nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise HarnessError(f"committed harness turn has invalid {name}")
    return value


def required_positive_int(value: Any, name: str) -> int:
    result = _required_nonnegative_int(value, name)
    if result < 1:
        raise HarnessError(f"harness frame has invalid {name}")
    return result


def required_digest(value: Any, name: str) -> str:
    result = required_string(value, name)
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise HarnessError(f"committed harness turn has invalid {name}")
    return result


def _nonnegative_int_mapping(value: Any, name: str) -> dict[str, int]:
    if not isinstance(value, dict) or any(
        not isinstance(key, str)
        or not key
        or isinstance(item, bool)
        or not isinstance(item, int)
        or item < 0
        for key, item in value.items()
    ):
        raise HarnessError(f"committed harness turn has invalid {name}")
    return dict(value)


def _tool_budget(value: Any, tool_calls: dict[str, int]) -> dict[str, dict[str, int]]:
    if not isinstance(value, dict):
        raise HarnessError("committed harness turn has invalid toolBudget")
    result: dict[str, dict[str, int]] = {}
    for name, raw in value.items():
        if not isinstance(name, str) or not name or not isinstance(raw, dict):
            raise HarnessError("committed harness turn has invalid toolBudget")
        assert_response_keys(raw, {"limit", "used", "remaining"})
        limit = _required_nonnegative_int(raw["limit"], f"toolBudget.{name}.limit")
        used = _required_nonnegative_int(raw["used"], f"toolBudget.{name}.used")
        remaining = _required_nonnegative_int(raw["remaining"], f"toolBudget.{name}.remaining")
        if used + remaining != limit or tool_calls.get(name, 0) != used:
            raise HarnessError("committed harness turn has inconsistent toolBudget")
        result[name] = {"limit": limit, "used": used, "remaining": remaining}
    return result


def assert_response_keys(response: dict[str, Any], expected: set[str]) -> None:
    if set(response) != expected:
        raise HarnessError("harness sidecar returned unexpected response fields")
