"""Minimal JSONL sidecar used by the shared harness client test."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from importlib.metadata import version


profiles: list[str] = []
print(json.dumps({"type": "ready", "protocolVersion": version("large-discovery-models")}), flush=True)


def submission_digest(submission, artifacts) -> str:
    body = json.dumps(
        {"artifacts": artifacts, "submission": submission},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(body).hexdigest()


for line in sys.stdin:
    frame = json.loads(line)
    request_id = frame["requestId"]
    common = {
        "requestId": request_id,
        "protocolVersion": frame["protocolVersion"],
        "campaignId": frame["campaignId"],
    }
    if frame["type"] == "bootstrap_secret":
        response = (
            {"type": "error", **common, "error": {"message": "secret inherited"}}
            if os.environ.get("HARNESS_TEST_SECRET") or os.environ.get("HARNESS_MCP_SECRET")
            else {"type": "secret_bootstrapped", **common}
        )
    elif frame["type"] == "initialize":
        assert frame["guestRuntime"] == {
            "imageRef": "ldm/fixture-research:aaaaaaaaaaaa",
            "recipeSha256": "a" * 64,
            "rootfsSize": "4G",
            "installPolicy": "session_overlay",
        }
        profiles = [item["profileId"] for item in frame["profiles"]]
        response = {"type": "initialized", **common, "profiles": profiles, "manifest": "manifest.json"}
    elif frame["type"] == "run_turn":
        turns = []
        for item in frame["turns"]:
            attempt_index = 0
            submission = {"candidates": [{"value": item["profileId"]}]}
            artifacts = []
            decision = {"decision": "accept", "errors": []}
            while not os.environ.get("HARNESS_TEST_SKIP_VALIDATION"):
                attempt_index += 1
                digest = submission_digest(submission, artifacts)
                print(json.dumps({
                    "type": "submission_validation_requested",
                    **common,
                    "validationId": f"{item['turnId']}-validation-{attempt_index}",
                    "profileId": item["profileId"],
                    "turnId": item["turnId"],
                    "attemptIndex": attempt_index,
                    "submission": submission,
                    "artifacts": artifacts,
                    "submissionDigest": digest,
                }), flush=True)
                validation = json.loads(next(sys.stdin))
                assert validation["type"] == "submission_validation_result"
                assert validation["submissionDigest"] == digest
                decision = validation
                if decision["decision"] != "retry":
                    break
                submission = {"candidates": [{"value": f"{item['profileId']}-{attempt_index + 1}"}]}
            digest = submission_digest(submission, artifacts)
            if os.environ.get("HARNESS_TEST_CHANGE_AFTER_VALIDATION"):
                submission = {"candidates": [{"value": "changed"}]}
            turns.append({
                "profileId": item["profileId"],
                "sessionId": f"session-{item['profileId']}",
                "turnId": item["turnId"],
                "roundIndex": item["roundIndex"],
                "historyFromSeq": item["historyFromSeq"],
                "historyToSeq": item["historyToSeq"],
                "historyDigest": item["historyDigest"],
                "inputDigest": item["inputDigest"],
                "replayed": False,
                "submissionStatus": (
                    "rejected" if decision["decision"] == "reject_turn" else "accepted"
                ),
                "submissionId": f"{item['turnId']}-submission-{max(attempt_index, 1)}",
                "submissionDigest": digest,
                "submission": submission,
                "submittedArtifacts": artifacts,
                "validationErrors": decision["errors"],
                "usage": {"providerCalls": 1, "toolCalls": {}, "artifactBytes": 12},
                "toolBudget": {},
                "artifacts": {"turn": f"turns/{item['turnId']}", "session": f"sessions/{item['profileId']}.jsonl"},
            })
        response = {"type": "turn_committed", **common, "turns": turns}
    elif frame["type"] == "close":
        response = {"type": "closed", **common}
        print(json.dumps(response), flush=True)
        break
    else:
        response = {"type": "error", **common, "error": {"message": "unknown frame"}}
    print(json.dumps(response), flush=True)
