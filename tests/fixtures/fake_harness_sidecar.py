"""Minimal JSONL sidecar used by the shared harness client test."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from importlib.metadata import version


profiles: list[str] = []
committed = {}
turn_requests = 0
print(json.dumps({"type": "ready", "protocolVersion": version("large-discovery-models")}), flush=True)


def submission_record(submission, artifacts) -> tuple[str, str]:
    body = json.dumps(
        {"artifacts": artifacts, "submission": submission},
        sort_keys=True,
        separators=(",", ":"),
    )
    return body, hashlib.sha256(body.encode()).hexdigest()


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
        profiles = [item["profileId"] for item in frame["profiles"]]
        store = Path(frame["artifactRoot"]) / "fake_committed.json"
        if store.exists():
            committed = json.loads(store.read_text())
        response = {"type": "initialized", **common, "profiles": profiles, "manifest": "manifest.json"}
    elif frame["type"] == "run_turn":
        turn_requests += 1
        failure = os.environ.get("HARNESS_TEST_TURN_FAILURE")
        if failure:
            error = {"message": "provider 502"}
            if failure != "unknown":
                error["turnUsage"] = [{
                    "profileId": item["profileId"],
                    "turnId": "wrong-turn" if failure == "wrong-turn" else item["turnId"],
                    "usage": {"providerCalls": 3, "toolCalls": {"bash": 2}, "artifactBytes": 120},
                } for item in frame["turns"]]
            print(json.dumps({"type": "error", **common, "error": error}), flush=True)
            continue
        turns = []
        for item in frame["turns"]:
            if os.environ.get("HARNESS_TEST_PROCESS_FAILURE") and item["turnId"] in committed:
                previous = committed[item["turnId"]]
                assert previous["inputDigest"] == item["inputDigest"]
                turns.append({**previous, "replayed": True})
                continue
            if os.environ.get("HARNESS_TEST_PARTIAL_FAILURE"):
                if turn_requests == 1 and item["profileId"] == profiles[-1]:
                    continue
                if item["turnId"] in committed:
                    previous = committed[item["turnId"]]
                    assert previous["inputDigest"] == item["inputDigest"]
                    turns.append({**previous, "replayed": True})
                    continue
            attempt_index = 0
            submission = {"candidates": [{"value": item["profileId"]}]}
            artifacts = []
            decision = {"decision": "accept", "errors": []}
            while not os.environ.get("HARNESS_TEST_SKIP_VALIDATION"):
                attempt_index += 1
                submission_json, digest = submission_record(submission, artifacts)
                print(json.dumps({
                    "type": "submission_validation_requested",
                    **common,
                    "validationId": f"{item['turnId']}-validation-{attempt_index}",
                    "profileId": item["profileId"],
                    "turnId": item["turnId"],
                    "attemptIndex": attempt_index,
                    "submission": submission,
                    "artifacts": artifacts,
                    "submissionJson": submission_json,
                    "submissionDigest": digest,
                }), flush=True)
                validation = json.loads(next(sys.stdin))
                assert validation["type"] == "submission_validation_result"
                assert validation["submissionDigest"] == digest
                decision = validation
                if decision["decision"] != "retry":
                    break
                submission = {"candidates": [{"value": f"{item['profileId']}-{attempt_index + 1}"}]}
            submission_json, digest = submission_record(submission, artifacts)
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
                "submissionJson": submission_json,
                "submissionDigest": digest,
                "submission": submission,
                "submittedArtifacts": artifacts,
                "validationErrors": decision["errors"],
                "usage": {"providerCalls": 1, "toolCalls": {}, "artifactBytes": 12},
                "toolBudget": {},
                "artifacts": {"turn": f"turns/{item['turnId']}", "session": f"sessions/{item['profileId']}.jsonl"},
            })
            committed[item["turnId"]] = turns[-1]
            if os.environ.get("HARNESS_TEST_PROCESS_FAILURE"):
                store.write_text(json.dumps(committed))
                if len(committed) == 1:
                    if os.environ["HARNESS_TEST_PROCESS_FAILURE"] == "timeout":
                        time.sleep(30)
                    sys.exit(1)
        if os.environ.get("HARNESS_TEST_PARTIAL_FAILURE") and turn_requests == 1:
            response = {"type": "error", **common, "error": {
                "code": "recoverable_turn_error", "message": "session wall-time limit reached: 1800s",
                "turnUsage": [{"profileId": t["profileId"], "turnId": t["turnId"], "usage": t["usage"]} for t in turns],
            }}
            print(json.dumps(response), flush=True)
            continue
        response = {"type": "turn_committed", **common, "turns": turns}
    elif frame["type"] == "close":
        response = {"type": "closed", **common}
        print(json.dumps(response), flush=True)
        break
    else:
        response = {"type": "error", **common, "error": {"message": "unknown frame"}}
    print(json.dumps(response), flush=True)
