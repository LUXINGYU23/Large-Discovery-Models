"""Minimal JSONL sidecar used by the shared harness client test."""

from __future__ import annotations

import json
import os
import sys


profiles: list[str] = []
for line in sys.stdin:
    frame = json.loads(line)
    request_id = frame["requestId"]
    if frame["type"] == "bootstrap_secret":
        response = (
            {"type": "error", "requestId": request_id, "error": {"message": "secret inherited"}}
            if os.environ.get("HARNESS_TEST_SECRET")
            else {"type": "secret_bootstrapped", "requestId": request_id}
        )
    elif frame["type"] == "initialize":
        profiles = [item["profileId"] for item in frame["profiles"]]
        response = {"type": "initialized", "requestId": request_id, "profiles": profiles}
    elif frame["type"] == "run_turn":
        turns = []
        for item in frame["turns"]:
            turns.append({
                "profileId": item["profileId"],
                "sessionId": f"session-{item['profileId']}",
                "turnId": item["turnId"],
                "inputDigest": item["inputDigest"],
                "submission": {
                    "submissionId": f"{item['turnId']}-submission",
                    "candidates": [{"value": item["profileId"]}],
                },
                "usage": {"providerCalls": 1, "webCalls": 0, "context7Calls": 0, "artifactBytes": 12},
                "artifacts": {"turn": f"turns/{item['turnId']}", "session": f"sessions/{item['profileId']}.jsonl"},
            })
        response = {"type": "turn_committed", "requestId": request_id, "turns": turns}
    elif frame["type"] == "close":
        response = {"type": "closed", "requestId": request_id}
        print(json.dumps(response), flush=True)
        break
    else:
        response = {"type": "error", "requestId": request_id, "error": {"message": "unknown frame"}}
    print(json.dumps(response), flush=True)
