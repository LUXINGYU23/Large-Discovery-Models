"""Explicitly synthetic JSONL sidecar used to exercise the real shared client.

This fixture never calls an LLM. It follows the Pi sidecar wire protocol:
provisional submissions, task validation, committed-turn replay, recoverable
turn errors with measured usage, and provider-capture files for token usage.
Every trace row is labeled synthetic_fixture.
"""

import hashlib
import json
import sys
from pathlib import Path

MODE = sys.argv[1] if len(sys.argv) > 1 else "repair"
TASK_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = TASK_ROOT.parents[1]
ROOT = None
COMMITTED = {}
PROFILES = []
POLICY = False
CASE = ""


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def trace(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(json.dumps({"synthetic_fixture": True, **row}) + "\n")


def policy_source(valid):
    expression = "np.full(len(query_features), 0.1)" if valid else "np.zeros((len(query_features), 1))"
    return f'''# Authored static protocol-test fixture; no generated code or real provider.
import numpy as np
POLICY_API_VERSION = 1
CAPABILITIES = {{"prior_mean": 1, "ldm_weights": 1}}
def compute_prior_mean(history_features, history_utilities, query_features, context):
    return {expression}
def choose_ldm_weights(context):
    return {{"stage": "fixture", "alpha": 0.5, "eta": 1.5}}
'''


def program(index):
    seed = (TASK_ROOT / "resources/cases/seeds" / f"{CASE}.py").read_text().rstrip()
    return seed + f"\n\n\ndef _fixture_variant_{index}(values):\n    return values * {index + 2}\n"


def provider_capture(turn_id):
    request_id = sha(turn_id)[:12]
    body = 'data: {"type":"response.completed","response":{"usage":{"input_tokens":1000,"output_tokens":200}}}\n'
    path = ROOT / "turns" / turn_id / "provider" / f"{request_id}.response.bin"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    trace(ROOT / "turns" / turn_id / "provider_index.jsonl", {
        "type": "provider_exchange", "turnId": turn_id, "requestId": request_id,
        "response": {"artifact": f"provider/{request_id}.response.bin", "status": 200}})


package = REPO_ROOT / "harnesses/pi/package.json"
print(json.dumps({"type": "ready", "protocolVersion": json.loads(package.read_text())["version"]}), flush=True)
for line in sys.stdin:
    frame = json.loads(line)
    common = {key: frame[key] for key in ("requestId", "protocolVersion", "campaignId")}
    if frame["type"] == "bootstrap_secret":
        response = {"type": "secret_bootstrapped", **common}
    elif frame["type"] == "initialize":
        ROOT = Path(frame["artifactRoot"])
        ROOT.mkdir(parents=True, exist_ok=True)
        PROFILES = [p["profileId"] for p in frame["profiles"]]
        POLICY = json.loads(frame["submissionContractJson"])["contractId"] == "optimization_policy"
        CASE = frame["caseId"].split(":")[0]
        if (ROOT / "fixture_committed.json").exists():
            COMMITTED = json.loads((ROOT / "fixture_committed.json").read_text())
        # Same identity fields the real sidecar pins in <artifact root>/manifest.json.
        (ROOT / "manifest.json").write_text(json.dumps({
            "synthetic_fixture": True, "protocolVersion": frame["protocolVersion"], "campaignId": frame["campaignId"],
            "taskId": frame["taskId"], "caseId": frame["caseId"], "seed": frame["seed"], "backend": "fixture",
            "profileSetSha256": frame["profileSetSha256"], "profiles": frame["profiles"], "limits": frame["limits"],
            "submissionContract": json.loads(frame["submissionContractJson"])}))
        response = {"type": "initialized", **common, "profiles": PROFILES, "manifest": "manifest.json"}
    elif frame["type"] == "run_turn":
        trace(ROOT / "fixture_requests.jsonl", {"turns": frame["turns"]})
        results, interrupted = [], False
        for item in frame["turns"]:
            if item["turnId"] in COMMITTED:
                prior = COMMITTED[item["turnId"]]
                assert prior["inputDigest"] == item["inputDigest"]
                results.append({**prior, "replayed": True})
                continue
            if MODE in ("partial", "fail_once") and results and not (ROOT / "fixture_interrupted").exists():
                (ROOT / "fixture_interrupted").touch()
                code = "recoverable_turn_error" if MODE == "partial" else "provider_unavailable"
                response = {"type": "error", **common, "error": {
                    "code": code, "message": "synthetic session interruption",
                    "turnUsage": [{"profileId": r["profileId"], "turnId": r["turnId"], "usage": r["usage"]} for r in results]}}
                interrupted = True
                break
            if not POLICY:
                payload = json.loads(item["message"].split("\n", 1)[1])
                count = payload["candidate_count"]
                history = json.loads((ROOT / "measured_history.json").read_text())["observations"]
                trace(ROOT / "fixture_provider.jsonl", {"profile_id": item["profileId"], "turn_id": item["turnId"],
                                                        "tool": "get_measured_history", "history_count": len(history)})
                provider_capture(item["turnId"])
            else:
                provider_capture(item["turnId"])
            for attempt in range(1, 5):
                if POLICY:
                    body, filename = policy_source(attempt > 1 and MODE != "policy_reject"), "optimization_policy.py"
                    submission = {"action": "replace", "artifact_path": filename}
                else:
                    offset = 0 if MODE == "agreement" else PROFILES.index(item["profileId"]) * 7
                    base = item["roundIndex"] * 31 + offset
                    if "replenishment" in payload:
                        base += 500 + 7 * PROFILES.index(item["profileId"]) + len(payload["replenishment"]["already_proposed_this_round"])
                    rows = [{"program": program(base + i), "change_summary": f"fixture variant {base + i}",
                             "rationale": f"fixture independent hypothesis {item['profileId']}"} for i in range(count)]
                    exhausted = MODE == "exhaust" or (MODE == "exhaust_second" and PROFILES.index(item["profileId"]) == 1)
                    if (attempt == 1 and MODE != "valid") or exhausted:
                        rows[0] = {**rows[0], "program": "def unrelated():\n    return 1\n"}
                    body, filename = json.dumps({"candidates": rows}), "candidates.json"
                    submission = {"artifact_path": filename}
                path = ROOT / "turns" / item["turnId"] / "attempts" / str(attempt) / filename
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(body)
                artifacts = [{"pathPointer": "/artifact_path", "relativePath": filename,
                              "snapshotPath": path.relative_to(ROOT).as_posix(), "sha256": sha(body),
                              "sizeBytes": len(body.encode())}]
                submission_json = canonical({"artifacts": artifacts, "submission": submission})
                digest = sha(submission_json)
                print(json.dumps({
                    "type": "submission_validation_requested", **common,
                    "validationId": item["turnId"] + f"-validation-{attempt}",
                    "profileId": item["profileId"], "turnId": item["turnId"], "attemptIndex": attempt,
                    "submission": submission, "artifacts": artifacts,
                    "submissionJson": submission_json, "submissionDigest": digest,
                }), flush=True)
                decision = json.loads(next(sys.stdin))
                assert decision["type"] == "submission_validation_result"
                trace(ROOT / "sessions" / (item["profileId"] + ".jsonl"), {
                    "turn_id": item["turnId"], "attempt": attempt, "decision": decision["decision"],
                    "errors": decision["errors"], "history_from_seq": item["historyFromSeq"],
                    "history_to_seq": item["historyToSeq"]})
                if decision["decision"] != "retry":
                    break
            result = {
                **{key: item[key] for key in ("profileId", "turnId", "roundIndex", "historyFromSeq", "historyToSeq",
                                               "historyDigest", "inputDigest")},
                "sessionId": "fixture-session-" + item["profileId"], "replayed": False,
                "submissionStatus": "accepted" if decision["decision"] == "accept" else "rejected",
                "submissionId": item["turnId"] + f"-submission-{attempt}",
                "submissionJson": submission_json, "submissionDigest": digest,
                "submission": submission, "submittedArtifacts": artifacts,
                "validationErrors": decision["errors"],
                "usage": {"providerCalls": attempt, "toolCalls": {"get_measured_history": 1, "bash": 1},
                          "artifactBytes": len(body.encode())},
                "toolBudget": {},
                "artifacts": {"turn": f"turns/{item['turnId']}", "session": f"sessions/{item['profileId']}.jsonl"},
            }
            COMMITTED[item["turnId"]] = result
            (ROOT / "fixture_committed.json").write_text(json.dumps(COMMITTED))
            results.append(result)
        if not interrupted:
            response = {"type": "turn_committed", **common, "turns": results}
    elif frame["type"] == "close":
        print(json.dumps({"type": "closed", **common}), flush=True)
        break
    else:
        response = {"type": "error", **common, "error": {"message": "unknown fixture frame"}}
    print(json.dumps(response), flush=True)
