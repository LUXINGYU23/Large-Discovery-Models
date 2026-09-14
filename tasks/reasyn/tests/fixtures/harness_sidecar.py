"""Explicitly synthetic JSONL provider used to exercise the real shared client.

This fixture never calls an LLM. Every trace row is labeled synthetic_fixture.
"""

import hashlib
import json
import sys
from pathlib import Path

MODE = sys.argv[1] if len(sys.argv) > 1 else "molecular"
SMILES = ("CCO", "CCN", "CCC", "CCCO", "CCCN", "CCCC")
ROOT = None
COMMITTED = {}
PROFILES = []
POLICY = False


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def trace(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(json.dumps({"synthetic_fixture": True, **row}) + "\n")


def source(valid):
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


# A task's independently locked environment does not install the root Python
# distribution. The wire release belongs to the actual sidecar package.
sidecar_package = Path(__file__).resolve().parents[4] / "harnesses/pi/package.json"
protocol_version = json.loads(sidecar_package.read_text())["version"]
print(json.dumps({"type": "ready", "protocolVersion": protocol_version}), flush=True)
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
        if (ROOT / "fixture_committed.json").exists():
            COMMITTED = json.loads((ROOT / "fixture_committed.json").read_text())
        response = {"type": "initialized", **common, "profiles": PROFILES, "manifest": "fixture_manifest.json"}
    elif frame["type"] == "run_turn":
        trace(ROOT / "fixture_requests.jsonl", {"turns": frame["turns"]})
        results = []
        interrupted = False
        for item in frame["turns"]:
            if item["turnId"] in COMMITTED:
                prior = COMMITTED[item["turnId"]]
                assert prior["inputDigest"] == item["inputDigest"]
                results.append({**prior, "replayed": True})
                continue
            if MODE == "partial" and results and not (ROOT / "fixture_interrupted").exists():
                (ROOT / "fixture_interrupted").touch()
                response = {"type": "error", **common, "error": {
                    "message": "synthetic session interruption",
                    "turnUsage": [{"profileId": r["profileId"], "turnId": r["turnId"], "usage": r["usage"]} for r in results],
                }}
                interrupted = True
                break
            if not POLICY:
                payload = json.loads(item["message"].split("\n", 1)[1])
                count = payload["candidate_count"]
                history = json.loads((ROOT / "measured_history.json").read_text())["observations"]
                trace(ROOT / "fixture_provider.jsonl", {
                    "profile_id": item["profileId"], "turn_id": item["turnId"],
                    "tool": "get_measured_history", "history_count": len(history),
                })
            for attempt in range(1, 5):
                if POLICY:
                    body = source(attempt > 1)
                    filename = "optimization_policy.py"
                    submission = {"action": "replace", "artifact_path": filename}
                else:
                    index = 0 if MODE == "agreement" else item["roundIndex"] * 2 + PROFILES.index(item["profileId"])
                    body = json.dumps({"candidates": [
                        {"target_smiles": SMILES[(index + i) % len(SMILES)] if attempt > 1 else "invalid",
                         "rationale": f"fixture independent hypothesis {item['profileId']}"} for i in range(count)
                    ]})
                    filename = "candidates.json"
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
                    "history_to_seq": item["historyToSeq"],
                })
                if decision["decision"] != "retry":
                    break
            result = {
                **{key: item[key] for key in ("profileId", "turnId", "roundIndex", "historyFromSeq", "historyToSeq", "historyDigest", "inputDigest")},
                "sessionId": "fixture-session-" + item["profileId"], "replayed": False,
                "submissionStatus": "accepted" if decision["decision"] == "accept" else "rejected",
                "submissionId": item["turnId"] + f"-submission-{attempt}",
                "submissionJson": submission_json, "submissionDigest": digest,
                "submission": submission, "submittedArtifacts": artifacts,
                "validationErrors": decision["errors"],
                "usage": {"providerCalls": attempt, "toolCalls": {"get_measured_history": 1, "bash": 1}, "artifactBytes": len(body.encode())},
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
