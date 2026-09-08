"""Actual shared JSONL client integration with a deterministic test sidecar.

The sidecar exercises real public geometry but contains no model/provider. Its
trace must not be used as live-agent qualification evidence.
"""

import json
import sys
from pathlib import Path

from ldm_tts.harness import HarnessClient, HarnessPoolConfig
from tasks.atomworld.core.harness import harness_profiles, submission_contract
from tasks.atomworld.core.workflow import parse_args, run

SIDECAR = r"""
import hashlib, json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[2])
from tasks.atomworld.core.geometry import execute_operations

root = Path(sys.argv[1])
root.mkdir(parents=True, exist_ok=True)
def send(frame, kind, **fields):
    print(json.dumps({**{k:frame[k] for k in ("protocolVersion", "requestId", "campaignId")},
        "type":kind, **fields}), flush=True)
def log(value):
    with (root / "protocol_fixture.jsonl").open("a") as stream:
        stream.write(json.dumps(value) + "\n")
print(json.dumps({"type":"ready", "protocolVersion":"1.0.0"}), flush=True)
for line in sys.stdin:
    frame = json.loads(line)
    if frame["type"] == "bootstrap_secret":
        send(frame, "secret_bootstrapped")
    elif frame["type"] == "initialize":
        send(frame, "initialized", profiles=[], manifest={})
    elif frame["type"] == "close":
        send(frame, "closed")
        break
    elif frame["type"] == "run_turn":
        results=[]
        for turn in frame["turns"]:
            context=json.loads(turn["message"])
            source=context["sample"]["input_cif"]
            output="<cif>" + execute_operations(source, [{"op":"move", "index":0, "d_pos":[1,0,0]}]) + "</cif>"
            for attempt, text in enumerate(["invalid CIF", output], 1):
                submission={"generated_output":text, "rationale":"Applied the public Cartesian displacement using the delivered geometry implementation."}
                encoded=json.dumps({"submission":submission, "artifacts":[]},sort_keys=True,separators=(",", ":"))
                digest=hashlib.sha256(encoded.encode()).hexdigest()
                send(frame, "submission_validation_requested", validationId=f"{turn['turnId']}_{attempt}",
                    profileId=turn["profileId"], turnId=turn["turnId"], attemptIndex=attempt,
                    submission=submission, artifacts=[], submissionJson=encoded, submissionDigest=digest)
                validation=json.loads(next(sys.stdin))
                log({"type":"validation", "turn":turn["turnId"], "decision":validation["decision"], "errors":validation["errors"]})
                assert validation["decision"] == ("retry" if attempt == 1 else "accept")
            log({"type":"public_turn", "turn":turn, "tool":"delivered_geometry", "session":turn["profileId"]})
            results.append({
                **{key:turn[key] for key in ("profileId","turnId","roundIndex","historyFromSeq","historyToSeq","historyDigest","inputDigest")},
                "sessionId":turn["profileId"], "replayed":False, "submissionStatus":"accepted",
                "submissionId":turn["turnId"], "submissionJson":encoded, "submissionDigest":digest,
                "submission":submission, "submittedArtifacts":[], "validationErrors":[],
                "usage":{"providerCalls":0,"toolCalls":{"geometry_fixture":1},"artifactBytes":0},
                "toolBudget":{}, "artifacts":{"session":"fixture-session", "turn":"fixture-turn"}})
        send(frame, "turn_committed", turns=results)
"""


def test_shared_client_runs_geometry_repair_history_and_evaluation(tmp_path):
    script = tmp_path / "sidecar.py"
    script.write_text(SIDECAR)
    roots = []

    def factory(args, root, sample, *, policy=False):
        roots.append(root)
        client = HarnessClient(
            [
                sys.executable,
                str(script),
                str(root),
                str(Path(__file__).resolve().parents[3]),
            ],
            api_key="fixture-only",
            config=HarnessPoolConfig(
                artifact_root=root,
                profiles=harness_profiles(args.harness_profile),
                campaign_id="fixture-campaign",
                task_id="atomworld",
                case_id=sample["sample_id"],
                seed=0,
                submission_contract=submission_contract(),
            ),
            response_timeout_seconds=30,
        )
        client.start()
        return client

    result = run(
        parse_args(
            [
                "--mock",
                "--search-method",
                "harness",
                "--iterations",
                "2",
                "--out-dir",
                str(tmp_path / "campaign"),
            ]
        ),
        harness_client_factory=factory,
    )
    traces = [
        json.loads(line)
        for line in (roots[0] / "protocol_fixture.jsonl").read_text().splitlines()
    ]
    turns = [row for row in traces if row["type"] == "public_turn"]
    assert len(turns) == 4
    assert turns[0]["session"] == turns[2]["session"]
    assert turns[2]["turn"]["historyToSeq"] == 1
    assert any(
        row["decision"] == "retry" for row in traces if row["type"] == "validation"
    )
    assert result.runtime.budget.counters["harness_tool_calls"] == 4
    assert result.runtime.budget.counters["harness_validation_submissions"] == 8
    assert result.runtime.budget.counters["benchmark_jobs"] >= 1
    assert len(result.projected["samples"][0]["attempts"]) == 2
