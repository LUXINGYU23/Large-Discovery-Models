"""Persistent sessions through the real shared HarnessClient and a protocol fixture."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from ldm_tts.contracts import RawProposal
from ldm_tts.engine.run_store import BudgetLedger
from ldm_tts.harness import HarnessError, HarnessProviderAuthorizationRequest, file_sha256
from tasks.researchgym.core import workflow
from tasks.researchgym.core.candidate import ProgramDomain
from tasks.researchgym.core.cases import load_case
from tasks.researchgym.core.harness import (
    PROPOSAL_SKILLS, RESOURCE_ROOT, TOOL_FILE, HarnessProgramSource, ProviderAuthorizationLedger, create_client,
    profiles, write_context,
)
from tasks.researchgym.core.proposals import ProposalExhausted
from tasks.researchgym.core.sampling import Q0_KEY, attach_empirical_base_measure
from tasks.researchgym.core.usage import ApiBudgetExhausted, ApiPricing, account_tokens, harness_turn_tokens

CASE = load_case("continual_learning")
FIXTURE = Path(__file__).with_name("fixtures") / "harness_sidecar.py"


class Runtime:
    def __init__(self, path):
        self.budget = BudgetLedger(limits={}, path=path)

    def consume_many(self, amounts, *, usage_key=None):
        return self.budget.consume_many(amounts, usage_key=usage_key)


def make_source(tmp_path, mode, sessions=2, runtime=None, pricing=None, reserve_usd=None):
    args = workflow.parse_args(["--mock", "--case", CASE.case_id, "--search-method", "ldm_harness",
                                "--harness-sessions", str(sessions), "--harness-command-json", "[]"])
    args.provider_api_key = "synthetic-fixture-key"
    root = tmp_path / "harness"
    write_context(root, CASE, {"models/base.py": "class BaseLearner: ..."}, {"harness_sessions": sessions})
    client = create_client(args, root, "fixture-campaign", CASE, command=[sys.executable, "-u", str(FIXTURE), mode])
    client.start()
    pricing = pricing or ApiPricing(1.0, 2.0)
    source = HarnessProgramSource(client, root, args, CASE, pricing=pricing)
    source.runtime = runtime or Runtime(tmp_path / "budget.json")
    source.configure_provider_authorization(ProviderAuthorizationLedger(
        root=root, runtime=source.runtime, pricing=pricing, campaign_id="fixture-campaign",
        usage_prefix="harness", reserve_usd=reserve_usd,
    ))
    return source


def collect(source, round_idx=0, count=4, records=(), measured=(), recovery_pass=0):
    return source.collect(round_idx=round_idx, count=count, batch_size=0, records=list(records),
                          measured_keys=set(measured), recovery_pass=recovery_pass)


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def test_profiles_and_tools_are_digest_bound():
    proposal = profiles(3)
    assert [p.profile_id for p in proposal] == ["program_research_01", "program_research_02", "program_research_03"]
    assert len({p.agents_sha256 for p in proposal}) == 1 and len(proposal[0].skill_dirs) == len(PROPOSAL_SKILLS)
    assert profiles(1, policy=True)[0].profile_id == "policy_architect"
    assert file_sha256(RESOURCE_ROOT / TOOL_FILE)


def test_researchgym_overrides_unsupported_first_tool_choice(tmp_path):
    source = make_source(tmp_path, "valid", sessions=1)
    try:
        assert source.client.config.provider_request_body["tool_choice"] == "auto"
    finally:
        source.client.close()


def test_sessions_repair_in_place_and_keep_lineage_and_usage(tmp_path):
    source = make_source(tmp_path, "repair")
    try:
        batch = collect(source)
    finally:
        source.client.close()
    assert len(batch.rows) == 4 and {r["note"]["profile_id"] for r in batch.rows} == {"program_research_01", "program_research_02"}
    assert all(r["note"]["source"] == "persistent_harness" and r["rationale"].startswith("fixture") for r in batch.rows)
    sessions = jsonl(tmp_path / "harness/sessions/program_research_01.jsonl")
    assert [s["decision"] for s in sessions] == ["retry", "accept"]
    assert sessions[0]["errors"][0]["path"] == "/artifact_path/candidates/0/program"
    counters = source.runtime.budget.counters
    assert counters["harness_turns"] == 2 and counters["harness_provider_calls"] == 4
    assert counters["harness_provider_requests"] == 4
    assert counters["harness_input_tokens"] == 4000 and counters["api_cost_usd"] == pytest.approx(4 * (1000 + 400) / 1e6)
    assert {json.loads(path.read_text())["status"] for path in (tmp_path / "harness/provider_authorizations").glob("*.json")} == {"reconciled"}
    assert len(batch.attempts) == 2 and all(a.metadata["ldm_proposal_attempt_receipt"].startswith("harness:")
                                            for a in batch.attempts)


def test_refill_smaller_than_pool_still_sends_all_profiles(tmp_path):
    source = make_source(tmp_path, "valid", sessions=2)
    try:
        batch = collect(source, count=1)
    finally:
        source.client.close()
    assert len(batch.rows) == 1
    frames = jsonl(tmp_path / "harness/fixture_requests.jsonl")
    assert [turn["profileId"] for turn in frames[0]["turns"]] == ["program_research_01", "program_research_02"]


def test_provider_authorization_reserves_budget_before_a_second_provider_call(tmp_path):
    pricing = ApiPricing(1.0, 2.0, budget_usd=0.002)
    source = make_source(tmp_path, "repair", pricing=pricing, reserve_usd=0.0015)
    try:
        with pytest.raises(ApiBudgetExhausted, match="would be exceeded"):
            collect(source)
    finally:
        source.client.close()
    assert source.runtime.budget.counters["harness_provider_requests"] == 1
    receipts = list((tmp_path / "harness/provider_authorizations").glob("*.json"))
    assert len(receipts) == 1 and json.loads(receipts[0].read_text())["status"] == "authorized"


def _authorization(turn, number):
    return HarnessProviderAuthorizationRequest("fixture-campaign", "program_research_01", turn,
                                               f"{turn}-provider-{number}", "a" * 64)


def _captured_response(root, turn, number, *, status=200, usage=True):
    request_id = f"{turn}-provider-{number}"
    turn_root = root / "sessions/program_research_01/turns" / turn
    response = turn_root / "provider" / f"{request_id}.response.bin"
    response.parent.mkdir(parents=True, exist_ok=True)
    response.write_text(json.dumps({"usage": {"input_tokens": 1000, "output_tokens": 200}}
                                   if usage else {"error": {"message": "invalid request"}}))
    with (turn_root / "provider_index.jsonl").open("a") as handle:
        handle.write(json.dumps({"type": "provider_exchange", "turnId": turn,
                                 "profileId": "program_research_01", "requestId": request_id,
                                 "response": {"status": status,
                                              "artifact": f"provider/{request_id}.response.bin"}}) + "\n")


def test_completed_request_releases_reserve_before_whole_turn_finishes(tmp_path):
    runtime = Runtime(tmp_path / "budget.json")
    meter = ProviderAuthorizationLedger(root=tmp_path / "harness", runtime=runtime,
                                        pricing=ApiPricing(1, 2, budget_usd=1.01), campaign_id="fixture-campaign",
                                        usage_prefix="harness", reserve_usd=1.0)
    meter.authorize(_authorization("turn-1", 1))
    _captured_response(tmp_path / "harness", "turn-1", 1)
    meter.authorize(_authorization("turn-1", 2))
    assert runtime.budget.counters["api_cost_usd"] == pytest.approx(0.0014)
    receipts = [json.loads(path.read_text()) for path in meter.root.glob("*.json")]
    assert sorted(row["status"] for row in receipts) == ["authorized", "reconciled"]
    _captured_response(tmp_path / "harness", "turn-1", 2)
    meter.reconcile_turn("turn-1", reported_provider_calls=2,
                         tokens={"input_tokens": 2000, "output_tokens": 400})
    meter.reconcile_turn("turn-1", reported_provider_calls=2,
                         tokens={"input_tokens": 2000, "output_tokens": 400})
    assert runtime.budget.counters["api_cost_usd"] == pytest.approx(0.0028)
    assert runtime.budget.counters["harness_input_tokens"] == 2000
    assert {json.loads(path.read_text())["status"] for path in meter.root.glob("*.json")} == {"reconciled"}


def test_failed_provider_response_fails_fast_without_releasing_reserve(tmp_path):
    runtime = Runtime(tmp_path / "budget.json")
    meter = ProviderAuthorizationLedger(root=tmp_path / "harness", runtime=runtime,
                                        pricing=ApiPricing(1, 2, budget_usd=2), campaign_id="fixture-campaign",
                                        usage_prefix="harness", reserve_usd=1.0)
    meter.authorize(_authorization("turn-1", 1))
    _captured_response(tmp_path / "harness", "turn-1", 1, status=400, usage=False)
    with pytest.raises(HarnessError, match="provider returned HTTP 400"):
        meter.authorize(_authorization("turn-1", 2))
    assert len(list(meter.root.glob("*.json"))) == 1
    assert runtime.budget.counters.get("api_cost_usd", 0) == 0


def test_settlement_is_shared_across_candidate_and_policy_pools(tmp_path):
    runtime = Runtime(tmp_path / "budget.json")
    pricing = ApiPricing(1, 2, budget_usd=1.01)
    policy = ProviderAuthorizationLedger(root=tmp_path / "policy_harness", runtime=runtime,
                                         pricing=pricing, campaign_id="fixture-campaign",
                                         usage_prefix="policy", reserve_usd=1.0)
    candidate = ProviderAuthorizationLedger(root=tmp_path / "harness", runtime=runtime,
                                            pricing=pricing, campaign_id="fixture-campaign",
                                            usage_prefix="harness", reserve_usd=1.0)
    policy.authorize(_authorization("policy-turn", 1))
    _captured_response(tmp_path / "policy_harness", "policy-turn", 1)
    candidate.authorize(_authorization("candidate-turn", 1))
    assert runtime.budget.counters["policy_input_tokens"] == 1000
    assert runtime.budget.counters["api_cost_usd"] == pytest.approx(0.0014)
    assert json.loads(next(policy.root.glob("*.json")).read_text())["status"] == "reconciled"


def test_recovery_reconciles_cumulative_captures_against_all_turn_receipts(tmp_path):
    runtime = Runtime(tmp_path / "budget.json")
    meter = ProviderAuthorizationLedger(root=tmp_path / "harness", runtime=runtime,
                                        pricing=ApiPricing(1, 1), campaign_id="fixture-campaign",
                                        usage_prefix="harness", reserve_usd=None)
    meter.authorize(_authorization("turn-1", 1))
    meter.reconcile_turn("turn-1", reported_provider_calls=1,
                         tokens={"input_tokens": 100, "output_tokens": 20})
    meter.authorize(_authorization("turn-1", 2))
    meter.reconcile_turn("turn-1", reported_provider_calls=2,
                         tokens={"input_tokens": 200, "output_tokens": 40})
    assert {json.loads(p.read_text())["status"] for p in meter.root.glob("*.json")} == {"reconciled"}


def test_candidate_and_policy_pools_share_outstanding_reservations(tmp_path):
    runtime = Runtime(tmp_path / "budget.json")
    pricing = ApiPricing(1, 1, budget_usd=1)
    policy = ProviderAuthorizationLedger(root=tmp_path / "policy_harness", runtime=runtime,
                                         pricing=pricing, campaign_id="fixture-campaign",
                                         usage_prefix="policy", reserve_usd=0.6)
    candidate = ProviderAuthorizationLedger(root=tmp_path / "harness", runtime=runtime,
                                            pricing=pricing, campaign_id="fixture-campaign",
                                            usage_prefix="harness", reserve_usd=0.6)
    policy.authorize(_authorization("policy-turn", 1))
    with pytest.raises(ApiBudgetExhausted, match="would be exceeded"):
        candidate.authorize(_authorization("candidate-turn", 1))


def test_partial_provider_capture_keeps_known_cost_and_unknown_reserve(tmp_path):
    capture = tmp_path / "captures"
    capture.mkdir()
    (capture / "response.json").write_text(json.dumps({"usage": {"input_tokens": 2000, "output_tokens": 400}}))
    (capture / "provider_index.jsonl").write_text("\n".join(json.dumps(row) for row in (
        {"turnId": "turn-1", "type": "provider_exchange", "response": {"artifact": "response.json"}},
        {"turnId": "turn-1", "type": "provider_transport_error"},
    )))
    tokens = harness_turn_tokens(tmp_path, "turn-1")
    assert tokens == {"input_tokens": 2000, "output_tokens": 400, "unknown_calls": 1.0}
    runtime = Runtime(tmp_path / "budget.json")
    account_tokens(runtime, ApiPricing(1, 2), "harness", tokens, usage_key="turn-1")
    assert runtime.budget.counters["harness_input_tokens"] == 2000
    assert runtime.budget.counters["harness_usage_unknown"] == 1


def test_cross_session_agreement_enters_q0(tmp_path):
    source = make_source(tmp_path, "agreement")
    try:
        batch = collect(source)
    finally:
        source.client.close()
    proposals = [RawProposal({"program": r["program"]}, "test", {"research_note": r["note"]}) for r in batch.rows]
    annotated = attach_empirical_base_measure(proposals, ProgramDomain(CASE), measured_keys=set())
    assert [p.metadata[Q0_KEY]["occurrence_count"] for p in annotated] == [2, 2, 2, 2]
    assert len(annotated[0].metadata["research_annotations"]) == 2


def test_recoverable_failure_replays_committed_session_and_continues_the_other(tmp_path):
    source = make_source(tmp_path, "partial")
    try:
        batch = collect(source)
    finally:
        source.client.close()
    assert len(batch.rows) == 4
    frames = jsonl(tmp_path / "harness/fixture_requests.jsonl")
    assert len(frames) == 2 and [t["turnId"] for t in frames[0]["turns"]] == [t["turnId"] for t in frames[1]["turns"]]
    counters = source.runtime.budget.counters
    assert counters["harness_turns"] == 2 and counters["harness_provider_calls"] == 4


def test_accepted_sessions_are_not_rerun_after_another_exhausts_validation(tmp_path):
    runtime = Runtime(tmp_path / "budget.json")
    source = make_source(tmp_path, "exhaust_second", runtime=runtime)
    try:
        with pytest.raises(ProposalExhausted) as raised:
            collect(source)
    finally:
        source.client.close()
    assert raised.value.metadata["sessions"][0]["profile_id"] == "program_research_02"
    assert len(raised.value.attempts) == 2
    retry = make_source(tmp_path, "valid", runtime=runtime)
    try:
        batch = collect(retry, recovery_pass=1)
    finally:
        retry.client.close()
    assert len(batch.rows) == 4
    frames = jsonl(tmp_path / "harness/fixture_requests.jsonl")
    assert [t["profileId"] for t in frames[-1]["turns"]] == ["program_research_01", "program_research_02"]
    assert frames[-1]["turns"][0]["turnId"] == frames[0]["turns"][0]["turnId"]
    # A process-level replay of the completed round issues no new turns.
    replay = make_source(tmp_path, "valid", runtime=runtime)
    try:
        assert len(collect(replay, recovery_pass=1).rows) == 4
    finally:
        replay.client.close()
    assert len(jsonl(tmp_path / "harness/fixture_requests.jsonl")) == len(frames)


def test_measured_programs_are_rejected_and_history_only_grows(tmp_path):
    source = make_source(tmp_path, "valid")
    try:
        first = collect(source)
        records = [{"seq": 0, "candidate_id": "rg-a", "canonical_key": CASE.canonical_key(first.rows[0]["program"]),
                    "round_idx": 0, "status": "succeeded", "objective": 1.0, "metrics": {}, "error": "",
                    "research_annotations": [], "program": first.rows[0]["program"]}]
        with pytest.raises(ProposalExhausted):
            # Round 1 of the fixture reuses round-offset programs; seed the measured set with them.
            measured = {CASE.canonical_key(r["program"]) for r in first.rows}
            source.collect(round_idx=0, count=4, batch_size=0, records=records, measured_keys=measured,
                           recovery_pass=5)
    finally:
        source.client.close()
    errors = [e for s in jsonl(tmp_path / "harness/sessions/program_research_01.jsonl") for e in s["errors"]]
    assert any(e["code"] == "already_measured" for e in errors)
    with pytest.raises(ValueError, match="changed or shrank"):
        from tasks.researchgym.core.history import write_history
        write_history(tmp_path / "harness/measured_history.json", [])


NODE_TEST = r'''
import assert from "node:assert/strict";
import { mkdirSync, writeFileSync } from "node:fs";
import { pathToFileURL } from "node:url";
const [tool, workspace] = process.argv.slice(2);
const tools = new Map();
(await import(pathToFileURL(tool).href)).default({ registerTool: spec => tools.set(spec.name, spec) });
const ctx = { cwd: workspace };
const call = async (name, params = {}) => (await tools.get(name).execute("fixture", params, undefined, undefined, ctx)).details;
const described = await call("describe_researchgym_case");
assert.equal(described.case_id, "continual_learning");
assert.ok(described.reference_files["models/base.py"] > 0 && described.guest_file.path.startsWith("/workspace/"));
const history = await call("get_measured_history", { sort_by: "objective", order: "desc", limit: 1 });
assert.equal(history.total, 2); assert.equal(history.observations[0].candidate_id, "rg-best");
assert.equal(history.observations[0].program, undefined);
const detailed = await call("get_measured_history", { candidate_ids: ["rg-best"], detail: "detailed" });
assert.ok(detailed.observations[0].program.includes("class Learner"));
const checked = await call("check_candidate_programs", { artifact_path: "candidates.json" });
assert.equal(checked.valid, false);
assert.deepEqual(checked.candidates.map(c => c.errors.map(e => e.code)), [["already_measured"], [], ["missing_entry"]]);
await assert.rejects(call("check_candidate_programs", { artifact_path: "../outside.json" }));
console.log("ok");
'''


@pytest.mark.skipif(shutil.which("node") is None or shutil.which("python3") is None, reason="node and python3 required")
def test_task_tools_filter_history_and_reuse_admission_rules(tmp_path):
    root = tmp_path / "artifacts"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    write_context(root, CASE, {"models/base.py": "class BaseLearner: ..."}, {})
    seed = CASE.seed_program()
    history = [
        {"seq": 0, "candidate_id": "rg-low", "canonical_key": "x", "round_idx": 0, "status": "succeeded",
         "objective": 70.0, "metrics": {}, "error": "", "research_annotations": [], "program": seed + "\nA = 1\n"},
        {"seq": 1, "candidate_id": "rg-best", "canonical_key": CASE.canonical_key(seed), "round_idx": 0,
         "status": "succeeded", "objective": 80.0, "metrics": {}, "error": "", "research_annotations": [], "program": seed},
    ]
    (root / "measured_history.json").write_text(json.dumps({"observations": history}))
    rows = [{"program": seed, "change_summary": "s", "rationale": "r"},
            {"program": seed + "\nB = 2\n", "change_summary": "s", "rationale": "r"},
            {"program": "def f():\n    pass\n", "change_summary": "s", "rationale": "r"}]
    (workspace / "candidates.json").write_text(json.dumps({"candidates": rows}))
    (tmp_path / "outside.json").write_text("{}")
    script = tmp_path / "tool_test.mjs"
    script.write_text(NODE_TEST)
    completed = subprocess.run(["node", str(script), str(RESOURCE_ROOT / TOOL_FILE), str(workspace)], capture_output=True,
                               text=True, timeout=60, env={"PATH": __import__("os").environ["PATH"],
                                                           "LDM_RG_CONTEXT": str(root / "context.json"),
                                                           "LDM_RG_HISTORY": str(root / "measured_history.json")})
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "ok"
