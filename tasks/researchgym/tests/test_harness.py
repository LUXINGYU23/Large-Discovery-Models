"""Persistent sessions through the real shared HarnessClient and a protocol fixture."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from ldm_tts.contracts import RawProposal
from ldm_tts.engine.run_store import BudgetLedger
from ldm_tts.harness import file_sha256
from tasks.researchgym.core import workflow
from tasks.researchgym.core.candidate import ProgramDomain
from tasks.researchgym.core.cases import load_case
from tasks.researchgym.core.harness import (
    PROPOSAL_SKILLS, RESOURCE_ROOT, TOOL_FILE, HarnessProgramSource, create_client, profiles, write_context,
)
from tasks.researchgym.core.proposals import ProposalExhausted
from tasks.researchgym.core.sampling import Q0_KEY, attach_empirical_base_measure
from tasks.researchgym.core.usage import ApiPricing

CASE = load_case("continual_learning")
FIXTURE = Path(__file__).with_name("fixtures") / "harness_sidecar.py"


class Runtime:
    def __init__(self, path):
        self.budget = BudgetLedger(limits={}, path=path)

    def consume_many(self, amounts, *, usage_key=None):
        return self.budget.consume_many(amounts, usage_key=usage_key)


def make_source(tmp_path, mode, sessions=2, runtime=None):
    args = workflow.parse_args(["--mock", "--case", CASE.case_id, "--search-method", "ldm_harness",
                                "--harness-sessions", str(sessions), "--harness-command-json", "[]"])
    args.provider_api_key = "synthetic-fixture-key"
    root = tmp_path / "harness"
    write_context(root, CASE, {"models/base.py": "class BaseLearner: ..."}, {"harness_sessions": sessions})
    client = create_client(args, root, "fixture-campaign", CASE, command=[sys.executable, "-u", str(FIXTURE), mode])
    client.start()
    source = HarnessProgramSource(client, root, args, CASE, pricing=ApiPricing(1.0, 2.0))
    source.runtime = runtime or Runtime(tmp_path / "budget.json")
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
    assert counters["harness_input_tokens"] == 2000 and counters["api_cost_usd"] == pytest.approx(2 * (1000 + 400) / 1e6)
    assert len(batch.attempts) == 2 and all(a.metadata["ldm_proposal_attempt_receipt"].startswith("harness:")
                                            for a in batch.attempts)


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
    assert [t["profileId"] for t in frames[-1]["turns"]] == ["program_research_02"]
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
