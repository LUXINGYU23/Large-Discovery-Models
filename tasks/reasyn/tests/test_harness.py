"""Actual shared JSONL client/runner integration with an explicitly synthetic provider."""

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from ldm_tts.contracts import Candidate, RawProposal, ReservoirBuilder
from ldm_tts.engine.expansion import ExpansionRequest
from ldm_tts.engine.run_store import BudgetLedger
from ldm_tts.harness import (
    HarnessClient, HarnessError, HarnessPoolConfig, HarnessSubmissionError,
    PolicyResearchController, file_sha256, policy_submission_contract,
)
from ldm_tts.harness.policy_execution import PolicyExecutionError, PolicyExecutionResult
from ldm_tts.optimization.records import BOObservation
from ldm_tts.transport import ProposalRequest
from tasks.reasyn.core.candidate import ReaSynDomain
from tasks.reasyn.core.evaluator import TDCOracleEvaluator
from tasks.reasyn.core.harness import HarnessTargetSource, RESOURCE_ROOT, profiles, submission_contract
from tasks.reasyn.core.optimization_policy import ReaSynPolicyAdapter
from tasks.reasyn.core.projector import Projector
from tasks.reasyn.core.proposals import ReaSynExpander
from tasks.reasyn.core.sampling import attach_empirical_base_measure, empirical_base_masses
from tasks.reasyn.core.selection import AcquisitionTiltedSelector, TanimotoGPSelector
from tasks.reasyn.core.surrogate import MoleculeEncoder
from tasks.reasyn.core import workflow
from tasks.reasyn.core.workflow import parse_args

FIXTURE = Path(__file__).with_name("fixtures") / "harness_sidecar.py"
RUNNER = Path(__file__).resolve().parents[3] / "harnesses/pi/policy_runner.py"


class Sink:
    def append(self, *args, **kwargs):
        pass


def args():
    options = parse_args(["--mock", "--benchmark", "tdc", "--evaluations-per-round", "2"])
    options.proposal_mode = "harness"
    return options


def client(root, *, mode="molecular", policy=False, count=2):
    root.mkdir(parents=True, exist_ok=True)
    return HarnessClient(
        (sys.executable, "-u", str(FIXTURE), mode), api_key="synthetic-fixture-secret",
        config=HarnessPoolConfig(
            artifact_root=root, profiles=profiles(count, policy=policy), campaign_id="protocol-fixture",
            task_id="reasyn", case_id="synthetic", seed=0,
            submission_contract=policy_submission_contract() if policy else submission_contract(),
        ), response_timeout_seconds=10,
    )


def request(round_idx=0, history=(), *, batch=0):
    return ProposalRequest(messages=({"role": "user", "content": "protocol fixture"},), metadata={
        "round_idx": round_idx, "count": 2, "minibatch_index": batch,
        "history": list(history), "rejection_feedback": [],
    })


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def harness_identity(options, target="", source_commit="fixture-source"):
    options.asset_digests = {}
    options.proposal_recovery_seed_span = max(
        workflow.DEFAULT_PROPOSAL_RECOVERY_SEED_SPAN,
        options.iterations,
        options.max_oracle_calls,
    )
    configuration = workflow._jsonable(options)
    configuration.update(
        proposal_samples=options.reservoir_size,
        proposal_candidates_per_request=options.proposal_batch_size,
        proposal_counting="bounded_minibatches",
        harness_component_digests=workflow._harness_component_digests(options),
    )
    contract = SimpleNamespace(benchmark={"source_commit": source_commit})
    return workflow._scientific_identity(options, configuration, target, contract)


def test_scientific_identity_freezes_harness_images_and_mcp_content(tmp_path):
    mcp = tmp_path / "mcp.yaml"
    mcp.write_text("servers: {}\n")
    base_argv = [
        "--mock",
        "--search-method",
        "harness",
        "--harness-mcp-config",
        str(mcp),
        "--harness-sidecar-image",
        "sidecar:one",
        "--policy-runner-image",
        "policy:one",
    ]
    identity = harness_identity(parse_args(base_argv))
    changed_image = harness_identity(parse_args([
        *base_argv[:-4],
        "--harness-sidecar-image",
        "sidecar:two",
        "--policy-runner-image",
        "policy:one",
    ]))
    assert identity["harness_sidecar_image"] == "sidecar:one"
    assert identity["policy_runner_image"] == "policy:one"
    assert identity["harness_mcp_config"] == str(mcp.resolve())
    assert identity["harness_component_digests"]["resource_tree"]
    assert identity != changed_image

    mcp.write_text("servers:\n  local:\n    transport: stdio\n    command: node\n    tools: [search]\n")
    changed_mcp = harness_identity(parse_args(base_argv))
    assert identity["harness_component_digests"]["harness_mcp_config"] != (
        changed_mcp["harness_component_digests"]["harness_mcp_config"]
    )
    assert identity != changed_mcp


def test_real_client_parallel_history_repair_projection_and_evaluation(tmp_path):
    options = args()
    root = tmp_path / "harness"
    ledger = BudgetLedger({})
    with client(root) as transport:
        source = HarnessTargetSource(transport, root, options, account=ledger.consume_many)
        expander = ReaSynExpander(options, Projector(options, tmp_path), Sink(), client=source)
        evaluator = TDCOracleEvaluator(options, tmp_path)
        observations = []
        first_sessions = None
        for round_idx in range(2):
            expanded = expander.expand(ExpansionRequest(round_idx, 2, observations=tuple(observations)))
            reservoir = ReservoirBuilder(ReaSynDomain("tdc", mock=True)).build(expanded.proposals)
            assert len(reservoir.candidates) == 2
            sessions = {p.metadata["harness_lineage"]["session_id"] for p in expanded.proposals}
            if first_sessions is None:
                first_sessions = sessions
            else:
                assert sessions == first_sessions
            for candidate in reservoir.candidates:
                measured = evaluator.evaluate(candidate)
                assert measured.succeeded
                observations.append(SimpleNamespace(candidate=candidate, candidate_id=candidate.candidate_id,
                    canonical_key=candidate.canonical_key, metrics=measured.metrics))
        assert len(evaluator.entries) == 4
    sent = jsonl(root / "fixture_requests.jsonl")
    assert [len(row["turns"]) for row in sent] == [2, 2]
    assert [(turn["historyFromSeq"], turn["historyToSeq"]) for turn in sent[1]["turns"]] == [(0, 2), (0, 2)]
    for path in (root / "sessions").glob("*.jsonl"):
        events = jsonl(path)
        assert [event["decision"] for event in events] == ["retry", "accept", "retry", "accept"]
        assert all(event["synthetic_fixture"] for event in events)
    provider = jsonl(root / "fixture_provider.jsonl")
    assert [row["history_count"] for row in provider] == [0, 0, 2, 2]
    assert ledger.counters["harness_turns"] == 4
    assert ledger.counters["harness_provider_calls"] == 8
    assert ledger.counters["harness_validation_submissions"] == 8
    authoritative = json.loads((root / "measured_history.json").read_text())["observations"]
    assert len(authoritative) == 2
    assert authoritative[0]["research_annotations"][0]["rationale"].startswith("fixture independent")


def test_small_batches_rotate_all_sessions_and_replay_durable_allocations(tmp_path):
    root = tmp_path / "harness"
    requests = [replace(request(batch=i), metadata={**request(batch=i).metadata, "count": 4})
                for i in range(3)]
    with client(root, count=8) as transport:
        source = HarnessTargetSource(transport, root, args())
        source.propose(requests[0])
        source.propose(requests[1])
    with client(root, count=8) as transport:
        source = HarnessTargetSource(transport, root, args())
        source.propose(requests[0])
        source.propose(requests[2])
    sent = jsonl(root / "fixture_requests.jsonl")
    assert [[turn["profileId"] for turn in row["turns"]] for row in sent] == [
        [p.profile_id for p in profiles(8)[:4]],
        [p.profile_id for p in profiles(8)[4:]],
        [p.profile_id for p in profiles(8)[:4]],
    ]


def test_reconstruction_policy_exposes_prior_and_effective_query_weights():
    adapter = ReaSynPolicyAdapter(MoleculeEncoder(mock=True), benchmark="reconstruction",
                                 alpha=1, eta=1, seed=0)
    assert set(adapter.capability_contract().enabled_capabilities) == {"prior_mean@1", "ldm_weights@1"}
    for alpha in ("0", "0.5", "2", "10"):
        assert parse_args(["--mock", "--benchmark", "reconstruction", "--acquisition-alpha", alpha]).acquisition_alpha == float(alpha)


def test_partial_session_recovery_replays_frozen_inputs_and_cumulative_usage(tmp_path):
    root = tmp_path / "harness"
    ledger = BudgetLedger({}, path=tmp_path / "budget.json")
    options = args()
    with client(root, mode="partial") as transport:
        source = HarnessTargetSource(transport, root, options, account=ledger.consume_many)
        with pytest.raises(HarnessError, match="synthetic session interruption"):
            source.propose(request())
        result = source.propose(request())
        assert len(json.loads(result.text)["candidates"]) == 2
        counters = dict(ledger.counters)
        assert source.propose(request()).to_dict() == result.to_dict()
        assert ledger.counters == counters
    sent = jsonl(root / "fixture_requests.jsonl")
    assert len(sent) == 2
    assert sent[0]["turns"] == sent[1]["turns"]
    assert ledger.counters["harness_turns"] == 2
    assert ledger.counters["harness_provider_calls"] == 4
    assert ledger.counters["harness_tool_calls"] == 4
    assert ledger.counters["harness_validation_submissions"] == 4
    # The first session researched only once even though the barrier was retried.
    assert len(jsonl(root / "fixture_provider.jsonl")) == 2


def test_proposal_recovery_pass_uses_separate_harness_turn_cache(tmp_path):
    root = tmp_path / "harness"
    ledger = BudgetLedger({})
    options = args()
    with client(root) as transport:
        source = HarnessTargetSource(transport, root, options, account=ledger.consume_many)
        source.propose(request())
        recovery = request()
        recovery.metadata["proposal_recovery_pass"] = 1
        source.propose(recovery)
    assert (root / "proposal_turns/round-000000/batch-000000/response.json").is_file()
    assert (
        root / "proposal_turns/recovery-000001/round-000000/batch-000000/response.json"
    ).is_file()


def test_committed_cursor_replays_native_turn_before_projection_cache_handoff(tmp_path):
    root = tmp_path / "harness"
    options = args()
    history = [{"candidate_id": "measured-1", "smiles": "CCCC", "metrics": {"oracle_score": 0.5}}]
    ledger = BudgetLedger({})
    with client(root) as transport:
        source = HarnessTargetSource(transport, root, options, account=ledger.consume_many)
        first = source.propose(request(1, history))
        # Simulate loss of only the normalization handoff; native committed turns
        # and their original input must still replay after local cursors advanced.
        (root / "proposal_turns/round-000001/batch-000000/response.json").unlink()
        second = HarnessTargetSource(transport, root, options, account=ledger.consume_many).propose(request(1, history))
    assert second.to_dict() == first.to_dict()
    sent = jsonl(root / "fixture_requests.jsonl")
    assert sent[0]["turns"] == sent[1]["turns"]
    assert sent[1]["turns"][0]["historyFromSeq"] == 0
    assert ledger.counters["harness_turns"] == 2
    assert ledger.counters["harness_provider_calls"] == 4


def test_compact_delta_full_history_and_agreement_annotations(tmp_path):
    root = tmp_path / "harness"
    history = [{"candidate_id": f"old-{i}", "smiles": "CCCC", "metrics": {"oracle_score": 0.1}} for i in range(40)]
    with client(root, mode="agreement") as transport:
        source = HarnessTargetSource(transport, root, args())
        response = source.propose(request(1, history))
        payload = json.loads(jsonl(root / "fixture_requests.jsonl")[0]["turns"][0]["message"].split("\n", 1)[1])
        assert len(payload["new_measured_observations"]) == 8
        assert payload["omitted_delta_observations"] == 32
        assert len(json.loads((root / "measured_history.json").read_text())["observations"]) == 40
        with pytest.raises(ValueError, match="changed or shrank"):
            source.propose(request(2, history[:-1]))
    proposals = tuple(RawProposal({"smiles": "CCO", "synthesis": "CCO", "projection_artifact": "fixture.json"},
        "reasyn_projector", {"pathway_verified": True, "harness_lineage": lineage})
        for lineage in response.metadata["harness_lineage"])
    annotated = attach_empirical_base_measure(proposals, benchmark="tdc", mock=True)
    reservoir = ReservoirBuilder(ReaSynDomain("tdc", mock=True)).build(annotated)
    assert len(reservoir.candidates) == 1
    notes = reservoir.candidates[0].metadata["research_annotations"]
    assert len(notes) == 2
    assert len({row["session_id"] for row in notes}) == 2


def policy_round_data(*, history_limit=2):
    encoder = MoleculeEncoder(mock=True)
    candidates = tuple(Candidate(s, {"smiles": s}, s) for s in ("CCO", "CCN"))
    representations = {c.candidate_id: encoder.encode(c) for c in candidates}
    history = [BOObservation.scalar(f"past-{i}", value, [0.01 * (i + 1)] * 32,
        feature_version=encoder.version, metadata={"round_idx": i}) for i, value in enumerate((0.9, 0.3, 0.5))]
    gp = TanimotoGPSelector(objective_name="utility", feature_version=encoder.version, history_limit=history_limit)
    gp.fit(history)
    baseline = gp.select(candidates, representations, count=2)
    adapter = ReaSynPolicyAdapter(encoder, alpha=1.0, eta=1.0, seed=0, history_limit=history_limit)
    inputs = adapter.build_selection_round(round_index=3, history=history, candidates=candidates,
        representations=representations, baseline_predictions=baseline.predictions,
        valid_proposal_occurrences=4, q0=np.asarray([0.75, 0.25]), requested_evaluation_batch=1)
    return adapter, inputs, history, candidates, representations, baseline


def test_policy_diagnostic_projection_matches_actual_bounded_gp_and_priors():
    adapter, inputs, history, candidates, representations, baseline = policy_round_data()
    location, scale = inputs.diagnostic_arrays["diagnostic_current_location_scale"]
    assert location == pytest.approx(0.4)  # recent two observations, not all three
    assert scale == pytest.approx(0.1)
    weights = inputs.diagnostic_arrays["diagnostic_current_weights"]
    assert weights.shape == (2, 3)
    assert np.all(weights[:, 0] == 0)
    projected = location + scale * (weights @ ((inputs.history_utilities - location) / scale))
    assert projected == pytest.approx([p.scalar_mean for p in baseline.predictions])
    assert len(inputs.execution_context["validation_folds"]) == 2
    bad = PolicyExecutionResult(np.zeros((3, 1)), np.zeros(2), "fixture", 1, 1, 0)
    assert adapter.validate_task_execution(bad, inputs)[0].code == "reasyn_prior_alignment"
    with pytest.raises(ValueError, match="align"):
        adapter.build_selection_round(round_index=3, history=history, candidates=candidates,
            representations=representations, baseline_predictions=baseline.predictions[::-1],
            valid_proposal_occurrences=4, q0=np.asarray([0.75, 0.25]), requested_evaluation_batch=1)


class AuthoredFixtureExecutor:
    """Run only this test's static artifact through the actual restricted runner.

    Production continues to use DockerPolicyExecutor. This local fixture is not
    a claim that arbitrary generated artifacts are isolated by this test helper.
    """
    def __init__(self):
        self.outputs = []

    def execute(self, artifact_path, input_directory, output_directory):
        assert Path(artifact_path).read_text().startswith("# Authored static protocol-test fixture;")
        output = Path(output_directory)
        diagnostic = RESOURCE_ROOT / "policy_diagnostics.py"
        completed = subprocess.run([sys.executable, str(RUNNER), "execute", "--artifact", str(artifact_path),
            "--input", str(input_directory), "--output", str(output),
            "--diagnostics", str(diagnostic), "--diagnostics-sha256", file_sha256(diagnostic)],
            capture_output=True, text=True, timeout=15, check=False)
        response = json.loads(completed.stdout)
        if completed.returncode:
            raise PolicyExecutionError(tuple(HarnessSubmissionError(**row) for row in response["errors"]))
        metadata = json.loads((output / "result.json").read_text())
        self.outputs.append(metadata)
        with np.load(output / "arrays.npz", allow_pickle=False) as arrays:
            return PolicyExecutionResult(arrays["history_prior_mean"].copy(), arrays["query_prior_mean"].copy(),
                metadata["stage"], metadata["alpha"], metadata["eta"], metadata["prior_clip_count"])


def test_real_policy_controller_shared_runner_repairs_shape_and_replays(tmp_path):
    root = tmp_path / "policy"
    adapter, inputs, history, candidates, representations, _ = policy_round_data()
    executor = AuthoredFixtureExecutor()
    with client(root, policy=True) as transport:
        controller = PolicyResearchController(client=transport, adapter=adapter, executor=executor, root=root)
        policy = controller.resolve(inputs)
        assert policy.source == "artifact"
        assert policy.history_prior_mean.shape == (3,)
        assert policy.query_prior_mean.shape == (2,)
        assert policy.alpha == 0.5 and policy.eta == 1.5
        assert controller.resolve(inputs).artifact_digest == policy.artifact_digest
    sessions = jsonl(root / "sessions/policy_architect.jsonl")
    assert [row["decision"] for row in sessions] == ["retry", "accept"]
    assert len(jsonl(root / "fixture_requests.jsonl")) == 1
    diagnostics = executor.outputs[-1]["draft_diagnostics"]
    assert diagnostics["held_out_count"] == 2
    assert diagnostics["current_pool"]["probability_total_variation"] > 0
    assert (root / "rounds/round_003/result.json").exists()


def test_actual_guest_tools_register_query_full_history_and_preserve_reconstruction_identity(tmp_path):
    import os
    import shutil
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required only for the Harness guest-tool integration test")
    context = tmp_path / "context.json"
    context.write_text(json.dumps({"benchmark": "tdc", "oracle": "jnk3"}))
    history = tmp_path / "history.json"
    rows = [{"candidate_id": f"c-{i}", "smiles": "CCO", "round_idx": i // 10, "metrics": {"oracle_score": i / 140},
             "research_annotations": [{"rationale": "measured hypothesis"}]} for i in range(140)]
    history.write_text(json.dumps({"observations": rows}))
    script = r'''
import assert from "node:assert/strict";
import { readFileSync, writeFileSync } from "node:fs";
import { pathToFileURL } from "node:url";
const tools = new Map();
const { default: register } = await import(pathToFileURL(process.argv[1]));
register({registerTool(tool) { assert.equal(tool.parameters.additionalProperties, false); tools.set(tool.name, tool); }});
assert.deepEqual([...tools.keys()].sort(), ["check_measured_product", "describe_reasyn_task", "get_measured_history"]);
const call = async (name, params = {}) => (await tools.get(name).execute("fixture", params)).details;
assert.equal((await call("describe_reasyn_task")).oracle, "jnk3");
const page = await call("get_measured_history", {offset: 128, limit: 12});
assert.equal(page.total, 140); assert.equal(page.observations.length, 12); assert.equal(page.next_offset, null);
assert.equal(page.observations[0].annotation_count, 1);
assert.equal(page.observations[0].research_annotations, undefined);
const detailed = await call("get_measured_history", {candidate_ids: ["c-0", "c-139"], detail: "detailed"});
assert.equal(detailed.observations.length, 2);
assert.equal(detailed.observations[0].research_annotations[0].rationale, "measured hypothesis");
const ranked = await call("get_measured_history", {round: 11, sort_by: "score", order: "desc", limit: 2});
assert.equal(ranked.total, 10);
assert.deepEqual(ranked.observations.map(row => row.candidate_id), ["c-119", "c-118"]);
const exact = await call("get_measured_history", {candidate_id: "c-0"});
assert.equal(exact.observations.length, 1);
const products = await call("check_measured_product", {smiles: ["CCO", "CCN"]});
assert.equal(products.identity_space, "canonical_product");
assert.deepEqual(products.products.map(row => row.already_measured), [true, false]);
writeFileSync(process.env.LDM_REASYN_CONTEXT, JSON.stringify({benchmark:"reconstruction", original_target:"CCO"}));
const queries = await call("check_measured_product", {smiles: ["CCO", "CCN"]});
assert.equal(queries.identity_space, "canonical_query_and_sampling_seed");
assert.deepEqual(queries.products.map(row => row.already_measured), [false, false]);
assert.deepEqual(queries.products.map(row => row.query_seen), [true, false]);
writeFileSync(process.env.LDM_REASYN_HISTORY, JSON.stringify({observations:[]}));
assert.equal((await call("get_measured_history")).total, 0);
await assert.rejects(() => call("get_measured_history", {limit:0}));
await assert.rejects(() => call("check_measured_product", {smiles:[]}));
console.log(JSON.stringify({status:"ok",tools:[...tools.keys()]}));
'''
    completed = subprocess.run([node, "--input-type=module", "-e", script,
        str(RESOURCE_ROOT / "tools/molecular_research.mjs")], env={**os.environ,
        "LDM_REASYN_CONTEXT": str(context), "LDM_REASYN_HISTORY": str(history)},
        capture_output=True, text=True, timeout=10, check=False)
    assert completed.returncode == 0, completed.stderr
    from tasks.reasyn.core.harness import TOOL_NAMES
    assert set(json.loads(completed.stdout)["tools"]) == set(TOOL_NAMES)
