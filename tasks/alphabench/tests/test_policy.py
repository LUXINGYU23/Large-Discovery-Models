from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import json
import os

import numpy as np
import pytest

from ldm_tts.contracts import RawProposal
from ldm_tts.harness import CompiledOptimizationPolicy, DockerPolicyExecutor, file_sha256
from ldm_tts.engine.run_store import CampaignRuntime
from ldm_tts.optimization.gp import RBFGPSurrogate
from ldm_tts.optimization.records import BOObservation
from tasks.alphabench.core.candidate import FactorDomain
from tasks.alphabench.core.collection import AcceptedActions
from tasks.alphabench.core.gateway import OracleGateway
from tasks.alphabench.core.harness import HarnessMeter
from tasks.alphabench.core.harness_runtime import build_policy_harness
from tasks.alphabench.core.policy import CompiledFactorSelector, T3PolicyAdapter
from tasks.alphabench.core.protocol import T3Protocol
from tasks.alphabench.core.selection import FEATURE_VERSION, FactorEncoder, FactorSelector


def _fixture():
    protocol = T3Protocol(method="ldm_harness_compiled", sessions=2, candidates_per_session=1, batch_size=1)
    encoder = FactorEncoder()
    domain = FactorDomain("qlib")
    expressions = ("Mean($close,5)", "Std($close,10)", "Ref($close,5)", "Mean($volume,20)")
    items = [domain.admit(RawProposal({"name": f"f{index}", "expression": expression}, "test"))
             for index, expression in enumerate(expressions)]
    history = tuple(BOObservation.scalar(item.candidate_id, score, encoder.encode(item).values,
        feature_version=FEATURE_VERSION, metadata={"round_idx": 0})
        for item, score in zip(items[:2], (.02, -.01)))
    candidates = tuple(replace(item, metadata={"q0": .5,
        "harness_lineage": [{"profile_id": f"gp_query_{index:02d}"}]})
        for index, item in enumerate(items[2:]))
    representations = {item.candidate_id: encoder.encode(item) for item in candidates}
    return protocol, history, candidates, representations


class _Controller:
    def __init__(self, protocol, initial_ids, history_prior, query_prior):
        self.adapter = T3PolicyAdapter(protocol, initial_ids)
        self.history_prior, self.query_prior = history_prior, query_prior
        self.recorded = []

    def resolve(self, round_input):
        self.round_input = round_input
        return CompiledOptimizationPolicy("epoch_000", "a" * 64,
            np.asarray(self.history_prior, dtype=float), np.asarray(self.query_prior, dtype=float),
            "test", 2.0, .25, "artifact", False,
            metadata={"action": "replace", "status": "accepted", "harness_turn": {
                "profile_id": "policy_architect", "turn_id": "policy-test", "usage": {"providerCalls": 1}}})

    def record_predictions(self, round_idx, rows):
        self.recorded.append((round_idx, rows))


class _Meter:
    def reconcile(self, profile_id, turn_id, usage):
        assert (profile_id, turn_id, usage["providerCalls"]) == ("policy_architect", "policy-test", 1)
        return {"host_authorizations": 1, "sidecar_provider_calls": 1}


def _selector(protocol, history, history_prior, query_prior):
    controller = _Controller(protocol, [item.candidate_id for item in history], history_prior, query_prior)
    runtime = SimpleNamespace(consume_many=lambda amounts, usage_key: None)
    gateway = SimpleNamespace(runtime=runtime, host=SimpleNamespace(run=lambda operation: operation()))
    selector = CompiledFactorSelector(protocol, mock=True,
        initial_candidate_ids=(item.candidate_id for item in history))
    selector.bind(controller, _Meter(), gateway)
    selector.fit(history)
    return selector, controller


def test_zero_prior_matches_original_gp_and_selection():
    protocol, history, candidates, representations = _fixture()
    baseline = FactorSelector("mock_rank_ic", seed=protocol.random_seed)
    baseline.fit(history)
    expected = baseline.select(candidates, representations, round_idx=0)
    selector, controller = _selector(protocol, history, [0, 0], [0, 0])
    result = selector.select(candidates, representations, round_idx=0)
    assert result.selected_candidate_ids == expected.selected_candidate_ids
    for actual, original in zip(result.predictions, expected.predictions):
        assert actual.scalar_mean == pytest.approx(original.scalar_mean)
        assert actual.scalar_std == pytest.approx(original.scalar_std)
        assert actual.acquisition_score == pytest.approx(original.acquisition_score)
    assert controller.round_input.history_rounds == (-1, -1)
    assert "q0" not in controller.round_input.execution_context["mean_context"]
    assert controller.round_input.execution_context["weight_context"]["occurrences"] == 2
    assert len(controller.recorded) == 1


def test_nonzero_prior_uses_shared_residual_gp_and_raw_target_scale():
    protocol, history, candidates, representations = _fixture()
    selector, controller = _selector(protocol, history, [.8, -.2], [.3, -.4])
    result = selector.select(candidates, representations, round_idx=0)
    residual = RBFGPSurrogate(history, lengthscale=1.5, noise=1e-4, prior_mean=0, prior_std=.25,
        min_training_observations=2, feature_scale_floor=1.0, target_scale_floor=.01,
        feature_version=FEATURE_VERSION, residual_prior_mean=[.8, -.2])
    for index, candidate in enumerate(candidates):
        expected = residual.predict_record(candidate.candidate_id,
            representations[candidate.candidate_id].values, beta=1, query_prior_mean=[.3, -.4][index])
        assert result.predictions[index].scalar_mean == pytest.approx(expected.scalar_mean)
        assert result.predictions[index].scalar_std == pytest.approx(expected.scalar_std)
    assert result.metadata["policy"]["target_scale"] == pytest.approx(selector.gp.y_std)
    assert controller.round_input.research_snapshot["history_mask"] == [True, True]


@pytest.mark.parametrize("action", ["replace", "keep", "disable"])
def test_accepted_policy_actions_export_public_snapshot_once(action, tmp_path, monkeypatch):
    monkeypatch.setenv("LDM_DATA_COLLECTION_DIR", str(tmp_path / "collection"))
    protocol, history, candidates, representations = _fixture()
    collection = AcceptedActions(tmp_path / "run")
    selector = CompiledFactorSelector(protocol, mock=True, collection=collection,
        initial_candidate_ids=(item.candidate_id for item in history))
    selector.fit(history)
    baseline = tuple(selector.gp.predict_record(item.candidate_id,
        representations[item.candidate_id].values, beta=selector.beta) for item in candidates)
    round_input = selector.adapter.build_selection_round(round_idx=0, history=history,
        candidates=candidates, representations=representations, baseline_predictions=baseline,
        requested_batch=1, gp=selector.gp)
    root = tmp_path / "policy_harness"
    directory = root / "rounds/round_000"
    directory.mkdir(parents=True)
    research = {**round_input.research_snapshot, "history_rounds": list(round_input.history_rounds),
                "history_candidate_ids": list(round_input.history_candidate_ids)}
    (directory / "research_snapshot.json").write_text(json.dumps(research))
    context = dict(round_input.execution_context)
    context["weight_context"] = {**context["weight_context"], "prediction_feedback": {
        "count": 1, "measurements": [{"candidate_id": history[0].candidate_id,
            "round_index": -1, "measured_utility": history[0].scalar_score}]}}
    (directory / "input.json").write_text(json.dumps({"execution_context": context}))
    (directory / "contract.json").write_text(json.dumps(selector.adapter.contract.to_dict()))
    np.savez(directory / "arrays.npz", history_features=round_input.history_features,
        history_utilities=round_input.history_utilities, query_features=round_input.query_features)
    names = ("contract.json", "research_snapshot.json", "input.json", "arrays.npz")
    (directory / "manifest.json").write_text(json.dumps({"round_index": 0, "input_sha256": "frozen-input",
        "files": {name: file_sha256(directory / name) for name in names}}))
    prior = None
    if action != "replace":
        prior = {"epoch_id": "epoch_999", "artifact_sha256": ""}
        prior_dir = root / "epochs/epoch_999"
        prior_dir.mkdir(parents=True)
        (prior_dir / "optimization_policy.py").write_text("# prior policy\n")
        prior["artifact_sha256"] = file_sha256(prior_dir / "optimization_policy.py")
    (root / "active_round.json").write_text(json.dumps({"round_index": 0,
        "input_sha256": "frozen-input", "active_policy": prior}))
    if action == "replace":
        epoch_dir = root / "epochs/epoch_000"
        epoch_dir.mkdir(parents=True)
        (epoch_dir / "optimization_policy.py").write_text("# accepted policy\n")
        epoch_id, artifact_hash = "epoch_000", file_sha256(epoch_dir / "optimization_policy.py")
    elif action == "keep":
        epoch_id, artifact_hash = prior["epoch_id"], prior["artifact_sha256"]
    else:
        epoch_id = artifact_hash = None
    policy = CompiledOptimizationPolicy(epoch_id, artifact_hash, np.zeros(2), np.zeros(2),
        "test", 2.0, .25, "artifact", False,
        metadata={"action": action, "status": "accepted", "submission_sha256": "submitted",
                  "harness_turn": {"profile_id": "policy_architect", "turn_id": "policy-turn"}})
    selector.controller = SimpleNamespace(root=root)
    selector.gateway = SimpleNamespace(runtime=SimpleNamespace(run_id="test-run"))
    private = tmp_path / "run/private/validation.json"
    private.parent.mkdir(parents=True)
    private.write_text("HOLDOUT_CANARY")

    selector._collect_policy_action(round_input, policy)
    selector._collect_policy_action(round_input, policy)
    published = next((tmp_path / "collection").rglob("ldm_ir.jsonl"))
    ir_rows = [json.loads(line) for line in published.read_text().splitlines()]
    sft_rows = [json.loads(line) for line in (published.parent / "ldm_sft.jsonl").read_text().splitlines()]
    assert len(ir_rows) == len(sft_rows) == 1
    assert ir_rows[0]["action"]["payload"]["candidates"][0]["action"] == action
    for candidate in (*candidates, *history):
        assert candidate.candidate_id not in json.dumps(ir_rows[0])
        assert candidate.candidate_id not in json.dumps(sft_rows[0])
    assert "history_index" in sft_rows[0]["instruction"]
    assert "HOLDOUT_CANARY" not in json.dumps(ir_rows[0]) + json.dumps(sft_rows[0])
    if action == "replace":
        assert "# accepted policy" in sft_rows[0]["output"]
    else:
        assert "# prior policy" in sft_rows[0]["instruction"]


@pytest.mark.parametrize("capabilities", [("prior_mean@1",), ("ldm_weights@1",),
    ("prior_mean@1", "ldm_weights@1")])
def test_policy_capability_variants_are_frozen(capabilities):
    protocol = T3Protocol(method="ldm_harness_compiled", policy_capabilities=capabilities)
    assert T3PolicyAdapter(protocol).capability_contract().enabled_capabilities == tuple(sorted(capabilities))
    assert protocol.to_dict()["policy_capabilities"] == list(capabilities)


@pytest.mark.parametrize("capabilities", [("prior_mean@1",), ("ldm_weights@1",),
    ("prior_mean@1", "ldm_weights@1")])
@pytest.mark.skipif(not os.environ.get("ALPHABENCH_T3_SIDECAR_IMAGE"), reason="requires policy runner image")
def test_all_capability_variants_execute_in_isolated_policy_runner(tmp_path, capabilities):
    protocol = T3Protocol(method="ldm_harness_compiled", policy_capabilities=capabilities)
    adapter = T3PolicyAdapter(protocol)
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "contract.json").write_text(json.dumps(adapter.contract.to_dict()))
    (input_dir / "input.json").write_text(json.dumps({"execution_context": {
        "mean_context": {"target_location": 0.01, "target_scale": 0.02},
        "weight_context": {"default_alpha": 2.0, "default_eta": .25}}}))
    np.savez(input_dir / "arrays.npz", history_features=np.zeros((2, len(adapter.contract.feature_names))),
             history_utilities=np.array([.01, .02]), query_features=np.ones((2, len(adapter.contract.feature_names))))
    declarations = {name.removesuffix("@1"): 1 for name in capabilities}
    code = f"POLICY_API_VERSION = 1\nCAPABILITIES = {declarations!r}\n"
    if "prior_mean@1" in capabilities:
        code += "import numpy as np\ndef compute_prior_mean(history_features, history_utilities, query_features, context):\n    return np.full(len(query_features), .5)\n"
    if "ldm_weights@1" in capabilities:
        code += "def choose_ldm_weights(context):\n    return {'stage': 'baseline', 'alpha': context['default_alpha'], 'eta': context['default_eta']}\n"
    artifact = tmp_path / "optimization_policy.py"
    artifact.write_text(code)
    executor = DockerPolicyExecutor(image=os.environ["ALPHABENCH_T3_SIDECAR_IMAGE"],
        container_user=f"{os.getuid()}:{os.getgid()}")
    result = executor.execute(artifact, input_dir, tmp_path / "output")
    assert not adapter.validate_task_execution(result, SimpleNamespace(history_features=np.zeros((2, 1)),
        query_features=np.zeros((2, 1))))
    assert list(result.query_prior_mean) == ([.5, .5] if "prior_mean@1" in capabilities else [0., 0.])
    assert (result.alpha, result.eta) == (2.0, .25)


@pytest.mark.skipif(not os.environ.get("ALPHABENCH_T3_SIDECAR_IMAGE"), reason="requires policy runner image")
def test_isolated_policy_runner_cannot_read_host_canary(tmp_path):
    protocol = T3Protocol(method="ldm_harness_compiled", policy_capabilities=("prior_mean@1",))
    contract = T3PolicyAdapter(protocol).contract
    secret = tmp_path / "private.canary"
    secret.write_text("host_only")
    artifact_dir = tmp_path / "policy"
    artifact_dir.mkdir()
    artifact = artifact_dir / "optimization_policy.py"
    artifact.write_text("import numpy as np\nPOLICY_API_VERSION = 1\nCAPABILITIES = {'prior_mean': 1}\n"
        "def compute_prior_mean(history_features, history_utilities, query_features, context):\n"
        "    builtins = __builtins__\n"
        "    reader = builtins['open'] if isinstance(builtins, dict) else getattr(builtins, 'open')\n"
        f"    try:\n        reader({str(secret)!r}).read()\n        return np.ones(len(query_features))\n"
        "    except OSError:\n        return np.zeros(len(query_features))\n")
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "contract.json").write_text(json.dumps(contract.to_dict()))
    (input_dir / "input.json").write_text(json.dumps({"execution_context": {
        "mean_context": {}, "weight_context": {}}}))
    np.savez(input_dir / "arrays.npz", history_features=np.zeros((2, len(contract.feature_names))),
             history_utilities=np.zeros(2), query_features=np.zeros((1, len(contract.feature_names))))
    executor = DockerPolicyExecutor(image=os.environ["ALPHABENCH_T3_SIDECAR_IMAGE"],
        container_user=f"{os.getuid()}:{os.getgid()}")
    result = executor.execute(artifact, input_dir, tmp_path / "output")
    assert list(result.query_prior_mean) == [0.]
    assert secret.read_text() == "host_only"


@pytest.mark.skipif(not os.environ.get("ALPHABENCH_T3_SIDECAR_IMAGE"), reason="requires built Pi sidecar")
def test_policy_sidecar_starts_with_separate_profile_and_skill(tmp_path):
    protocol = T3Protocol(method="ldm_harness_compiled")
    runtime = CampaignRuntime.open(tmp_path / "run", task="alphabench", budget_limits={"model_requests": 2})
    gateway = OracleGateway(protocol, runtime, mock=True)
    client, controller = build_policy_harness(protocol, gateway, T3PolicyAdapter(protocol),
        HarnessMeter(runtime, gateway.host), api_key="test-key",
        sidecar_image=os.environ["ALPHABENCH_T3_SIDECAR_IMAGE"])
    try:
        assert client.config.profiles[0].profile_id == "policy_architect"
        assert client.config.profiles[0].skill_dirs == (Path("/resources/skills/compile-ldm-policy"),)
        assert client.config.mcp_servers[0].tools == ("inspect_policy_contract", "validate_policy_draft")
        assert controller.account is None
        client.start()
    finally:
        client.close()
