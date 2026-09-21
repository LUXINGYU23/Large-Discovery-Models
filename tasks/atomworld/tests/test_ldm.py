"""Actual LDM history, residual-GP, frequency sampling and compiled execution."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from ldm_tts.contracts import ReservoirBuilder
from ldm_tts.harness import HarnessError
from ldm_tts.optimization.records import BOObservation
from ldm_tts.transport import CallableProposalClient
from tasks.atomworld.core import selection
from tasks.atomworld.core.proposals import AtomWorldDomain
from tasks.atomworld.core.selection import GeometryEncoder, AtomWorldLDMSelector, pool_result
from tasks.atomworld.core.workflow import parse_args, run
from tasks.atomworld.tests.test_harness import ResearchClient, fixture
from tasks.atomworld.tests.test_policy import PolicyClient, KnownArtifactExecutor


class MeasuredPolicyClient(PolicyClient):
    expects_measured = True
    source = '''import numpy as np
POLICY_API_VERSION = 1
CAPABILITIES = {"prior_mean": 1, "ldm_weights": 1}
def compute_prior_mean(history_features, history_utilities, query_features, context):
    center = np.mean(history_features[:, 3]) if len(history_features) else 0.0
    return np.clip(0.5 + query_features[:, 3] - center, -2, 2)
def choose_ldm_weights(context):
    return {"stage": "measured_fixture", "alpha": 2.0, "eta": 0.5}
'''


def options(tmp_path, method="ldm_harness_compiled", rounds=3):
    return parse_args(["--mock", "--search-method", method, "--iterations", str(rounds),
                       "--out-dir", str(tmp_path / "run")])


def selection_receipts(root):
    return [json.loads(p.read_text())["selection"] for p in sorted((root / "attempts").glob("*.json"))]


def test_compiled_campaign_fits_measured_gp_and_executes_both_capabilities(tmp_path):
    clients = []

    def factory(*args, **kwargs):
        client = (MeasuredPolicyClient if kwargs.get("policy") else ResearchClient)(*args, **kwargs)
        clients.append(client)
        return client

    args = options(tmp_path)
    result = run(args, harness_client_factory=factory, policy_executor=KnownArtifactExecutor())
    assert result.projected["ldm_applicability"] == "measured_feedback_optimization"
    assert result.projected["extended_final_accuracy"] == 1
    receipts = selection_receipts(result.runtime.run_dir)
    assert receipts[0]["warm_start"] and "compiled_policy" not in receipts[0]
    assert [r["training_observations"] for r in receipts] == [0, 1, 2]
    for receipt in receipts[1:]:
        assert receipt["surrogate"]["fit_status"] == "fitted"
        assert receipt["compiled_policy"]["source"] == "artifact"
        assert receipt["alpha"] == 2 and receipt["eta"] == 0.5
        assert sum(r["selection_probability"] for r in receipt["distribution"]) == pytest.approx(1)
    assert receipts[1]["compiled_policy"]["prediction_mean_abs_change"] > 0
    policy = next(c for c in clients if isinstance(c, MeasuredPolicyClient))
    assert len(policy.turns) == 2
    assert result.runtime.budget.counters["policy_harness_turns"] == 2
    assert result.runtime.budget.counters["expensive_evaluation_attempts"] == 3
    snapshots = sorted((policy.root / "rounds").glob("round_*/arrays.npz"))
    with np.load(snapshots[0]) as arrays:
        assert arrays["history_utilities"].tolist() == [0.0]
    with np.load(snapshots[1]) as arrays:
        assert arrays["history_utilities"].tolist() == [0.0, 1.0]
    research = next(c for c in clients if isinstance(c, ResearchClient))
    assert json.loads(research.calls[1][0].message)["new_public_drafts"][0]["correct"] == 0
    assert all(c.closed for c in clients)
    before = result.runtime.budget.counters.copy()
    args.resume = True
    resumed = run(args, harness_client_factory=lambda *a, **k: pytest.fail("completed run replayed"))
    assert {k: v for k, v in resumed.runtime.budget.counters.items() if v} == {k: v for k, v in before.items() if v}


def test_research_cache_resume_before_selection_has_no_new_provider_draws(tmp_path, monkeypatch):
    clients = []
    def factory(*args, **kwargs):
        client = ResearchClient(*args, **kwargs)
        clients.append(client)
        return client
    args = options(tmp_path, "ldm_harness", rounds=1)
    original = AtomWorldLDMSelector.select
    def interrupt(*a, **k):
        raise HarnessError("fixture interruption after durable pool")
    monkeypatch.setattr(AtomWorldLDMSelector, "select", interrupt)
    with pytest.raises(HarnessError):
        run(args, harness_client_factory=factory)
    budget = json.loads((args.out_dir / "budget.json").read_text())["counters"]
    monkeypatch.setattr(AtomWorldLDMSelector, "select", original)
    args.resume = True
    result = run(args, harness_client_factory=lambda *a, **k: pytest.fail("cached provider draw repeated"))
    assert result.runtime.budget.counters["llm_requests"] == budget["llm_requests"]
    assert result.runtime.budget.counters["harness_turns"] == 2
    assert len(clients) == 1


def test_direct_ldm_has_a_real_frequency_reservoir_and_immutable_weights(tmp_path):
    data = fixture()
    texts = data["mock_outputs"][data["public"][0]["sample_id"]]
    client = CallableProposalClient(lambda request: texts[int(request.metadata["batch_idx"] == 7)])
    args = options(tmp_path, "ldm", rounds=2)
    result = run(args, client=client)
    receipt = selection_receipts(result.runtime.run_dir)[0]
    assert receipt["proposal_reservoir_size"] == 2
    assert sorted(r["q0"] for r in receipt["distribution"]) == pytest.approx([1 / 8, 7 / 8])
    assert result.runtime.budget.counters["proposal_attempts"] == 16
    assert result.runtime.budget.counters["mock_model_requests"] == 16
    args.resume, args.acquisition_alpha = True, 2
    with pytest.raises(ValueError, match="Resume requires"):
        run(args, client=client)


def test_q0_alpha_eta_and_compiled_prior_change_actual_selector_distribution(tmp_path, monkeypatch):
    args = options(tmp_path)
    sample = fixture()["public"][0]
    encoder = GeometryEncoder([sample], mock=True)
    texts = ["<cif>data_a</cif>", "<cif>data_longer_draft</cif>"]
    record = {"sample_id": sample["sample_id"], "action_name": sample["action_name"], "round_idx": 1,
              "drafts": [{"generated_output": text, "rationale": "fixture"} for text in [texts[0]] * 3 + [texts[1]]]}
    candidates = ReservoirBuilder(AtomWorldDomain([sample], mock=True)).build(pool_result(record).proposals).candidates
    features = {c.candidate_id: encoder.encode(c) for c in candidates}
    history = [BOObservation.scalar("measured", 0, features[candidates[0].candidate_id].values,
               feature_version=encoder.version, metadata={"round_idx": 0})]
    choice = {"alpha": 1.0, "eta": 0.0, "prior": 0.0}
    class Controller:
        def resolve(self, inputs):
            assert inputs.history_utilities.tolist() == [0.0]
            return SimpleNamespace(history_prior_mean=np.zeros(1),
                query_prior_mean=np.arange(len(inputs.query_features)) * choice["prior"],
                alpha=choice["alpha"], eta=choice["eta"], metadata={},
                epoch_id="fixture", artifact_digest="digest", source="artifact", degraded=False, stage="fixture")
        def record_predictions(self, *a):
            pass
    monkeypatch.setattr("tasks.atomworld.core.compiled_policy.compiled_controller", lambda *a: Controller())
    def select(name):
        root = tmp_path / name
        selector = AtomWorldLDMSelector(args, [sample], root)
        selector.fit(history)
        result = selector.select(candidates, features, round_idx=1)
        return result.metadata["distribution"]
    default = select("default")
    assert sorted(r["selection_probability"] for r in default) == pytest.approx([0.25, 0.75])
    choice["alpha"] = 2
    weighted = select("weighted")
    assert sorted(r["selection_probability"] for r in weighted) == pytest.approx([0.1, 0.9])
    choice.update(eta=1, prior=2)
    compiled = select("compiled")
    assert [r["selection_probability"] for r in compiled] != pytest.approx([r["selection_probability"] for r in weighted])
    assert any(r["active_mean"] != r["baseline_mean"] for r in compiled)
    assert all(r["active_std"] == r["baseline_std"] for r in compiled)


def test_ldm_gp_and_research_histories_do_not_cross_questions(tmp_path):
    data = fixture()
    samples = [data["public"][0], {**data["public"][0], "sample_id": "second"}]
    args = options(tmp_path, "ldm_harness", rounds=2)
    selector = AtomWorldLDMSelector(args, samples, tmp_path)
    selector.fit([BOObservation.scalar("first_sample", 1, [0] * 12,
        feature_version=selection.FEATURE_VERSION, metadata={"round_idx": 1})])
    record = {"sample_id": "second", "action_name": samples[1]["action_name"], "round_idx": 2,
              "drafts": [{"generated_output": "<cif>data_second</cif>", "rationale": "test"}]}
    candidate = AtomWorldDomain(samples, mock=True).admit(pool_result(record).proposals[0])
    result = selector.select([candidate], {candidate.candidate_id: GeometryEncoder(samples, mock=True).encode(candidate)}, round_idx=2)
    assert result.metadata["training_observations"] == 0


def test_task_contract_advertises_full_compiled_ldm(tmp_path):
    from tasks.atomworld.core.task_spec import describe_ldm_task
    spec = describe_ldm_task(options(tmp_path))
    assert spec.metadata["ldm_applicable"] is True
    assert spec.surrogate.kind == "vector"
    assert spec.reservoir.max_size == 2
    assert spec.metadata["policy_capabilities"] == ["prior_mean@1", "ldm_weights@1"]


def test_resume_after_compiled_selection_replays_same_policy_and_counters(tmp_path, monkeypatch):
    args = options(tmp_path, rounds=2)
    clients = []
    def factory(*a, **k):
        client = (MeasuredPolicyClient if k.get("policy") else ResearchClient)(*a, **k)
        clients.append(client)
        return client
    original = AtomWorldLDMSelector.select
    def interrupted(self, *a, **k):
        result = original(self, *a, **k)
        if k["round_idx"] == 1:
            raise HarnessError("fixture interruption after compiled selection")
        return result
    monkeypatch.setattr(AtomWorldLDMSelector, "select", interrupted)
    with pytest.raises(HarnessError):
        run(args, harness_client_factory=factory, policy_executor=KnownArtifactExecutor())
    before = json.loads((args.out_dir / "budget.json").read_text())["counters"]
    receipt = (args.out_dir / "attempts/000001.json").read_text()
    monkeypatch.setattr(AtomWorldLDMSelector, "select", original)
    args.resume = True
    result = run(args, harness_client_factory=factory, policy_executor=KnownArtifactExecutor())
    assert (args.out_dir / "attempts/000001.json").read_text() == receipt
    for key in ("llm_requests", "harness_turns", "proposal_attempts", "policy_harness_turns"):
        assert result.runtime.budget.counters[key] == before[key]
    assert sum(len(c.turns) for c in clients if isinstance(c, MeasuredPolicyClient)) == 1


def test_policy_keeps_full_measurement_cursor_when_gp_window_is_bounded(tmp_path):
    clients = []
    def factory(*a, **k):
        client = (MeasuredPolicyClient if k.get("policy") else ResearchClient)(*a, **k)
        clients.append(client)
        return client
    args = options(tmp_path, rounds=4)
    args.gp_history_limit = 1
    result = run(args, harness_client_factory=factory, policy_executor=KnownArtifactExecutor())
    receipts = selection_receipts(result.runtime.run_dir)
    assert [r["training_observations"] for r in receipts] == [0, 1, 1, 1]
    assert [r["measured_history_size"] for r in receipts] == [0, 1, 2, 3]
    policy = next(c for c in clients if isinstance(c, MeasuredPolicyClient))
    assert [len(json.loads(t.message)["new_measured_observations"]) for t in policy.turns] == [1, 1, 1]
    path = sorted((policy.root / "rounds").glob("round_*/arrays.npz"))[-1]
    with np.load(path) as arrays:
        assert arrays["history_utilities"].shape == (3,)
        assert np.all(arrays["diagnostic_current_weights"][:, :-1] == 0)


def test_final_submission_pilot_checks_compiled_policy_receipts(tmp_path):
    import csv
    from ldm_tts.pilot_evaluation.submission_reporting import submission_trajectory

    def factory(*a, **k):
        return (MeasuredPolicyClient if k.get("policy") else ResearchClient)(*a, **k)
    args = options(tmp_path, rounds=2)
    result = run(args, harness_client_factory=factory, policy_executor=KnownArtifactExecutor())
    spec = SimpleNamespace(task="atomworld", iterations=2, result_fields={},
        trajectory=SimpleNamespace(step_kind="round", step_column="round", objective_column="correct"))
    with (args.out_dir / "trajectory.csv").open() as handle:
        records = list(csv.DictReader(handle))
    observations = json.loads((args.out_dir / "checkpoint.json").read_text())["state"]["observations"]
    assert submission_trajectory(spec, records, observations, run_dir=args.out_dir,
                                 result=result.projected) == [0.0, 1.0]
    path = args.out_dir / "attempts/000001.json"
    receipt = json.loads(path.read_text())
    receipt["selection"]["compiled_policy"]["eta"] = 99
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="policy receipt: eta"):
        submission_trajectory(spec, records, observations, run_dir=args.out_dir, result=result.projected)
