import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from ldm_tts.contracts import RawProposal
from ldm_tts.engine.runtime import LDMEngine
from tasks.atomworld.core import workflow
from tasks.atomworld.core.evaluator import AtomWorldEvaluator
from tasks.atomworld.core.proposals import AtomWorldDomain, extract_cif
from tasks.atomworld.tests.test_harness import ResearchClient, fixture


@pytest.mark.parametrize("method", ["llm", "ldm", "harness", "ldm_harness"])
@pytest.mark.parametrize("interrupt_at", ["receipt", "checkpoint"])
@pytest.mark.parametrize("interrupt_round", [0, 1])
def test_completed_judge_replays_without_losing_final_round(tmp_path, monkeypatch, method, interrupt_at, interrupt_round):
    tmp_path = tmp_path / "run"
    calls = []
    interrupted = False
    checkpoint = LDMEngine._checkpoint

    def judge(target, text, **kwargs):
        calls.append(text)
        return SimpleNamespace(correct=extract_cif(text) == target, wrong_type=None, rmsd=None, max_dist=None)

    class Evaluator(AtomWorldEvaluator):
        def __init__(self, targets, run_dir, **kwargs):
            super().__init__(targets, run_dir, mock=False, official=judge)

        def evaluate(self, candidate):
            nonlocal interrupted
            result = super().evaluate(candidate)
            if interrupt_at == "receipt" and len(calls) == interrupt_round + 1 and not interrupted:
                interrupted = True
                raise KeyboardInterrupt()
            return result

    def crash(self, state):
        nonlocal interrupted
        if interrupt_at == "checkpoint" and not interrupted and len(state.observations) == interrupt_round + 1:
            interrupted = True
            raise KeyboardInterrupt()
        checkpoint(self, state)

    monkeypatch.setattr(workflow, "AtomWorldEvaluator", Evaluator)
    monkeypatch.setattr(LDMEngine, "_checkpoint", crash)
    clients = {}
    def factory(args, root, sample, **kwargs):
        if root not in clients:
            clients[root] = ResearchClient(args, root, sample, **kwargs)
        return clients[root]
    args = workflow.parse_args(["--mock", "--search-method", method, "--iterations", "2",
                                "--out-dir", str(tmp_path)])
    with pytest.raises(KeyboardInterrupt):
        workflow.run(args, harness_client_factory=factory)
    assert len(calls) == interrupt_round + 1
    assert json.loads((tmp_path / "budget.json").read_text())["counters"]["external_evaluations"] == interrupt_round + 1
    args.resume = True
    resumed = workflow.run(args, harness_client_factory=factory)
    assert len(calls) == 2
    assert len(resumed.engine.state.observations) == 2
    assert resumed.engine.state.next_round == 2
    assert resumed.projected["schedule_completed"]
    assert resumed.projected["extended_final_accuracy"] == 1
    for name in ("external_evaluations", "expensive_evaluation_attempts", "successful_evaluations", "benchmark_jobs"):
        assert resumed.runtime.budget.counters[name] == 2
    assert json.loads((tmp_path / "status.json").read_text())["status"] == "completed"


def candidate():
    data = fixture()
    sample = data["public"][0]
    return AtomWorldDomain([sample], mock=True).admit(RawProposal(
        {"sample_id": sample["sample_id"], "action_name": sample["action_name"],
         "generated_output": data["mock_outputs"][sample["sample_id"]][0]},
        "test", metadata={"round_idx": 0},
    )), {sample["sample_id"]: data["private"][0]["target_cif"]}


def test_receipt_binds_target_round_submission_and_result(tmp_path):
    c, targets = candidate()
    evaluator = AtomWorldEvaluator(targets, tmp_path, mock=True)
    first = evaluator.evaluate(c)
    assert evaluator.evaluate(c) == first
    later = replace(c, metadata={"round_idx": 1})
    assert evaluator.evaluation_attempt_usage_key(c) != evaluator.evaluation_attempt_usage_key(later)
    assert evaluator.evaluate(later).artifacts != first.artifacts
    for changed in (replace(c, canonical_key="other"), replace(c, payload={**c.payload, "generated_output": "other"})):
        with pytest.raises(ValueError, match="identity mismatch"):
            evaluator.evaluation_attempt_usage_key(changed)
    with pytest.raises(ValueError, match="identity mismatch"):
        AtomWorldEvaluator({k: "changed" for k in targets}, tmp_path, mock=True).evaluate(c)
    path = tmp_path / first.artifacts["evaluation"]
    receipt = json.loads(path.read_text())
    receipt["result"]["metrics"]["correct"] = 1
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="result mismatch"):
        evaluator.evaluate(c)


def test_interrupted_judge_is_not_silently_retried(tmp_path):
    c, targets = candidate()
    calls = []
    def judge(*args, **kwargs):
        calls.append(True)
        raise RuntimeError("judge crashed")
    evaluator = AtomWorldEvaluator(targets, tmp_path, official=judge)
    with pytest.raises(RuntimeError, match="judge crashed"):
        evaluator.evaluate(c)
    with pytest.raises(RuntimeError, match="ambiguous charged"):
        evaluator.evaluate(c)
    assert len(calls) == 1


def test_incomplete_schedule_cli_returns_nonzero(tmp_path, monkeypatch):
    from ldm_tts.engine.expansion import ExpansionResult
    tmp_path = tmp_path / "run"
    monkeypatch.setattr(workflow.BlindRefinementExpander, "expand", lambda *args: ExpansionResult(proposals=(RawProposal({}, "test"),)))
    assert workflow.main(["--mock", "--iterations", "2", "--out-dir", str(tmp_path)]) == 1
    assert not json.loads((tmp_path / "result.json").read_text())["schedule_completed"]
    assert json.loads((tmp_path / "status.json").read_text())["status"] == "stopped"
