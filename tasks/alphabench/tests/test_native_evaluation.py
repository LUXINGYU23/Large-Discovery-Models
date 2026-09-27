import json
import subprocess
import sys
from threading import Barrier

import pytest

from ldm_tts.contracts import EvaluationResult, Observation, RawProposal
from ldm_tts.contracts.evaluation import EvaluationPaused
from ldm_tts.engine.run_store import CampaignRuntime
from ldm_tts.transport import CallableProposalClient
from tasks.alphabench.core.candidate import FactorDomain
from tasks.alphabench.core.collection import AcceptedActions
from tasks.alphabench.core.gateway import OracleGateway, mock_oracle
from tasks.alphabench.core.generator import Generator
from tasks.alphabench.core.native_evaluation import NativeEvaluator, NativeSearchBudgetExhausted
from tasks.alphabench.core.native_generator import NativeGenerator
from tasks.alphabench.core.native_runtime import NativeRuntime, NativeStop
from tasks.alphabench.core.native_source import load_algorithms
from tasks.alphabench.core.protocol import T3Protocol
from tasks.alphabench.tests.test_native import SEEDS, source


FACTORS = [{"name": "open", "expression": "$open"}, {"name": "close", "expression": "$close"}]


def build(root, *, evaluations=4, workers=2, limits=None, method="ea"):
    protocol = T3Protocol(method="alphabench_" + method, evaluations=evaluations)
    runtime = CampaignRuntime.open(root, task="alphabench", resume=(root / "campaign.json").exists(),
        budget_limits={**protocol.budgets, "expensive_evaluation_attempts": evaluations, **(limits or {})})
    gateway = OracleGateway(protocol, runtime, mock=True)
    scheduler = NativeRuntime(runtime, gateway.host)
    initial = []
    for seed in SEEDS:
        candidate = FactorDomain().admit(RawProposal(seed, "initialization"))
        initial.append(Observation(candidate, EvaluationResult(candidate.candidate_id, "succeeded",
            metrics={"mock_" + key: value for key, value in seed["metrics"].items()})))
    evaluator = NativeEvaluator(scheduler, gateway, initial, workers=workers)
    return evaluator, scheduler


def batch(evaluator, scheduler, factors=FACTORS):
    return scheduler.run(lambda: evaluator.callbacks()["batch_evaluate_fn"](factors))


def test_non_search_budget_rejects_batch_without_partial_dispatch(tmp_path, monkeypatch):
    evaluator, scheduler = build(tmp_path, limits={"oracle_job_slots": 3})
    monkeypatch.setattr(evaluator.gateway, "_send", lambda _: pytest.fail("rejected batch was dispatched"))
    before = (tmp_path / "budget.json").read_bytes()
    with pytest.raises(EvaluationPaused):
        batch(evaluator, scheduler)
    assert (tmp_path / "budget.json").read_bytes() == before
    assert evaluator.observations() == []
    assert evaluator.settle()["reserved_but_unstarted"] == []


def test_final_batch_measures_only_remaining_search_allowance(tmp_path, monkeypatch):
    evaluator, scheduler = build(tmp_path, evaluations=1)
    calls = []
    def send(request):
        calls.append(request["phase"])
        return mock_oracle(request)
    monkeypatch.setattr(evaluator.gateway, "_send", send)
    with pytest.raises(NativeSearchBudgetExhausted):
        batch(evaluator, scheduler)
    assert evaluator.settle()["reserved_but_unstarted"] == []
    assert [item.candidate.payload["name"] for item in evaluator.observations()] == ["open"]
    assert evaluator.runtime.budget.counters["expensive_evaluation_attempts"] == 1
    assert calls == ["search", "validation"]


def test_parallel_search_keeps_input_order_and_commits_private_validation_before_return(tmp_path, monkeypatch):
    evaluator, scheduler = build(tmp_path)
    gateway = evaluator.gateway
    overlap, calls = Barrier(2), []
    def send(request):
        calls.append((request["phase"], request["expression"]))
        if request["phase"] == "search":
            overlap.wait(timeout=3)
        result = mock_oracle(request)
        if request["phase"] == "validation":
            result["metrics"] = {"rank_ic": 999}
        return result
    monkeypatch.setattr(gateway, "_send", send)
    result = batch(evaluator, scheduler)
    assert [row["name"] for row in result] == ["open", "close"]
    assert all(row["metrics"]["rank_ic"] != 999 for row in result)
    observations = evaluator.observations()
    assert len(observations) == 2 and len(calls) == 4
    for observation in observations:
        candidate = observation.candidate
        assert gateway.receipts.load(gateway.identity("validation", candidate.metadata["attempt_position"], candidate))["state"] == "completed"
    counters = gateway.runtime.budget.counters
    assert counters["expensive_evaluation_attempts"] == 2
    assert counters["oracle_job_slots"] == 8 and counters["benchmark_jobs"] == 4
    before = (tmp_path / "budget.json").read_bytes()
    replay, restored = build(tmp_path)
    monkeypatch.setattr(replay.gateway, "_send", lambda _: pytest.fail("completed callback was re-dispatched"))
    assert batch(replay, restored) == result
    assert (tmp_path / "budget.json").read_bytes() == before
    assert [item.to_dict() for item in replay.observations()] == [item.to_dict() for item in observations]


def test_seed_role_reuses_initial_information_but_reproposals_pay_new_attempts(tmp_path, monkeypatch):
    evaluator, scheduler = build(tmp_path)
    calls = []
    def send(request):
        calls.append(request["phase"])
        return mock_oracle(request)
    monkeypatch.setattr(evaluator.gateway, "_send", send)
    def stage():
        callbacks = evaluator.callbacks()
        with scheduler.seed():
            seed = callbacks["batch_evaluate_fn"](FACTORS[:1])
        first = callbacks["batch_evaluate_fn"](FACTORS[:1])
        second = callbacks["batch_evaluate_fn"](FACTORS[:1])
        return seed, first, second
    seed, first, second = scheduler.run(stage)
    assert seed[0]["cached"] is True and first == second
    assert len(calls) == 4 and evaluator.runtime.budget.counters["expensive_evaluation_attempts"] == 2
    assert len(evaluator.observations()) == 2
    assert len({tuple(item.candidate.metadata["attempt_position"]) for item in evaluator.observations()}) == 2


def test_concurrent_batches_cannot_exceed_remaining_E(tmp_path):
    evaluator, scheduler = build(tmp_path, evaluations=3)
    overlap = Barrier(2)
    def stage():
        def work():
            overlap.wait(timeout=3)
            return evaluator.callbacks()["batch_evaluate_fn"](FACTORS)
        with scheduler.executor(max_workers=2) as pool:
            futures = [pool.submit(work) for _ in range(2)]
            return [future.result() for future in futures]
    with pytest.raises(NativeSearchBudgetExhausted):
        scheduler.run(stage)
    evaluator.settle()
    charged = evaluator.runtime.budget.counters["expensive_evaluation_attempts"]
    assert 2 <= charged <= 3
    assert len(evaluator.observations()) == charged
    assert evaluator.runtime.budget.remaining("expensive_evaluation_attempts") == 3 - charged


@pytest.mark.parametrize("unknown", [False, True])
def test_stop_cancels_unstarted_requests_but_drains_and_reconciles_begun_work(tmp_path, monkeypatch, unknown):
    evaluator, scheduler = build(tmp_path, workers=1)
    calls = []
    def send(request):
        calls.append(request["phase"])
        if request["phase"] == "search":
            scheduler.stop(NativeSearchBudgetExhausted("peer exhausted E"), notify=False)
            if unknown:
                raise OSError("lost response")
        return mock_oracle(request)
    monkeypatch.setattr(evaluator.gateway, "_send", send)
    with pytest.raises(NativeSearchBudgetExhausted):
        batch(evaluator, scheduler)
    if unknown:
        with pytest.raises(EvaluationPaused, match="reconciliation"):
            evaluator.settle()
        assert calls == ["search"] and not evaluator.observations()
    else:
        settled = evaluator.settle()
        assert settled["reserved_but_unstarted"] == []
        assert len(evaluator.observations()) == 2 and calls == ["search", "validation", "search", "validation"]
    assert evaluator.runtime.budget.counters["expensive_evaluation_attempts"] == 2


def test_provider_authorization_blocks_repairs_after_stop_but_allows_read_only_replay(tmp_path):
    evaluator, scheduler = build(tmp_path)
    calls = []
    def model(request):
        calls.append(request)
        return '{"generated": []}'
    generator = Generator(evaluator.protocol, evaluator.runtime, CallableProposalClient(model), evaluator.gateway, None)
    messages = [{"role": "user", "content": "test"}]
    original = generator.request("first", messages)
    scheduler.stop(NativeSearchBudgetExhausted("stopped"))
    assert generator.request("first", messages).text == original.text
    with pytest.raises(NativeStop):
        generator.request("repair", messages)
    assert len(calls) == 1 and evaluator.runtime.budget.counters["model_requests"] == 1


def test_failed_search_is_charged_without_validation_or_invented_scores(tmp_path, monkeypatch):
    evaluator, scheduler = build(tmp_path)
    calls = []
    def send(request):
        calls.append(request["phase"])
        return {**mock_oracle(request), "success": False, "metrics": {}, "error": "all missing"}
    monkeypatch.setattr(evaluator.gateway, "_send", send)
    result = batch(evaluator, scheduler, FACTORS[:1])
    assert result[0]["metrics"] == {} and result[0]["success"] is False
    assert calls == ["search"]
    assert evaluator.runtime.budget.counters["expensive_evaluation_attempts"] == 1
    assert evaluator.observations()[0].evaluation.status == "invalid"


def test_validation_disconnect_preserves_search_cost_and_reconciles_before_publishing(tmp_path, monkeypatch):
    evaluator, scheduler = build(tmp_path, workers=1)
    calls, lost = [], {}
    def send(request):
        calls.append(request["phase"])
        response = mock_oracle(request)
        if request["phase"] == "validation":
            lost.update(response)
            raise OSError("lost validation response")
        return response
    monkeypatch.setattr(evaluator.gateway, "_send", send)
    with pytest.raises(EvaluationPaused):
        batch(evaluator, scheduler, FACTORS[:1])
    assert evaluator.observations() == []
    assert evaluator.runtime.budget.counters["benchmark_jobs"] == 1
    restored, replay = build(tmp_path, workers=1)
    monkeypatch.setattr(restored.gateway, "_send", lambda _: pytest.fail("begun request was retried"))
    monkeypatch.setattr(restored.gateway, "_reconcile", lambda request: lost)
    assert batch(restored, replay, FACTORS[:1])[0]["success"] is True
    assert len(restored.observations()) == 1
    assert restored.runtime.budget.counters["benchmark_jobs"] == 2
    assert calls == ["search", "validation"]


@pytest.mark.parametrize("jobs", [[{}], [{"job_id": 7}], [None]])
def test_malformed_job_receipt_pauses_before_upstream_can_treat_it_as_formula_failure(tmp_path, monkeypatch, jobs):
    evaluator, scheduler = build(tmp_path)
    monkeypatch.setattr(evaluator.gateway, "_send", lambda request: {**mock_oracle(request), "jobs": jobs})
    with pytest.raises(EvaluationPaused, match="accounting"):
        batch(evaluator, scheduler, FACTORS[:1])
    assert not evaluator.observations()


@pytest.mark.parametrize("method", ["cot", "tot", "ea"])
def test_all_actual_native_algorithms_use_metered_generator_and_private_evaluator(tmp_path, source, monkeypatch, method):
    monkeypatch.setattr("tasks.alphabench.core.native_generator.time.sleep", lambda _: None)
    evaluator, scheduler = build(tmp_path, evaluations=50, method=method)
    gateway, protocol = evaluator.gateway, evaluator.protocol
    generated = iter(range(100))
    def model(request):
        index = next(generated)
        return json.dumps({"generated": [{"name": f"factor_{index}_{offset}",
            "expression": f"Ref($close, {index * 2 + offset + 1})"} for offset in range(2)]})
    client = CallableProposalClient(model)
    generator = NativeGenerator(protocol, evaluator.runtime, client, gateway, AcceptedActions(tmp_path), source)
    def stage(module):
        algo = module.create_algo(method, {"rounds": 2, "workers": 2, "N": 2, "top_k": 2,
            "model": protocol.model, "temperature": protocol.temperature},
            search_fn=scheduler.callback("search_fn", generator), **evaluator.callbacks())
        return algo.run(SEEDS, str(tmp_path / "native"))
    with load_algorithms(source, tmp_path / "private", scheduler, matched=True) as module:
        result = scheduler.run(lambda: stage(module))
    assert result["history"] and evaluator.observations()
    assert evaluator.runtime.budget.counters["validation_evaluations"] == len(evaluator.observations())
    assert evaluator.runtime.budget.counters["expensive_evaluation_attempts"] == len(evaluator.observations())
    if method != "ea":
        assert any(event["event_type"] == "native_seed_reused" for event in evaluator.runtime.events())
    assert evaluator.settle()["reserved_but_unstarted"] == []


@pytest.mark.parametrize("point", ["after_reservation", "after_observation"])
def test_process_death_resumes_paid_batch_without_repaying_or_redispatching(tmp_path, point):
    script = r'''
import json, os, sys
from pathlib import Path
from tasks.alphabench.tests.test_native_evaluation import build, batch, FACTORS
from tasks.alphabench.core.gateway import mock_oracle
root, point = Path(sys.argv[1]), sys.argv[2]
evaluator, scheduler = build(root, workers=1)
record = evaluator.runtime.record
def instrument(kind, *args, **kwargs):
    marker = root / "killed"
    if point == "after_reservation" and kind == "native_batch_reserved" and not marker.exists():
        marker.write_text(point)
        os._exit(91)
    result = record(kind, *args, **kwargs)
    if point == "after_observation" and kind == "native_observation" and not marker.exists():
        marker.write_text(point)
        os._exit(91)
    return result
evaluator.runtime.record = instrument
def send(request):
    with (root / "physical.jsonl").open("a") as stream:
        stream.write(json.dumps(request) + "\n")
    return mock_oracle(request)
evaluator.gateway._send = send
batch(evaluator, scheduler, FACTORS[:1])
assert len(evaluator.observations()) == 1
'''
    command = [sys.executable, "-c", script, str(tmp_path), point]
    first = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert first.returncode == 91, first.stderr
    resumed = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert resumed.returncode == 0, resumed.stderr
    assert len((tmp_path / "physical.jsonl").read_text().splitlines()) == 2
    counters = json.loads((tmp_path / "budget.json").read_text())["counters"]
    assert counters["expensive_evaluation_attempts"] == 1 and counters["benchmark_jobs"] == 2
