from concurrent.futures import ThreadPoolExecutor
import json
from threading import Barrier, get_ident

import pytest

from ldm_tts.contracts import RawProposal
from ldm_tts.contracts.evaluation import EvaluationPaused
from ldm_tts.engine.run_store import BudgetExceededError, CampaignRuntime
from ldm_tts.transport import CallableProposalClient
from tasks.alphabench.core.candidate import FactorDomain
from tasks.alphabench.core.gateway import OracleGateway, mock_oracle
from tasks.alphabench.core.generator import Generator
from tasks.alphabench.core.host import HostDispatcher
from tasks.alphabench.core.protocol import T3Protocol


def parallel(function, count=2):
    with ThreadPoolExecutor(max_workers=count) as pool:
        return list(pool.map(function, range(count)))


def metered_gateway(tmp_path, monkeypatch, **limits):
    runtime = CampaignRuntime.open(tmp_path, task="alphabench", budget_limits=limits)
    owner, consume = get_ident(), runtime.consume_many
    def checked(*args, **kwargs):
        assert get_ident() == owner, "only the Host thread may mutate the ledger"
        return consume(*args, **kwargs)
    monkeypatch.setattr(runtime, "consume_many", checked)
    return OracleGateway(T3Protocol(), runtime, mock=True)


def test_competing_requests_reserve_before_dispatch_without_exceeding_one_slot(tmp_path, monkeypatch):
    gateway = metered_gateway(tmp_path, monkeypatch, dynamic_checks=1, oracle_job_slots=2)
    candidate = FactorDomain().admit(RawProposal({"expression": "Mean($close,5)"}, "test"))
    calls = []
    def send(request):
        assert get_ident() != gateway.host.owner
        record = json.loads((gateway.receipts.root / (request["request_id"] + ".json")).read_text())
        assert record["state"] == "dispatch_intent"
        calls.append(request["request_id"])
        return mock_oracle(request)
    monkeypatch.setattr(gateway, "_send", send)
    barrier = Barrier(2)
    def evaluate(index):
        barrier.wait(timeout=3)
        try:
            return gateway.evaluate(candidate, phase="check", position=index)["success"]
        except BudgetExceededError:
            return "budget"
    assert sorted(gateway.host.run(lambda: parallel(evaluate)), key=str) == [True, "budget"]
    assert len(calls) == 1
    budget = json.loads((tmp_path / "budget.json").read_text())["counters"]
    assert budget["dynamic_checks"] == 1 and budget["oracle_job_slots"] == 2 and budget["benchmark_jobs"] == 1


def test_parallel_oracles_overlap_io_and_completed_receipts_replay_without_cost(tmp_path, monkeypatch):
    gateway = metered_gateway(tmp_path, monkeypatch, dynamic_checks=2, oracle_job_slots=4)
    candidate = FactorDomain().admit(RawProposal({"expression": "$close"}, "test"))
    overlap, calls = Barrier(2), []
    def send(request):
        overlap.wait(timeout=3)  # Serializing the entire request would deadlock here.
        calls.append(request["request_id"])
        return mock_oracle(request)
    monkeypatch.setattr(gateway, "_send", send)
    def stage():
        return parallel(lambda index: gateway.evaluate(candidate, phase="check", position=index))
    first = gateway.host.run(stage)
    before = {str(file): file.read_bytes() for file in tmp_path.rglob("*.json") if file.name != "status.json"}
    assert gateway.host.run(stage) == first
    assert before == {str(file): file.read_bytes() for file in tmp_path.rglob("*.json") if file.name != "status.json"}
    assert len(calls) == 2


def test_model_preparation_and_settlement_use_same_host_writer(tmp_path, monkeypatch):
    gateway = metered_gateway(tmp_path, monkeypatch, model_requests=1, proposal_attempts=1)
    calls = []
    def model(request):
        assert get_ident() != gateway.host.owner
        calls.append(request)
        return '{"candidates": []}'
    generator = Generator(gateway.protocol, gateway.runtime, CallableProposalClient(model), gateway, None)
    def request(index):
        try:
            return generator.request(index, [{"role": "user", "content": "fixture"}]).text
        except BudgetExceededError:
            return "budget"
    answers = gateway.host.run(lambda: parallel(request))
    assert answers.count("budget") == 1 and len(calls) == 1
    assert gateway.runtime.budget.counters["proposal_attempts"] == 1


def test_unknown_outcome_is_preserved_across_worker_handoffs(tmp_path, monkeypatch):
    gateway = metered_gateway(tmp_path, monkeypatch, dynamic_checks=1)
    candidate = FactorDomain().admit(RawProposal({"expression": "$close"}, "test"))
    calls = []
    def disconnect(request):
        calls.append(request)
        raise OSError("response lost")
    monkeypatch.setattr(gateway, "_send", disconnect)
    for _ in range(2):
        with pytest.raises(EvaluationPaused):
            gateway.host.run(lambda: gateway.evaluate(candidate, phase="check", position=0))
    assert len(calls) == 1 and gateway.runtime.budget.counters["dynamic_checks"] == 1
    assert gateway.receipts.load(gateway.identity("check", 0, candidate))["state"] == "dispatch_intent"


@pytest.mark.parametrize("changes", [
    {"check_kind": "lint"}, {"nan_ratio": None}, {"non_finite_ratio": 1.1},
    {"metrics": {"ic": .5}}, {"elapsed_seconds": -1},
])
def test_invalid_check_evidence_pauses_without_losing_or_repeating_paid_work(tmp_path, monkeypatch, changes):
    gateway = metered_gateway(tmp_path, monkeypatch, dynamic_checks=1, oracle_job_slots=2)
    candidate = FactorDomain().admit(RawProposal({"expression": "$close"}, "test"))
    calls = []
    def send(request):
        calls.append(request)
        return mock_oracle(request) | changes
    monkeypatch.setattr(gateway, "_send", send)
    for _ in range(2):
        with pytest.raises(EvaluationPaused, match="check response"):
            gateway.evaluate(candidate, phase="check", position=0)
    assert len(calls) == 1 and gateway.runtime.budget.counters["benchmark_jobs"] == 1
    assert gateway.runtime.budget.counters["dynamic_checks"] == 1


@pytest.mark.parametrize("phase", ["check", "search"])
def test_missing_calendar_pauses_instead_of_becoming_a_bad_factor_and_preserves_cost(tmp_path, monkeypatch, phase):
    gateway = metered_gateway(tmp_path, monkeypatch, oracle_job_slots=2)
    candidate = FactorDomain().admit(RawProposal({"expression": "$close"}, "test"))
    calls = []
    def send(request):
        calls.append(request)
        return mock_oracle(request) | {"success": False, "pause_status": "paused_data_coverage",
            "error": "calendar lacks future sessions", "metrics": {}, "daily": [], "scores": []}
    monkeypatch.setattr(gateway, "_send", send)
    for _ in range(2):
        with pytest.raises(EvaluationPaused, match="calendar") as failure:
            gateway.evaluate(candidate, phase=phase, position=0)
        assert failure.value.status == "paused_data_coverage"
    assert len(calls) == 1 and gateway.runtime.budget.counters["benchmark_jobs"] == 1


def test_full_evaluation_requires_portfolio_without_repeating_a_paid_job(tmp_path, monkeypatch):
    gateway = metered_gateway(tmp_path, monkeypatch, oracle_job_slots=8, test_evaluations=1)
    candidate = FactorDomain().admit(RawProposal({"expression": "$close"}, "test"))
    calls = []
    def send(request):
        calls.append(request)
        return mock_oracle(request) | {"portfolio": None}
    monkeypatch.setattr(gateway, "_send", send)
    for _ in range(2):
        with pytest.raises(EvaluationPaused, match="required portfolio"):
            gateway.evaluate(candidate, phase="test", position=0, fast=False)
    assert len(calls) == 1
    assert gateway.runtime.budget.counters["test_evaluations"] == 1
    assert gateway.runtime.budget.counters["benchmark_jobs"] == 1


def test_interrupted_host_releases_waiting_callback_before_stage_shutdown(monkeypatch):
    host = HostDispatcher()
    original_get = host.messages.get
    fired = False
    def interrupted_get(*args, **kwargs):
        nonlocal fired
        if not fired:
            fired = True
            raise KeyboardInterrupt()
        return original_get(*args, **kwargs)
    monkeypatch.setattr(host.messages, "get", interrupted_get)
    calls = []
    with pytest.raises(KeyboardInterrupt):
        host.run(lambda: host.call(lambda: calls.append("must not execute")))
    assert not calls and not host.active
    assert host.run(lambda: host.call(lambda: 3)) == 3
    with ThreadPoolExecutor(max_workers=1) as pool:
        with pytest.raises(RuntimeError, match="active Host"):
            pool.submit(host.call, lambda: calls.append("late callback")).result(timeout=3)
    assert not calls


def test_bounded_queue_backpressure_keeps_all_reservations_on_host(tmp_path, monkeypatch):
    gateway = metered_gateway(tmp_path, monkeypatch, model_requests=80)
    def callback(index):
        return gateway.host.call(lambda: gateway.runtime.consume_many({"model_requests": 1}, usage_key=str(index)))
    result = gateway.host.run(lambda: parallel(callback, 80))
    assert len(result) == 80 and gateway.runtime.budget.counters["model_requests"] == 80
