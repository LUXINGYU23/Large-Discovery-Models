from concurrent.futures import ThreadPoolExecutor
import io
import json
from threading import Barrier, get_ident

import pytest

from ldm_tts.contracts import RawProposal
from ldm_tts.contracts.evaluation import EvaluationPaused
from ldm_tts.engine.run_store import BudgetExceededError, CampaignRuntime
from ldm_tts.harness import HarnessProviderAuthorizationRequest
from ldm_tts.transport import CallableProposalClient
from tasks.alphabench.core.candidate import FactorDomain
from tasks.alphabench.core.gateway import OracleGateway, mock_oracle
from tasks.alphabench.core.generator import Generator
from tasks.alphabench.core.host import HostDispatcher
from tasks.alphabench.core.harness import HarnessMeter
from tasks.alphabench.core.protocol import T3Protocol, digest


def parallel(function, count=2):
    with ThreadPoolExecutor(max_workers=count) as pool:
        return list(pool.map(function, range(count)))


def metered_gateway(tmp_path, monkeypatch, **limits):
    runtime = CampaignRuntime.open(tmp_path, task="alphabench", budget_limits=limits)
    owner, consume, consume_groups = get_ident(), runtime.consume_many, runtime.consume_groups
    def checked(*args, **kwargs):
        assert get_ident() == owner, "only the Host thread may mutate the ledger"
        return consume(*args, **kwargs)
    monkeypatch.setattr(runtime, "consume_many", checked)
    def checked_groups(*args, **kwargs):
        assert get_ident() == owner, "only the Host thread may mutate the ledger"
        return consume_groups(*args, **kwargs)
    monkeypatch.setattr(runtime, "consume_groups", checked_groups)
    return OracleGateway(T3Protocol(), runtime, mock=True)


def test_preflight_freezes_oracle_config_across_same_run(tmp_path, monkeypatch):
    gateway = metered_gateway(tmp_path, monkeypatch)
    gateway.mock = False
    gateway.endpoint = "http://oracle.invalid"
    health = {key: getattr(gateway.protocol, key) for key in
              ("backend", "market", "data_digest", "environment_digest")}
    health["capabilities"] = ["durable_requests", "worker_permits", "daily_ic", "factor_scores",
                              "portfolio", "dynamic_check"]
    digest = "a" * 64

    def response(url, timeout):
        assert url == "http://oracle.invalid/t3/health"
        return io.BytesIO(json.dumps(health | {"config_digest": digest}).encode())

    monkeypatch.setattr("tasks.alphabench.core.gateway.urllib.request.urlopen", response)
    assert gateway.preflight()["config_digest"] == digest
    digest = "b" * 64
    with pytest.raises(ValueError, match="config changed across this run"):
        gateway.preflight()
    assert gateway.receipts.load("service_identity")["config_digest"] == "a" * 64


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


def test_timed_out_oracle_post_waits_for_same_durable_request(tmp_path, monkeypatch):
    gateway = metered_gateway(tmp_path, monkeypatch, dynamic_checks=1, oracle_job_slots=2)
    gateway.mock, gateway.endpoint = False, "http://oracle.invalid"
    candidate = FactorDomain().admit(RawProposal({"expression": "$close"}, "test"))
    posts, polls = [], []
    def disconnected(request, *, timeout):
        posts.append(request)
        raise TimeoutError("response lost")
    def reconcile(request):
        polls.append(request["request_id"])
        return mock_oracle(request) if len(polls) == 2 else None
    monkeypatch.setattr("tasks.alphabench.core.gateway.urllib.request.urlopen", disconnected)
    monkeypatch.setattr("tasks.alphabench.core.gateway.time.sleep", lambda _: None)
    monkeypatch.setattr(gateway, "_reconcile", reconcile)

    result = gateway.evaluate(candidate, phase="check", position=0)
    assert result["success"] and len(posts) == 1 and len(polls) == 2
    assert gateway.runtime.budget.counters["dynamic_checks"] == 1
    assert gateway.receipts.load(gateway.identity("check", 0, candidate))["state"] == "completed"


def test_resume_reposts_missing_oracle_request_with_same_identity_and_budget(tmp_path, monkeypatch):
    gateway = metered_gateway(tmp_path, monkeypatch, dynamic_checks=1, oracle_job_slots=2)
    gateway.mock, gateway.endpoint = False, "http://oracle.invalid"
    candidate = FactorDomain().admit(RawProposal({"expression": "$close"}, "test"))
    posts, server = [], {}

    def send(request):
        posts.append(request)
        if len(posts) == 1:
            raise TimeoutError("request never reached the oracle")
        response = mock_oracle(request)
        server[request["request_id"]] = {"state": "completed", "request_digest": digest(request),
                                         "response": response, "response_digest": digest(response)}
        return response

    monkeypatch.setattr(gateway, "_send", send)
    monkeypatch.setattr(gateway, "_service_receipt", lambda request: server.get(request["request_id"]))
    with pytest.raises(EvaluationPaused, match="reconciliation"):
        gateway.evaluate(candidate, phase="check", position=0)
    assert gateway.evaluate(candidate, phase="check", position=0)["success"]
    assert posts[0] == posts[1]
    assert gateway.receipts.load(gateway.identity("check", 0, candidate))["state"] == "completed"
    assert gateway.runtime.budget.counters == {
        "dynamic_checks": 1, "oracle_job_slots": 2, "benchmark_jobs": 1}


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


def test_harness_provider_authorization_is_single_writer_and_finite(tmp_path, monkeypatch):
    gateway = metered_gateway(tmp_path, monkeypatch, model_requests=1, proposal_attempts=2, harness_turns=2)
    meter = HarnessMeter(gateway.runtime, gateway.host)
    campaign = gateway.runtime.run_id
    requests = [HarnessProviderAuthorizationRequest(campaign, profile, "round-0-" + profile,
                "round-0-" + profile + "-provider-1", "a" * 64) for profile in ("one", "two")]

    def stage():
        meter.reserve_turns([request.turn_id for request in requests])
        return parallel(lambda index: meter.authorize(requests[index]))

    with pytest.raises(BudgetExceededError):
        gateway.host.run(stage)
    assert gateway.runtime.budget.counters == {"model_requests": 1, "proposal_attempts": 2, "harness_turns": 2}
    records = list(meter.receipts.root.glob("*.json"))
    assert len(records) == 1
    authorized = json.loads(records[0].read_text(encoding="utf-8"))
    assert authorized["request_digest"] == "a" * 64
    summary = meter.reconcile(authorized["profile_id"], authorized["turn_id"], {"providerCalls": 1})
    assert summary == {"host_authorizations": 1, "sidecar_provider_calls": 1}
    with pytest.raises(EvaluationPaused, match="exceeds Host authorizations"):
        meter.reconcile(authorized["profile_id"], authorized["turn_id"], {"providerCalls": 2})


def test_harness_strict_barrier_reserves_all_turns_or_none(tmp_path, monkeypatch):
    gateway = metered_gateway(tmp_path, monkeypatch, proposal_attempts=2, harness_turns=1)
    meter = HarnessMeter(gateway.runtime, gateway.host)
    with pytest.raises(BudgetExceededError):
        gateway.host.run(lambda: meter.reserve_turns(["round-0-one", "round-0-two"]))
    assert gateway.runtime.budget.counters == {}


def test_harness_provider_id_cannot_be_reused_or_reauthorized_after_ledger_only_crash(tmp_path, monkeypatch):
    gateway = metered_gateway(tmp_path, monkeypatch, model_requests=2)
    meter = HarnessMeter(gateway.runtime, gateway.host)
    request = HarnessProviderAuthorizationRequest(gateway.runtime.run_id, "one", "round-0-one",
                                                   "round-0-one-provider-1", "a" * 64)
    assert gateway.host.run(lambda: meter.authorize(request)) is True
    with pytest.raises(EvaluationPaused, match="already authorized"):
        gateway.host.run(lambda: meter.authorize(request))
    changed = HarnessProviderAuthorizationRequest(request.campaign_id, request.profile_id, request.turn_id,
                                                  request.provider_request_id, "b" * 64)
    with pytest.raises(ValueError, match="different digest"):
        gateway.host.run(lambda: meter.authorize(changed))
    meter.receipts.path({"campaign_id": request.campaign_id, "profile_id": request.profile_id,
                         "turn_id": request.turn_id, "provider_request_id": request.provider_request_id}).unlink()
    with pytest.raises(EvaluationPaused, match="requires reconciliation"):
        gateway.host.run(lambda: meter.authorize(request))
    assert gateway.runtime.budget.counters["model_requests"] == 1
