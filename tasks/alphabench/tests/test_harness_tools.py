import json
import socket
import tempfile
from pathlib import Path

import pytest

from ldm_tts.contracts import RawProposal
from ldm_tts.contracts.evaluation import EvaluationResult, Observation
from ldm_tts.engine.expansion import ExpansionRequest
from ldm_tts.engine.run_store import CampaignRuntime
from tasks.alphabench.core.candidate import FactorDomain
from tasks.alphabench.core.collection import AcceptedActions
from tasks.alphabench.core.gateway import OracleGateway
from tasks.alphabench.core.harness_tools import HarnessToolService, check_position
from tasks.alphabench.core import harness_runtime
from tasks.alphabench.core.protocol import T3Protocol


def call(service, tool, **arguments):
    with socket.socket(socket.AF_UNIX) as connection:
        connection.connect(str(service.socket_path))
        connection.sendall(json.dumps({"tool": tool, "arguments": arguments}).encode() + b"\n")
        with connection.makefile("rb") as stream:
            return json.loads(stream.readline())


def observed(window, score, *, status="succeeded"):
    candidate = FactorDomain().admit(RawProposal({"name": "old", "expression": f"Mean($close,{window})"}, "test"))
    metrics = {"mock_rank_ic": score} if status == "succeeded" else {}
    return Observation(candidate, EvaluationResult(candidate.candidate_id, status, metrics=metrics), round_idx=0)


@pytest.mark.parametrize("backend", ["qlib", "assay"])
def test_host_tools_expose_public_contract_and_frozen_gp_without_private_labels(tmp_path, backend):
    protocol = T3Protocol(method="ldm_harness", backend=backend, harness_surrogate_query=True,
        budgets={**T3Protocol().budgets, "surrogate_queries": 400})
    runtime = CampaignRuntime.open(tmp_path / "run", task="alphabench", budget_limits={
        "dynamic_checks": 2, "oracle_job_slots": 4, "benchmark_jobs": 2,
        "surrogate_queries": 2})
    gateway = OracleGateway(protocol, runtime, mock=True)
    history = (observed(5, .1), observed(6, .2), observed(7, 0, status="invalid"))
    request = ExpansionRequest(1, 8, observations=history)
    with HarnessToolService(protocol, gateway, Path(tempfile.gettempdir()), surrogate_query=True) as tools:
        def research():
            frozen = tools.prepare(request)
            contract = call(tools, "get_contract")
            first = call(tools, "get_history", offset=0, limit=2)
            second = call(tools, "get_history", offset=2, limit=2)
            found = call(tools, "get_observation", candidate_id=history[2].candidate_id)
            query = call(tools, "query_surrogate", expression="Mean($close,8)", snapshot_id=frozen["snapshot_id"])
            again = call(tools, "query_surrogate", expression="Mean($close,8)", snapshot_id=frozen["snapshot_id"])
            stale = call(tools, "query_surrogate", expression="Mean($close,8)", snapshot_id="0" * 64)
            checked = call(tools, "check_expression", expression="Mean($close,8)")
            replay = call(tools, "check_expression", expression="Mean($close,8)")
            candidate = FactorDomain(backend).admit(RawProposal({"expression": "Mean($close,8)"}, "test"))
            gateway.evaluate(candidate, phase="check", position=check_position(request.round_idx, candidate))
            tools.clear()
            inactive = call(tools, "get_history", offset=0)
            return frozen, contract, first, second, found, query, again, stale, checked, replay, inactive
        frozen, contract, first, second, found, query, again, stale, checked, replay, inactive = gateway.host.run(research)
    assert contract["ok"] and set(contract["result"]["operator_signatures"]) == ({"qlib", "assay"} if backend == "assay" else {"qlib"})
    assert contract["result"]["surrogate_queries_remaining"] == 2
    assert first["result"]["total"] == 3 and first["result"]["next_offset"] == 2
    assert len(second["result"]["observations"]) == 1
    assert found["result"]["observation"]["status"] == "invalid"
    assert query["ok"] and again["ok"] and query["result"]["fit_status"] == "fitted"
    assert query["result"]["surrogate_queries_remaining"] == 1
    assert again["result"]["surrogate_queries_remaining"] == 0
    assert all(query["result"][key] == again["result"][key] for key in ("mean", "latent_std", "ucb", "snapshot_id"))
    assert query["result"]["snapshot_id"] == frozen["snapshot_id"]
    assert set(query["result"]) >= {"mean", "latent_std", "ucb", "history_digest"}
    assert stale["ok"] is False
    assert checked["ok"] and checked == replay and checked["result"]["success"] is True
    assert runtime.budget.counters["dynamic_checks"] == 1
    assert runtime.budget.counters["surrogate_queries"] == 2
    assert inactive["ok"] is False


def test_direct_harness_does_not_offer_gp_query_and_rejects_changed_snapshot(tmp_path):
    protocol = T3Protocol(method="harness")
    runtime = CampaignRuntime.open(tmp_path / "run", task="alphabench", budget_limits={})
    gateway = OracleGateway(protocol, runtime, mock=True)
    with HarnessToolService(protocol, gateway, Path(tempfile.gettempdir()), surrogate_query=False) as tools:
        def research():
            tools.prepare(ExpansionRequest(0, 2))
            validated = call(tools, "validate_expression", expression="Mean($close,5)")
            forbidden = call(tools, "query_surrogate", expression="Mean($close,5)", snapshot_id="anything")
            with pytest.raises(ValueError, match="snapshot changed"):
                tools.prepare(ExpansionRequest(0, 2, observations=(observed(5, .1),)))
            return validated, forbidden
        validated, forbidden = gateway.host.run(research)
    assert validated["ok"] and validated["result"]["canonical_expression"] == "Mean($close,5)"
    assert forbidden["ok"] is False


@pytest.mark.parametrize("method,query,count", [("harness", False, 1), ("ldm_harness", False, 2),
    ("ldm_harness", True, 2)])
def test_pi_runtime_exposes_only_the_frozen_research_variant(tmp_path, monkeypatch, method, query, count):
    monkeypatch.setattr(harness_runtime, "SOCKET_ROOT", Path(tempfile.gettempdir()))
    monkeypatch.setattr(harness_runtime, "CACHE_ROOT", tmp_path / "cache")
    protocol = T3Protocol(method=method, harness_surrogate_query=query,
        budgets={**T3Protocol().budgets, **({"surrogate_queries": 400} if query else {})})
    runtime = CampaignRuntime.open(tmp_path / "run", task="alphabench", budget_limits={})
    gateway = OracleGateway(protocol, runtime, mock=True)
    client, tools, _ = harness_runtime.build_harness(protocol, gateway, AcceptedActions(runtime.run_dir),
        api_key="test-only", sidecar_image="ldm-pi-t3:local")
    try:
        assert len(client.config.profiles) == count
        assert ("query_surrogate" in client.config.mcp_servers[0].tools) is query
        assert client.config.context7_enabled is False
        assert client.config.initialize_payload()["forceFirstToolCall"] is False
        assert client.config.network_policy.allowed_hosts
        mounts = [client.command[index + 1] for index, part in enumerate(client.command) if part == "--mount"]
        assert any("dst=/t3-host" in mount for mount in mounts)
        assert any("dst=/resources,readonly" in mount for mount in mounts)
        assert all(str(runtime.run_dir / "private") not in mount for mount in mounts)
    finally:
        client.close()
        tools.close()
