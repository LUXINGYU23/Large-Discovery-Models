from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import ldm_tts.cli.runner as runner_module
from ldm_tts.engine.run_store import BudgetExceededError, BudgetLedger, CampaignStatus
from ldm_tts.transport.openai import (
    EndpointCircuitBreaker,
    EndpointCircuitOpen,
    EndpointRequestError,
    call_with_circuit_breaker,
)
from ldm_tts.registration.experiment import (
    ExperimentContractError,
    load_experiment_contract,
    snapshot_experiment_contract,
    validate_profile_args,
)


def _write_qualified_contract(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "task_id": "contract_test",
        "qualification": "qualified",
        "benchmark": {
            "source_url": "https://example.test/benchmark",
            "source_commit": "0123456789abcdef",
        },
        "metrics": {
            "reported": [{"name": "objective", "direction": "maximize"}],
            "optimized": [{"name": "search_score", "direction": "maximize"}],
            "diagnostic": [],
        },
        "evaluation": {
            "datasets": ["validation"],
            "settings": {"epochs": 100},
            "per_candidate_limits": {"training_hours": 1},
        },
        "budget": {"epochs_per_evaluation": 100},
        "profiles": {
            "official": {
                "description": "Pinned test profile.",
                "budget": {"expensive_evaluation_attempts": 2},
                "locked_args": {"epochs": 100},
            }
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_qualified_contract_exposes_metric_roles_and_pinned_source(tmp_path: Path) -> None:
    contract = load_experiment_contract(
        _write_qualified_contract(tmp_path / "experiment.json")
    )

    assert contract.task_id == "contract_test"
    assert contract.qualification == "qualified"
    assert contract.benchmark["source_commit"] == "0123456789abcdef"
    assert contract.proposal_provider == {
        "kind": "unspecified",
        "requires_endpoint_preflight": False,
        "supports_collection": False,
    }
    assert [item["name"] for item in contract.metrics["reported"]] == ["objective"]
    assert [item["name"] for item in contract.metrics["optimized"]] == ["search_score"]
    assert contract.profile("official").budget["expensive_evaluation_attempts"] == 2


def test_contract_validates_provider_capabilities_and_metric_modes(tmp_path: Path) -> None:
    path = _write_qualified_contract(tmp_path / "experiment.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["proposal_provider"] = {
        "kind": "model_endpoint",
        "requires_endpoint_preflight": True,
        "supports_collection": True,
    }
    payload["metrics"]["optimized"][0]["modes"] = ["real"]
    path.write_text(json.dumps(payload), encoding="utf-8")

    contract = load_experiment_contract(path)

    assert contract.proposal_provider["requires_endpoint_preflight"] is True
    assert contract.metrics["optimized"][0]["modes"] == ["real"]

    payload["proposal_provider"]["requires_endpoint_preflight"] = False
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ExperimentContractError, match="must require endpoint preflight"):
        load_experiment_contract(path)


@pytest.mark.parametrize(
    ("invalid_field", "error_pattern"),
    [
        ("source_commit", r"must pin benchmark\.source_commit"),
        ("per_candidate_limits", r"must define evaluation\.per_candidate_limits"),
        ("profiles", "must define at least one campaign profile"),
    ],
)
def test_qualified_contract_requires_qualification_evidence(
    tmp_path: Path,
    invalid_field: str,
    error_pattern: str,
) -> None:
    contract_path = _write_qualified_contract(tmp_path / "source" / "experiment.json")
    payload = json.loads(contract_path.read_text(encoding="utf-8"))
    if invalid_field == "source_commit":
        payload["benchmark"]["source_commit"] = "unqualified"
    elif invalid_field == "per_candidate_limits":
        payload["evaluation"]["per_candidate_limits"] = {}
    else:
        payload["profiles"] = {}
    path = tmp_path / "experiment.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ExperimentContractError, match=error_pattern):
        load_experiment_contract(path)


def test_contract_profile_rejects_training_budget_drift(tmp_path: Path) -> None:
    contract = load_experiment_contract(
        _write_qualified_contract(tmp_path / "experiment.json")
    )
    profile = contract.profile("official")
    args = dict(profile.locked_args)
    args["epochs"] = 99

    with pytest.raises(ExperimentContractError, match="--epochs=99, expected 100"):
        validate_profile_args(contract, profile.name, args)


def test_runner_enforces_selected_contract_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    relative_contract = Path("tasks/contract_test/experiment.json")
    contract_path = _write_qualified_contract(tmp_path / relative_contract)
    contract = load_experiment_contract(contract_path)
    profile = contract.profile("official")
    definition = SimpleNamespace(
        relative_root=Path("tasks/contract_test"),
        module="tasks.contract_test.ldm_task.procedure",
        experiment_contract_path=relative_contract,
    )
    monkeypatch.setattr(runner_module, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(runner_module, "get_task_definition", lambda task_id: definition)
    config = {
        "name": "contract_test",
        "task": "contract_test",
        "algorithm": "gp_ucb",
        "mode": "real",
        "contract_profile": profile.name,
        "args": dict(profile.locked_args),
    }
    config_path = tmp_path / "config/contract_test/real.yaml"

    plan = runner_module.build_plan(config, config_path)
    assert plan["contract_profile"] == profile.name
    assert plan["contract_sha256"] == contract.digest

    config["args"]["epochs"] = 10
    with pytest.raises(SystemExit, match="violates experiment contract profile"):
        runner_module.build_plan(config, config_path)


def test_contract_snapshot_records_digest_and_profile(tmp_path: Path) -> None:
    contract = load_experiment_contract(
        _write_qualified_contract(tmp_path / "experiment.json")
    )
    snapshot_dir = tmp_path / "snapshot"
    snapshot_dir.mkdir()
    destination = snapshot_experiment_contract(
        contract, snapshot_dir, profile="official"
    )
    payload = json.loads(destination.read_text(encoding="utf-8"))

    assert payload["snapshot"]["sha256"] == contract.digest
    assert payload["snapshot"]["profile"] == "official"


def test_budget_ledger_persists_and_prevents_overflow(tmp_path: Path) -> None:
    path = tmp_path / "budget.json"
    ledger = BudgetLedger(
        limits={"expensive_evaluation_attempts": 2},
        path=path,
    )
    ledger.consume("expensive_evaluation_attempts")
    ledger.consume("expensive_evaluation_attempts")

    with pytest.raises(BudgetExceededError, match="would be exceeded"):
        ledger.consume("expensive_evaluation_attempts")

    restored = BudgetLedger.load(path)
    assert restored.counters["expensive_evaluation_attempts"] == 2
    assert restored.remaining("expensive_evaluation_attempts") == 0


def test_cumulative_usage_survives_resume_and_overflow_atomically(tmp_path: Path) -> None:
    path = tmp_path / "budget.json"
    ledger = BudgetLedger(limits={"calls": 5}, path=path)
    ledger.consume_many({"calls": 2, "tools": 1}, usage_key="turn:a")
    ledger = BudgetLedger.load(path)
    ledger.consume_many({"calls": 2}, usage_key="turn:a")
    ledger.consume_many({"calls": 1}, usage_key="turn:a")
    ledger.consume_many({"calls": 3, "tools": 2}, usage_key="turn:a")
    ledger.consume_many({"calls": 2}, usage_key="turn:b")
    assert ledger.counters == {"calls": 5, "tools": 2}
    before = path.read_bytes()
    with pytest.raises(BudgetExceededError):
        ledger.consume_many({"tools": 4, "calls": 4}, usage_key="turn:a")
    assert path.read_bytes() == before
    assert ledger.counters == {"calls": 5, "tools": 2}
    assert ledger.metadata["cumulative_usage"]["turn:a"] == {"calls": 3, "tools": 2}


def test_budget_snapshot_includes_zero_counters_and_normalizes_integral_floats() -> None:
    ledger = BudgetLedger(
        limits={"whole": 60.0, "fractional": 2.5, "unused": 1},
        counters={"whole": 60.0, "fractional": 0.25},
    )

    assert ledger.snapshot()["counters"] == {
        "fractional": 0.25,
        "unused": 0,
        "whole": 60,
    }


def test_campaign_status_embeds_current_budget(tmp_path: Path) -> None:
    ledger = BudgetLedger(limits={"outer_iterations": 3})
    ledger.consume("outer_iterations")
    writer = CampaignStatus(tmp_path / "status.json", "task", "run")
    writer.update("running", phase="evaluation", iteration=1, budget=ledger)

    payload = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    assert payload["status"] == "running"
    assert payload["phase"] == "evaluation"
    assert payload["budget"]["remaining"]["outer_iterations"] == 2


def test_endpoint_circuit_opens_at_failure_threshold() -> None:
    breaker = EndpointCircuitBreaker(failure_threshold=2, recovery_timeout_seconds=60)
    calls = 0

    def unavailable() -> None:
        nonlocal calls
        calls += 1
        raise EndpointRequestError("upstream timeout")

    with pytest.raises(EndpointRequestError, match="upstream timeout"):
        call_with_circuit_breaker(breaker, unavailable)
    with pytest.raises(EndpointCircuitOpen, match="opened after 2 failures"):
        call_with_circuit_breaker(breaker, unavailable)
    with pytest.raises(EndpointCircuitOpen, match="circuit is open"):
        call_with_circuit_breaker(breaker, unavailable)

    assert calls == 2
    assert breaker.snapshot()["state"] == "open"


def test_endpoint_circuit_recovers_from_zero_timestamp() -> None:
    breaker = EndpointCircuitBreaker(
        failure_threshold=1,
        recovery_timeout_seconds=10,
    )
    breaker.record_failure(EndpointRequestError("unavailable"), now=0)

    with pytest.raises(EndpointCircuitOpen, match="circuit is open"):
        breaker.before_request(now=9)
    breaker.before_request(now=10)

    assert breaker.state == "half_open"
