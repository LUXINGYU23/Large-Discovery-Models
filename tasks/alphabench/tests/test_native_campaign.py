import ast
from dataclasses import replace
import json
from pathlib import Path

import pytest

from tasks.alphabench.core import workflow
from tasks.alphabench.core.gateway import OracleGateway
from tasks.alphabench.core.protocol import T3Protocol
from tasks.alphabench.rebuild_report import rebuild
from tasks.alphabench.core.source_profiles import definition, resolve_source
from tasks.alphabench.core.oracle_worker import load_source
from tasks.alphabench.core.receipts import Receipts
from tasks.alphabench.ldm_task.procedure import main
from tasks.alphabench.tests.test_native import SOURCE, source


def protocol(method, **kwargs):
    parameters = {"oracle_workers": 2, "enable_reason": True, "accept_threshold": 0.0}
    parameters.update({"N": 2, "mutation_rate": .5, "crossover_rate": .5, "pool_size": 4, "seeds_top_k": 2}
                      if method == "ea" else {"workers": 2})
    if method == "tot":
        parameters.update(N=2, top_k=1)
    return replace(T3Protocol(method="alphabench_" + method, evaluations=10, cold_seed_count=2,
                              native_parameters=parameters), **kwargs)


def arguments(root, contract):
    path = root / "protocol.json"
    path.write_text(json.dumps(contract.to_dict()))
    return ["--mock", "--protocol-file", str(path), "--upstream-root", str(SOURCE), "--out-dir", str(root / "run")]


def source_protocol(method, backend="qlib", **changes):
    path = "searcher/config.yaml" if method == "tot" else f"searcher/configs/{method}_config.yaml"
    source = resolve_source(SOURCE, path, method, backend=backend)["effective_source"]
    algorithm, initial = source["algorithm"], source["initialization"]
    portfolio = {}
    if backend == "assay":
        config_type = load_source("test_assay_portfolio_config", SOURCE.parent / "Assay/src/assay/portfolio/config.py").PortfolioBacktestConfig
        portfolio = config_type.preset("A", universe="CSI300", period_start="2016-01-01", period_end="2025-01-01",
            benchmark="custom", benchmark_symbol="SH000300", slippage_model="zero", st_filter=False,
            new_listing_lockout_days=0, ipo_lockout_days=0, rebalance_around_index=False, save_position_log=True).to_dict()
    base = T3Protocol(method="alphabench_" + method, profile="upstream_searcher_v1", backend=backend,
        filter_profile=backend + "_code_filter_v1", native_parameters={"source_config": path},
        label=source["search"]["label"], assay_portfolio=portfolio,
        init_mode=initial["mode"], rounds=algorithm["rounds"], temperature=algorithm["temperature"], evaluations=1000)
    return replace(base, budgets={**base.budgets, "validation_evaluations": 1000,
        "dynamic_checks": 2000, "quality_checks": 2000, "oracle_job_slots": 12000,
        "benchmark_jobs": 12000}, **changes)


def example_protocol(method="ea", backend="qlib", **changes):
    path = "example/search/configs/search_csi300.yaml"
    effective = resolve_source(SOURCE, path, method, backend=backend)["effective_source"]
    portfolio = {}
    if backend == "assay":
        config_type = load_source("test_assay_example_config", SOURCE.parent / "Assay/src/assay/portfolio/config.py").PortfolioBacktestConfig
        portfolio = config_type.preset("A", universe="CSI300", period_start="2024-07-15", period_end="2024-12-20",
            benchmark="custom", benchmark_symbol="SH000300", slippage_model="zero", st_filter=False,
            new_listing_lockout_days=0, ipo_lockout_days=0, rebalance_around_index=False, save_position_log=True).to_dict()
    base = T3Protocol(method="alphabench_" + method, profile="upstream_benchmark_v1", backend=backend,
        native_parameters={"source_config": path}, init_mode="alpha158",
        alpha158_groups=("kbar", "rolling", "price"), filter_profile=backend + "_code_filter_v1",
        label=effective["search"]["label"], assay_portfolio=portfolio,
        rounds=effective["algorithm"]["rounds"], temperature=effective["algorithm"]["temperature"],
        evaluations=3000)
    return replace(base, budgets={**base.budgets, "model_requests": 500,
        "proposal_attempts": 500, "dynamic_checks": 5000, "quality_checks": 5000,
        "validation_evaluations": 3000, "oracle_job_slots": 50000,
        "benchmark_jobs": 50000}, **changes)


@pytest.mark.parametrize("method,backend", [("ea", "qlib"), ("cot", "qlib"), ("tot", "qlib"), ("ea", "assay")])
def test_example_runs_full_baseline_search_extension_and_replay(tmp_path, source, monkeypatch, method, backend):
    monkeypatch.setattr("tasks.alphabench.core.native_generator.time.sleep", lambda _: None)
    args = arguments(tmp_path, example_protocol(method, backend))
    assert main(args) == 0
    run = tmp_path / "run"
    report = json.loads((run / "result.json").read_text())
    execution = report["execution"]
    assert report["initialization"]["seed_count"] == 42
    assert len(execution["algorithm_result"]["baseline"]) == 42
    native_output = execution["algorithm_result"][method]
    assert native_output["final_pool"] if method == "ea" else native_output["results"]
    if method == "ea":
        assert [(row["name"], row["expression"]) for row in execution["native_final_pool"]] == [
            (row["name"], row["expression"]) for row in native_output["final_pool"]]
    assert execution["algorithm_completed"] and report["search"]["attempts"] > 0
    assert report["final_pool"] and report["test"] and report["independent_combination"]
    assert report["diversity"]["final_pool"]["interval"] == ["2024-01-15", "2024-06-28"]
    contract = Receipts(run / "private/stages").load("source_entry_contract")
    assert any(row["field"] == "validation/test" for row in contract["protocol_delta"])
    receipts = [json.loads(path.read_text()) for root in (run / "private/oracle", run / "initialization/private/oracle")
                for path in root.glob("*.json")]
    assert {row["request"]["phase"] for row in receipts} >= {"initialization", "search", "validation", "test"}
    assert all(row["request"]["fast"] is (row["request"]["phase"] not in {"initialization", "search", "test", "analysis"})
               for row in receipts)
    if backend == "assay":
        assert all(row["request"]["protocol"]["label"] == "open_return" for row in receipts)
        assert all(row["response"]["check_kind"] == "lint" for row in receipts if row["request"]["operation"] == "check")
    frozen = {path.relative_to(run): path.read_bytes() for path in run.rglob("*") if path.is_file()}
    assert main(args + ["--resume-run", str(run)]) == 0
    assert {path.relative_to(run): path.read_bytes() for path in run.rglob("*") if path.is_file()} == frozen


@pytest.mark.parametrize("change,error", [
    ({"alpha158_groups": ("kbar", "rolling")}, "all three pinned Alpha158 groups"),
    ({"rounds": 2}, "differs from the pinned source config"),
    ({"filter_profile": "paper_filter_v1"}, "differs from the pinned source config"),
])
def test_example_contract_rejects_unfrozen_settings_before_baseline(tmp_path, source, change, error):
    args = arguments(tmp_path, example_protocol(**change))
    with pytest.raises(ValueError, match=error):
        main(args + ["--dry-run"])
    assert not (tmp_path / "run/initialization").exists()


def test_example_requires_budget_for_all_42_baseline_factors(tmp_path, source):
    base = example_protocol()
    args = arguments(tmp_path, replace(base, budgets={**base.budgets, "initialization_evaluations": 41}))
    with pytest.raises(ValueError, match="complete seed pool"):
        main(args + ["--dry-run"])
    assert not (tmp_path / "run/initialization").exists()


def test_example_batch_budget_stop_does_not_publish_truncated_result(tmp_path, source, monkeypatch):
    monkeypatch.setattr("tasks.alphabench.core.native_generator.time.sleep", lambda _: None)
    args = arguments(tmp_path, example_protocol(evaluations=25))
    assert main(args) == 2
    run = tmp_path / "run"
    assert json.loads((run / "status.json").read_text())["status"] == "paused_budget"
    assert not (run / "result.json").exists() and not (run / "selection_frozen.json").exists()
    assert json.loads((run / "budget.json").read_text())["counters"]["expensive_evaluation_attempts"] == 0


def test_example_sp500_binds_every_evaluation_to_the_selected_market(tmp_path, source, monkeypatch):
    monkeypatch.setattr("tasks.alphabench.core.native_generator.time.sleep", lambda _: None)
    args = arguments(tmp_path, example_protocol(market="sp500"))
    assert main(args) == 0
    run = tmp_path / "run"
    report = json.loads((run / "result.json").read_text())
    assert report["execution"]["algorithm_completed"] and report["test"]
    receipts = [json.loads(path.read_text()) for root in (run / "private/oracle", run / "initialization/private/oracle")
                for path in root.glob("*.json")]
    assert {row["request"]["phase"] for row in receipts} >= {"initialization", "search", "validation", "test"}
    assert all(row["request"]["protocol"]["market"] == "sp500" for row in receipts)


@pytest.mark.parametrize("method", ["cot", "tot", "ea"])
@pytest.mark.parametrize("backend", ["qlib", "assay"])
def test_source_searcher_freezes_entry_and_tests_only_its_actual_final_pool(tmp_path, source, monkeypatch, method, backend):
    monkeypatch.setattr("tasks.alphabench.core.native_generator.time.sleep", lambda _: None)
    monkeypatch.setattr(workflow, "run_campaign", lambda *_args, **_kwargs: pytest.fail("source started a search LDMEngine"))
    args = arguments(tmp_path, source_protocol(method, backend=backend))
    assert main(args) == 0
    run = tmp_path / "run"
    report = json.loads((run / "result.json").read_text())
    execution = report["execution"]
    assert execution["algorithm_completed"] and execution["config"]["algorithm"]["rounds"] == 10
    assert report["initialization"]["seed_count"] == (125 if method == "ea" else 30)
    assert [(f["name"], f["expression"]) for f in report["final_pool"]] == [
        (f["name"], f["expression"]) for f in execution["native_final_pool"]]
    assert report["diversity"]["final_pool"]["scope"] == "native algorithm final pool"
    assert len(report["final_pool"]) < report["initialization"]["seed_count"] + report["search"]["attempts"]
    assert report["test"] and report["independent_combination"] and report["quality_audit"]["complete"]
    receipts = [json.loads(p.read_text()) for p in (run / "private/oracle").glob("*.json")]
    receipts += [json.loads(p.read_text()) for p in (run / "initialization/private/oracle").glob("*.json")]
    requests = [row["request"] for row in receipts]
    assert all(row["fast"] is (row["phase"] not in {"test", "analysis"}) for row in requests)
    if backend == "assay":
        assert all(row["protocol"]["label"] == "open_return" and row["protocol"]["assay_portfolio"] for row in requests)
        assert all(row["response"]["check_kind"] == "lint" for row in receipts if row["request"]["operation"] == "check")
        assert execution["config"]["source_resolution"]["effective_source"]["factor_backend"]["portfolio"] is False
    private = {row["identity"]["candidate"]: row["response"]["metrics"] for row in receipts
               if row["identity"]["phase"] == "validation"}
    from tasks.alphabench.core.candidate import FactorDomain
    from ldm_tts.contracts import RawProposal
    factors = [{**row, "val_metrics": private[FactorDomain().admit(RawProposal(row, "comparison")).candidate_id]}
               for row in execution["native_final_pool"]]
    node = definition((source / "searcher/run_test_backtest.py").read_text(), "rank_factors")
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "source_rank_factors", "exec"), namespace)
    expected = namespace["rank_factors"](factors)[:50]
    assert [(row["candidate"]["name"], row["candidate"]["expression"]) for row in report["test"]] == [
        (row["name"], row["expression"]) for row in expected]
    if method == "cot":
        signal = report["diversity"]["final_pool"]["signal"]
        assert any(pair["left"] == pair["right"] and pair["left_index"] != pair["right_index"] for pair in signal["pairs"])
    frozen = {p.relative_to(run): p.read_bytes() for p in run.rglob("*") if p.is_file()}
    assert main(args + ["--resume-run", str(run)]) == 0
    assert {p.relative_to(run): p.read_bytes() for p in run.rglob("*") if p.is_file()} == frozen


def test_source_budget_stop_does_not_publish_a_truncated_native_result(tmp_path, source, monkeypatch):
    monkeypatch.setattr("tasks.alphabench.core.native_generator.time.sleep", lambda _: None)
    args = arguments(tmp_path, source_protocol("ea", evaluations=25))
    assert main(args) == 2
    run = tmp_path / "run"
    assert json.loads((run / "status.json").read_text())["status"] == "paused_budget"
    assert not (run / "result.json").exists() and not (run / "selection_frozen.json").exists()
    assert json.loads((run / "budget.json").read_text())["counters"]["expensive_evaluation_attempts"] == 20
    budget = (run / "budget.json").read_bytes()
    receipts = {p.name: p.read_bytes() for p in (run / "private/model").glob("*.json")}
    assert main(args + ["--resume-run", str(run)]) == 2
    assert {p.name: p.read_bytes() for p in (run / "private/model").glob("*.json")} == receipts
    assert (run / "budget.json").read_bytes() == budget


@pytest.mark.parametrize("change", [{"rounds": 2}, {"cold_seed_count": 3}, {"filter_profile": "paper_filter_v1"}])
def test_source_settings_cannot_silently_become_a_matched_pilot(tmp_path, source, change):
    args = arguments(tmp_path, source_protocol("cot", **change))
    with pytest.raises(ValueError, match="differs from the pinned source config"):
        main(args + ["--dry-run"])
    with pytest.raises(ValueError, match="differs from the pinned source config"):
        main(args)
    assert not (tmp_path / "run/initialization").exists()


@pytest.mark.parametrize("change,error", [
    ({"assay_portfolio": {}}, "full independent PortfolioBacktestConfig"),
    ({"label": "close_return"}, "differs from the pinned source config"),
    ({"filter_profile": "paper_filter_v1"}, "differs from the pinned source config"),
])
def test_assay_source_contract_is_checked_before_initialization(tmp_path, source, change, error):
    args = arguments(tmp_path, source_protocol("cot", backend="assay", **change))
    with pytest.raises(ValueError, match=error):
        main(args + ["--dry-run"])
    with pytest.raises(ValueError, match=error):
        main(args)
    assert not (tmp_path / "run/initialization").exists()


def test_source_validation_failure_cannot_select_test_factors_using_search_scores(tmp_path, source, monkeypatch):
    monkeypatch.setattr("tasks.alphabench.core.native_generator.time.sleep", lambda _: None)
    send = OracleGateway._send
    search_failed, validated, phases = [], [], []
    def respond(self, request):
        result = send(self, request)
        phases.append(request["phase"])
        if request["phase"] == "search":
            search_failed.append(request["expression"])
            result.update(success=False, metrics={})
        if request["phase"] == "validation":
            validated.append(request["expression"])
            result.update(success=False, metrics={})
        return result
    monkeypatch.setattr(OracleGateway, "_send", respond)
    args = arguments(tmp_path, source_protocol("ea"))
    assert main(args) == 2
    run = tmp_path / "run"
    assert search_failed and set(search_failed) <= set(validated)
    assert "test" not in phases and "analysis" not in phases
    assert json.loads((run / "status.json").read_text())["status"] == "paused_incomplete_validation"
    assert not (run / "selection_frozen.json").exists()


@pytest.mark.parametrize("method", ["cot", "tot", "ea"])
@pytest.mark.parametrize("backend", ["qlib", "assay"])
def test_native_workflow_runs_full_stages_without_starting_search_engine(tmp_path, source, monkeypatch, method, backend):
    monkeypatch.setattr("tasks.alphabench.core.native_generator.time.sleep", lambda _: None)
    monkeypatch.setenv("LDM_DATA_COLLECTION_ENABLED", "1")
    monkeypatch.setenv("LDM_DATA_COLLECTION_DIR", str(tmp_path / "collection"))
    monkeypatch.setattr(workflow, "run_campaign", lambda *_args, **_kwargs: pytest.fail("native search started LDMEngine"))
    args = arguments(tmp_path, protocol(method, backend=backend))
    run = tmp_path / "run"
    assert main(args) == 0
    report = json.loads((run / "result.json").read_text())
    assert report["execution"]["kind"] == "native_reference"
    assert report["execution"]["engine_native"] is False
    assert report["execution"]["algorithm_completed"] is True
    assert report["execution"]["algorithm_result"]["history"]
    assert report["execution"]["native_final_pool"] == report["execution"]["algorithm_result"]["final_pool"]
    assert report["test"] and report["independent_combination"] and report["quality_audit"]["complete"]
    assert report["search"]["attempts"] > 0
    if method == "ea":
        update = report["search"]["ea_update"]
        assert update["rounds"] == 2 and len(update["events"]) == 2
        assert update["sum_U_t"] == sum(event["U_t"] for event in update["events"])
        assert update["reason"] is None
        assert len(rebuild(run, tmp_path / "rebuilt")["matched"]) == 4
    else:
        assert report["search"]["ea_update"] == {"applicable": False, "reason": "not_ea"}
    assert report["initialization"]["seed_count"] == 2
    assert report["complete_t3"] is False
    cold = json.loads(next((run / "initialization/generation").glob("*.json")).read_text())
    assert cold["native_result"]["success"] is True
    pointers = list((tmp_path / "collection").rglob("current.json"))
    assert len(pointers) == 2
    for pointer in pointers:
        published = json.loads(pointer.read_text())
        assert published["count"] > 0
        directory = pointer.parent / published["generation"]
        assert len((directory / "ldm_ir.jsonl").read_text().splitlines()) == published["count"]
        assert len((directory / "ldm_sft.jsonl").read_text().splitlines()) == published["count"]
    before = {p.relative_to(run): p.read_bytes() for p in run.rglob("*") if p.is_file()}
    assert main(args + ["--resume-run", str(run)]) == 0
    assert {p.relative_to(run): p.read_bytes() for p in run.rglob("*") if p.is_file()} == before


@pytest.mark.parametrize("method", ["cot", "tot", "ea"])
def test_native_budget_stop_finalizes_every_paid_search_attempt(tmp_path, source, monkeypatch, method):
    monkeypatch.setattr("tasks.alphabench.core.native_generator.time.sleep", lambda _: None)
    args = arguments(tmp_path, protocol(method, evaluations=1))
    assert main(args) == 0
    report = json.loads((tmp_path / "run/result.json").read_text())
    assert report["execution"]["algorithm_completed"] is False
    assert report["execution"]["stop_reason"] == "search_batch_budget_exhausted"
    assert report["completeness"]["search"]["status"] == "complete"
    assert report["search"]["attempts"] == 1
    assert report["execution"]["remaining_search_allowance"] == 0
    assert report["quality_audit"]["complete"]
    assert main(args + ["--resume-run", str(tmp_path / "run")]) == 0


@pytest.mark.parametrize("method", ["cot", "tot", "ea"])
def test_matched_native_trajectory_contains_every_planned_search_step(tmp_path, source, monkeypatch, method):
    monkeypatch.setattr("tasks.alphabench.core.native_generator.time.sleep", lambda _: None)
    args = arguments(tmp_path, protocol(method, evaluations=5, rounds=3))
    assert main(args) == 0
    run = tmp_path / "run"
    report = json.loads((run / "result.json").read_text())
    assert report["search"]["attempts"] == 5
    assert report["budget"]["counters"]["expensive_evaluation_attempts"] == 5
    assert report["completeness"]["search"]["status"] == "complete"
    assert len((run / "trajectory.csv").read_text().splitlines()) == 6


def test_native_finalization_pause_resumes_without_restarting_search_or_generation(tmp_path, source, monkeypatch):
    monkeypatch.setattr("tasks.alphabench.core.native_generator.time.sleep", lambda _: None)
    send = OracleGateway._send
    lost, calls = {}, []
    def disconnect(self, request):
        calls.append(request["phase"])
        result = send(self, request)
        if request["phase"] == "test" and not lost:
            lost.update(result)
            raise OSError("test response lost")
        return result
    monkeypatch.setattr(OracleGateway, "_send", disconnect)
    args = arguments(tmp_path, protocol("ea"))
    run = tmp_path / "run"
    assert main(args) == 2
    assert not (run / "result.json").exists()
    assert json.loads((run / "status.json").read_text())["status"].startswith("paused_")
    search_calls = calls.count("search")
    model_receipts = {p.name: p.read_bytes() for p in (run / "private/model").glob("*.json")}
    frozen = (run / "selection_frozen.json").read_bytes()
    monkeypatch.setattr(OracleGateway, "_reconcile", lambda self, request: lost if request["request_id"] == lost["request_id"] else None)
    assert main(args + ["--resume-run", str(run)]) == 0
    assert calls.count("search") == search_calls
    assert (run / "selection_frozen.json").read_bytes() == frozen
    assert {p.name: p.read_bytes() for p in (run / "private/model").glob("*.json")} == model_receipts


def test_native_configuration_is_checked_before_any_initialization_cost(tmp_path, source):
    invalid = protocol("ea", native_parameters={"N": 2})
    with pytest.raises(ValueError, match="explicitly contain"):
        main(arguments(tmp_path, invalid))
    assert not (tmp_path / "run/initialization").exists()


def test_named_runner_profile_binds_protocol_contents_before_initialization(tmp_path, source, monkeypatch):
    root = Path(__file__).resolve().parents[3]
    monkeypatch.setenv("LDM_EXPERIMENT_CONTRACT_PATH", str(root / "tasks/alphabench/experiment.json"))
    monkeypatch.setenv("LDM_EXPERIMENT_CONTRACT_PROFILE", "mock_native_ea")
    changed = json.loads((root / "tasks/alphabench/resources/protocols/native_ea_qualification.json").read_text())
    changed["budgets"]["model_requests"] += 1
    with pytest.raises(ValueError, match="locked runner profile"):
        main(arguments(tmp_path, T3Protocol(**changed)))
    assert not (tmp_path / "run/initialization").exists()


def test_named_campaign_preserves_runner_identity_on_direct_resume(tmp_path, source, monkeypatch):
    root = Path(__file__).resolve().parents[3]
    monkeypatch.setattr("tasks.alphabench.core.native_generator.time.sleep", lambda _: None)
    monkeypatch.setenv("LDM_EXPERIMENT_CONTRACT_PATH", str(root / "tasks/alphabench/experiment.json"))
    monkeypatch.setenv("LDM_EXPERIMENT_CONTRACT_PROFILE", "mock_native_ea")
    contract = T3Protocol(**json.loads((root / "tasks/alphabench/resources/protocols/native_ea_qualification.json").read_text()))
    args = arguments(tmp_path, contract)
    assert main(args) == 0
    run = tmp_path / "run"
    snapshot = json.loads((run / "experiment_contract.json").read_text())
    campaign = json.loads((run / "campaign.json").read_text())
    assert campaign["contract_sha256"] == snapshot["snapshot"]["sha256"]
    assert campaign["contract_profile"] == "mock_native_ea"
    monkeypatch.delenv("LDM_EXPERIMENT_CONTRACT_PATH")
    monkeypatch.delenv("LDM_EXPERIMENT_CONTRACT_PROFILE")
    before = {p.relative_to(run): p.read_bytes() for p in run.rglob("*") if p.is_file()}
    assert main(args + ["--resume-run", str(run)]) == 0
    assert {p.relative_to(run): p.read_bytes() for p in run.rglob("*") if p.is_file()} == before
    snapshot["evaluation"]["settings"]["reasoning_effort"] = "low"
    (run / "experiment_contract.json").write_text(json.dumps(snapshot))
    with pytest.raises(ValueError, match="snapshot integrity"):
        main(args + ["--resume-run", str(run)])


def test_native_matched_workflow_imports_the_same_verified_shared_seed_information(tmp_path, source, monkeypatch):
    from tasks.alphabench.tests.test_seed_bundle import make_source
    monkeypatch.setattr("tasks.alphabench.core.native_generator.time.sleep", lambda _: None)
    original, bundle = make_source(tmp_path)
    contract = protocol("tot", cold_seed_count=original.cold_seed_count,
                        budgets={**original.budgets, "initialization_evaluations": 0})
    args = arguments(tmp_path, contract)
    assert main(args + ["--initialization-bundle", str(bundle)]) == 0
    created = json.loads((bundle / "seed_manifest.json").read_text())
    imported = json.loads((tmp_path / "run/initialization/seed_manifest.json").read_text())
    assert imported["observations"] == created["observations"] and imported["pool"] == created["pool"]
    report = json.loads((tmp_path / "run/result.json").read_text())
    assert report["initialization"]["public_information_digest"] == created["public_information_digest"]
    assert all(value == 0 for value in report["initialization"]["budget"]["counters"].values())
    assert report["search"]["attempts"] > 0


@pytest.mark.parametrize("method", ["cot", "ea"])
def test_empty_initialization_retains_each_native_algorithms_actual_seed_requirement(tmp_path, source, monkeypatch, method):
    monkeypatch.setattr("tasks.alphabench.core.native_generator.time.sleep", lambda _: None)
    args = arguments(tmp_path, protocol(method, cold_seed_count=0))
    run = tmp_path / "run"
    assert main(args) == (2 if method == "cot" else 0)
    if method == "cot":
        assert json.loads((run / "status.json").read_text())["status"] == "paused_no_valid_seeds"
        assert not list((run / "private/model").glob("*.json"))
    else:
        report = json.loads((run / "result.json").read_text())
        assert report["initialization"]["seed_count"] == 0 and report["search"]["attempts"] > 0
