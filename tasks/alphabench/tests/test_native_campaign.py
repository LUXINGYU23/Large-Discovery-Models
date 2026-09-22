from dataclasses import replace
import json
from pathlib import Path

import pytest

from tasks.alphabench.core import workflow
from tasks.alphabench.core.gateway import OracleGateway
from tasks.alphabench.core.protocol import T3Protocol
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
def test_native_budget_stop_finalizes_committed_state_without_partial_batch_scores(tmp_path, source, monkeypatch, method):
    monkeypatch.setattr("tasks.alphabench.core.native_generator.time.sleep", lambda _: None)
    args = arguments(tmp_path, protocol(method, evaluations=1))
    assert main(args) == 0
    report = json.loads((tmp_path / "run/result.json").read_text())
    assert report["execution"]["algorithm_completed"] is False
    assert report["execution"]["stop_reason"] == "search_batch_budget_exhausted"
    assert report["search"]["attempts"] <= 1
    assert report["execution"]["remaining_search_allowance"] == (0 if method == "cot" else 1)
    assert report["quality_audit"]["complete"]
    assert main(args + ["--resume-run", str(tmp_path / "run")]) == 0


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
