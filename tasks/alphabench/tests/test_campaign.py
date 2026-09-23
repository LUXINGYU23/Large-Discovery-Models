from dataclasses import replace
import csv
import json

import pytest

from tasks.alphabench.core.protocol import T3Protocol
from tasks.alphabench.core.reporting import render_report
from tasks.alphabench.ldm_task.procedure import main


@pytest.mark.parametrize("method", ["llm", "ldm"])
def test_initialized_campaign_resume_has_no_new_calls_or_cost(method, tmp_path, monkeypatch):
    monkeypatch.setenv("LDM_DATA_COLLECTION_ENABLED", "1")
    monkeypatch.setenv("LDM_DATA_COLLECTION_DIR", str(tmp_path / "collection"))
    run = tmp_path / method
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps(replace(T3Protocol(), method=method, cold_seed_count=3).to_dict()))
    arguments = ["--mock", "--protocol-file", str(protocol), "--out-dir", str(run)]
    assert main(arguments) == 0
    checkpoint = json.loads((run / "checkpoint.json").read_text(encoding="utf-8"))
    assert len(checkpoint["state"]["observations"]) == 7
    budget = (run / "budget.json").read_bytes()
    model = {p.name: p.read_bytes() for p in (run / "private/model").glob("*.json")}
    selection = (run / "selection_frozen.json").read_bytes()
    published = {}
    for pointer in (tmp_path / "collection").rglob("current.json"):
        current = json.loads(pointer.read_text(encoding="utf-8"))
        generation = pointer.parent / current["generation"]
        assert len((generation / "ldm_ir.jsonl").read_text(encoding="utf-8").splitlines()) == current["count"]
        assert len((generation / "ldm_sft.jsonl").read_text(encoding="utf-8").splitlines()) == current["count"]
        published[pointer] = pointer.read_bytes()
    assert len(published) == 2  # Separate initialization and search collections.
    report_files = {name: (run / name).read_bytes() for name in ("result.json", "report.md", "trajectory.csv")}
    assert main(arguments + ["--resume-run", str(run)]) == 0
    assert (run / "budget.json").read_bytes() == budget
    assert {p.name: p.read_bytes() for p in (run / "private/model").glob("*.json")} == model
    assert (run / "selection_frozen.json").read_bytes() == selection
    assert all(path.read_bytes() == content for path, content in published.items())
    assert all((run / name).read_bytes() == content for name, content in report_files.items())
    report = json.loads((run / "result.json").read_text(encoding="utf-8"))
    assert render_report(report).encode("utf-8") == report_files["report.md"]
    assert report["search"]["attempts"] == 4
    assert report["initialization"]["seed_count"] == 3
    assert report["completeness"]["search"]["status"] == "complete"
    assert report["completeness"]["selection_frozen"]["status"] == "complete"
    assert report["completeness"]["report"]["status"] == "complete"
    assert report["completeness"]["data"]["status"] == "synthetic"
    assert report["costs"]["tool_calls"] == 0
    assert report["costs"]["model_requests"] == (
        report["initialization"]["budget"]["counters"]["model_requests"] +
        report["budget"]["counters"]["model_requests"])
    assert report["costs"]["model_requests_by_stage"]["initialization"] == 1
    assert "| independent_combination | complete |" in (run / "report.md").read_text(encoding="utf-8")
    with (run / "trajectory.csv").open(newline="") as stream:
        trajectory = list(csv.DictReader(stream))
    assert len(trajectory) == 4
    assert trajectory[-1]["successful_search_observations"] == "4"
    assert trajectory[-1]["cumulative_search_jobs"] == "4"
    assert trajectory[-1]["model_requests_run_total"] == str(report["costs"]["model_requests"])
    for filename in ("events.jsonl", "checkpoint.json", "summary.json"):
        assert (run / filename).exists()


def test_real_run_requires_data_and_frozen_protocol(tmp_path):
    with pytest.raises(ValueError, match="frozen"):
        main(["--out-dir", str(tmp_path)])


def test_shared_search_reports_partial_when_rounds_end_before_attempt_target(tmp_path):
    protocol = replace(T3Protocol(), cold_seed_count=0, rounds=1, evaluations=4, batch_size=2)
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(protocol.to_dict()))
    run = tmp_path / "run"
    assert main(["--mock", "--protocol-file", str(path), "--out-dir", str(run)]) == 0
    report = json.loads((run / "result.json").read_text())
    assert report["search"]["attempts"] == 2
    assert report["completeness"]["search"] == {
        "status": "partial", "attempts": 2, "planned_max_attempts": 4, "reason": "iteration_budget"}


@pytest.mark.parametrize("metric,winner", [("ic", 3), ("rank_ic", 0), ("icir", 1), ("rank_icir", 2)])
def test_frozen_validation_ranking_is_independent_of_search_objective(metric, winner, tmp_path, monkeypatch):
    from tasks.alphabench.core import gateway
    original = gateway.mock_oracle
    measured = []
    def response(request):
        result = original(request)
        if request["phase"] == "validation":
            index = len(measured)
            measured.append(request["expression"])
            result["metrics"].update(ic=index, rank_ic=-index, icir=-abs(index-1), rank_icir=-abs(index-2))
        return result
    monkeypatch.setattr(gateway, "mock_oracle", response)
    protocol = replace(T3Protocol(), cold_seed_count=0, validation_metric=metric, factor_select_n=1)
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(protocol.to_dict()))
    run = tmp_path / "run"
    assert main(["--mock", "--protocol-file", str(path), "--out-dir", str(run)]) == 0
    selection = json.loads((run / "selection_frozen.json").read_text())
    assert selection["validation_metric"] == metric
    assert selection["selected"][0]["candidate"]["payload"]["expression"] == measured[winner]
    report = json.loads((run / "result.json").read_text())
    assert report["protocol"]["objective"] == "rank_ic"
    assert len(report["diversity"]["final_pool"]["signal"]["candidate_ids"]) == 4
    assert len(report["diversity"]["test_selection"]["signal"]["candidate_ids"]) == 1
    assert report["diversity"]["final_pool"]["signal"]["reason"] is None
    assert report["diversity"]["test_selection"]["signal"]["reason"] == "no_defined_pairs"
