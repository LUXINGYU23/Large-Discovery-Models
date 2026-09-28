from dataclasses import replace
import json

import pytest

from tasks.alphabench.core.protocol import T3Protocol
from tasks.alphabench.core import workflow
from tasks.alphabench.ldm_task.procedure import main
from tasks.alphabench.rebuild_report import rebuild


def test_offline_rebuild_recreates_report_after_result_is_removed(tmp_path):
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps(replace(T3Protocol(), cold_seed_count=1).to_dict()))
    run = tmp_path / "run"
    assert main(["--mock", "--protocol-file", str(protocol), "--out-dir", str(run)]) == 0
    expected = (run / "result.json").read_bytes()
    budget = (run / "budget.json").read_bytes()
    oracle = {path.name: path.read_bytes() for path in (run / "private/oracle").glob("*.json")}
    first = rebuild(run, tmp_path / "rebuilt")
    assert set(first["matched"]) == {"result.json", "report.md", "trajectory.csv", "report_manifest.json"}
    with pytest.raises(ValueError, match="outside the source"):
        rebuild(run, run / "nested_rebuild")
    (run / "result.json").unlink()
    second = rebuild(run, tmp_path / "rebuilt_without_result")
    assert (tmp_path / "rebuilt_without_result/result.json").read_bytes() == expected
    assert "result.json" not in second["matched"]
    assert (run / "budget.json").read_bytes() == budget
    assert {path.name: path.read_bytes() for path in (run / "private/oracle").glob("*.json")} == oracle


def test_offline_rebuild_rejects_modified_oracle_receipt(tmp_path):
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps(replace(T3Protocol(), cold_seed_count=0).to_dict()))
    run = tmp_path / "run"
    assert main(["--mock", "--protocol-file", str(protocol), "--out-dir", str(run)]) == 0
    receipt_path = next(path for path in (run / "private/oracle").glob("*.json")
                        if json.loads(path.read_text())["identity"]["phase"] == "test")
    receipt = json.loads(receipt_path.read_text())
    receipt["response"]["metrics"]["ic"] += .1
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="receipt failed offline identity"):
        rebuild(run, tmp_path / "rejected")


def test_offline_rebuild_preserves_incomplete_quality_coverage(tmp_path, monkeypatch):
    monkeypatch.setattr(workflow, "mock_response", lambda request, native_count=None: json.dumps({
        "candidates": [{"name": "first", "expression": "Mean($close,5)"},
                       {"name": "tail", "expression": "Mean($close,8)"}]}))
    base = T3Protocol()
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps(replace(base, cold_seed_count=0, rounds=1, evaluations=1,
        batch_size=1, budgets={**base.budgets, "quality_checks": 0}).to_dict()))
    run = tmp_path / "run"
    assert main(["--mock", "--protocol-file", str(protocol), "--out-dir", str(run)]) == 0
    report = json.loads((run / "result.json").read_text())
    assert report["quality_audit"]["complete"] is False
    assert report["quality_audit"]["rows"][1]["reason"] == "budget_exhausted"
    assert len(rebuild(run, tmp_path / "rebuilt")["matched"]) == 4
