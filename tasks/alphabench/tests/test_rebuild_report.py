from dataclasses import replace
import json

import pytest

from tasks.alphabench.core.protocol import T3Protocol, digest
from tasks.alphabench.core.data import sha256, verify_data_manifest
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


def test_offline_rebuild_preserves_verified_partial_data_audit(tmp_path):
    data_root = tmp_path / "market"
    for relative in ("calendars/day.txt", "instruments/csi300.txt", "instruments/all.txt",
                     "features/stock/close.day.bin"):
        target = data_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"DATA")
    files = {"features/stock/close.day.bin": sha256(data_root / "features/stock/close.day.bin")}
    manifest = {"backend": "qlib", "market": "csi300", "qualification": "blocked",
                "issues": [{"code": "unresolved_price_gaps"}],
                "gaps": [{"missing_sessions": 7}],
                "instrument_interval_conflicts": [{"instrument": "S00"}],
                "data_root": str(data_root), "source": "fixture", "archive_sha256": "a" * 64,
                "files_sha256": digest(files), "calendar_sha256": sha256(data_root / "calendars/day.txt"),
                "universe_sha256": sha256(data_root / "instruments/csi300.txt"),
                "all_instruments_sha256": sha256(data_root / "instruments/all.txt"),
                "adjustment": "split", "fields": ["close"], "start": "2015-01-01",
                "end": "2025-02-01", "benchmark": "SH000300", "historical_universe": True}
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    manifest_path.with_suffix(".files.json").write_text(json.dumps(files))
    protocol = replace(T3Protocol(), cold_seed_count=0, data_digest=digest(manifest),
                       data_policy="partial_comparison")
    assert verify_data_manifest(manifest_path, protocol) == manifest
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol.to_dict()))
    run = tmp_path / "run"
    assert main(["--mock", "--protocol-file", str(protocol_path), "--out-dir", str(run)]) == 0
    config_path = run / "config.json"
    config = json.loads(config_path.read_text())
    config_path.write_text(json.dumps(config | {"mock": False}))
    (run / "data_manifest.json").write_text(json.dumps(manifest))
    for name in ("result.json", "report.md", "trajectory.csv", "report_manifest.json"):
        (run / name).unlink()
    rebuilt = tmp_path / "rebuilt"
    rebuild(run, rebuilt)
    data = json.loads((rebuilt / "result.json").read_text())["completeness"]["data"]
    assert data["source_qualification"] == "blocked"
    assert data["issue_codes"] == ["unresolved_price_gaps"]
    assert data["unresolved_sessions"] == 7
    assert data["instrument_interval_conflicts"] == 1
    (run / "data_manifest.json").unlink()
    reconstructed = tmp_path / "reconstructed-from-manifest"
    rebuild(run, reconstructed, data_manifest_path=manifest_path)
    for name in ("result.json", "report.md", "trajectory.csv", "report_manifest.json"):
        assert (rebuilt / name).read_bytes() == (reconstructed / name).read_bytes()
