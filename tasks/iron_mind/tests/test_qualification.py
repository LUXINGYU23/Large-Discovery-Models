"""Contracts for compact, source-traceable Iron Mind qualification records."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tasks.iron_mind.core.qualification import (
    QualificationRecordError,
    build_mock_verification_record,
    build_registered_record,
)
from tasks.iron_mind.core.workflow import main


TASK_ROOT = Path(__file__).resolve().parents[1]
MOCK_CONFIG = TASK_ROOT.parents[1] / "config" / "iron_mind" / "mock.yaml"


def test_registered_record_pins_the_registered_task_and_contract() -> None:
    record = build_registered_record()

    assert record["status"] == "passed"
    assert record["task"] == "iron_mind"
    assert record["task_manifest"]["path"] == "tasks/iron_mind/task.json"
    assert record["experiment_contract"]["path"] == "tasks/iron_mind/experiment.json"


def test_mock_record_accepts_a_complete_exact_completed_campaign(
    tmp_path: Path, monkeypatch
) -> None:
    run_dir = _completed_mock_run(tmp_path, monkeypatch)
    record = build_mock_verification_record(run_dir=run_dir, config_path=MOCK_CONFIG)

    assert record["status"] == "passed"
    assert record["budget"]["counters"]["valid_search_candidates"] == 4
    assert record["collection"] == {"ir_rows": 4, "sft_rows": 4}
    assert "summary.json" in record["artifacts"]


@pytest.mark.parametrize(
    "corruption",
    ("missing_summary", "failed_status", "wrong_budget", "unknown_contract"),
)
def test_mock_record_rejects_missing_or_inconsistent_campaign_artifacts(
    corruption: str, tmp_path: Path, monkeypatch
) -> None:
    run_dir = _completed_mock_run(tmp_path, monkeypatch)
    _corrupt_run(run_dir, corruption)

    with pytest.raises(QualificationRecordError):
        build_mock_verification_record(run_dir=run_dir, config_path=MOCK_CONFIG)


def _completed_mock_run(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.setenv("LDM_DATA_COLLECTION_ENABLED", "1")
    out_dir = tmp_path / "runs"
    assert main(["--mock", "--out-dir", str(out_dir)]) == 0
    return out_dir / "mock"


def _corrupt_run(run_dir: Path, corruption: str) -> None:
    if corruption == "missing_summary":
        (run_dir / "summary.json").unlink()
        return
    path, field, value = _corruption_target(corruption)
    artifact = run_dir / path
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    if field == "selected_candidates":
        payload["counters"][field] = value
    else:
        payload[field] = value
    artifact.write_text(json.dumps(payload), encoding="utf-8")


def _corruption_target(corruption: str) -> tuple[str, str, object]:
    targets = {
        "failed_status": ("status.json", "status", "failed"),
        "wrong_budget": ("budget.json", "selected_candidates", 0),
        "unknown_contract": ("campaign.json", "contract_sha256", "unknown"),
    }
    try:
        return targets[corruption]
    except KeyError as exc:
        raise AssertionError(f"Unknown corruption: {corruption}") from exc
