"""End-to-end matrix execution through the shared runner on an Iron Mind mock."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from ldm_tts.pilot_evaluation.config import load_pilot_evaluation_spec
from ldm_tts.pilot_evaluation.execution import run_evaluation


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_mock_matrix_runs_all_three_methods_and_writes_reports(tmp_path: Path) -> None:
    base_path = tmp_path / "base.yaml"
    evaluation_path = tmp_path / "evaluation.yaml"
    output_root = tmp_path / "evaluation"
    _write_yaml(base_path, _base_config())
    _write_yaml(evaluation_path, _evaluation_config(base_path, output_root))

    assert run_evaluation(load_pilot_evaluation_spec(evaluation_path), resume=False, dry_run=False) == 0

    manifest = _json(output_root / "evaluation_manifest.json")
    summary = _json(output_root / "summary.json")
    assert manifest["state"] == "completed"
    assert manifest["integrity"]["valid"] is True
    assert len(manifest["runs"]) == 9
    assert summary["cases"][0]["verdict"] in {"promising", "mixed", "not_promising"}
    assert (output_root / "best_so_far.png").is_file()


def test_partial_matrix_can_resume_without_publishing_reports(tmp_path: Path) -> None:
    base_path = tmp_path / "base.yaml"
    evaluation_path = tmp_path / "evaluation.yaml"
    output_root = tmp_path / "evaluation"
    _write_yaml(base_path, _base_config())
    _write_yaml(evaluation_path, _evaluation_config(base_path, output_root))
    spec = load_pilot_evaluation_spec(evaluation_path)

    assert run_evaluation(spec, resume=False, dry_run=False, methods=("bo",)) == 0

    manifest = _json(output_root / "evaluation_manifest.json")
    assert manifest["state"] == "partial"
    assert set(manifest["runs"]) == {"mock/bo/seed_0", "mock/bo/seed_1", "mock/bo/seed_2"}
    assert not (output_root / "summary.json").exists()

    assert run_evaluation(spec, resume=True, dry_run=False) == 0
    assert _json(output_root / "evaluation_manifest.json")["state"] == "completed"


def test_reporting_failure_marks_manifest_failed(tmp_path: Path, monkeypatch) -> None:
    base_path = tmp_path / "base.yaml"
    evaluation_path = tmp_path / "evaluation.yaml"
    output_root = tmp_path / "evaluation"
    _write_yaml(base_path, _base_config())
    _write_yaml(evaluation_path, _evaluation_config(base_path, output_root))

    def fail_reporting(*_args) -> None:
        raise ValueError("invalid evaluation trajectory")

    monkeypatch.setattr(
        "ldm_tts.pilot_evaluation.execution.write_evaluation_reports",
        fail_reporting,
    )

    with pytest.raises(ValueError, match="invalid evaluation trajectory"):
        run_evaluation(load_pilot_evaluation_spec(evaluation_path), resume=False, dry_run=False)

    manifest = _json(output_root / "evaluation_manifest.json")
    assert manifest["state"] == "failed"
    assert manifest["error"] == {
        "stage": "reporting",
        "type": "ValueError",
        "message": "invalid evaluation trajectory",
    }


def test_matrix_rejects_a_task_label_that_differs_from_its_base_config(tmp_path: Path) -> None:
    base_path = tmp_path / "base.yaml"
    evaluation_path = tmp_path / "evaluation.yaml"
    _write_yaml(base_path, _base_config())
    payload = _evaluation_config(base_path, tmp_path / "evaluation")
    payload["task"] = "synthonbench"
    _write_yaml(evaluation_path, payload)

    with pytest.raises(ValueError, match="must match base config task"):
        run_evaluation(load_pilot_evaluation_spec(evaluation_path), resume=False, dry_run=True)


def _base_config() -> dict[str, object]:
    return {
        "name": "pilot_mock", "task": "iron_mind", "algorithm": "test", "mode": "mock",
        "args": {
            "mock": True, "proposal-mode": "callable", "dataset-id": "buchwald_hartwig",
            "iterations": 2, "proposal-samples": 64, "bo-pool-size": 32,
            "proposal-max-workers": 8, "evaluations-per-round": 1,
            "acquisition-beta": 1.0, "alpha": 1.0, "eta": 3.0, "z-clip": 5.0,
            "prompt-policy": "portfolio_v1",
        },
    }


def _evaluation_config(base_path: Path, output_root: Path) -> dict[str, object]:
    return {
        "schema_version": 1, "name": "pilot_mock", "task": "iron_mind",
        "base_config": str(base_path),
        "cases": [{"id": "mock", "overrides": []}],
        "methods": ["ldm", "bo", "llm"], "seeds": [0, 1, 2],
        "method_overrides": {"ldm": [], "bo": [], "llm": []},
        "optimization_rounds": 1, "initialization_mode": "shared_random",
        "output_root": str(output_root),
        "trajectory": {
            "step_column": "round", "step_kind": "round",
            "objective_column": "reaction_score", "direction": "maximize",
        },
        "result_fields": {},
    }


def _write_yaml(path: Path, payload: dict[str, object]) -> None:
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


def _json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload
