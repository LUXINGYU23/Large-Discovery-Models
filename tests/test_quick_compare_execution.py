"""End-to-end matrix execution through the shared runner on an Iron Mind mock."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from ldm_tts.quick_compare.config import load_quick_compare_spec
from ldm_tts.quick_compare.execution import run_comparison


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_mock_matrix_runs_all_three_methods_and_writes_reports(tmp_path: Path) -> None:
    base_path = tmp_path / "base.yaml"
    quick_path = tmp_path / "quick.yaml"
    output_root = tmp_path / "comparison"
    _write_yaml(base_path, _base_config())
    _write_yaml(quick_path, _quick_config(base_path, output_root))

    assert run_comparison(load_quick_compare_spec(quick_path), resume=False, dry_run=False) == 0

    manifest = _json(output_root / "comparison_manifest.json")
    summary = _json(output_root / "summary.json")
    assert manifest["state"] == "completed"
    assert manifest["integrity"]["valid"] is True
    assert len(manifest["runs"]) == 9
    assert summary["cases"][0]["verdict"] in {"promising", "mixed", "not_promising"}
    assert (output_root / "best_so_far.png").is_file()


def test_partial_matrix_can_resume_without_publishing_reports(tmp_path: Path) -> None:
    base_path = tmp_path / "base.yaml"
    quick_path = tmp_path / "quick.yaml"
    output_root = tmp_path / "comparison"
    _write_yaml(base_path, _base_config())
    _write_yaml(quick_path, _quick_config(base_path, output_root))
    spec = load_quick_compare_spec(quick_path)

    assert run_comparison(spec, resume=False, dry_run=False, methods=("bo",)) == 0

    manifest = _json(output_root / "comparison_manifest.json")
    assert manifest["state"] == "partial"
    assert set(manifest["runs"]) == {"mock/bo/seed_0", "mock/bo/seed_1", "mock/bo/seed_2"}
    assert not (output_root / "summary.json").exists()

    assert run_comparison(spec, resume=True, dry_run=False) == 0
    assert _json(output_root / "comparison_manifest.json")["state"] == "completed"


def test_matrix_rejects_a_task_label_that_differs_from_its_base_config(tmp_path: Path) -> None:
    base_path = tmp_path / "base.yaml"
    quick_path = tmp_path / "quick.yaml"
    _write_yaml(base_path, _base_config())
    payload = _quick_config(base_path, tmp_path / "comparison")
    payload["task"] = "synthonbench"
    _write_yaml(quick_path, payload)

    with pytest.raises(ValueError, match="must match base config task"):
        run_comparison(load_quick_compare_spec(quick_path), resume=False, dry_run=True)


def _base_config() -> dict[str, object]:
    return {
        "name": "quick_mock", "task": "iron_mind", "algorithm": "test", "mode": "mock",
        "args": {
            "mock": True, "proposal-mode": "callable", "dataset-id": "buchwald_hartwig",
            "iterations": 2, "proposal-samples": 64, "bo-pool-size": 32,
            "proposal-max-workers": 8, "evaluations-per-round": 1,
            "acquisition-beta": 1.0, "alpha": 1.0, "eta": 3.0, "z-clip": 5.0,
            "prompt-policy": "portfolio_v1",
        },
    }


def _quick_config(base_path: Path, output_root: Path) -> dict[str, object]:
    return {
        "schema_version": 1, "name": "quick_mock", "task": "iron_mind",
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
