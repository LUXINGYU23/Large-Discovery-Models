"""Configuration contracts for task-neutral pilot evaluations."""

from __future__ import annotations

from pathlib import Path

from ldm_tts.pilot_evaluation.config import load_pilot_evaluation_spec


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_iron_mind_matrix_expands_to_the_planned_two_case_design(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("IRON_MIND_RUNS_ROOT", str(tmp_path / "iron_runs"))

    spec = load_pilot_evaluation_spec(REPO_ROOT / "config" / "pilot_evaluation" / "iron_mind.yaml")

    assert spec.task == "iron_mind"
    assert spec.methods == ("ldm", "bo", "llm")
    assert len(spec.cases) == 2
    assert spec.seeds == (0, 1, 2)
    assert spec.iterations == 6
    assert spec.method_overrides["llm"] == (
        'contract_profile="pilot_evaluation_direct_llm"',
        'args.prompt-policy="direct_v1"',
    )


def test_synthonbench_matrix_declares_batch_trajectory_mapping(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("SYNTHONBENCH_RUNS_ROOT", str(tmp_path / "syn_runs"))

    spec = load_pilot_evaluation_spec(REPO_ROOT / "config" / "pilot_evaluation" / "synthonbench.yaml")

    assert spec.trajectory.step_kind == "evaluation_index"
    assert spec.methods == ("ldm", "harness", "bo", "llm")
    assert spec.result_fields["best_found_utility"] == "best_found_utility"
    assert spec.method_overrides["llm"][-2:] == (
        "args.proposal-samples=16",
        "args.proposal-max-workers=4",
    )
    assert spec.method_overrides["harness"][0] == 'contract_profile="pilot_evaluation_harness"'


def test_extended_matrices_use_the_separate_twelve_round_profiles(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("IRON_MIND_RUNS_ROOT", str(tmp_path / "iron_runs"))
    monkeypatch.setenv("SYNTHONBENCH_RUNS_ROOT", str(tmp_path / "syn_runs"))

    iron = load_pilot_evaluation_spec(REPO_ROOT / "config" / "pilot_evaluation" / "iron_mind_extended.yaml")
    synthon = load_pilot_evaluation_spec(REPO_ROOT / "config" / "pilot_evaluation" / "synthonbench_extended.yaml")

    assert iron.iterations == synthon.iterations == 12
    assert iron.methods == ("ldm", "bo", "llm")
    assert synthon.methods == ("ldm", "harness", "bo", "llm")
    assert iron.method_overrides["llm"][0] == 'contract_profile="pilot_evaluation_extended_direct_llm"'
    assert synthon.method_overrides["llm"][0] == 'contract_profile="pilot_evaluation_extended_direct_llm"'
    assert synthon.method_overrides["llm"][-1] == "args.proposal-max-workers=4"
    assert synthon.method_overrides["harness"][0] == 'contract_profile="pilot_evaluation_extended_harness"'
