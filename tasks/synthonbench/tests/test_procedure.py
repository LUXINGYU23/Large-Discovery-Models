"""Workflow-level checks for the SynthonBench LDM task."""

from __future__ import annotations

import json
from pathlib import Path

from ldm_tts.registration.experiment import (
    load_experiment_contract,
    validate_profile_args,
)
from tasks.synthonbench.core.workflow import describe_ldm_task, main, parse_args

TASK_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = TASK_ROOT.parents[1]


def test_task_spec_declares_independent_oversampled_proposals() -> None:
    spec = describe_ldm_task(parse_args(["--mock"]))

    assert spec.task == "synthonbench"
    assert spec.candidate_domain.kind == "reaction_synthon_tuple"
    assert spec.reservoir.max_size == 64
    assert spec.proposal_search.name == "parallel_independent_requests"
    assert spec.proposal_search.breadth == 64
    assert spec.proposal_search.parameters["one_candidate_per_request"] is True
    assert spec.metadata["bo_pool_size"] == 32
    assert spec.acquisition.name == "ucb_tilted"
    assert spec.acquisition.parameters["pool_size"] == 32
    assert spec.acquisition.parameters["proposal_sample_count"] == 64
    assert spec.surrogate.dimension == 257
    assert spec.surrogate.metadata["kernel"] == "count_tanimoto"
    assert spec.surrogate.metadata["landmark_count"] == 256
    assert spec.surrogate.metadata["reaction_weight"] == 1.0
    assert spec.acquisition.parameters["eta_acquisition_tilt"] == 1.0
    assert spec.acquisition.parameters["base_acquisition_parameters"]["surrogate"] == (
        "online_nystrom_fitc_count_tanimoto_gaussian_process"
    )


def test_proposal_defaults_disable_thinking() -> None:
    args = parse_args(["--mock"])

    assert args.llm_extra_body_json == '{"thinking":{"type":"disabled"}}'
    assert args.llm_max_tokens == 256


def test_quick_compare_arguments_expose_a_task_local_bo_pool_size() -> None:
    args = parse_args(["--mock", "--search-method", "bo", "--proposal-mode", "none"])

    assert args.bo_search_samples == 64


def test_mock_campaign_uses_official_example_task(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("LDM_DATA_COLLECTION_ENABLED", "1")

    assert main([
        "--mock",
        "--iterations", "1",
        "--proposal-samples", "8",
        "--bo-pool-size", "4",
        "--slate-size", "4",
        "--out-dir", str(tmp_path),
        "--run-name", "official_example",
    ]) == 0

    payload = json.loads(capsys.readouterr().out)
    run_dir = Path(payload["run_dir"])
    result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    budget = json.loads((run_dir / "budget.json").read_text(encoding="utf-8"))

    assert result["mode"] == "mock"
    assert result["official_calls"] == 1
    assert result["official_metrics"]["submitted_calls"] == 1.0
    assert budget["counters"]["proposal_attempts"] == 8
    assert budget["counters"]["benchmark_jobs"] == 1
    for filename in (
        "submission.csv",
        "trajectory.csv",
        "search_manifest.json",
        "selection_record.json",
        "evaluation_manifest.json",
        "ldm_data/ldm_ir.jsonl",
    ):
        assert (run_dir / filename).is_file()


def test_real_profiles_lock_the_scientific_method_arguments() -> None:
    contract = load_experiment_contract(TASK_ROOT / "experiment.json")
    required = {
        "proposal-samples", "bo-pool-size", "proposal-max-workers", "fingerprint-bits",
        "gp-landmarks", "gp-kernel-jitter", "gp-signal-std", "gp-mean-std",
        "gp-observation-noise-std", "gp-reaction-weight", "acquisition-beta", "alpha", "eta", "z-clip",
        "llm-max-tokens", "llm-temperature", "llm-extra-body-json",
    }

    for profile in contract.profiles.values():
        assert required <= set(profile.locked_args)
        filename = "quick_compare_base.yaml" if profile.name == "quick_compare" else f"{profile.name}.yaml"
        config_path = REPO_ROOT / "config" / "synthonbench" / filename
        config = _load_yaml(config_path)
        validate_profile_args(contract, profile.name, config["args"])


def test_qualification_record_covers_the_source_pinned_real_tracks() -> None:
    evidence = _load_json(TASK_ROOT / "resources" / "qualification_evidence.json")
    record = _load_json(TASK_ROOT / "resources" / "verification_record.json")

    assert evidence["stage"] == "tiny_campaign_verified"
    assert evidence["gates"]["tiny_campaign_verified"]["status"] == "passed"
    assert record["method"]["algorithm"] == "ldm_tilted_synthon_tanimoto_gp_ucb"
    assert record["method"]["surrogate"] == "online_nystrom_fitc_count_tanimoto_gaussian_process"
    assert record["method"]["proposal_request_defaults"]["llm_extra_body_json"] == (
        '{"thinking":{"type":"disabled"}}'
    )
    assert record["surrogate_1m_qualification"]["status"] == "succeeded"
    assert record["glide_1m_qualification"]["status"] == "succeeded"


def _load_yaml(path: Path) -> dict[str, object]:
    import yaml

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload
