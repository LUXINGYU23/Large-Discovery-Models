from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from ldm_tts.cli.runner import build_plan, load_config
from ldm_tts.registration.experiment import load_experiment_contract
from tasks.nucleobench.core.cases import CaseCatalogError, load_case_catalog
from tasks.nucleobench.core.constants import CATALOG_PATH
from tasks.nucleobench.core.workflow import describe_ldm_task, resolve_provider_settings
from tasks.nucleobench.ldm_task.procedure import main, parse_args

TASK_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = TASK_ROOT.parents[1] / "config" / "nucleobench"


def test_catalog_covers_the_pinned_public_suite() -> None:
    cases = load_case_catalog()

    assert len(cases) == 17
    assert len({case.case_id for case in cases}) == 17
    assert Counter(case.model_family for case in cases) == {
        "malinois": 3,
        "bpnet": 12,
        "rinalmo": 1,
        "enformer": 1,
    }
    assert {case.state for case in cases} == {"planned"}


def test_experiment_contract_is_draft_with_both_termination_profiles() -> None:
    contract = load_experiment_contract(TASK_ROOT / "experiment.json")
    case_ids = {case.case_id for case in load_case_catalog()}

    assert contract.qualification == "draft"
    assert set(contract.evaluation["datasets"]) == case_ids
    assert set(contract.profiles) == {
        "pilot_evaluation_malinois_k562",
        "official_benchmark_malinois_k562",
    }
    assert (
        contract.profile("pilot_evaluation_malinois_k562").locked_args[
            "termination-kind"
        ]
        == "rounds"
    )
    assert (
        contract.profile("official_benchmark_malinois_k562").locked_args[
            "termination-kind"
        ]
        == "wall_time"
    )


def test_dry_run_describes_the_selected_case(capsys) -> None:
    assert parse_args(["--case-id", "malinois_k562", "--dry-run"]).dry_run
    assert main(["--case-id", "malinois_k562", "--dry-run"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["case"]["case_id"] == "malinois_k562"
    assert payload["ldm_task_spec"]["task"] == "nucleobench"
    assert payload["ldm_task_spec"]["candidate_domain"]["kind"] == (
        "nucleotide_mutation_patch"
    )
    assert payload["ldm_task_spec"]["response_spaces"][0]["name"] == (
        "mutation_patch_batch_json"
    )


def test_direct_method_contracts_match_the_required_request_shapes() -> None:
    ldm = describe_ldm_task(
        parse_args(
            [
                "--case-id",
                "malinois_k562",
                "--search-method",
                "ldm",
                "--evaluations-per-round",
                "4",
                "--proposal-max-workers",
                "3",
            ]
        )
    )
    direct = describe_ldm_task(
        parse_args(
            [
                "--case-id",
                "malinois_k562",
                "--search-method",
                "llm",
                "--evaluations-per-round",
                "4",
            ]
        )
    )

    assert ldm.reservoir.max_size == 16
    assert ldm.proposal_search.name == "parallel_independent_minibatch_requests"
    assert ldm.proposal_search.parameters == {
        "request_count": 4,
        "candidates_per_request": 4,
        "max_workers": 3,
    }
    assert ldm.response_spaces[0].schema["properties"]["candidates"]["minItems"] == 4
    assert direct.reservoir.max_size == 4
    assert (
        direct.proposal_search.name == "parallel_independent_single_candidate_requests"
    )
    assert direct.proposal_search.parameters["request_count"] == 4
    assert direct.surrogate.kind == "none"


def test_provider_configuration_is_user_defined_and_secret_free(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    key_file = tmp_path / "api_key"
    key_file.write_text("test-secret", encoding="utf-8")
    monkeypatch.setenv("LLM_BASE_URL", "https://provider.example")
    monkeypatch.setenv("LLM_MODEL_NAME", "model-name")
    args = parse_args(
        [
            "--case-id",
            "malinois_k562",
            "--dry-run",
            "--llm-wire-api",
            "responses",
            "--llm-reasoning",
            "max",
            "--api-key-file",
            str(key_file),
        ]
    )

    provider = resolve_provider_settings(args)
    assert provider.api_key == "test-secret"
    assert (
        main(
            [
                "--case-id",
                "malinois_k562",
                "--dry-run",
                "--llm-wire-api",
                "responses",
                "--llm-reasoning",
                "max",
                "--api-key-file",
                str(key_file),
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    payload = json.loads(output)

    assert payload["proposal_provider"] == {
        "configured": True,
        "reasoning": "max",
        "required": True,
        "wire_api": "responses",
    }
    assert "test-secret" not in output
    assert str(key_file) not in output


def test_bo_dry_run_does_not_require_or_read_provider_credentials(capsys) -> None:
    assert (
        main(
            [
                "--case-id",
                "malinois_k562",
                "--search-method",
                "bo",
                "--api-key-file",
                "missing-secret-file",
                "--dry-run",
            ]
        )
        == 0
    )

    assert json.loads(capsys.readouterr().out)["proposal_provider"] == {
        "required": False
    }

    assert (
        main(
            [
                "--mock",
                "--case-id",
                "mock_dna",
                "--api-key-file",
                "missing-secret-file",
                "--dry-run",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["proposal_provider"] == {
        "required": False
    }


def test_execution_is_rejected_before_qualification() -> None:
    with pytest.raises(SystemExit, match="not qualified"):
        main(["--case-id", "malinois_k562"])


def test_mock_config_uses_the_registered_shared_runner() -> None:
    config_path = CONFIG_ROOT / "mock.yaml"
    config = load_config(config_path)
    plan = build_plan(config, config_path)

    assert config["mode"] == "mock"
    assert config["args"] == {
        "mock": True,
        "case-id": "mock_dna",
        "iterations": 2,
        "reservoir-size": 4,
        "evaluations-per-round": 1,
        "out-dir": "runs",
    }
    assert plan["module"] == "tasks.nucleobench.ldm_task.procedure"
    assert "--mock" in plan["argv"]


def test_mock_campaign_writes_the_complete_shared_engine_artifacts(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setenv("LDM_DATA_COLLECTION_ENABLED", "1")
    assert (
        main(
            [
                "--mock",
                "--case-id",
                "mock_dna",
                "--iterations",
                "2",
                "--reservoir-size",
                "4",
                "--evaluations-per-round",
                "1",
                "--out-dir",
                str(tmp_path),
                "--run-name",
                "campaign",
            ]
        )
        == 0
    )

    output = json.loads(capsys.readouterr().out)
    run_dir = Path(output["run_dir"])
    assert output["ldm_task_spec"]["response_spaces"][0]["name"] == (
        "mutation_patch_json"
    )
    assert "model_requests_per_round" not in output["ldm_task_spec"]["metadata"]
    expected = {
        "campaign.json",
        "config.json",
        "ldm_task_spec.json",
        "experiment_contract.json",
        "budget.json",
        "status.json",
        "events.jsonl",
        "checkpoint.json",
        "summary.json",
        "result.json",
        "trajectory.csv",
    }
    assert expected.issubset({path.name for path in run_dir.iterdir()})

    result = json.loads((run_dir / "result.json").read_text())
    summary = json.loads((run_dir / "summary.json").read_text())
    status = json.loads((run_dir / "status.json").read_text())
    budget = json.loads((run_dir / "budget.json").read_text())
    contract = json.loads((run_dir / "experiment_contract.json").read_text())
    events = [
        json.loads(line) for line in (run_dir / "events.jsonl").read_text().splitlines()
    ]

    assert result["finished"] is True
    assert result["evaluation_count"] == 2
    assert summary["observation_count"] == 2
    assert summary["rounds_run"] == 2
    assert status["status"] == "completed"
    assert budget["counters"]["external_evaluations"] == 2
    assert contract["snapshot"]["sha256"] == output["contract_sha256"]
    assert [
        event["payload"]["metadata"]["phase"]
        for event in events
        if event["event_type"] == "reservoir_expanded"
    ] == ["shared_initialization", "active_search"]
    assert sum(event["event_type"] == "candidate_evaluated" for event in events) == 2
    assert len((run_dir / "trajectory.csv").read_text().splitlines()) == 3

    ir_path = run_dir / "ldm_data" / "ldm_ir.jsonl"
    assert ir_path.is_file()
    assert "AAAAAAAA" not in ir_path.read_text()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("state", "enabled", "Unknown state"),
        ("model_family", "unknown", "Unknown model_family"),
        ("max_seconds", 1, "Incorrect max_seconds"),
        ("sequence_length", 201, "Incorrect sequence_length"),
    ],
)
def test_catalog_rejects_protocol_drift(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    payload = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    payload["cases"][0][field] = value
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(CaseCatalogError, match=message):
        load_case_catalog(path)
