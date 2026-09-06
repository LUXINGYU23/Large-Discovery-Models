from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from ldm_tts.cli.runner import build_plan, load_config
from ldm_tts.registration.dependencies import has_failures
from ldm_tts.registration.experiment import load_experiment_contract
from tasks.nucleobench.core import dependencies
from tasks.nucleobench.core.cases import CaseCatalogError, load_case_catalog
from tasks.nucleobench.core.constants import CATALOG_PATH
from tasks.nucleobench.core.workflow import describe_ldm_task, resolve_provider_settings
from tasks.nucleobench.ldm_task.procedure import main, parse_args

TASK_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = TASK_ROOT.parents[1] / "config" / "nucleobench"


def test_real_dependency_check_requires_the_pinned_official_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = {"task": "nucleobench", "mode": "real", "argv": []}

    monkeypatch.setattr(
        dependencies.metadata,
        "version",
        lambda _name: dependencies.UPSTREAM_PACKAGE_VERSION,
    )
    checks = dependencies.check_task_dependencies(plan)
    assert not has_failures(checks)

    def missing(_name: str) -> str:
        raise dependencies.metadata.PackageNotFoundError

    monkeypatch.setattr(dependencies.metadata, "version", missing)
    checks = dependencies.check_task_dependencies(plan)
    assert has_failures(checks)
    assert checks[-1].name == "official runtime"


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
    states = {case.case_id: case.state for case in cases}
    assert states["malinois_k562"] == "qualified"
    assert states["rinalmo_mrl"] == "planned"
    assert set(states.values()) == {"planned", "prepared", "qualified"}
    assert sum(state == "prepared" for state in states.values()) == 15

    upstream = json.loads(
        (TASK_ROOT / "resources" / "upstream_contract.json").read_text(encoding="utf-8")
    )
    assert set(upstream["case_preparation"]) == set(states)
    assert all(
        "start_set_sha256" in upstream["case_preparation"][case_id]
        for case_id, state in states.items()
        if state in {"prepared", "qualified"}
    )
    assert "start_set_sha256" not in upstream["case_preparation"]["rinalmo_mrl"]


def test_experiment_contract_is_qualified_with_both_termination_profiles() -> None:
    contract = load_experiment_contract(TASK_ROOT / "experiment.json")
    case_ids = {case.case_id for case in load_case_catalog()}

    assert contract.qualification == "qualified"
    assert set(contract.evaluation["datasets"]) == case_ids
    assert set(contract.profiles) == {"pilot_evaluation", "official_benchmark"}
    assert (
        contract.profile("pilot_evaluation").locked_args["termination-kind"] == "rounds"
    )
    assert (
        contract.profile("official_benchmark").locked_args["termination-kind"]
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
    assert payload["execution"] == {
        "active_optimization_rounds": 1,
        "benchmark_comparable": False,
        "execution_profile": "qualification",
        "hardware_profile": "n1-highmem-16-cpu",
        "initialization_evaluations": 1,
        "termination_kind": "rounds",
        "total_rounds": 2,
    }


def test_every_declared_case_builds_the_same_dry_run_workflow(capsys) -> None:
    for case in load_case_catalog():
        assert main(["--case-id", case.case_id, "--dry-run"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["case_id"] == case.case_id
        assert payload["ldm_task_spec"]["metadata"]["model_family"] == (
            case.model_family
        )
        assert payload["ldm_task_spec"]["metadata"]["case_state"] == case.state


def test_unqualified_case_cannot_run_pilot_or_official_execution() -> None:
    with pytest.raises(SystemExit, match="bpnet_atac.*not qualified"):
        main(
            [
                "--case-id",
                "bpnet_atac",
                "--execution-profile",
                "pilot_evaluation",
                "--termination-kind",
                "rounds",
                "--iterations",
                "12",
                "--initialization-mode",
                "shared_start",
            ]
        )


def test_direct_method_contracts_match_the_required_request_shapes() -> None:
    ldm_args = parse_args(
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
    ldm = describe_ldm_task(ldm_args)
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
    assert ldm_args.alpha == 2.0
    assert ldm_args.eta == 0.25
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


@pytest.mark.parametrize(
    "argv",
    [
        [
            "--execution-profile",
            "official_benchmark",
            "--termination-kind",
            "wall_time",
            "--max-seconds",
            "28800",
            "--iterations",
            "12",
            "--initialization-mode",
            "official_start",
        ],
        [
            "--execution-profile",
            "pilot_evaluation",
            "--termination-kind",
            "rounds",
            "--iterations",
            "12",
            "--max-seconds",
            "28800",
            "--initialization-mode",
            "shared_start",
        ],
        [
            "--execution-profile",
            "pilot_evaluation",
            "--termination-kind",
            "wall_time",
            "--max-seconds",
            "28800",
            "--initialization-mode",
            "shared_start",
        ],
    ],
)
def test_execution_profiles_reject_mixed_or_mismatched_termination(
    argv: list[str],
) -> None:
    with pytest.raises(SystemExit):
        parse_args(argv)


def test_release_configs_use_one_real_workflow_and_distinct_termination_modes() -> None:
    names = {
        "malinois_k562_seed_qualification.yaml": ("qualification", "rounds"),
        "malinois_k562_tiny_campaign.yaml": ("qualification", "rounds"),
        "malinois_k562_pilot_base.yaml": ("pilot_evaluation", "rounds"),
        "malinois_k562_official_base.yaml": ("official_benchmark", "wall_time"),
    }

    for filename, expected in names.items():
        config_path = CONFIG_ROOT / filename
        config = load_config(config_path)
        plan = build_plan(config, config_path)
        assert plan["module"] == "tasks.nucleobench.ldm_task.procedure"
        assert config["args"]["execution-profile"] == expected[0]
        assert config["args"]["termination-kind"] == expected[1]
        assert config["env"]["CUDA_VISIBLE_DEVICES"] == ""

    official = load_config(CONFIG_ROOT / "malinois_k562_official_base.yaml")
    assert "iterations" not in official["args"]
    assert official["args"]["max-seconds"] == 28800


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
