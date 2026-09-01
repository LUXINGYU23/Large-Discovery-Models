from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from ldm_tts.registration.experiment import load_experiment_contract
from tasks.nucleobench.core.cases import CaseCatalogError, load_case_catalog
from tasks.nucleobench.core.constants import CATALOG_PATH
from tasks.nucleobench.ldm_task.procedure import main, parse_args


TASK_ROOT = Path(__file__).resolve().parents[1]


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
    assert contract.profile("pilot_evaluation_malinois_k562").locked_args[
        "termination-kind"
    ] == "rounds"
    assert contract.profile("official_benchmark_malinois_k562").locked_args[
        "termination-kind"
    ] == "wall_time"


def test_dry_run_describes_the_selected_case(capsys) -> None:
    assert parse_args(["--case-id", "malinois_k562", "--dry-run"]).dry_run
    assert main(["--case-id", "malinois_k562", "--dry-run"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["case"]["case_id"] == "malinois_k562"
    assert payload["ldm_task_spec"]["task"] == "nucleobench"
    assert payload["ldm_task_spec"]["candidate_domain"]["kind"] == (
        "nucleotide_mutation_patch"
    )


def test_execution_is_rejected_before_qualification() -> None:
    with pytest.raises(SystemExit, match="not qualified"):
        main(["--case-id", "malinois_k562"])


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
