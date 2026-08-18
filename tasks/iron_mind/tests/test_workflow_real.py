"""Real-mode workflow wiring without calling an external model endpoint."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from ldm_tts.transport import ProposalResponse
from ldm_tts.registration.experiment import (
    ACTIVE_CONTRACT_PATH_ENV,
    ACTIVE_CONTRACT_PROFILE_ENV,
)

from tasks.iron_mind.core import workflow
from tasks.iron_mind.core.workflow import main, parse_args


class StaticEndpoint:
    def __init__(self, table) -> None:
        self.table = table

    def preflight(self):
        return {"model": "test-model"}

    def propose(self, _request):
        return ProposalResponse(text=workflow._mock_response(self.table))


def test_real_mode_uses_the_pinned_loader_and_endpoint_client(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    table = workflow._load_mock_table(workflow._schema_for("buchwald_hartwig"))
    calls: dict[str, object] = {}
    _install_real_fakes(monkeypatch, table, calls)
    _activate_real_tiny(monkeypatch)
    code = main(_real_args(tmp_path))

    payload = json.loads(capsys.readouterr().out)
    run_dir = Path(payload["run_dir"])
    campaign = json.loads((run_dir / "campaign.json").read_text(encoding="utf-8"))
    snapshot = json.loads(
        (run_dir / "experiment_contract.json").read_text(encoding="utf-8")
    )
    assert code == 0
    assert calls["dataset_id"] == "buchwald_hartwig"
    assert calls["data_root"] == tmp_path / "pinned"
    assert calls["client"]["base_url"] == "https://example.invalid/v1"
    assert payload["engine_summary"]["successful_evaluation_count"] == 1
    assert campaign["contract_profile"] == "real_tiny"
    assert snapshot["snapshot"]["profile"] == "real_tiny"


def _install_real_fakes(monkeypatch, table, calls: dict[str, object]) -> None:
    def load_table(*, dataset_id: str, data_root: Path):
        calls["dataset_id"] = dataset_id
        calls["data_root"] = data_root
        return table

    def build_client(**kwargs):
        calls["client"] = kwargs
        return StaticEndpoint(table)

    monkeypatch.setenv("LDM_LLM_API_KEY", "test-workflow-key")
    monkeypatch.setattr(workflow, "load_pinned_reaction_table", load_table, raising=False)
    monkeypatch.setattr(workflow, "build_deepseek_reaction_client", build_client)


def _activate_real_tiny(monkeypatch) -> None:
    task_root = Path(__file__).resolve().parents[1]
    monkeypatch.setenv(ACTIVE_CONTRACT_PATH_ENV, str(task_root / "experiment.json"))
    monkeypatch.setenv(ACTIVE_CONTRACT_PROFILE_ENV, "real_tiny")


def _real_args(tmp_path: Path) -> list[str]:
    return [
        "--proposal-mode", "openai", "--dataset-id", "buchwald_hartwig",
        "--data-dir", str(tmp_path / "pinned"), "--llm-url", "https://example.invalid/v1",
        "--llm-model-name", "test-model", "--out-dir", str(tmp_path),
        "--run-name", "real-wiring",
    ]


def test_parse_args_accepts_every_operational_workflow_flag(tmp_path: Path) -> None:
    args = parse_args(
        [
            "--mock",
            "--proposal-mode",
            "callable",
            "--dataset-id",
            "buchwald_hartwig",
            "--iterations",
            "1",
            "--reservoir-size",
            "4",
            "--evaluations-per-round",
            "1",
            "--out-dir",
            str(tmp_path / "runs"),
            "--run-name",
            "smoke",
            "--resume-from",
            str(tmp_path / "resume"),
            "--data-dir",
            str(tmp_path / "data"),
            "--llm-url",
            "https://example.invalid/v1",
            "--llm-model-name",
            "test-model",
            "--llm-timeout",
            "12.5",
            "--llm-max-tokens",
            "512",
            "--acquisition-beta",
            "1.0",
            "--qualification-input",
            str(tmp_path / "qualification.json"),
            "--dry-run",
        ]
    )

    assert args.mock is True
    assert args.qualification_input == tmp_path / "qualification.json"
    assert args.dry_run is True


def test_campaign_rejects_qualification_input_before_the_seed_stage(
    tmp_path: Path
) -> None:
    with pytest.raises(SystemExit, match="not available before the seed stage"):
        main(
            [
                "--mock",
                "--qualification-input",
                str(tmp_path / "qualification.json"),
            ]
        )
