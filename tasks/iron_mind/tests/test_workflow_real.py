"""Real-mode workflow wiring without contacting an external endpoint."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from ldm_tts.registration.experiment import (
    ACTIVE_CONTRACT_PATH_ENV,
    ACTIVE_CONTRACT_PROFILE_ENV,
)
from ldm_tts.transport import ProposalResponse

from tasks.iron_mind.core import workflow
from tasks.iron_mind.core.mock import mock_proposal_response
from tasks.iron_mind.core.workflow import main, parse_args


TASK_ROOT = Path(__file__).resolve().parents[1]


class StaticEndpoint:
    def __init__(self, table, calls: dict[str, object]) -> None:
        self.table = table
        self.calls = calls

    def preflight(self) -> dict[str, object]:
        model = self.calls["client"]["model"]
        return {
            "status": "ok",
            "request_model": model,
            "response_model": model,
            "model_visible": True,
            "model_count": 1,
            "latency_seconds": 0.0,
        }

    def propose(self, request) -> ProposalResponse:
        self.calls["proposal_request"] = request
        return mock_proposal_response(
            self.table,
            proposal_index=int(request.metadata["proposal_index"]),
        )


def test_real_mode_connects_pinned_data_endpoint_and_shared_engine(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    table = workflow._load_mock_table(workflow._schema_for("buchwald_hartwig"))
    data_root = tmp_path / "official-data"
    calls: dict[str, object] = {}
    _install_real_fakes(monkeypatch, table, calls)
    _activate_smoke_profile(monkeypatch)

    code = main(_real_args(tmp_path, data_root))

    payload = json.loads(capsys.readouterr().out)
    run_dir = Path(payload["run_dir"])
    campaign = json.loads((run_dir / "campaign.json").read_text(encoding="utf-8"))
    snapshot = json.loads(
        (run_dir / "experiment_contract.json").read_text(encoding="utf-8")
    )
    checkpoint = json.loads((run_dir / "checkpoint.json").read_text(encoding="utf-8"))
    assert code == 0
    assert calls["dataset_id"] == "buchwald_hartwig"
    assert calls["data_root"] == data_root
    assert calls["client"]["base_url"] == "https://example.invalid/v1"
    assert calls["client"]["model"] == "test-model"
    assert calls["client"]["json_mode"] is False
    assert calls["client"]["extra_body"] == {"thinking": {"type": "disabled"}}
    assert payload["engine_summary"]["successful_evaluation_count"] == 1
    assert campaign["contract_profile"] == "ldm_official_smoke"
    assert snapshot["snapshot"]["profile"] == "ldm_official_smoke"
    assert len(checkpoint["state"]["observations"]) == 1
    prompt = calls["proposal_request"].messages[1]["content"]
    assert "Do-not-repeat canonical keys: []" in prompt
    assert "Required slot focus (hard allocation):" in prompt
    assert calls["proposal_request"].metadata["proposal_index"] == 3
    assert calls["proposal_request"].metadata["prompt_policy"] == "portfolio_v1"


def test_real_mode_requires_an_external_data_root(tmp_path: Path) -> None:
    args = parse_args(
        [
            "--proposal-mode",
            "openai",
            "--dataset-id",
            "buchwald_hartwig",
            "--out-dir",
            str(tmp_path),
        ]
    )

    with pytest.raises(SystemExit, match="--data-dir"):
        workflow._validate_args(args)


def test_parse_args_accepts_public_operational_flags(tmp_path: Path) -> None:
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
            "--proposal-max-workers",
            "3",
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
            "--api-key",
            "test-api-key",
            "--llm-timeout",
            "12.5",
            "--llm-max-tokens",
            "512",
            "--llm-temperature",
            "0.7",
            "--llm-json-mode",
            "--llm-extra-body-json",
            '{"thinking":{"type":"disabled"}}',
            "--prompt-policy",
            "baseline_v1",
            "--campaign-index",
            "3",
            "--acquisition-beta",
            "1.0",
            "--dry-run",
        ]
    )

    assert args.mock is True
    assert args.campaign_index == 3
    assert args.proposal_max_workers == 3
    assert args.api_key == "test-api-key"
    assert args.dry_run is True
    assert args.llm_json_mode is True
    assert args.llm_extra_body_json == '{"thinking":{"type":"disabled"}}'
    assert args.prompt_policy == "baseline_v1"


def _install_real_fakes(monkeypatch, table, calls: dict[str, object]) -> None:
    def load_table(*, dataset_id: str, data_root: Path):
        calls["dataset_id"] = dataset_id
        calls["data_root"] = data_root
        return table

    def build_client(**kwargs):
        calls["client"] = kwargs
        return StaticEndpoint(table, calls)

    monkeypatch.setenv("LLM_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("LLM_MODEL_NAME", "test-model")
    monkeypatch.setenv("LLM_API_KEY", "test-workflow-key")
    monkeypatch.setenv("LDM_DATA_COLLECTION_ENABLED", "1")
    monkeypatch.setattr(workflow, "load_pinned_reaction_table", load_table)
    monkeypatch.setattr(workflow, "build_openai_reaction_client", build_client)


def _activate_smoke_profile(monkeypatch) -> None:
    monkeypatch.setenv(ACTIVE_CONTRACT_PATH_ENV, str(TASK_ROOT / "experiment.json"))
    monkeypatch.setenv(ACTIVE_CONTRACT_PROFILE_ENV, "ldm_official_smoke")


def _real_args(tmp_path: Path, data_root: Path) -> list[str]:
    return [
        "--proposal-mode",
        "openai",
        "--dataset-id",
        "buchwald_hartwig",
        "--reservoir-size",
        "4",
        "--data-dir",
        str(data_root),
        "--out-dir",
        str(tmp_path),
        "--run-name",
        "real-wiring",
        "--llm-extra-body-json",
        '{"thinking":{"type":"disabled"}}',
    ]
