"""Real-mode workflow wiring without calling an external model endpoint."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from ldm_tts.contracts import Candidate
from ldm_tts.optimization import BOObservation
from ldm_tts.transport import ProposalResponse
from ldm_tts.registration.experiment import (
    ACTIVE_CONTRACT_PATH_ENV,
    ACTIVE_CONTRACT_PROFILE_ENV,
)

from tasks.iron_mind.core import workflow
from tasks.iron_mind.core.surrogate import ReactionOneHotEncoder
from tasks.iron_mind.core.tiny_campaign import (
    TinyCampaignRecordError,
    build_tiny_campaign_record,
)
from tasks.iron_mind.core.workflow import main, parse_args


TASK_ROOT = Path(__file__).resolve().parents[1]
REAL_TINY_CONFIG_PATH = TASK_ROOT.parents[1] / "config" / "iron_mind" / "real_tiny.yaml"
REAL_DATA_ROOT = Path(
    "/mnt/data1/ldm-for-sci/data/iron_mind/pinned/olympus-7b4bb35c04eb31dc57a8e46cc79a9cab71dee06d"
)


class StaticEndpoint:
    def __init__(self, table, calls: dict[str, object]) -> None:
        self.table = table
        self.calls = calls

    def preflight(self):
        model = self.calls["client"]["model"]
        return {
            "status": "ok",
            "request_model": model,
            "response_model": model,
            "model_visible": True,
            "model_count": 1,
            "latency_seconds": 0.0,
        }

    def propose(self, request):
        self.calls["proposal_request"] = request
        return ProposalResponse(text=workflow._mock_response(self.table))


def test_real_mode_uses_the_pinned_loader_and_endpoint_client(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    table = workflow._load_mock_table(workflow._schema_for("buchwald_hartwig"))
    calls: dict[str, object] = {}
    seed_prior = _seed_prior(table)
    _install_real_fakes(monkeypatch, table, seed_prior, calls)
    _activate_real_tiny(monkeypatch)
    code = main(_real_args(tmp_path))

    payload = json.loads(capsys.readouterr().out)
    run_dir = Path(payload["run_dir"])
    campaign = json.loads((run_dir / "campaign.json").read_text(encoding="utf-8"))
    snapshot = json.loads(
        (run_dir / "experiment_contract.json").read_text(encoding="utf-8")
    )
    checkpoint = json.loads((run_dir / "checkpoint.json").read_text(encoding="utf-8"))
    assert code == 0
    assert calls["dataset_id"] == "buchwald_hartwig"
    assert calls["data_root"] == REAL_DATA_ROOT
    assert calls["client"]["base_url"] == "https://api.deepseek.com"
    assert payload["engine_summary"]["successful_evaluation_count"] == 1
    assert campaign["contract_profile"] == "real_tiny"
    assert snapshot["snapshot"]["profile"] == "real_tiny"
    assert calls["seed_priors"] == (seed_prior.observation,)
    assert calls["blocked_canonical_keys"] == seed_prior.blocked_canonical_keys
    assert calls["seed_loader"]["input_path"] == TASK_ROOT / "resources" / "qualification_input.json"
    prompt = calls["proposal_request"].messages[1]["content"]
    assert seed_prior.blocked_canonical_keys[0] in prompt
    observations = checkpoint["state"]["observations"]
    assert len(observations) == 1
    assert observations[0]["candidate"]["candidate_id"] != seed_prior.observation.candidate_id


def test_tiny_campaign_record_captures_one_seeded_real_tiny_run(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    table = workflow._load_mock_table(workflow._schema_for("buchwald_hartwig"))
    calls: dict[str, object] = {}
    seed_prior = _seed_prior(table)
    _install_real_fakes(monkeypatch, table, seed_prior, calls)
    _activate_real_tiny(monkeypatch)

    code = main(_real_args(tmp_path))
    run_dir = Path(json.loads(capsys.readouterr().out)["run_dir"])
    record = build_tiny_campaign_record(
        run_dir=run_dir,
        config_path=REAL_TINY_CONFIG_PATH,
    )

    assert code == 0
    assert record["mode"] == "real"
    assert record["contract_profile"] == "real_tiny"
    assert record["provider"] == {
        "kind": "openai_compatible",
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-v4-flash",
    }
    assert record["seed"]["canonical_key"] == seed_prior.blocked_canonical_keys[0]
    assert record["seed"]["excluded_from_campaign_budget"] is True
    assert len(record["candidate_canonical_keys"]) == 4
    assert seed_prior.blocked_canonical_keys[0] not in record["candidate_canonical_keys"]
    assert record["selected_canonical_key"] in record["candidate_canonical_keys"]
    assert record["evaluation"]["benchmark_jobs"] == 1
    assert record["collection"] == {"ir_rows": 4, "sft_rows": 4}
    assert record["dataset"]["schema_sha256"] == table.schema.schema_sha256
    assert record["budget"]["counters"]["llm_requests"] == 1
    assert "test-workflow-key" not in json.dumps(record)
    assert str(run_dir) not in json.dumps(record)


def test_tiny_campaign_record_rejects_a_counter_drift(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    table = workflow._load_mock_table(workflow._schema_for("buchwald_hartwig"))
    calls: dict[str, object] = {}
    _install_real_fakes(monkeypatch, table, _seed_prior(table), calls)
    _activate_real_tiny(monkeypatch)

    assert main(_real_args(tmp_path)) == 0
    run_dir = Path(json.loads(capsys.readouterr().out)["run_dir"])
    budget_path = run_dir / "budget.json"
    budget = json.loads(budget_path.read_text(encoding="utf-8"))
    budget["counters"]["llm_requests"] = 0
    budget_path.write_text(json.dumps(budget), encoding="utf-8")

    with pytest.raises(TinyCampaignRecordError, match="budget"):
        build_tiny_campaign_record(run_dir=run_dir, config_path=REAL_TINY_CONFIG_PATH)


def _install_real_fakes(monkeypatch, table, seed_prior, calls: dict[str, object]) -> None:
    def load_table(*, dataset_id: str, data_root: Path):
        calls["dataset_id"] = dataset_id
        calls["data_root"] = data_root
        return table

    def build_client(**kwargs):
        calls["client"] = kwargs
        return StaticEndpoint(table, calls)

    def load_seed_prior(**kwargs):
        calls["seed_loader"] = kwargs
        return seed_prior

    original_build_components = workflow.build_campaign_components

    def build_components(options):
        calls["seed_priors"] = options.seed_priors
        calls["blocked_canonical_keys"] = options.blocked_canonical_keys
        return original_build_components(options)

    monkeypatch.setenv("LDM_LLM_API_KEY", "test-workflow-key")
    monkeypatch.setenv("LDM_DATA_COLLECTION_ENABLED", "1")
    monkeypatch.setattr(workflow, "load_pinned_reaction_table", load_table, raising=False)
    monkeypatch.setattr(workflow, "build_deepseek_reaction_client", build_client)
    monkeypatch.setattr(
        workflow,
        "load_tracked_qualification_seed_prior",
        load_seed_prior,
        raising=False,
    )
    monkeypatch.setattr(workflow, "build_campaign_components", build_components)


def _activate_real_tiny(monkeypatch) -> None:
    monkeypatch.setenv(ACTIVE_CONTRACT_PATH_ENV, str(TASK_ROOT / "experiment.json"))
    monkeypatch.setenv(ACTIVE_CONTRACT_PROFILE_ENV, "real_tiny")


def _real_args(tmp_path: Path, *, include_qualification_input: bool = True) -> list[str]:
    args = [
        "--proposal-mode", "openai", "--dataset-id", "buchwald_hartwig",
        "--data-dir", str(REAL_DATA_ROOT), "--llm-url", "https://api.deepseek.com",
        "--llm-model-name", "deepseek-v4-flash", "--out-dir", str(tmp_path),
        "--run-name", "real-wiring",
    ]
    if include_qualification_input:
        args.extend(["--qualification-input", str(TASK_ROOT / "resources" / "qualification_input.json")])
    return args


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


def test_real_mode_requires_a_qualification_input(tmp_path: Path) -> None:
    args = parse_args(_real_args(tmp_path, include_qualification_input=False))

    with pytest.raises(SystemExit, match="require --qualification-input"):
        workflow._validate_args(args)


def test_mock_campaign_rejects_a_qualification_input(tmp_path: Path) -> None:
    args = parse_args(
        ["--mock", "--qualification-input", str(tmp_path / "qualification.json")]
    )

    with pytest.raises(SystemExit, match="only supported by non-mock campaigns"):
        workflow._validate_args(args)


def _seed_prior(table) -> SimpleNamespace:
    row = table.rows[0]
    seed_record = json.loads(
        (TASK_ROOT / "resources" / "seed_evaluation_record.json").read_text(encoding="utf-8")
    )
    candidate = Candidate(
        candidate_id=seed_record["candidate"]["candidate_id"],
        payload={"dataset_id": table.schema.dataset_id, "conditions": dict(row.conditions)},
        canonical_key=seed_record["candidate"]["canonical_key"],
        source="qualification_seed",
    )
    encoder = ReactionOneHotEncoder(table.schema)
    observation = BOObservation.scalar(
        candidate.candidate_id,
        row.measurements["yield"],
        encoder.encode(candidate).values,
        feature_version=encoder.version,
    )
    return SimpleNamespace(
        observation=observation,
        blocked_canonical_keys=(candidate.canonical_key,),
    )
