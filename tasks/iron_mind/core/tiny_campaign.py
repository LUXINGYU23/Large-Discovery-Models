"""Verify and persist compact evidence for one real Iron Mind tiny campaign."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ldm_tts.cli.runner import load_config
from ldm_tts.registration.experiment import ExperimentContract

from tasks.iron_mind.core.qualification import COLLECTION_ARTIFACTS, MOCK_ARTIFACTS
from tasks.iron_mind.core.qualification_support import (
    REPOSITORY_ROOT,
    TASK_ID,
    contract_reference,
    file_reference,
    load_iron_mind_contract,
    read_json_object,
    sha256_file,
)
from tasks.iron_mind.core.seed_evaluation import SEED_EVALUATION_RECORD_PATH
from tasks.iron_mind.core.tiny_campaign_support import (
    PROVIDER,
    TinyCampaignRecordError,
    candidate_key as _candidate_key,
    event_payload as _event_payload,
    integer_mapping as _integer_mapping,
    one_event as _one_event,
    read_json_lines as _read_json_lines,
    require_endpoint_preflight as _require_endpoint_preflight,
    require_mapping as _mapping,
    sha256_digest as _sha256_digest,
)
from tasks.iron_mind.core.tiny_campaign_validation import (
    require_collection,
    require_evaluation,
    require_reports,
    require_search_and_selection,
)


PROFILE_NAME = "real_tiny"
SEED_EVENT_TYPE = "qualification_seed_prior_loaded"


def build_tiny_campaign_record(*, run_dir: Path, config_path: Path) -> dict[str, Any]:
    """Return passed evidence only for one exact completed real_tiny campaign."""
    contract = load_iron_mind_contract()
    config_args = _load_real_config(config_path, contract)
    run_dir = Path(run_dir).resolve()
    artifacts = {name: sha256_file(run_dir / name) for name in MOCK_ARTIFACTS + COLLECTION_ARTIFACTS}
    campaign = read_json_object(run_dir / "campaign.json", "campaign")
    status = read_json_object(run_dir / "status.json", "campaign status")
    snapshot = read_json_object(run_dir / "experiment_contract.json", "contract snapshot")
    budget = read_json_object(run_dir / "budget.json", "campaign budget")
    summary = read_json_object(run_dir / "summary.json", "campaign summary")
    result = read_json_object(run_dir / "result.json", "campaign result")
    run_config = read_json_object(run_dir / "config.json", "campaign config")
    _require_campaign_contract(campaign, status, snapshot, contract)
    _require_runtime_config(run_config, config_args)
    _require_completion(status, summary, result)
    expected_budget = dict(contract.profile(PROFILE_NAME).budget)
    _require_exact_budget(budget, expected_budget)
    events = _read_json_lines(run_dir / "events.jsonl")
    _require_endpoint_preflight(events)
    seed = _require_seed_prior(events)
    candidate_keys, selected_key = require_search_and_selection(run_dir, events, seed["canonical_key"])
    score = require_evaluation(run_dir, events, selected_key, seed)
    require_reports(run_dir, result, selected_key, score)
    return {
        "schema_version": 1,
        "record_type": "tiny_campaign",
        "task": TASK_ID,
        "status": "passed",
        "mode": "real",
        "contract_profile": PROFILE_NAME,
        "config": file_reference(config_path),
        "experiment_contract": contract_reference(contract),
        "seed": {
            "evaluation_record": file_reference(SEED_EVALUATION_RECORD_PATH),
            **seed,
        },
        "provider": dict(PROVIDER),
        "dataset": {
            "id": seed["dataset_id"],
            "schema_sha256": seed["schema_sha256"],
            "data_sha256": seed["data_sha256"],
        },
        "candidate_canonical_keys": candidate_keys,
        "selected_canonical_key": selected_key,
        "evaluation": {"reaction_score": score, "benchmark_jobs": 1},
        "budget": {"limits": expected_budget, "counters": expected_budget},
        "collection": require_collection(run_dir),
        "artifacts": artifacts,
    }


def _load_real_config(config_path: Path, contract: ExperimentContract) -> Mapping[str, Any]:
    config = load_config(Path(config_path))
    if config.get("task") != TASK_ID or config.get("mode") != "real":
        raise TinyCampaignRecordError("Tiny campaign config must identify the real Iron Mind task.")
    if config.get("contract_profile") != PROFILE_NAME:
        raise TinyCampaignRecordError("Tiny campaign config must select the real_tiny contract profile.")
    args = _mapping(config.get("args"), "tiny campaign config args")
    locked = dict(contract.profile(PROFILE_NAME).locked_args)
    if any(args.get(key) != value for key, value in locked.items()):
        raise TinyCampaignRecordError("Tiny campaign config does not match its locked contract args.")
    if {
        "kind": "openai_compatible",
        "base_url": args.get("llm-url"),
        "model": args.get("llm-model-name"),
    } != PROVIDER:
        raise TinyCampaignRecordError("Tiny campaign config does not match the locked endpoint provider.")
    env = _mapping(config.get("env"), "tiny campaign config env")
    if env.get("LDM_DATA_COLLECTION_ENABLED") != "1":
        raise TinyCampaignRecordError("Tiny campaign config must enable data collection.")
    return args


def _require_campaign_contract(
    campaign: Mapping[str, Any],
    status: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    contract: ExperimentContract,
) -> None:
    for label, payload in (("campaign", campaign), ("campaign status", status)):
        if payload.get("task") != TASK_ID or payload.get("contract_sha256") != contract.digest:
            raise TinyCampaignRecordError(f"{label.capitalize()} does not match the tracked contract.")
        if payload.get("contract_profile") != PROFILE_NAME:
            raise TinyCampaignRecordError(f"{label.capitalize()} does not identify the real_tiny profile.")
    details = _mapping(snapshot.get("snapshot"), "contract snapshot")
    if details.get("sha256") != contract.digest or details.get("profile") != PROFILE_NAME:
        raise TinyCampaignRecordError("Campaign contract snapshot does not match real_tiny.")


def _require_runtime_config(run_config: Mapping[str, Any], args: Mapping[str, Any]) -> None:
    expected = {
        "proposal_mode": args["proposal-mode"],
        "dataset_id": args["dataset-id"],
        "iterations": args["iterations"],
        "reservoir_size": args["reservoir-size"],
        "evaluations_per_round": args["evaluations-per-round"],
        "acquisition_beta": args["acquisition-beta"],
        "llm_url": args["llm-url"],
        "llm_model_name": args["llm-model-name"],
        "data_dir": args["data-dir"],
    }
    if run_config.get("mock") is not False or any(
        run_config.get(key) != value for key, value in expected.items()
    ):
        raise TinyCampaignRecordError("Campaign runtime config does not match real_tiny settings.")
    expected_input = (REPOSITORY_ROOT / str(args["qualification-input"])).resolve()
    run_input = run_config.get("qualification_input")
    if not isinstance(run_input, str) or Path(run_input).resolve() != expected_input:
        raise TinyCampaignRecordError("Campaign runtime did not use the tracked qualification input.")


def _require_completion(
    status: Mapping[str, Any], summary: Mapping[str, Any], result: Mapping[str, Any]
) -> None:
    if status.get("status") != "completed" or result.get("finished") is not True:
        raise TinyCampaignRecordError("Tiny campaign must be completed before verification.")
    if summary.get("successful_evaluation_count") != 1 or result.get("evaluation_count") != 1:
        raise TinyCampaignRecordError("Tiny campaign must contain exactly one successful evaluation.")


def _require_exact_budget(budget: Mapping[str, Any], expected: Mapping[str, int]) -> None:
    limits = _integer_mapping(budget.get("limits"), "budget limits")
    counters = _integer_mapping(budget.get("counters"), "budget counters")
    if limits != expected or counters != expected:
        raise TinyCampaignRecordError("Tiny campaign budget does not match the exact real_tiny budget.")


def _require_seed_prior(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    record = read_json_object(SEED_EVALUATION_RECORD_PATH, "seed evaluation record")
    candidate = _mapping(record.get("candidate"), "seed evaluation candidate")
    candidate_id = _candidate_key(candidate.get("candidate_id"))
    canonical_key = candidate.get("canonical_key")
    dataset_id = record.get("dataset_id")
    schema_sha256 = _sha256_digest(record.get("schema_sha256"), "seed schema digest")
    data_sha256 = _sha256_digest(record.get("data_sha256"), "seed data digest")
    if record.get("status") != "passed" or record.get("excluded_from_campaign_budget") is not True:
        raise TinyCampaignRecordError("Tracked seed evaluation record is not campaign-excluded evidence.")
    if canonical_key != candidate_id[1] or not isinstance(dataset_id, str):
        raise TinyCampaignRecordError("Tracked seed evaluation record has an inconsistent candidate key.")
    payload = _event_payload(_one_event(events, SEED_EVENT_TYPE), SEED_EVENT_TYPE)
    expected = {
        "candidate_id": candidate_id[0],
        "canonical_key": canonical_key,
        "excluded_from_campaign_budget": True,
    }
    if payload != expected:
        raise TinyCampaignRecordError("Campaign did not load the tracked excluded seed prior.")
    return {
        "candidate_id": candidate_id[0],
        "canonical_key": canonical_key,
        "excluded_from_campaign_budget": True,
        "dataset_id": dataset_id,
        "schema_sha256": schema_sha256,
        "data_sha256": data_sha256,
    }


__all__ = [
    "TinyCampaignRecordError",
    "build_tiny_campaign_record",
]
