"""Build compact registered and mock qualification records for Iron Mind."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ldm_tts.engine.run_store import atomic_json_write
from ldm_tts.registration.experiment import ExperimentContract
from ldm_tts.registration.registry import get_task_definition

from tasks.iron_mind.core.contract_verification import build_contract_verification_record
from tasks.iron_mind.core.qualification_support import (
    TASK_ID,
    TASK_MANIFEST_PATH,
    QualificationRecordError,
    contract_reference,
    file_reference,
    load_iron_mind_contract,
    read_json_object,
    sha256_file,
)


MOCK_BUDGET = {
    "outer_iterations": 1,
    "llm_requests": 0,
    "proposal_attempts": 1,
    "valid_search_candidates": 4,
    "selected_candidates": 1,
    "external_evaluations": 1,
    "expensive_evaluation_attempts": 1,
    "successful_evaluations": 1,
    "benchmark_jobs": 1,
}
MOCK_ARTIFACTS = (
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
    "search_manifest.json",
    "selection_record.json",
    "evaluation_manifest.json",
    "ldm_data/dataset_info.json",
)
COLLECTION_ARTIFACTS = ("ldm_data/ldm_ir.jsonl", "ldm_data/ldm_sft.jsonl")


def build_registered_record() -> dict[str, Any]:
    """Return a passed record only when the tracked task remains registered."""

    definition = get_task_definition(TASK_ID)
    manifest = read_json_object(TASK_MANIFEST_PATH, "task manifest")
    contract = load_iron_mind_contract()
    if manifest.get("task_id") != TASK_ID or manifest.get("schema_version") != 1:
        raise QualificationRecordError("Tracked task manifest does not identify Iron Mind schema v1.")
    if definition.relative_root != Path("tasks") / TASK_ID:
        raise QualificationRecordError("Registered task root does not match Iron Mind.")
    if definition.experiment_contract_path != Path("tasks") / TASK_ID / "experiment.json":
        raise QualificationRecordError("Registered experiment contract does not match Iron Mind.")
    return {
        "schema_version": 1,
        "record_type": "registered",
        "task": TASK_ID,
        "status": "passed",
        "task_manifest": file_reference(TASK_MANIFEST_PATH),
        "experiment_contract": contract_reference(contract),
    }


def build_mock_verification_record(*, run_dir: Path, config_path: Path) -> dict[str, Any]:
    """Verify one exact completed mock campaign before returning a passed record."""

    contract = load_iron_mind_contract()
    run_dir = Path(run_dir).resolve()
    artifacts = _artifact_digests(run_dir, MOCK_ARTIFACTS + COLLECTION_ARTIFACTS)
    _validate_mock_campaign(run_dir, contract)
    collection = _collection_counts(run_dir)
    if collection != {"ir_rows": 4, "sft_rows": 4}:
        raise QualificationRecordError("Mock collection must contain exactly four IR and SFT rows.")
    return {
        "schema_version": 1,
        "record_type": "mock_verification",
        "task": TASK_ID,
        "status": "passed",
        "mode": "mock",
        "config": file_reference(config_path),
        "experiment_contract": contract_reference(contract),
        "budget": {"limits": dict(MOCK_BUDGET), "counters": dict(MOCK_BUDGET)},
        "collection": collection,
        "artifacts": artifacts,
    }


def write_qualification_record(path: Path, record: Mapping[str, Any]) -> Path:
    """Persist one builder-produced passed record atomically."""

    if record.get("status") != "passed":
        raise QualificationRecordError("Only passed qualification records may be persisted.")
    destination = Path(path)
    atomic_json_write(destination, dict(record))
    return destination


def _validate_mock_campaign(run_dir: Path, contract: ExperimentContract) -> None:
    campaign = read_json_object(run_dir / "campaign.json", "campaign")
    status = read_json_object(run_dir / "status.json", "campaign status")
    snapshot = read_json_object(run_dir / "experiment_contract.json", "contract snapshot")
    budget = read_json_object(run_dir / "budget.json", "campaign budget")
    summary = read_json_object(run_dir / "summary.json", "campaign summary")
    result = read_json_object(run_dir / "result.json", "campaign result")
    _require_contract_digest(campaign, contract, "campaign")
    _require_contract_digest(status, contract, "campaign status")
    _require_snapshot_digest(snapshot, contract)
    if status.get("status") != "completed" or result.get("finished") is not True:
        raise QualificationRecordError("Mock campaign must be completed before verification.")
    if summary.get("successful_evaluation_count") != 1 or result.get("evaluation_count") != 1:
        raise QualificationRecordError("Mock campaign must contain exactly one successful evaluation.")
    _require_exact_budget(budget)


def _artifact_digests(run_dir: Path, names: tuple[str, ...]) -> dict[str, str]:
    return {name: sha256_file(run_dir / name) for name in names}


def _collection_counts(run_dir: Path) -> dict[str, int]:
    return {
        "ir_rows": _jsonl_object_count(run_dir / COLLECTION_ARTIFACTS[0]),
        "sft_rows": _jsonl_object_count(run_dir / COLLECTION_ARTIFACTS[1]),
    }


def _jsonl_object_count(path: Path) -> int:
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    except (OSError, json.JSONDecodeError) as exc:
        raise QualificationRecordError(f"Collection artifact is not valid JSONL: {path}") from exc
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise QualificationRecordError(f"Collection artifact must contain JSON objects: {path}")
    return len(rows)


def _require_exact_budget(payload: Mapping[str, Any]) -> None:
    limits = _integer_mapping(payload.get("limits"), "budget limits")
    counters = _integer_mapping(payload.get("counters"), "budget counters")
    if limits != MOCK_BUDGET or counters != MOCK_BUDGET:
        raise QualificationRecordError("Mock campaign budget does not match the exact verification budget.")


def _integer_mapping(value: Any, label: str) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise QualificationRecordError(f"{label} must be an object.")
    parsed: dict[str, int] = {}
    for key, item in value.items():
        if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item):
            raise QualificationRecordError(f"{label} contains a non-finite numeric value.")
        if int(item) != item:
            raise QualificationRecordError(f"{label} contains a non-integral verification value.")
        parsed[str(key)] = int(item)
    return parsed


def _require_contract_digest(payload: Mapping[str, Any], contract: ExperimentContract, label: str) -> None:
    if payload.get("task") != TASK_ID or payload.get("contract_sha256") != contract.digest:
        raise QualificationRecordError(f"{label.capitalize()} does not match the tracked experiment contract.")


def _require_snapshot_digest(snapshot: Mapping[str, Any], contract: ExperimentContract) -> None:
    details = snapshot.get("snapshot")
    if not isinstance(details, Mapping) or details.get("sha256") != contract.digest:
        raise QualificationRecordError("Campaign contract snapshot does not match the tracked contract.")


__all__ = [
    "QualificationRecordError",
    "build_contract_verification_record",
    "build_mock_verification_record",
    "build_registered_record",
    "sha256_file",
    "write_qualification_record",
]
