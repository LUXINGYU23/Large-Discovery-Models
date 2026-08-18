"""Source-pinned data verification for Iron Mind qualification evidence."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from tasks.iron_mind.core.dependencies import check_task_dependencies, load_pinned_reaction_table
from tasks.iron_mind.core.evaluator import chan_lam_row_score
from tasks.iron_mind.core.qualification_support import (
    TASK_ID,
    TASK_ROOT,
    UPSTREAM_CONTRACT_PATH,
    QualificationRecordError,
    contract_reference,
    file_reference,
    load_iron_mind_contract,
    read_json_object,
    sha256_file,
)


def build_contract_verification_record(*, data_root: Path) -> dict[str, Any]:
    """Verify both pinned reaction tables without contacting a proposal endpoint."""

    contract = load_iron_mind_contract()
    upstream = read_json_object(UPSTREAM_CONTRACT_PATH, "upstream contract")
    _require_matching_source_commit(contract.benchmark, upstream)
    tables = {
        "buchwald_hartwig": _load_checked_table(data_root, "buchwald_hartwig", "real_tiny"),
        "chan_lam_full": _load_checked_table(
            data_root, "chan_lam_full", "chan_lam_contract_validation"
        ),
    }
    return {
        "schema_version": 1,
        "record_type": "contract_verification",
        "task": TASK_ID,
        "status": "passed",
        "benchmark_commit": str(contract.benchmark["source_commit"]),
        "experiment_contract": contract_reference(contract),
        "upstream_contract": file_reference(UPSTREAM_CONTRACT_PATH),
        "revision_manifest_sha256": sha256_file(Path(data_root) / "revision_manifest.json"),
        "datasets": _dataset_records(tables, upstream),
        "chan_lam_formula_vector": _chan_lam_formula_vector(tables["chan_lam_full"]),
    }


def _load_checked_table(data_root: Path, dataset_id: str, profile: str):
    checks = check_task_dependencies(
        TASK_ID,
        {"dataset-id": dataset_id, "data-dir": str(data_root)},
        {},
        TASK_ROOT,
        mode="real",
        include_optional=True,
        contract_profile=profile,
    )
    if not checks or any(check.status != "ok" for check in checks):
        raise QualificationRecordError(f"Pinned source checks failed for {dataset_id}.")
    return load_pinned_reaction_table(dataset_id=dataset_id, data_root=Path(data_root))


def _require_matching_source_commit(benchmark: Mapping[str, Any], upstream: Mapping[str, Any]) -> None:
    sources = upstream.get("sources")
    if not isinstance(sources, Mapping):
        raise QualificationRecordError("Upstream contract does not define sources.")
    source = sources.get("iron_mind_public")
    if not isinstance(source, Mapping) or source.get("revision") != benchmark.get("source_commit"):
        raise QualificationRecordError("Experiment contract and upstream source commit do not match.")


def _dataset_records(tables: Mapping[str, Any], upstream: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_datasets = upstream.get("datasets")
    if not isinstance(raw_datasets, Mapping):
        raise QualificationRecordError("Upstream contract does not define datasets.")
    return [_dataset_record(dataset_id, tables[dataset_id], raw_datasets) for dataset_id in sorted(tables)]


def _dataset_record(dataset_id: str, table: Any, raw_datasets: Mapping[str, Any]) -> dict[str, Any]:
    raw = raw_datasets.get(dataset_id)
    if not isinstance(raw, Mapping) or not isinstance(raw.get("artifacts"), Mapping):
        raise QualificationRecordError(f"Upstream contract is missing {dataset_id} artifacts.")
    artifacts = raw["artifacts"]
    return {
        "dataset_id": dataset_id,
        "row_count": len(table.rows),
        "one_hot_dimension": table.schema.one_hot_dimension,
        "schema_sha256": table.schema.schema_sha256,
        "config_sha256": _artifact_digest(artifacts, "config", dataset_id),
        "data_sha256": _artifact_digest(artifacts, "data", dataset_id),
    }


def _artifact_digest(artifacts: Mapping[str, Any], name: str, dataset_id: str) -> str:
    artifact = artifacts.get(name)
    if not isinstance(artifact, Mapping) or not isinstance(artifact.get("sha256"), str):
        raise QualificationRecordError(f"Upstream contract is missing {dataset_id} {name} digest.")
    return artifact["sha256"]


def _chan_lam_formula_vector(table: Any) -> dict[str, Any]:
    row = min(table.rows, key=lambda item: (item.raw_row_sha256, item.row_id))
    desired = _finite_measurement(row.measurements.get("desired_yield"), "desired_yield")
    undesired = _finite_measurement(row.measurements.get("undesired_yield"), "undesired_yield")
    score = chan_lam_row_score(row)
    if not math.isfinite(score):
        raise QualificationRecordError("Pinned Chan-Lam formula vector is non-finite.")
    return {
        "row_id": row.row_id,
        "raw_row_sha256": row.raw_row_sha256,
        "desired_yield": desired,
        "undesired_yield": undesired,
        "reaction_score": score,
    }


def _finite_measurement(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise QualificationRecordError(f"Chan-Lam vector is missing finite {name}.")
    return float(value)


__all__ = ["build_contract_verification_record"]
