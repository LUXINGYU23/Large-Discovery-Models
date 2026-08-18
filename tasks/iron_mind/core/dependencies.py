"""Lightweight dependency checks for source-pinned Iron Mind campaigns."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ldm_tts.registration.dependencies import (
    DependencyCheck,
    arg_value,
    fail,
    ok,
    resolve_task_path,
)
from tasks.iron_mind.core.data import FrozenReactionTable, load_frozen_reaction_table
from tasks.iron_mind.core.schema import ReactionDatasetSchema, load_reaction_schemas


TASK_ROOT = Path(__file__).resolve().parents[1]
M0_M2_PROFILES = {
    "real_tiny": "buchwald_hartwig",
    "chan_lam_contract_validation": "chan_lam_full",
}


@dataclass(frozen=True)
class DependencyResources:
    """Tracked resource paths, injectable only for focused preflight tests."""

    schema_path: Path
    contract_path: Path
    mock_oracle_path: Path


DEFAULT_RESOURCES = DependencyResources(
    schema_path=TASK_ROOT / "resources" / "reaction_schemas.json",
    contract_path=TASK_ROOT / "resources" / "upstream_contract.json",
    mock_oracle_path=TASK_ROOT / "resources" / "mock_oracle.csv",
)


def check_task_dependencies(
    task: str,
    args: dict[str, Any],
    env: dict[str, str],
    cwd: Path,
    *,
    mode: str,
    include_optional: bool,
    contract_profile: str = "",
    resources: DependencyResources = DEFAULT_RESOURCES,
) -> list[DependencyCheck]:
    """Check only assets needed by the selected mock or real campaign boundary."""

    del env, include_optional
    if mode == "mock" or bool(args.get("mock")):
        return _mock_checks(task, resources)
    return _real_checks(task, args, cwd, contract_profile, resources)


def _mock_checks(task: str, resources: DependencyResources) -> list[DependencyCheck]:
    try:
        schema = load_reaction_schemas(resources.schema_path)["buchwald_hartwig"]
        _validate_mock_oracle(resources.mock_oracle_path, schema)
    except (KeyError, OSError, ValueError) as exc:
        return [fail(task, "mock assets", str(exc))]
    return [ok(task, "mock assets", "Tracked mock schema and oracle are valid.")]


def _real_checks(
    task: str,
    args: dict[str, Any],
    cwd: Path,
    contract_profile: str,
    resources: DependencyResources,
) -> list[DependencyCheck]:
    gate, dataset_id = _source_gate(task, args, contract_profile)
    if gate.status == "fail":
        return [gate]
    try:
        contract = _read_json_object(resources.contract_path, "upstream contract")
        schema = load_reaction_schemas(resources.schema_path)[dataset_id]
        dataset_contract = _dataset_contract(contract, dataset_id)
    except (KeyError, OSError, ValueError) as exc:
        return [gate, fail(task, "tracked resources", str(exc))]
    data_root = resolve_task_path(arg_value(args, "data-dir"), cwd)
    if data_root is None:
        return [gate, fail(task, "frozen data root", "Set --data-dir to pinned Iron Mind data.")]
    return [
        gate,
        _source_revision_check(task, data_root, contract),
        _frozen_table_check(task, data_root, schema, dataset_contract),
    ]


def _source_gate(
    task: str, args: Mapping[str, Any], contract_profile: str
) -> tuple[DependencyCheck, str]:
    profile = contract_profile.strip()
    dataset_id = M0_M2_PROFILES.get(profile)
    if dataset_id is None:
        return (
            fail(
                task,
                "M0-M2 source gate",
                f"Contract profile {profile!r} is not supported by the M0-M2 source contract.",
            ),
            "",
        )
    requested = arg_value(dict(args), "dataset-id")
    if requested != dataset_id:
        return (
            fail(
                task,
                "M0-M2 source gate",
                f"Profile {profile!r} requires dataset_id {dataset_id!r}.",
            ),
            "",
        )
    return (
        ok(task, "M0-M2 source gate", "Profile and dataset match the supported source contract."),
        dataset_id,
    )


def _source_revision_check(
    task: str, data_root: Path, contract: Mapping[str, Any]
) -> DependencyCheck:
    try:
        manifest = _read_json_object(data_root / "revision_manifest.json", "revision manifest")
        expected = _expected_revisions(contract)
        actual = _manifest_revisions(manifest)
        mismatches = [
            name for name, revision in expected.items() if actual.get(name) != revision
        ]
        if mismatches:
            raise ValueError("Source revision mismatch: " + ", ".join(sorted(mismatches)) + ".")
    except (KeyError, OSError, ValueError) as exc:
        return fail(task, "source revisions", str(exc), str(data_root))
    return ok(task, "source revisions", "Pinned source revisions match.", str(data_root))


def _frozen_table_check(
    task: str,
    data_root: Path,
    schema: ReactionDatasetSchema,
    dataset_contract: Mapping[str, Any],
) -> DependencyCheck:
    name = f"{schema.dataset_id} frozen table"
    try:
        _load_contract_table(data_root, schema, dataset_contract)
    except (OSError, ValueError) as exc:
        return fail(task, name, str(exc), str(data_root))
    return ok(task, name, "Config, data, row count, and schema digest match.", str(data_root))


def load_pinned_reaction_table(
    *,
    dataset_id: str,
    data_root: Path,
    resources: DependencyResources = DEFAULT_RESOURCES,
) -> FrozenReactionTable:
    """Load one tracked dataset through the same artifact contract as preflight."""

    contract = _read_json_object(resources.contract_path, "upstream contract")
    schema = load_reaction_schemas(resources.schema_path)[dataset_id]
    dataset_contract = _dataset_contract(contract, dataset_id)
    return _load_contract_table(Path(data_root), schema, dataset_contract)


def _load_contract_table(
    data_root: Path,
    schema: ReactionDatasetSchema,
    dataset_contract: Mapping[str, Any],
) -> FrozenReactionTable:
    expected_digest = _required_string(dataset_contract.get("schema_sha256"), "schema SHA-256")
    if schema.schema_sha256 != expected_digest:
        raise ValueError("Tracked schema SHA-256 does not match the dataset contract.")
    artifacts = _required_mapping(dataset_contract.get("artifacts"), "dataset artifacts")
    config_path = _artifact_path(data_root, artifacts.get("config"), "config")
    data_path = _artifact_path(data_root, artifacts.get("data"), "data")
    return load_frozen_reaction_table(
        schema=schema,
        config_path=config_path,
        data_path=data_path,
        artifact_contract=dataset_contract,
    )


def _validate_mock_oracle(path: Path, schema: ReactionDatasetSchema) -> None:
    if not path.is_file():
        raise ValueError(f"Mock oracle does not exist: {path}.")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        expected = ("dataset_id", *schema.factor_names, "reaction_score")
        if tuple(reader.fieldnames or ()) != expected:
            raise ValueError("Mock oracle header does not match the tracked Buchwald schema.")
        rows = list(reader)
    if not rows:
        raise ValueError("Mock oracle must contain at least one row.")
    if any(row["dataset_id"] != schema.dataset_id for row in rows):
        raise ValueError("Mock oracle contains a row for an unexpected dataset.")


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Could not parse {label}: {path}.") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label.capitalize()} must be a JSON object.")
    return payload


def _dataset_contract(contract: Mapping[str, Any], dataset_id: str) -> Mapping[str, Any]:
    datasets = _required_mapping(contract.get("datasets"), "upstream contract datasets")
    return _required_mapping(datasets.get(dataset_id), f"dataset contract {dataset_id!r}")


def _expected_revisions(contract: Mapping[str, Any]) -> dict[str, str]:
    sources = _required_mapping(contract.get("sources"), "upstream contract sources")
    return {
        "iron-mind-public": _required_string(
            _required_mapping(sources.get("iron_mind_public"), "iron-mind-public source").get(
                "revision"
            ),
            "iron-mind-public revision",
        ),
        "olympus": _required_string(
            _required_mapping(sources.get("olympus"), "olympus source").get("revision"),
            "olympus revision",
        ),
    }


def _manifest_revisions(manifest: Mapping[str, Any]) -> dict[str, str]:
    sources = manifest.get("sources")
    if not isinstance(sources, list):
        raise ValueError("Revision manifest sources must be an array.")
    revisions = {}
    for source in sources:
        entry = _required_mapping(source, "revision manifest source")
        name = _required_string(entry.get("name"), "revision manifest source name")
        revisions[name] = _required_string(entry.get("revision"), "revision manifest revision")
    return revisions


def _artifact_path(data_root: Path, raw_artifact: Any, label: str) -> Path:
    artifact = _required_mapping(raw_artifact, f"{label} artifact")
    relative = Path(_required_string(artifact.get("path"), f"{label} artifact path"))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label} artifact path must stay under the pinned data root.")
    return data_root / relative


def _required_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label.capitalize()} must be an object.")
    return value


def _required_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label.capitalize()} must be a non-empty string.")
    return value


__all__ = [
    "DependencyResources",
    "check_task_dependencies",
    "load_pinned_reaction_table",
]
