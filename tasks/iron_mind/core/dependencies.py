"""Dependency checks for source-pinned complete Iron Mind campaigns."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping

from ldm_tts.registration.dependencies import (
    DependencyCheck,
    arg_value,
    check_llm_settings,
    fail,
    ok,
    plan_check_context,
    resolve_task_path,
)
from tasks.iron_mind.core.data import FrozenReactionTable, load_frozen_reaction_table
from tasks.iron_mind.core.provider import (
    API_KEY_ENV_NAMES,
    BASE_URL_ENV_NAMES,
    MODEL_ENV_NAMES,
)
from tasks.iron_mind.core.schema import (
    ReactionDatasetSchema,
    load_reaction_schema_from_config,
    load_reaction_schemas,
)


TASK_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = TASK_ROOT / "resources" / "reaction_schemas.json"
CONTRACT_PATH = TASK_ROOT / "resources" / "upstream_contract.json"
MOCK_ORACLE_PATH = TASK_ROOT / "resources" / "mock_oracle.csv"


def check_task_dependencies(
    plan: dict[str, Any], *, include_optional: bool = True
) -> list[DependencyCheck]:
    """Check only assets needed by the selected mock or real campaign."""

    del include_optional
    task, args, env, cwd, mode = plan_check_context(plan)
    if mode == "mock" or bool(args.get("mock")):
        return _mock_checks(task)
    return _real_checks(task, args, env, cwd)


def _mock_checks(task: str) -> list[DependencyCheck]:
    try:
        schema = load_reaction_schemas(SCHEMA_PATH)["buchwald_hartwig"]
        _validate_mock_oracle(MOCK_ORACLE_PATH, schema)
    except (KeyError, OSError, ValueError) as exc:
        return [fail(task, "mock assets", str(exc))]
    return [ok(task, "mock assets", "Tracked mock schema and oracle are valid.")]


def _real_checks(
    task: str,
    args: dict[str, Any],
    env: dict[str, str],
    cwd: Path,
) -> list[DependencyCheck]:
    try:
        contract = _read_json_object(CONTRACT_PATH, "upstream contract")
    except (OSError, ValueError) as exc:
        return [fail(task, "tracked resources", str(exc))]
    gate, dataset_id = _source_gate(task, args, contract)
    if gate.status == "fail":
        return [gate]
    data_root = resolve_task_path(arg_value(args, "data-dir"), cwd)
    if data_root is None:
        return [gate, fail(task, "frozen data root", "Set --data-dir to pinned Iron Mind data.")]
    try:
        dataset_contract = _dataset_contract(contract, dataset_id)
    except (KeyError, ValueError) as exc:
        return [gate, fail(task, "tracked resources", str(exc))]
    provider_checks = _provider_checks(task, args, env)
    return [
        gate,
        *provider_checks,
        _source_revision_check(task, data_root, contract),
        _frozen_table_check(task, data_root, dataset_id, dataset_contract),
    ]


def _provider_checks(
    task: str, args: Mapping[str, Any], env: Mapping[str, str]
) -> list[DependencyCheck]:
    if arg_value(dict(args), "search-method", default="ldm") == "bo":
        return [ok(task, "proposal provider", "Pure BO does not use a model endpoint.")]
    return check_llm_settings(
        task,
        dict(args),
        dict(env),
        url_arg="llm-url",
        model_arg="llm-model-name",
        api_arg="api-key",
        url_env=BASE_URL_ENV_NAMES,
        model_env=MODEL_ENV_NAMES,
        api_env=API_KEY_ENV_NAMES,
        required=True,
    )


def _source_gate(
    task: str,
    args: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> tuple[DependencyCheck, str]:
    dataset_id = str(arg_value(dict(args), "dataset-id") or "")
    allowed = _suite_dataset_ids(contract, "public_union")
    if dataset_id not in allowed:
        return fail(task, "official source gate", f"Unknown official dataset_id {dataset_id!r}."), ""
    return ok(task, "official source gate", "Dataset matches the official contract."), dataset_id


def _source_revision_check(
    task: str, data_root: Path, contract: Mapping[str, Any]
) -> DependencyCheck:
    try:
        manifest = _read_json_object(data_root / "revision_manifest.json", "revision manifest")
        expected = _expected_revisions(contract)
        actual = _manifest_revisions(manifest)
        mismatches = [name for name, revision in expected.items() if actual.get(name) != revision]
        if mismatches:
            raise ValueError("Source revision mismatch: " + ", ".join(sorted(mismatches)) + ".")
    except (KeyError, OSError, ValueError) as exc:
        return fail(task, "source revisions", str(exc), str(data_root))
    return ok(task, "source revisions", "Pinned source revisions match.", str(data_root))


def _frozen_table_check(
    task: str,
    data_root: Path,
    dataset_id: str,
    dataset_contract: Mapping[str, Any],
) -> DependencyCheck:
    name = f"{dataset_id} frozen table"
    try:
        _load_contract_table(data_root, dataset_id, dataset_contract)
    except (OSError, ValueError) as exc:
        return fail(task, name, str(exc), str(data_root))
    return ok(task, name, "Config, data, row count, and schema digest match.", str(data_root))


def load_pinned_reaction_table(
    *,
    dataset_id: str,
    data_root: Path,
) -> FrozenReactionTable:
    """Load one official dataset through the same artifact contract as preflight."""

    contract = _read_json_object(CONTRACT_PATH, "upstream contract")
    if dataset_id not in _suite_dataset_ids(contract, "public_union"):
        raise ValueError(f"Unknown official Iron Mind dataset: {dataset_id!r}.")
    return _load_contract_table(
        Path(data_root), dataset_id, _dataset_contract(contract, dataset_id)
    )


def _load_contract_table(
    data_root: Path,
    dataset_id: str,
    dataset_contract: Mapping[str, Any],
) -> FrozenReactionTable:
    artifacts = _required_mapping(dataset_contract.get("artifacts"), "dataset artifacts")
    config_path = _artifact_path(data_root, artifacts.get("config"), "config")
    data_path = _artifact_path(data_root, artifacts.get("data"), "data")
    schema = load_reaction_schema_from_config(
        config_path,
        dataset_id=dataset_id,
        observation_policy=_required_string(
            dataset_contract.get("observation_policy"), "observation policy"
        ),
        expected_sha256=_required_string(
            dataset_contract.get("schema_sha256"), "schema SHA-256"
        ),
    )
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
    if not rows or any(row["dataset_id"] != schema.dataset_id for row in rows):
        raise ValueError("Mock oracle contains no valid Buchwald rows.")


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


def _suite_dataset_ids(contract: Mapping[str, Any], suite: str) -> tuple[str, ...]:
    suites = _required_mapping(contract.get("suites"), "upstream contract suites")
    values = suites.get(suite)
    if not isinstance(values, list) or not all(isinstance(item, str) and item for item in values):
        raise ValueError(f"Upstream contract suite {suite!r} must be a dataset ID array.")
    return tuple(values)


def _expected_revisions(contract: Mapping[str, Any]) -> dict[str, str]:
    sources = _required_mapping(contract.get("sources"), "upstream contract sources")
    return {
        "iron-mind-public": _source_revision(sources, "iron_mind_public"),
        "olympus": _source_revision(sources, "olympus"),
    }


def _source_revision(sources: Mapping[str, Any], name: str) -> str:
    source = _required_mapping(sources.get(name), f"{name} source")
    return _required_string(source.get("revision"), f"{name} revision")


def _manifest_revisions(manifest: Mapping[str, Any]) -> dict[str, str]:
    sources = manifest.get("sources")
    if not isinstance(sources, list):
        raise ValueError("Revision manifest sources must be an array.")
    revisions = {}
    for source in sources:
        entry = _required_mapping(source, "revision manifest source")
        revisions[_required_string(entry.get("name"), "source name")] = _required_string(
            entry.get("revision"), "source revision"
        )
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


__all__ = ["check_task_dependencies", "load_pinned_reaction_table"]
