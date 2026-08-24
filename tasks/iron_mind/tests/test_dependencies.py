"""Tests for Iron Mind dependency preflight and component assembly."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import MappingProxyType

from ldm_tts.data import DataCollectionSink
from ldm_tts.engine import LDMEngine
from ldm_tts.engine.run_store import CampaignRuntime
from ldm_tts.transport import CallableProposalClient
from tasks.iron_mind.core.data import FrozenReactionTable, ReactionRow
from tasks.iron_mind.core.dependencies import DependencyResources, check_task_dependencies
from tasks.iron_mind.core.factory import CampaignComponentOptions, build_campaign_components
from tasks.iron_mind.core.ldm_selector import AcquisitionTiltedSelector
from tasks.iron_mind.core.reaction_gp import ReactionCategoricalGPUCBSelector
from tasks.iron_mind.core.schema import (
    ReactionDatasetSchema,
    ReactionFactor,
    canonical_schema_payload,
    load_reaction_schemas,
    schema_sha256,
)
from tasks.iron_mind.ldm_task.dependencies import check_dependencies


TASK_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = TASK_ROOT / "resources" / "reaction_schemas.json"
IRON_MIND_REVISION = "1" * 40
OLYMPUS_REVISION = "2" * 40


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def _write_dependency_schema(tmp_path: Path) -> tuple[Path, str]:
    factor = ReactionFactor(name="base", categories=("A",))
    digest = schema_sha256(
        canonical_schema_payload(
            dataset_id="buchwald_hartwig", factors=(factor,), measurements=("yield",),
            objective="reaction_score", direction="maximize",
            observation_policy="single_row",
        )
    )
    schema_path = tmp_path / "reaction_schemas.json"
    _write_json(
        schema_path,
        {
            "schema_version": 1,
            "datasets": {
                "buchwald_hartwig": {
                    "schema_version": 1,
                    "dataset_id": "buchwald_hartwig",
                    "factors": [{"name": "base", "categories": ["A"]}],
                    "measurements": ["yield"],
                    "objective": {"name": "reaction_score", "direction": "maximize"},
                    "observation_policy": "single_row",
                    "schema_sha256": digest,
                }
            },
        },
    )
    return schema_path, digest


def _write_dependency_assets(tmp_path: Path, digest: str) -> tuple[Path, Path, Path]:
    data_root = tmp_path / "pinned"
    config_path = data_root / "olympus" / "config.json"
    data_path = data_root / "olympus" / "data.csv"
    _write_json(
        config_path,
        {
            "parameters": [{"name": "base", "type": "categorical", "options": ["A"]}],
            "measurements": [{"name": "yield", "type": "continuous"}],
        },
    )
    data_path.write_bytes(b"A,2.5\n")
    contract_path = tmp_path / "upstream_contract.json"
    artifacts = {
        "config": {"path": "olympus/config.json", "bytes": config_path.stat().st_size, "sha256": _sha256(config_path)},
        "data": {"path": "olympus/data.csv", "bytes": data_path.stat().st_size, "sha256": _sha256(data_path)},
    }
    _write_json(
        contract_path,
        {
            "schema_version": 2,
            "sources": {
                "iron_mind_public": {"revision": IRON_MIND_REVISION},
                "olympus": {"revision": OLYMPUS_REVISION},
            },
            "suites": {
                "paper_v2": ["buchwald_hartwig"],
                "public_union": ["buchwald_hartwig"],
            },
            "datasets": {
                "buchwald_hartwig": {
                    "row_count": 1,
                    "observation_policy": "single_row",
                    "schema_sha256": digest,
                    "artifacts": artifacts,
                }
            },
        },
    )
    _write_json(
        data_root / "revision_manifest.json",
        {
            "schema_version": 1,
            "sources": [
                {"name": "iron-mind-public", "revision": IRON_MIND_REVISION},
                {"name": "olympus", "revision": OLYMPUS_REVISION},
            ],
        },
    )
    return data_root, data_path, contract_path


def _dependency_fixture(tmp_path: Path) -> tuple[DependencyResources, Path, Path]:
    schema_path, digest = _write_dependency_schema(tmp_path)
    data_root, data_path, contract_path = _write_dependency_assets(tmp_path, digest)
    mock_oracle_path = tmp_path / "mock_oracle.csv"
    mock_oracle_path.write_text(
        "dataset_id,base,reaction_score\nbuchwald_hartwig,A,2.5\n", encoding="utf-8"
    )
    return (
        DependencyResources(
            schema_path=schema_path,
            contract_path=contract_path,
            mock_oracle_path=mock_oracle_path,
        ),
        data_root,
        data_path,
    )


def _real_checks(
    resources: DependencyResources,
    data_root: Path,
    *,
    profile: str = "ldm_official_smoke",
) -> list:
    return check_task_dependencies(
        "iron_mind",
        {"data-dir": str(data_root), "dataset-id": "buchwald_hartwig"},
        {
            "LLM_BASE_URL": "https://example.invalid/v1",
            "LLM_MODEL_NAME": "test-model",
            "LLM_API_KEY": "test-key",
        },
        data_root,
        mode="real",
        include_optional=False,
        contract_profile=profile,
        resources=resources,
    )


def test_mock_dependencies_validate_only_tracked_mock_assets(tmp_path: Path) -> None:
    resources, data_root, _data_path = _dependency_fixture(tmp_path)

    checks = check_task_dependencies(
        "iron_mind",
        {"mock": True},
        {},
        data_root,
        mode="mock",
        include_optional=False,
        resources=resources,
    )

    assert [(check.name, check.status) for check in checks] == [("mock assets", "ok")]
    assert all("GPU" not in check.name and "LLM" not in check.name for check in checks)


def test_real_dependencies_verify_revisions_and_frozen_table_contracts(tmp_path: Path) -> None:
    resources, data_root, data_path = _dependency_fixture(tmp_path)

    checks = _real_checks(resources, data_root)

    statuses = {check.name: check.status for check in checks}
    assert statuses == {
        "official source gate": "ok",
        "LLM URL": "ok",
        "LLM model": "ok",
        "LLM API key": "ok",
        "source revisions": "ok",
        "buchwald_hartwig frozen table": "ok",
    }
    data_path.write_bytes(b"A,3.5\n")
    failed = _real_checks(resources, data_root)
    table_check = next(check for check in failed if check.name == "buchwald_hartwig frozen table")
    assert table_check.status == "fail"
    assert "SHA-256" in table_check.message


def test_prompt_baseline_profiles_use_the_same_official_source_gate(tmp_path: Path) -> None:
    resources, data_root, _data_path = _dependency_fixture(tmp_path)

    for profile in ("ldm_prompt_baseline_smoke", "ldm_prompt_baseline_20"):
        checks = _real_checks(resources, data_root, profile=profile)

        assert checks[0].name == "official source gate"
        assert checks[0].status == "ok"


def test_extended_compare_profiles_use_the_same_official_source_gate(tmp_path: Path) -> None:
    resources, data_root, _data_path = _dependency_fixture(tmp_path)

    for profile in ("extended_compare", "extended_compare_direct_llm"):
        checks = _real_checks(resources, data_root, profile=profile)

        assert checks[0].name == "official source gate"
        assert checks[0].status == "ok"


def test_unsupported_profile_fails_before_data_access(tmp_path: Path) -> None:
    resources, data_root, _data_path = _dependency_fixture(tmp_path)

    checks = _real_checks(resources, data_root, profile="all_six_datasets")

    assert [(check.name, check.status) for check in checks] == [("official source gate", "fail")]
    assert "all_six_datasets" in checks[0].message
    assert "Unsupported" in checks[0].message


def test_dependency_adapter_forwards_the_contract_profile(tmp_path: Path) -> None:
    _resources, data_root, _data_path = _dependency_fixture(tmp_path)
    plan = {
        "task": "iron_mind",
        "argv": ["--data-dir", str(data_root), "--dataset-id", "buchwald_hartwig"],
        "cwd": str(data_root),
        "mode": "real",
        "contract_profile": "all_six_datasets",
    }

    checks = check_dependencies(plan, include_optional=False)

    assert [(check.name, check.status) for check in checks] == [("official source gate", "fail")]
    assert "all_six_datasets" in checks[0].message


def _factory_schema() -> ReactionDatasetSchema:
    return load_reaction_schemas(SCHEMA_PATH)["buchwald_hartwig"]

def _factory_table(schema: ReactionDatasetSchema, *, yield_value: float) -> FrozenReactionTable:
    conditions = {factor.name: factor.categories[0] for factor in schema.factors}
    row = ReactionRow(
        row_id=1,
        conditions=MappingProxyType(conditions),
        measurements=MappingProxyType({"yield": yield_value}),
        raw_row_sha256=hashlib.sha256(b"factory-row").hexdigest(),
    )
    key = tuple(conditions[name] for name in schema.factor_names)
    return FrozenReactionTable(
        schema=schema,
        rows=(row,),
        rows_by_conditions=MappingProxyType({key: (row,)}),
    )


def _factory_options(
    tmp_path: Path,
    schema: ReactionDatasetSchema,
    table: FrozenReactionTable,
    *,
    run_name: str,
) -> CampaignComponentOptions:
    runtime = CampaignRuntime.open(tmp_path / run_name, task="iron_mind")
    return CampaignComponentOptions(
        client=CallableProposalClient(lambda _request: "{}"),
        schema=schema,
        table=table,
        sink=DataCollectionSink.disabled(),
        runtime=runtime,
        proposal_samples=4,
        bo_pool_size=2,
    )


def test_factory_assembles_one_shared_engine_shape_for_mock_and_real(tmp_path: Path) -> None:
    schema = _factory_schema()
    mock = build_campaign_components(
        _factory_options(tmp_path, schema, _factory_table(schema, yield_value=1.0), run_name="mock")
    )
    real = build_campaign_components(
        _factory_options(tmp_path, schema, _factory_table(schema, yield_value=2.0), run_name="real")
    )

    for name in ("domain", "expander", "encoder", "selector", "engine"):
        assert type(getattr(mock, name)) is type(getattr(real, name))
    assert isinstance(mock.engine, LDMEngine)
    assert isinstance(mock.selector, AcquisitionTiltedSelector)
    assert isinstance(mock.selector.base_selector, ReactionCategoricalGPUCBSelector)
    assert mock.expander.domain is mock.domain
    assert mock.engine.task_spec == mock.task_spec
    assert mock.task_spec.surrogate == mock.encoder.describe()
    assert mock.task_spec.acquisition == mock.selector.describe()
    assert mock.evaluator.table is not real.evaluator.table
