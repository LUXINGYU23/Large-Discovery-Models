"""Tests for Iron Mind dependency preflight."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tasks.iron_mind.core import dependencies
from tasks.iron_mind.core.dependencies import check_task_dependencies
from tasks.iron_mind.core.schema import (
    ReactionFactor,
    canonical_schema_payload,
    schema_sha256,
)


IRON_MIND_REVISION = "1" * 40
OLYMPUS_REVISION = "2" * 40


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_dependency_schema(tmp_path: Path) -> tuple[Path, str]:
    factor = ReactionFactor(name="base", options=("A",))
    digest = schema_sha256(
        canonical_schema_payload(
            dataset_id="buchwald_hartwig",
            factors=(factor,),
            measurements=("yield",),
            objective="reaction_score",
            direction="maximize",
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
            "parameters": [
                {"name": "base", "type": "categorical", "options": ["A"]}
            ],
            "measurements": [{"name": "yield", "type": "continuous"}],
        },
    )
    data_path.write_bytes(b"A,2.5\n")
    contract_path = tmp_path / "upstream_contract.json"
    artifacts = {
        "config": {
            "path": "olympus/config.json",
            "bytes": config_path.stat().st_size,
            "sha256": _sha256(config_path),
        },
        "data": {
            "path": "olympus/data.csv",
            "bytes": data_path.stat().st_size,
            "sha256": _sha256(data_path),
        },
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


def _dependency_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path]:
    schema_path, digest = _write_dependency_schema(tmp_path)
    data_root, data_path, contract_path = _write_dependency_assets(tmp_path, digest)
    mock_oracle_path = tmp_path / "mock_oracle.csv"
    mock_oracle_path.write_text(
        "dataset_id,base,reaction_score\nbuchwald_hartwig,A,2.5\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(dependencies, "SCHEMA_PATH", schema_path)
    monkeypatch.setattr(dependencies, "CONTRACT_PATH", contract_path)
    monkeypatch.setattr(dependencies, "MOCK_ORACLE_PATH", mock_oracle_path)
    return data_root, data_path


def _real_checks(data_root: Path) -> list:
    return check_task_dependencies(
        {
            "task": "iron_mind",
            "argv": [
                "--data-dir",
                str(data_root),
                "--dataset-id",
                "buchwald_hartwig",
            ],
            "cwd": str(data_root),
            "mode": "real",
            "env_overrides": {
                "LLM_BASE_URL": "https://example.invalid/v1",
                "LLM_MODEL_NAME": "test-model",
                "LLM_API_KEY": "test-key",
            },
        },
        include_optional=False,
    )


def test_mock_dependencies_validate_only_tracked_mock_assets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root, _data_path = _dependency_fixture(tmp_path, monkeypatch)

    checks = check_task_dependencies(
        {
            "task": "iron_mind",
            "argv": ["--mock"],
            "cwd": str(data_root),
            "mode": "mock",
        },
        include_optional=False,
    )

    assert [(check.name, check.status) for check in checks] == [
        ("mock assets", "ok")
    ]


def test_real_dependencies_verify_revisions_and_frozen_table_contracts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root, data_path = _dependency_fixture(tmp_path, monkeypatch)

    statuses = {check.name: check.status for check in _real_checks(data_root)}

    assert statuses == {
        "official source gate": "ok",
        "LLM URL": "ok",
        "LLM model": "ok",
        "LLM API key": "ok",
        "source revisions": "ok",
        "buchwald_hartwig frozen table": "ok",
    }
    data_path.write_bytes(b"A,3.5\n")
    failed = _real_checks(data_root)
    table_check = next(
        check for check in failed if check.name == "buchwald_hartwig frozen table"
    )
    assert table_check.status == "fail"
    assert "SHA-256" in table_check.message


def test_harness_dependency_check_accepts_a_protected_key_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root, _ = _dependency_fixture(tmp_path, monkeypatch)
    key_path = tmp_path / "provider_key"
    key_path.write_text("test-key", encoding="utf-8")

    checks = check_task_dependencies(
        {
            "task": "iron_mind",
            "argv": [
                "--data-dir",
                str(data_root),
                "--dataset-id",
                "buchwald_hartwig",
                "--search-method",
                "ldm_harness",
                "--harness-api-key-file",
                str(key_path),
            ],
            "cwd": str(data_root),
            "mode": "real",
            "env_overrides": {
                "LLM_BASE_URL": "https://example.invalid/v1",
                "LLM_MODEL_NAME": "test-model",
            },
        },
        include_optional=False,
    )
    statuses = {check.name: check.status for check in checks}

    assert "LLM API key" not in statuses
    assert statuses["Harness API key file"] == "ok"
