"""Tests for Iron Mind's pinned reaction-table contracts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from tasks.iron_mind.core.data import load_frozen_reaction_table
from tasks.iron_mind.core.schema import load_reaction_schemas


TASK_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = TASK_ROOT / "resources" / "reaction_schemas.json"
CONTRACT_PATH = TASK_ROOT / "resources" / "upstream_contract.json"
BUCHWALD_SCHEMA_SHA256 = "9c25bd7cab474077bece3915866471a6d00a785a7ba2cb6c9dc35b6428466dc8"
CHAN_LAM_SCHEMA_SHA256 = "0b4c963fe7cf2a9d1c10088ef8963db6a730dff22ca4d7f3d765cc8ae25849b6"


def test_tracked_schemas_match_the_frozen_scientific_contract() -> None:
    schemas = load_reaction_schemas(SCHEMA_PATH)

    buchwald = schemas["buchwald_hartwig"]
    assert buchwald.factor_names == ("base", "ligand", "aryl_halide", "additive")
    assert buchwald.category_counts == (3, 4, 16, 24)
    assert buchwald.one_hot_dimension == 47
    assert buchwald.measurements == ("yield",)
    assert buchwald.objective == "reaction_score"
    assert buchwald.direction == "maximize"
    assert buchwald.schema_sha256 == BUCHWALD_SCHEMA_SHA256

    chan_lam = schemas["chan_lam_full"]
    assert chan_lam.factor_names == (
        "boronic_acid_reactant",
        "sulfonamide_reactant",
        "catalyst_catalyst",
        "base_reagent",
        "solvent",
    )
    assert chan_lam.category_counts == (2, 10, 4, 6, 4)
    assert chan_lam.one_hot_dimension == 26
    assert chan_lam.measurements == ("desired_yield", "undesired_yield")
    assert chan_lam.schema_sha256 == CHAN_LAM_SCHEMA_SHA256


def test_upstream_contract_pins_the_four_runtime_assets() -> None:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))

    assert contract["sources"]["iron_mind_public"]["revision"] == (
        "476c555e45e2556e2ee4b24c726e774c2bfb7762"
    )
    assert contract["sources"]["olympus"]["revision"] == (
        "7b4bb35c04eb31dc57a8e46cc79a9cab71dee06d"
    )
    assert contract["datasets"]["buchwald_hartwig"]["row_count"] == 4599
    assert contract["datasets"]["chan_lam_full"]["row_count"] == 5684
    assert contract["datasets"]["buchwald_hartwig"]["artifacts"]["data"]["sha256"] == (
        "5cfeb90fecdedc6eeab1fa4aba01654a2813ec45785d5bd61bb79e3afdef7bdd"
    )
    assert contract["datasets"]["chan_lam_full"]["artifacts"]["data"]["sha256"] == (
        "58945ee6c495abad30e665ee9e060f522ab526739a7d6d8d8eeb16f351802d00"
    )


def test_loader_binds_headerless_columns_in_config_order(tmp_path: Path) -> None:
    schema = load_reaction_schemas(SCHEMA_PATH)["buchwald_hartwig"]
    rows = [("P2Et", "XPhos", "None", "None", "26.8886154")]
    config_path, data_path = _write_fixture(tmp_path, schema, rows)

    table = load_frozen_reaction_table(
        schema=schema,
        config_path=config_path,
        data_path=data_path,
        artifact_contract=_artifact_contract(config_path, data_path, row_count=1),
    )

    row = table.rows[0]
    assert row.row_id == 1
    assert row.conditions == {
        "base": "P2Et",
        "ligand": "XPhos",
        "aryl_halide": "None",
        "additive": "None",
    }
    assert row.measurements == {"yield": 26.8886154}
    assert row.raw_row_sha256 == _sha256_text(",".join(rows[0]))
    assert table.rows_for_conditions(row.conditions) == (row,)


@pytest.mark.parametrize(
    ("row", "error"),
    [
        (("P2Et", "XPhos", "None", "26.8886154"), "column count"),
        (("UNKNOWN", "XPhos", "None", "None", "26.8886154"), "Unknown category"),
        (("P2Et", "XPhos", "None", "None", "nan"), "Non-finite measurement"),
        (("P2Et", "XPhos", "None", "None", "inf"), "Non-finite measurement"),
    ],
)
def test_loader_rejects_invalid_headerless_rows(
    tmp_path: Path, row: tuple[str, ...], error: str
) -> None:
    schema = load_reaction_schemas(SCHEMA_PATH)["buchwald_hartwig"]
    config_path, data_path = _write_fixture(tmp_path, schema, [row])

    with pytest.raises(ValueError, match=error):
        load_frozen_reaction_table(
            schema=schema,
            config_path=config_path,
            data_path=data_path,
            artifact_contract=_artifact_contract(config_path, data_path, row_count=1),
        )


def test_loader_rejects_artifact_mismatch_before_table_construction(tmp_path: Path) -> None:
    schema = load_reaction_schemas(SCHEMA_PATH)["buchwald_hartwig"]
    rows = [("P2Et", "XPhos", "None", "None", "26.8886154")]
    config_path, data_path = _write_fixture(tmp_path, schema, rows)
    contract = _artifact_contract(config_path, data_path, row_count=1)
    contract["artifacts"]["config"]["sha256"] = "0" * 64

    with pytest.raises(ValueError, match="config artifact SHA-256 mismatch"):
        load_frozen_reaction_table(
            schema=schema,
            config_path=config_path,
            data_path=data_path,
            artifact_contract=contract,
        )


def test_loader_rejects_artifact_byte_size_mismatch_before_table_construction(
    tmp_path: Path,
) -> None:
    schema = load_reaction_schemas(SCHEMA_PATH)["buchwald_hartwig"]
    rows = [("P2Et", "XPhos", "None", "None", "26.8886154")]
    config_path, data_path = _write_fixture(tmp_path, schema, rows)
    contract = _artifact_contract(config_path, data_path, row_count=1)
    contract["artifacts"]["config"]["bytes"] += 1

    with pytest.raises(ValueError, match="config artifact byte size mismatch"):
        load_frozen_reaction_table(
            schema=schema,
            config_path=config_path,
            data_path=data_path,
            artifact_contract=contract,
        )


def test_loader_rejects_config_schema_drift(tmp_path: Path) -> None:
    schema = load_reaction_schemas(SCHEMA_PATH)["buchwald_hartwig"]
    rows = [("P2Et", "XPhos", "None", "None", "26.8886154")]
    config_path, data_path = _write_fixture(tmp_path, schema, rows)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["parameters"] = list(reversed(config["parameters"]))
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="Config schema digest mismatch"):
        load_frozen_reaction_table(
            schema=schema,
            config_path=config_path,
            data_path=data_path,
            artifact_contract=_artifact_contract(config_path, data_path, row_count=1),
        )


def test_buchwald_rejects_duplicate_observation_keys(tmp_path: Path) -> None:
    schema = load_reaction_schemas(SCHEMA_PATH)["buchwald_hartwig"]
    rows = [
        ("P2Et", "XPhos", "None", "None", "26.8886154"),
        ("P2Et", "XPhos", "None", "None", "24.0632241"),
    ]
    config_path, data_path = _write_fixture(tmp_path, schema, rows)

    with pytest.raises(ValueError, match="Duplicate observation key"):
        load_frozen_reaction_table(
            schema=schema,
            config_path=config_path,
            data_path=data_path,
            artifact_contract=_artifact_contract(config_path, data_path, row_count=2),
        )


def test_chan_lam_keeps_replicates_and_stable_raw_row_identity(tmp_path: Path) -> None:
    schema = load_reaction_schemas(SCHEMA_PATH)["chan_lam_full"]
    conditions = tuple(factor.categories[0] for factor in schema.factors)
    rows = [conditions + ("78.8", "1.04"), conditions + ("77.22", "0.93")]
    config_path, data_path = _write_fixture(tmp_path, schema, rows)

    table = load_frozen_reaction_table(
        schema=schema,
        config_path=config_path,
        data_path=data_path,
        artifact_contract=_artifact_contract(config_path, data_path, row_count=2),
    )

    assert [row.row_id for row in table.rows] == [1, 2]
    assert [row.raw_row_sha256 for row in table.rows] == [
        _sha256_text(",".join(rows[0])),
        _sha256_text(",".join(rows[1])),
    ]
    assert table.rows_for_conditions(table.rows[0].conditions) == table.rows


def _write_fixture(
    tmp_path: Path, schema: Any, rows: list[tuple[str, ...]]
) -> tuple[Path, Path]:
    config_path = tmp_path / "config.json"
    data_path = tmp_path / "data.csv"
    config = {
        "parameters": [
            {"name": factor.name, "type": "categorical", "options": list(factor.categories)}
            for factor in schema.factors
        ],
        "measurements": [
            {"name": measurement, "type": "continuous"}
            for measurement in schema.measurements
        ],
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")
    data_path.write_text("\n".join(",".join(row) for row in rows) + "\n", encoding="utf-8")
    return config_path, data_path


def _artifact_contract(config_path: Path, data_path: Path, *, row_count: int) -> dict[str, Any]:
    return {
        "artifacts": {
            "config": _artifact_entry(config_path),
            "data": _artifact_entry(data_path),
        },
        "row_count": row_count,
    }


def _artifact_entry(path: Path) -> dict[str, Any]:
    return {"bytes": path.stat().st_size, "sha256": _sha256_file(path)}


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
