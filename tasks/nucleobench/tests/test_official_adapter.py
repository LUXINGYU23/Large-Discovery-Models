from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from tasks.nucleobench.scripts.prepare_official_data import (
    prepare_case_data,
    require_clean_revision,
)


def _start_sequences() -> list[str]:
    alphabet = "ACGT"
    sequences = []
    for index in range(100):
        value = index
        suffix = []
        for _ in range(4):
            suffix.append(alphabet[value % 4])
            value //= 4
        sequences.append("A" * 196 + "".join(reversed(suffix)))
    return sequences


def _write_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    starts = tmp_path / "starts.json"
    starts.write_text(json.dumps(_start_sequences()), encoding="utf-8")
    model = tmp_path / "malinois.tar.gz"
    model.write_bytes(b"source-pinned-model")
    positions = tmp_path / "positions.json"
    positions.write_text(json.dumps(list(range(200))), encoding="utf-8")
    return starts, model, positions


def _start_set_sha256(sequences: list[str]) -> str:
    return hashlib.sha256(
        json.dumps(sequences, separators=(",", ":")).encode()
    ).hexdigest()


def test_prepare_case_data_writes_a_digest_bound_manifest(tmp_path: Path) -> None:
    starts, model, positions = _write_inputs(tmp_path)
    output = tmp_path / "prepared"
    model_sha256 = hashlib.sha256(model.read_bytes()).hexdigest()

    manifest = prepare_case_data(
        case_id="malinois_k562",
        starts_path=starts,
        model_artifact=model,
        output_dir=output,
        editable_positions_path=positions,
        expected_model_sha256=model_sha256,
        expected_start_set_sha256=_start_set_sha256(_start_sequences()),
        bending_factor=1.0,
    )

    assert manifest["case_id"] == "malinois_k562"
    assert manifest["start_set"]["count"] == 100
    assert manifest["start_set"]["sha256"] == _start_set_sha256(_start_sequences())
    assert manifest["editable_positions"]["count"] == 200
    assert manifest["model_artifact"]["sha256"] == model_sha256
    assert manifest["model_init_args"] == {
        "a_max": 6.0,
        "a_min": -2.0,
        "bending_factor": 1.0,
        "flank_length": 200,
        "target_alpha": 1.0,
        "target_feature": 0,
    }
    assert json.loads((output / "prepared_manifest.json").read_text()) == manifest
    assert json.loads((output / "starts.json").read_text()) == _start_sequences()
    assert json.loads((output / "editable_positions.json").read_text()) == list(
        range(200)
    )


def test_prepare_case_data_rejects_unverified_or_invalid_inputs(tmp_path: Path) -> None:
    starts, model, positions = _write_inputs(tmp_path)

    with pytest.raises(ValueError, match="model artifact SHA-256 mismatch"):
        prepare_case_data(
            case_id="malinois_k562",
            starts_path=starts,
            model_artifact=model,
            output_dir=tmp_path / "bad-model",
            editable_positions_path=positions,
            expected_model_sha256="0" * 64,
            expected_start_set_sha256=_start_set_sha256(_start_sequences()),
            bending_factor=1.0,
        )

    duplicate_starts = tmp_path / "duplicate-starts.json"
    duplicate_starts.write_text(json.dumps(["A" * 200] * 100), encoding="utf-8")
    with pytest.raises(ValueError, match="100 unique start sequences"):
        prepare_case_data(
            case_id="malinois_k562",
            starts_path=duplicate_starts,
            model_artifact=model,
            output_dir=tmp_path / "bad-starts",
            editable_positions_path=positions,
            expected_model_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
            expected_start_set_sha256="0" * 64,
            bending_factor=1.0,
        )

    invalid_positions = tmp_path / "invalid-positions.json"
    invalid_positions.write_text(json.dumps([*range(199), 200]), encoding="utf-8")
    with pytest.raises(ValueError, match="outside the sequence"):
        prepare_case_data(
            case_id="malinois_k562",
            starts_path=starts,
            model_artifact=model,
            output_dir=tmp_path / "bad-positions",
            editable_positions_path=invalid_positions,
            expected_model_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
            expected_start_set_sha256=_start_set_sha256(_start_sequences()),
            bending_factor=1.0,
        )


def test_source_revision_check_rejects_dirty_or_mismatched_checkouts(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    subprocess.run(
        ["git", "-C", str(source), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(source), "config", "user.name", "Test"], check=True
    )
    (source / "README.md").write_text("fixture\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(source), "add", "README.md"], check=True)
    subprocess.run(["git", "-C", str(source), "commit", "-qm", "fixture"], check=True)
    revision = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    require_clean_revision(source, revision)
    with pytest.raises(ValueError, match="revision mismatch"):
        require_clean_revision(source, "0" * 40)

    (source / "README.md").write_text("dirty\n", encoding="utf-8")
    with pytest.raises(ValueError, match="dirty"):
        require_clean_revision(source, revision)
