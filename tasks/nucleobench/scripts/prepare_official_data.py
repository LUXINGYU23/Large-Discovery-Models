"""Prepare digest-bound NucleoBench inputs outside the Git repository."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from tasks.nucleobench.core.cases import get_case, load_case_catalog
from tasks.nucleobench.core.constants import (
    OFFICIAL_START_COUNT,
    TASK_ID,
    UPSTREAM_COMMIT,
    UPSTREAM_URL,
)
from tasks.nucleobench.core.source import prepare_source_checkout

TASK_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = TASK_ROOT / "resources" / "upstream_contract.json"
PREPARED_MANIFEST = "prepared_manifest.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        choices=[case.case_id for case in load_case_catalog()],
        required=True,
    )
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--starts-file", type=Path, required=True)
    parser.add_argument("--model-artifact", type=Path, required=True)
    parser.add_argument("--editable-positions-file", type=Path)
    parser.add_argument("--expected-start-set-sha256")
    parser.add_argument("--bending-factor", type=float, default=1.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    prepare_source_checkout(args.source_dir.resolve())
    manifest = prepare_case_data(
        case_id=args.case,
        starts_path=args.starts_file,
        model_artifact=args.model_artifact,
        output_dir=args.output_dir,
        editable_positions_path=args.editable_positions_file,
        expected_start_set_sha256=args.expected_start_set_sha256,
        bending_factor=args.bending_factor,
    )
    print(
        json.dumps(
            {
                "case_id": manifest["case_id"],
                "output_dir": str(args.output_dir.resolve()),
                "prepared_manifest": str(args.output_dir.resolve() / PREPARED_MANIFEST),
            },
            indent=2,
        )
    )
    return 0


def prepare_case_data(
    *,
    case_id: str,
    starts_path: Path,
    model_artifact: Path,
    output_dir: Path,
    editable_positions_path: Path | None = None,
    expected_start_set_sha256: str | None = None,
    bending_factor: float = 1.0,
) -> dict[str, Any]:
    case = get_case(case_id)
    if case.model_family == "malinois" and not math.isfinite(bending_factor):
        raise ValueError("bending_factor must be finite")
    starts_path = starts_path.resolve()
    model_artifact = model_artifact.resolve()
    output_dir = output_dir.resolve()
    if output_dir == TASK_ROOT or TASK_ROOT in output_dir.parents:
        raise ValueError("prepared data must be written outside the task repository")
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(f"output directory path is not a directory: {output_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"output directory is not empty: {output_dir}")

    contract = _load_json(CONTRACT_PATH, "upstream contract")
    case_contract = _mapping(contract, "case_preparation").get(case_id, {})
    if not isinstance(case_contract, Mapping):
        raise TypeError(f"invalid preparation contract for {case_id}")
    starts, embedded_positions = _load_starts(
        starts_path,
        case_id=case_id,
        sequence_length=case.sequence_length,
        case_contract=case_contract,
        artifacts=_mapping(contract, "artifacts"),
    )
    start_set_sha256 = _canonical_json_sha256(starts)
    official_start_sha256 = case_contract.get("start_set_sha256")
    expected_start = official_start_sha256 or expected_start_set_sha256
    if not expected_start:
        raise ValueError(
            f"{case_id} has no published paired-start digest; "
            "provide --expected-start-set-sha256 for an explicitly versioned set"
        )
    _require_expected_digest(start_set_sha256, str(expected_start), "start set")

    positions = _load_editable_positions(
        editable_positions_path,
        embedded_positions=embedded_positions,
        sequence_length=case.sequence_length,
        expected_count=case.editable_position_count,
    )
    positions_sha256 = _canonical_json_sha256(positions)
    if "editable_positions_sha256" in case_contract:
        _require_expected_digest(
            positions_sha256,
            str(case_contract["editable_positions_sha256"]),
            "editable positions",
        )
    model_digest = _digest(model_artifact)
    model_contract = _model_contract(contract, case_contract)
    _verify_model_artifact(
        model_artifact,
        model_digest,
        model_contract,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    prepared_starts = output_dir / "starts.json"
    prepared_positions = output_dir / "editable_positions.json"
    _write_json(prepared_starts, starts)
    _write_json(prepared_positions, positions)
    starts_file_digest = _digest(prepared_starts)
    positions_file_digest = _digest(prepared_positions)

    manifest = {
        "schema_version": 1,
        "task": TASK_ID,
        "case_id": case.case_id,
        "benchmark_source": {"url": UPSTREAM_URL, "revision": UPSTREAM_COMMIT},
        "start_set": {
            "path": prepared_starts.name,
            "count": len(starts),
            "sha256": start_set_sha256,
            "file_bytes": starts_file_digest["bytes"],
            "file_sha256": starts_file_digest["sha256"],
            "source_file": {"name": starts_path.name, **_digest(starts_path)},
        },
        "editable_positions": {
            "path": prepared_positions.name,
            "mode": "per_start"
            if positions and isinstance(positions[0], list)
            else "global",
            "count": case.editable_position_count,
            "sha256": positions_sha256,
            "file_bytes": positions_file_digest["bytes"],
            "file_sha256": positions_file_digest["sha256"],
        },
        "model_artifact": {"path": str(model_artifact), **model_digest},
        "model_init_args": _model_init_args(case_id, bending_factor),
    }
    _write_json(output_dir / PREPARED_MANIFEST, manifest)
    return manifest


def _load_starts(
    path: Path,
    *,
    case_id: str,
    sequence_length: int,
    case_contract: Mapping[str, Any],
    artifacts: Mapping[str, Any],
) -> tuple[list[str], Any | None]:
    start_source = case_contract.get("start_source")
    embedded_positions = None
    if path.suffix.lower() == ".csv":
        if (
            not isinstance(start_source, Mapping)
            or start_source.get("kind") != "csv_block"
        ):
            raise ValueError(f"{case_id} does not declare an official CSV start block")
        _verify_source_artifact(path, artifacts, str(start_source["artifact"]))
        payload = _load_start_sequences_csv(
            path,
            first_index=int(start_source["first_index"]),
            last_index=int(start_source["last_index"]),
            sequence_column=str(start_source.get("sequence_column", "0")),
        )
    elif path.suffix.lower() in {".parquet", ".pq"}:
        if (
            not isinstance(start_source, Mapping)
            or start_source.get("kind") != "parquet"
        ):
            raise ValueError(
                f"{case_id} does not declare an official Parquet start set"
            )
        _verify_source_artifact(path, artifacts, str(start_source["artifact"]))
        payload, embedded_positions = _load_enformer_parquet(path)
    else:
        payload = _load_json(path, "start sequence file")
    starts = _validate_starts(payload, sequence_length)
    return starts, embedded_positions


def _load_start_sequences_csv(
    path: Path,
    *,
    first_index: int,
    last_index: int,
    sequence_column: str,
) -> list[str]:
    csv.field_size_limit(max(csv.field_size_limit(), 1_000_000))
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header is None or sequence_column not in header:
            raise ValueError(
                f"start sequence CSV must contain the official {sequence_column!r} column"
            )
        sequence_index = header.index(sequence_column)
        selected: dict[int, str] = {}
        for row_number, row in enumerate(reader, start=2):
            if len(row) <= sequence_index:
                raise ValueError(f"start sequence CSV row {row_number} is incomplete")
            try:
                source_index = int(row[0])
            except (IndexError, ValueError) as exc:
                raise ValueError(
                    f"start sequence CSV row {row_number} has an invalid source index"
                ) from exc
            if first_index <= source_index <= last_index:
                selected[source_index] = row[sequence_index].strip()
    expected = list(range(first_index, last_index + 1))
    if sorted(selected) != expected:
        raise ValueError(
            "start sequence CSV does not contain the declared source block"
        )
    return [selected[index] for index in expected]


def _load_enformer_parquet(path: Path) -> tuple[list[str], list[list[int]]]:
    try:
        pandas = importlib.import_module("pandas")
        frame = pandas.read_parquet(path)
    except ImportError as exc:
        raise RuntimeError(
            "reading official Enformer starts requires pandas and a Parquet engine"
        ) from exc
    required = {"sequence", "positions_to_mutate"}
    if not required.issubset(frame.columns):
        raise ValueError(
            "official Enformer Parquet file must contain sequence and "
            "positions_to_mutate columns"
        )
    return (
        [str(value) for value in frame["sequence"].tolist()],
        [
            [int(position) for position in value]
            for value in frame["positions_to_mutate"]
        ],
    )


def _validate_starts(payload: Any, sequence_length: int) -> list[str]:
    if not isinstance(payload, list) or len(payload) != OFFICIAL_START_COUNT:
        raise ValueError(
            f"start sequence file must contain exactly {OFFICIAL_START_COUNT} sequences"
        )
    if any(not isinstance(sequence, str) for sequence in payload):
        raise ValueError("every start sequence must be a string")
    starts = list(payload)
    if len(set(starts)) != OFFICIAL_START_COUNT:
        raise ValueError("start sequence file must contain 100 unique start sequences")
    for index, sequence in enumerate(starts):
        if len(sequence) != sequence_length:
            raise ValueError(
                f"start sequence {index} has length {len(sequence)}; expected {sequence_length}"
            )
        if not set(sequence).issubset({"A", "C", "G", "T"}):
            raise ValueError(f"start sequence {index} contains a non-DNA base")
    return starts


def _load_editable_positions(
    path: Path | None,
    *,
    embedded_positions: Any | None,
    sequence_length: int,
    expected_count: int,
) -> list[int] | list[list[int]]:
    if path is not None:
        payload = _load_json(path.resolve(), "editable positions file")
    elif embedded_positions is not None:
        payload = embedded_positions
    elif expected_count == sequence_length:
        payload = list(range(sequence_length))
    else:
        raise ValueError(
            "this case requires an explicit per-start editable-position set"
        )
    if not isinstance(payload, list):
        raise TypeError("editable positions file must contain a JSON array")
    masks = payload if payload and isinstance(payload[0], list) else [payload]
    if len(masks) not in {1, OFFICIAL_START_COUNT}:
        raise ValueError("editable positions must contain one global mask or 100 masks")
    validated = [
        _validate_position_mask(mask, sequence_length, expected_count) for mask in masks
    ]
    return validated if len(validated) == OFFICIAL_START_COUNT else validated[0]


def _validate_position_mask(
    payload: Any,
    sequence_length: int,
    expected_count: int,
) -> list[int]:
    if not isinstance(payload, list) or any(
        isinstance(position, bool) or not isinstance(position, int)
        for position in payload
    ):
        raise ValueError("editable positions must contain only integers")
    positions = [int(position) for position in payload]
    if len(positions) != expected_count or len(set(positions)) != expected_count:
        raise ValueError(
            f"editable positions must contain {expected_count} unique entries"
        )
    if any(position < 0 or position >= sequence_length for position in positions):
        raise ValueError("editable positions contains a position outside the sequence")
    return sorted(positions)


def _model_init_args(case_id: str, bending_factor: float) -> dict[str, Any]:
    case = get_case(case_id)
    if case.model_family == "malinois":
        return {
            **case.model_selector,
            "bending_factor": float(bending_factor),
            "a_min": -2.0,
            "a_max": 6.0,
            "target_alpha": 1.0,
            "flank_length": 200,
        }
    if case.model_family == "enformer":
        return {
            **case.model_selector,
            "spatial_bins_to_aggregate": None,
            "run_sanity_checks": True,
        }
    return dict(case.model_selector)


def _model_contract(
    contract: Mapping[str, Any], case_contract: Mapping[str, Any]
) -> Mapping[str, Any]:
    key = case_contract.get("model_artifact")
    artifacts = _mapping(contract, "artifacts")
    if not isinstance(key, str) or key not in artifacts:
        raise ValueError("case does not declare an official model artifact")
    value = artifacts[key]
    if not isinstance(value, Mapping):
        raise TypeError(f"invalid model artifact contract: {key}")
    return value


def _verify_source_artifact(
    path: Path,
    artifacts: Mapping[str, Any],
    key: str,
) -> None:
    contract = artifacts.get(key)
    if not isinstance(contract, Mapping):
        raise TypeError(f"unknown source artifact contract: {key}")
    digest = _digest(path)
    if digest["bytes"] != contract.get("bytes") or digest["sha256"] != contract.get(
        "sha256"
    ):
        raise ValueError(f"source artifact does not match the pinned {key} bytes")


def _verify_model_artifact(
    path: Path,
    digest: Mapping[str, int | str],
    contract: Mapping[str, Any],
) -> None:
    if digest["bytes"] != contract.get("bytes"):
        raise ValueError("model artifact size does not match the pinned artifact")
    if "sha256" in contract:
        _require_expected_digest(
            str(digest["sha256"]), str(contract["sha256"]), "model artifact"
        )
    elif "md5" in contract:
        actual_md5 = _hash_file(path, "md5")
        if actual_md5 != str(contract["md5"]).lower():
            raise ValueError("model artifact MD5 mismatch")
    else:
        raise ValueError("model artifact contract has no digest")


def _load_json(path: Path, label: str) -> Any:
    if not path.is_file():
        raise ValueError(f"{label} does not exist: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON: {path}") from exc


def _mapping(parent: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise TypeError(f"{key} must be an object")
    return value


def _canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _require_expected_digest(actual: str, expected: str, label: str) -> None:
    if actual != expected.lower():
        raise ValueError(f"{label} SHA-256 mismatch: expected {expected}, got {actual}")


def _digest(path: Path) -> dict[str, int | str]:
    if not path.is_file():
        raise ValueError(f"artifact does not exist: {path}")
    return {
        "bytes": path.stat().st_size,
        "sha256": _hash_file(path, "sha256"),
    }


def _hash_file(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
