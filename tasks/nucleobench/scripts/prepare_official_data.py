"""Validate pinned NucleoBench source and prepare one official case outside Git."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from tasks.nucleobench.core.cases import get_case
from tasks.nucleobench.core.constants import (
    OFFICIAL_START_COUNT,
    TASK_ID,
    UPSTREAM_COMMIT,
    UPSTREAM_URL,
)
from tasks.nucleobench.core.source import prepare_source_checkout


TASK_ROOT = Path(__file__).resolve().parents[1]
PREPARED_MANIFEST = "prepared_manifest.json"
SUPPORTED_CASE = "malinois_k562"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=[SUPPORTED_CASE], required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--starts-file", type=Path, required=True)
    parser.add_argument("--model-artifact", type=Path, required=True)
    parser.add_argument("--editable-positions-file", type=Path)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--start-set-sha256", required=True)
    parser.add_argument("--bending-factor", type=float, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    source_dir = args.source_dir.resolve()
    prepare_source_checkout(source_dir)
    manifest = prepare_case_data(
        case_id=args.case,
        starts_path=args.starts_file,
        model_artifact=args.model_artifact,
        output_dir=args.output_dir,
        editable_positions_path=args.editable_positions_file,
        expected_model_sha256=args.model_sha256,
        expected_start_set_sha256=args.start_set_sha256,
        bending_factor=args.bending_factor,
    )
    print(
        json.dumps(
            {
                "case_id": manifest["case_id"],
                "output_dir": str(args.output_dir.resolve()),
                "prepared_manifest": str(
                    args.output_dir.resolve() / PREPARED_MANIFEST
                ),
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
    expected_model_sha256: str,
    expected_start_set_sha256: str,
    bending_factor: float,
) -> dict[str, Any]:
    """Validate one case's external inputs and write a digest-bound manifest."""

    if case_id != SUPPORTED_CASE:
        raise ValueError(f"only {SUPPORTED_CASE} is currently preparable")
    if not math.isfinite(bending_factor):
        raise ValueError("bending_factor must be finite")
    case = get_case(case_id)
    starts_path = starts_path.resolve()
    model_artifact = model_artifact.resolve()
    output_dir = output_dir.resolve()
    if output_dir == TASK_ROOT or TASK_ROOT in output_dir.parents:
        raise ValueError("prepared data must be written outside the task repository")
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(f"output directory path is not a directory: {output_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"output directory is not empty: {output_dir}")

    starts = _load_start_sequences(starts_path, case.sequence_length)
    start_set_sha256 = _canonical_json_sha256(starts)
    _require_expected_digest(
        start_set_sha256,
        expected_start_set_sha256,
        "start set",
    )
    positions = _load_editable_positions(
        editable_positions_path,
        sequence_length=case.sequence_length,
        expected_count=case.editable_position_count,
    )
    model_digest = _digest(model_artifact)
    _require_expected_digest(
        str(model_digest["sha256"]),
        expected_model_sha256,
        "model artifact",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    prepared_starts = output_dir / "starts.json"
    prepared_positions = output_dir / "editable_positions.json"
    _write_json(prepared_starts, starts)
    _write_json(prepared_positions, positions)
    prepared_starts_digest = _digest(prepared_starts)

    manifest = {
        "schema_version": 1,
        "task": TASK_ID,
        "case_id": case.case_id,
        "benchmark_source": {
            "url": UPSTREAM_URL,
            "revision": UPSTREAM_COMMIT,
        },
        "start_set": {
            "path": prepared_starts.name,
            "count": len(starts),
            "sha256": start_set_sha256,
            "file_bytes": prepared_starts_digest["bytes"],
            "file_sha256": prepared_starts_digest["sha256"],
        },
        "editable_positions": {
            "path": prepared_positions.name,
            "count": len(positions),
            **_digest(prepared_positions),
        },
        "model_artifact": {
            "path": str(model_artifact),
            **model_digest,
        },
        "model_init_args": {
            "target_feature": case.model_selector["target_feature"],
            "bending_factor": float(bending_factor),
            "a_min": -2.0,
            "a_max": 6.0,
            "target_alpha": 1.0,
            "flank_length": 200,
        },
    }
    _write_json(output_dir / PREPARED_MANIFEST, manifest)
    return manifest


def _load_start_sequences(path: Path, sequence_length: int) -> list[str]:
    payload = _load_json(path, "start sequence file")
    if not isinstance(payload, list) or len(payload) != OFFICIAL_START_COUNT:
        raise ValueError(
            f"start sequence file must contain exactly {OFFICIAL_START_COUNT} sequences"
        )
    if any(not isinstance(sequence, str) for sequence in payload):
        raise ValueError("every start sequence must be a string")
    starts = [str(sequence) for sequence in payload]
    if len(set(starts)) != OFFICIAL_START_COUNT:
        raise ValueError(
            f"start sequence file must contain {OFFICIAL_START_COUNT} unique start sequences"
        )
    for index, sequence in enumerate(starts):
        if len(sequence) != sequence_length:
            raise ValueError(
                f"start sequence {index} has length {len(sequence)}; expected {sequence_length}"
            )
        if not set(sequence).issubset(set("ACGT")):
            raise ValueError(f"start sequence {index} contains a non-DNA base")
    return starts


def _load_editable_positions(
    path: Path | None,
    *,
    sequence_length: int,
    expected_count: int,
) -> list[int]:
    payload: Any = list(range(sequence_length)) if path is None else _load_json(
        path.resolve(), "editable positions file"
    )
    if not isinstance(payload, list):
        raise ValueError("editable positions file must contain a JSON array")
    if any(
        isinstance(position, bool) or not isinstance(position, int)
        for position in payload
    ):
        raise ValueError("editable positions must contain only integers")
    positions = [int(position) for position in payload]
    if len(positions) != len(set(positions)):
        raise ValueError("editable positions must not contain duplicates")
    if any(position < 0 or position >= sequence_length for position in positions):
        raise ValueError("editable positions contains a position outside the sequence")
    if len(positions) != expected_count:
        raise ValueError(
            f"editable positions count must equal {expected_count}, got {len(positions)}"
        )
    return sorted(positions)


def _load_json(path: Path, label: str) -> Any:
    if not path.is_file():
        raise ValueError(f"{label} does not exist: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON: {path}") from exc


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
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
