"""Publication-facing records for official NucleoBench output files."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


def inventory_official_outputs(output_dir: Path, run_dir: Path) -> list[dict[str, Any]]:
    output_dir = Path(output_dir).resolve()
    run_dir = Path(run_dir).resolve()
    if not output_dir.is_relative_to(run_dir):
        raise ValueError("official output directory must be inside the campaign directory")
    for marker in ("START.txt", "SUCCESS.txt"):
        if not (output_dir / marker).is_file():
            raise ValueError(f"official runner did not write {marker}")
    files = sorted(path for path in output_dir.rglob("*") if path.is_file())
    if not any(path.suffix == ".parquet" for path in files):
        raise ValueError("official runner did not write a Parquet result")
    return [
        {
            "path": path.relative_to(run_dir).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in files
    ]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ["inventory_official_outputs"]
