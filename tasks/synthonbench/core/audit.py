"""Exact official submission export and at-scale SynthonBench scoring."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from tasks.synthonbench.core.constants import TOP_K_BY_SCALE
from tasks.synthonbench.core.data import score_table_path


def write_submission_csv(observed_ids: set[str], path: Path) -> Path:
    """Export exactly the charged official product IDs in deterministic order."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["product_id"])
        writer.writeheader()
        writer.writerows({"product_id": product_id} for product_id in sorted(observed_ids))
    return path


def run_official_submission_audit(
    *, source_dir: Path, data_dir: Path, scale: str, target: str, oracle_kind: str,
    submission_path: Path, output_path: Path, timeout_seconds: float,
) -> dict[str, Any]:
    """Execute the source-pinned official score_submission.py unchanged."""

    command = _audit_command(source_dir, data_dir, scale, target, oracle_kind, submission_path, output_path)
    completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=timeout_seconds)
    if completed.returncode != 0:
        raise RuntimeError(f"official SynthonBench audit failed: {completed.stderr.strip()}")
    return _load_audit_output(output_path, completed.stdout)


def _audit_command(source_dir, data_dir, scale, target, oracle_kind, submission_path, output_path) -> list[str]:
    return [
        sys.executable,
        str(Path(source_dir) / "scripts" / "score_submission.py"),
        "--score-table", str(score_table_path(Path(data_dir), scale, oracle_kind)),
        "--target", target,
        "--kind", oracle_kind,
        "--submission", str(submission_path),
        "--top-k", str(TOP_K_BY_SCALE[scale]),
        "--direction", "maximize",
        "--output", str(output_path),
    ]


def _load_audit_output(path: Path, stdout: str) -> dict[str, Any]:
    source = path.read_text(encoding="utf-8") if path.is_file() else stdout
    try:
        payload = json.loads(source)
    except json.JSONDecodeError as exc:
        raise RuntimeError("official SynthonBench audit did not emit JSON metrics") from exc
    if not isinstance(payload, dict):
        raise TypeError("official SynthonBench audit emitted a non-object result")
    return payload


__all__ = ["run_official_submission_audit", "write_submission_csv"]
