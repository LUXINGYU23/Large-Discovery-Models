"""Shared primitives for compact Iron Mind qualification records."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ldm_tts.registration.experiment import ExperimentContract, load_experiment_contract


TASK_ID = "iron_mind"
TASK_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = TASK_ROOT.parents[1]
TASK_MANIFEST_PATH = TASK_ROOT / "task.json"
EXPERIMENT_CONTRACT_PATH = TASK_ROOT / "experiment.json"
UPSTREAM_CONTRACT_PATH = TASK_ROOT / "resources" / "upstream_contract.json"


class QualificationRecordError(ValueError):
    """Raised when evidence cannot support a passed qualification record."""


def load_iron_mind_contract() -> ExperimentContract:
    """Load the sole tracked experiment contract for this task."""

    return load_experiment_contract(EXPERIMENT_CONTRACT_PATH)


def sha256_file(path: Path) -> str:
    """Return the SHA-256 for one required evidence artifact."""

    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise QualificationRecordError(f"Required evidence artifact is unavailable: {path}") from exc
    return digest.hexdigest()


def read_json_object(path: Path, label: str) -> dict[str, Any]:
    """Read one required JSON object with an evidence-specific error."""

    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise QualificationRecordError(f"Could not read {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise QualificationRecordError(f"{label.capitalize()} must be a JSON object.")
    return payload


def file_reference(path: Path) -> dict[str, str]:
    """Build a repository-relative immutable reference for one tracked file."""

    resolved = Path(path).resolve()
    try:
        relative = resolved.relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError as exc:
        raise QualificationRecordError(f"Evidence file is outside the repository: {resolved}") from exc
    return {"path": relative, "sha256": sha256_file(resolved)}


def contract_reference(contract: ExperimentContract) -> dict[str, str]:
    """Return the canonical file path and digest for an experiment contract."""

    return {"path": file_reference(contract.path)["path"], "sha256": contract.digest}
