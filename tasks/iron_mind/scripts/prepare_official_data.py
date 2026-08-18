"""Materialize the exact complete Iron Mind data snapshot from pinned checkouts."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Mapping


TASK_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = TASK_ROOT / "resources" / "upstream_contract.json"
HASH_CHUNK_BYTES = 1024 * 1024


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iron-mind-checkout", type=Path, required=True)
    parser.add_argument("--olympus-checkout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    contract = _read_contract()
    _require_revision(
        args.iron_mind_checkout,
        contract["sources"]["iron_mind_public"]["revision"],
        "iron-mind-public",
    )
    _require_revision(
        args.olympus_checkout,
        contract["sources"]["olympus"]["revision"],
        "olympus",
    )
    args.output.mkdir(parents=True, exist_ok=True)
    copied = _materialize_datasets(contract, args.olympus_checkout, args.output)
    _write_manifest(contract, args.output, copied)
    print(json.dumps({"status": "ok", "output": str(args.output), "artifacts": copied}, indent=2))
    return 0


def _materialize_datasets(
    contract: Mapping[str, Any], olympus_checkout: Path, output: Path
) -> int:
    copied = 0
    for dataset_id, dataset in contract["datasets"].items():
        for label, artifact in dataset["artifacts"].items():
            source = olympus_checkout / artifact["upstream_path"]
            destination = output / artifact["path"]
            _require_artifact(source, artifact, f"{dataset_id} {label} source")
            if destination.exists():
                _require_artifact(destination, artifact, f"{dataset_id} {label} destination")
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            _require_artifact(destination, artifact, f"{dataset_id} {label} destination")
            copied += 1
    return copied


def _write_manifest(contract: Mapping[str, Any], output: Path, copied: int) -> None:
    payload = {
        "schema_version": 2,
        "snapshot": "iron_mind_official_complete",
        "sources": [
            {
                "name": "iron-mind-public",
                "revision": contract["sources"]["iron_mind_public"]["revision"],
            },
            {
                "name": "olympus",
                "revision": contract["sources"]["olympus"]["revision"],
            },
        ],
        "suites": contract["suites"],
        "artifact_count": sum(
            len(dataset["artifacts"]) for dataset in contract["datasets"].values()
        ),
        "copied_artifact_count": copied,
    }
    (output / "revision_manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _require_revision(checkout: Path, expected: str, label: str) -> None:
    if not checkout.is_dir():
        raise ValueError(f"{label} checkout does not exist: {checkout}")
    result = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    actual = result.stdout.strip()
    if actual != expected:
        raise ValueError(f"{label} revision mismatch: expected {expected}, got {actual}")


def _require_artifact(path: Path, contract: Mapping[str, Any], label: str) -> None:
    if not path.is_file():
        raise ValueError(f"{label} does not exist: {path}")
    if path.stat().st_size != contract["bytes"]:
        raise ValueError(f"{label} byte size mismatch")
    if _sha256(path) != contract["sha256"]:
        raise ValueError(f"{label} SHA-256 mismatch")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_contract() -> dict[str, Any]:
    payload = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 2:
        raise ValueError("Complete Iron Mind upstream contract must use schema version 2.")
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
