"""Prepare source-pinned SynthonBench code and released data outside the repository."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections.abc import Sequence
from pathlib import Path

from tasks.synthonbench.core.constants import (
    DATA_MANIFEST_NAME,
    OFFICIAL_DATASET_REPOSITORY,
    OFFICIAL_DATASET_REVISION,
    OFFICIAL_SOURCE_COMMIT,
    OFFICIAL_SOURCE_URL,
    SCALES,
    TARGETS,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--scale", choices=(*SCALES, "all"), default="1M")
    parser.add_argument("--hf-endpoint", default=None)
    parser.add_argument("--no-seeds", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    source_dir = args.source_dir.resolve()
    data_dir = args.data_dir.resolve()
    _prepare_source(source_dir)
    _download_data(data_dir, args.scale, include_seeds=not args.no_seeds, endpoint=args.hf_endpoint)
    _write_manifest(
        data_dir,
        source_dir,
        args.scale,
        include_seeds=not args.no_seeds,
        endpoint=args.hf_endpoint,
    )
    print(json.dumps({"data_dir": str(data_dir), "source_dir": str(source_dir), "scale": args.scale}, indent=2))
    return 0


def _prepare_source(source_dir: Path) -> None:
    if source_dir.exists():
        _validate_existing_checkout(source_dir)
    else:
        _run(["git", "clone", OFFICIAL_SOURCE_URL, str(source_dir)])
    _ensure_commit_available(source_dir)
    _run(["git", "-C", str(source_dir), "checkout", "--detach", OFFICIAL_SOURCE_COMMIT])
    _validate_existing_checkout(source_dir)


def _validate_existing_checkout(source_dir: Path) -> None:
    if not (source_dir / ".git").exists():
        raise SystemExit(f"--source-dir exists but is not a Git checkout: {source_dir}")
    status = _run(["git", "-C", str(source_dir), "status", "--porcelain"], capture=True)
    if status.stdout.strip():
        raise SystemExit(f"official source checkout is dirty: {source_dir}")


def _ensure_commit_available(source_dir: Path) -> None:
    present = _run(["git", "-C", str(source_dir), "cat-file", "-e", f"{OFFICIAL_SOURCE_COMMIT}^{{commit}}"], check=False)
    if present.returncode == 0:
        return
    _run(["git", "-C", str(source_dir), "fetch", "origin", OFFICIAL_SOURCE_COMMIT])


def _download_data(
    data_dir: Path, scale: str, *, include_seeds: bool, endpoint: str | None
) -> None:
    from huggingface_hub import snapshot_download

    data_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=OFFICIAL_DATASET_REPOSITORY,
        repo_type="dataset",
        revision=OFFICIAL_DATASET_REVISION,
        local_dir=str(data_dir),
        allow_patterns=_patterns(scale, include_seeds),
        endpoint=endpoint,
    )


def _patterns(scale: str, include_seeds: bool) -> list[str]:
    scales = SCALES if scale == "all" else (scale,)
    patterns = ["README.md", "spaces/reactions.tsv"]
    for item in scales:
        patterns.extend((
            f"spaces/synthon_space_{item}.synthons.tsv",
            f"spaces/synthon_space_{item}.properties.csv",
            f"scores/surrogate_{item}.parquet",
        ))
    if "1M" in scales:
        patterns.append("scores/glide_1M.parquet")
    if include_seeds:
        patterns.extend(f"seeds/seeds_{target}_s*.parquet" for target in TARGETS)
    return patterns


def _write_manifest(
    data_dir: Path,
    source_dir: Path,
    scale: str,
    *,
    include_seeds: bool,
    endpoint: str | None,
) -> None:
    payload = {
        "schema_version": 1,
        "source_url": OFFICIAL_SOURCE_URL,
        "source_commit": OFFICIAL_SOURCE_COMMIT,
        "dataset_repository": OFFICIAL_DATASET_REPOSITORY,
        "dataset_revision": OFFICIAL_DATASET_REVISION,
        "dataset_endpoint": endpoint or "https://huggingface.co",
        "source_dir": str(source_dir),
        "requested_scale": scale,
        "include_seeds": include_seeds,
        "artifacts": _artifact_digests(data_dir),
    }
    path = data_dir / DATA_MANIFEST_NAME
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _artifact_digests(data_dir: Path) -> dict[str, dict[str, object]]:
    files = sorted(path for path in data_dir.rglob("*") if path.is_file() and ".cache" not in path.parts)
    return {str(path.relative_to(data_dir).as_posix()): _digest(path) for path in files}


def _digest(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def _run(command: Sequence[str], *, capture: bool = False, check: bool = True):
    completed = subprocess.run(command, check=False, capture_output=capture, text=True, timeout=300)
    if check and completed.returncode != 0:
        detail = completed.stderr.strip() if capture else ""
        raise SystemExit(f"command failed ({' '.join(command)}): {detail}")
    return completed


if __name__ == "__main__":
    raise SystemExit(main())
