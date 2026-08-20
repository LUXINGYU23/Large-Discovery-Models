"""Source-pinned SynthonBench loading with no automatic data download."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tasks.synthonbench.core.constants import (
    DATA_MANIFEST_NAME,
    OFFICIAL_DATASET_REPOSITORY,
    OFFICIAL_DATASET_REVISION,
    OFFICIAL_PACKAGE_VERSION,
    OFFICIAL_SOURCE_COMMIT,
    OFFICIAL_SOURCE_URL,
)


@dataclass(frozen=True)
class LoadedSynthonBenchmark:
    """The exact official task and provenance needed by one LDM campaign."""

    task: Any
    scale: str
    target: str
    oracle_kind: str
    data_dir: Path | None
    source_dir: Path | None
    mode: str


def load_mock_benchmark(*, budget: int, seed: int) -> LoadedSynthonBenchmark:
    """Use SynthonBench's bundled official example space and Pairwise oracle."""

    _require_official_package()
    from synthonbench.benchmark import make_example_task

    return LoadedSynthonBenchmark(make_example_task(budget=budget, seed=seed), "example", "mock", "pairwise", None, None, "mock")


def load_official_benchmark(
    *, data_dir: Path, source_dir: Path, scale: str, target: str, oracle_kind: str,
    budget: int, seed: int, allowed_reactions: Sequence[str] | None = None,
) -> LoadedSynthonBenchmark:
    """Load the released space and score table directly from pinned local files."""

    _require_official_package()
    validate_prepared_data(data_dir, source_dir, scale, oracle_kind)
    from synthonbench.oracle import ScoreTableOracle
    from synthonbench.space import SynthonSpace
    from synthonbench.task import GlobalSynthonTask

    space = SynthonSpace.from_tsv(space_path(data_dir, scale))
    oracle = _load_oracle(ScoreTableOracle, score_table_path(data_dir, scale, oracle_kind), target, oracle_kind)
    task = GlobalSynthonTask(space, oracle, budget=budget, seed=seed, allowed_reactions=allowed_reactions,
                             task_id=f"{target}-{scale}-{oracle_kind}")
    return LoadedSynthonBenchmark(task, scale, target, oracle_kind, data_dir, source_dir, "real")


def validate_prepared_data(data_dir: Path, source_dir: Path, scale: str, oracle_kind: str) -> None:
    """Reject unpinned or incomplete data instead of downloading implicitly."""

    manifest = _read_manifest(data_dir / DATA_MANIFEST_NAME)
    _validate_manifest(manifest)
    _validate_source_checkout(source_dir)
    required = (space_path(data_dir, scale), score_table_path(data_dir, scale, oracle_kind))
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ValueError("prepared SynthonBench data is missing: " + ", ".join(missing))
    _validate_required_artifacts(manifest, data_dir, required)


def space_path(data_dir: Path, scale: str) -> Path:
    return Path(data_dir) / "spaces" / f"synthon_space_{scale}.synthons.tsv"


def score_table_path(data_dir: Path, scale: str, oracle_kind: str) -> Path:
    filename = "glide_1M.parquet" if oracle_kind == "glide" else f"surrogate_{scale}.parquet"
    return Path(data_dir) / "scores" / filename


def _load_oracle(score_class, path: Path, target: str, oracle_kind: str):
    if oracle_kind == "glide":
        return score_class.from_glide_parquet(path, target)
    return score_class.from_surrogate_parquet(path, target)


def _require_official_package() -> None:
    import synthonbench

    if synthonbench.__version__ != OFFICIAL_PACKAGE_VERSION:
        raise RuntimeError(
            f"synthonbench version must be {OFFICIAL_PACKAGE_VERSION}, got {synthonbench.__version__}"
        )


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"prepared SynthonBench manifest is missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"prepared SynthonBench manifest is invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise TypeError("prepared SynthonBench manifest must be an object")
    return payload


def _validate_manifest(manifest: dict[str, Any]) -> None:
    expected = {
        "source_url": OFFICIAL_SOURCE_URL,
        "source_commit": OFFICIAL_SOURCE_COMMIT,
        "dataset_repository": OFFICIAL_DATASET_REPOSITORY,
        "dataset_revision": OFFICIAL_DATASET_REVISION,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"prepared SynthonBench manifest has unexpected {key!r}")


def _validate_required_artifacts(
    manifest: dict[str, Any], data_dir: Path, required: Sequence[Path]
) -> None:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise TypeError("prepared SynthonBench manifest has no artifact digests")
    for path in required:
        relative = path.relative_to(data_dir).as_posix()
        expected = artifacts.get(relative)
        if not isinstance(expected, dict):
            raise TypeError(f"prepared SynthonBench manifest has no digest for {relative}")
        actual = _artifact_digest(path)
        if actual != expected:
            raise ValueError(f"prepared SynthonBench artifact digest differs for {relative}")


def _artifact_digest(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def _validate_source_checkout(source_dir: Path) -> None:
    script = Path(source_dir) / "scripts" / "score_submission.py"
    if not script.is_file():
        raise ValueError(f"official SynthonBench scorer is missing: {script}")
    completed = subprocess.run(
        ["git", "-C", str(source_dir), "rev-parse", "HEAD"],
        check=False, capture_output=True, text=True, timeout=20,
    )
    if completed.returncode != 0 or completed.stdout.strip() != OFFICIAL_SOURCE_COMMIT:
        raise ValueError("official SynthonBench checkout does not match the pinned source commit")
    status = subprocess.run(
        ["git", "-C", str(source_dir), "status", "--porcelain"],
        check=False, capture_output=True, text=True, timeout=20,
    )
    if status.returncode != 0 or status.stdout.strip():
        raise ValueError("official SynthonBench checkout must be clean")


__all__ = [
    "LoadedSynthonBenchmark",
    "load_mock_benchmark",
    "load_official_benchmark",
    "score_table_path",
    "space_path",
    "validate_prepared_data",
]
