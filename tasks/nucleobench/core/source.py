"""Pinned official-source checkout validation."""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from pathlib import Path

from tasks.nucleobench.core.constants import UPSTREAM_COMMIT, UPSTREAM_URL


def prepare_source_checkout(
    source_dir: Path,
    *,
    source_url: str = UPSTREAM_URL,
    revision: str = UPSTREAM_COMMIT,
) -> None:
    source_dir = Path(source_dir).resolve()
    if source_dir.exists():
        _require_checkout(source_dir)
        _require_clean(source_dir)
    else:
        source_dir.parent.mkdir(parents=True, exist_ok=True)
        _run(["git", "clone", source_url, str(source_dir)])

    present = _run(
        ["git", "-C", str(source_dir), "cat-file", "-e", f"{revision}^{{commit}}"],
        check=False,
    )
    if present.returncode != 0:
        _run(["git", "-C", str(source_dir), "fetch", "origin", revision])
    _run(["git", "-C", str(source_dir), "checkout", "--detach", revision])
    require_clean_revision(source_dir, revision)


def require_clean_revision(source_dir: Path, expected_revision: str) -> None:
    source_dir = Path(source_dir).resolve()
    _require_checkout(source_dir)
    _require_clean(source_dir)
    actual = _run(
        ["git", "-C", str(source_dir), "rev-parse", "HEAD"]
    ).stdout.strip()
    if actual != expected_revision:
        raise ValueError(
            f"official source revision mismatch: expected {expected_revision}, got {actual}"
        )


def _require_checkout(source_dir: Path) -> None:
    if not source_dir.is_dir() or not (source_dir / ".git").exists():
        raise ValueError(f"official source is not a Git checkout: {source_dir}")


def _require_clean(source_dir: Path) -> None:
    status = _run(
        ["git", "-C", str(source_dir), "status", "--porcelain"]
    ).stdout.strip()
    if status:
        raise ValueError(f"official source checkout is dirty: {source_dir}")


def _run(
    command: Sequence[str],
    *,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if check and completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise ValueError(f"command failed ({' '.join(command)}): {detail}")
    return completed


__all__ = ["prepare_source_checkout", "require_clean_revision"]
