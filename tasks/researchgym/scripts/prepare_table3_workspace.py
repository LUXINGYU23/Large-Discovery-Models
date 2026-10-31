#!/usr/bin/env python3
"""Create one clean, writable Table 3 task workspace from tracked source only.

This deliberately does *not* copy or symlink data, checkpoints, virtual
environments, logs, or prior outputs.  The resulting ``input_mounts.json`` is
consumed by the container launcher, which binds every declared input read-only
at runtime.  This keeps an LDM code-writing trajectory from modifying a shared
asset or observing an old run's result.
"""

from __future__ import annotations

import argparse
import io
import json
import subprocess
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any

from tasks.researchgym.table3_protocol import TASKS, UPSTREAM_COMMIT, ProtocolError


class WorkspaceError(RuntimeError):
    """The pinned source or an input mount is unsafe for a campaign workspace."""


def _git(source_root: Path, *args: str) -> str:
    result = subprocess.run(["git", "-C", str(source_root), *args], text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if result.returncode:
        raise WorkspaceError(result.stderr.strip() or f"git {' '.join(args)} failed")
    return result.stdout.strip()


def _relative_destination(raw: str) -> str:
    candidate = PurePosixPath(raw)
    if not raw or candidate.is_absolute() or ".." in candidate.parts or raw == ".":
        raise WorkspaceError(f"input mount destination must be a workspace-relative path: {raw!r}")
    return candidate.as_posix()


def parse_input_mount(raw: str) -> dict[str, str]:
    """Parse ``HOST_PATH:workspace/relative/path`` without accepting escapes."""
    try:
        source_raw, destination = raw.rsplit(":", 1)
    except ValueError as exc:
        raise WorkspaceError("--input-mount must be HOST_PATH:workspace/relative/path") from exc
    source = Path(source_raw).expanduser().resolve()
    if not source.exists():
        raise WorkspaceError(f"input mount source does not exist: {source}")
    return {"source": str(source), "destination": _relative_destination(destination)}


def _extract_tracked_task(source_root: Path, task_id: str, destination: Path) -> None:
    task = TASKS[task_id]
    prefix = PurePosixPath("tasks") / "test" / task.upstream_dir
    archive = subprocess.run(["git", "-C", str(source_root), "archive", "--format=tar", UPSTREAM_COMMIT,
                              "--", prefix.as_posix()], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             check=False)
    if archive.returncode:
        raise WorkspaceError(archive.stderr.decode("utf-8", errors="replace").strip() or "git archive failed")
    with tarfile.open(fileobj=io.BytesIO(archive.stdout), mode="r:") as bundle:
        members: list[tarfile.TarInfo] = []
        for member in bundle.getmembers():
            path = PurePosixPath(member.name)
            try:
                relative = path.relative_to(prefix)
            except ValueError:
                continue
            # ``git archive`` includes the task-root directory itself.  It is
            # not a file to reproduce inside the workspace; every descendant
            # is handled below after its path has been made relative.
            if relative == PurePosixPath("."):
                continue
            if not relative.parts or ".." in relative.parts:
                raise WorkspaceError(f"unsafe archive entry: {member.name}")
            member.name = relative.as_posix()
            members.append(member)
        if not members:
            raise WorkspaceError(f"pinned source has no tracked files for {task.upstream_dir}")
        # ``filter=`` is only available in newer Python releases.  The member
        # names above are validated and this archive is addressed by a pinned
        # Git object, so the Python 3.10/3.11-compatible extraction is safe.
        bundle.extractall(destination, members=members)


def prepare_workspace(*, source_root: Path, task_id: str, workspace: Path,
                      input_mounts: list[dict[str, str]]) -> dict[str, Any]:
    """Write a task-only clean workspace and an immutable source/mount receipt."""
    if task_id not in TASKS:
        raise WorkspaceError(f"unknown task: {task_id}")
    source_root, workspace = source_root.resolve(), workspace.resolve()
    if workspace.exists():
        raise WorkspaceError(f"workspace already exists: {workspace}")
    if _git(source_root, "rev-parse", "HEAD") != UPSTREAM_COMMIT:
        raise WorkspaceError(f"source is not pinned to {UPSTREAM_COMMIT}")
    if _git(source_root, "status", "--porcelain", "--untracked-files=all"):
        raise WorkspaceError("source worktree is not clean; use a detached clean worktree")
    workspace.mkdir(parents=True)
    try:
        _extract_tracked_task(source_root, task_id, workspace)
        receipt = {
            "schema_version": 1,
            "task_id": task_id,
            "upstream_dir": TASKS[task_id].upstream_dir,
            "source_root": str(source_root),
            "source_commit": UPSTREAM_COMMIT,
            "inputs": input_mounts,
            "mount_policy": "all inputs must be bound read-only by the container launcher",
        }
        (workspace / "input_mounts.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except Exception:
        # The caller has a deterministic new path per job.  Leaving a partial
        # directory could make a later retry silently reuse contaminated files.
        for child in sorted(workspace.rglob("*"), reverse=True):
            if child.is_file() or child.is_symlink():
                child.unlink()
            elif child.is_dir():
                child.rmdir()
        workspace.rmdir()
        raise
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--task", choices=tuple(TASKS), required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--input-mount", action="append", default=[], metavar="HOST:DEST",
                        help="Repeat for each raw input; it will be RO-mounted at DEST by the launcher.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        mounts = [parse_input_mount(item) for item in args.input_mount]
        receipt = prepare_workspace(source_root=args.source_root, task_id=args.task,
                                    workspace=args.workspace, input_mounts=mounts)
    except (OSError, WorkspaceError, ProtocolError) as exc:
        raise SystemExit(f"Table 3 workspace rejected: {exc}") from exc
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
