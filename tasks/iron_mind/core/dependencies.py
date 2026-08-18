"""Draft dependency boundary for the Iron Mind task."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ldm_tts.registration.dependencies import DependencyCheck, warn


def check_task_dependencies(
    task: str,
    args: dict[str, Any],
    env: dict[str, str],
    cwd: Path,
    *,
    mode: str,
    include_optional: bool,
) -> list[DependencyCheck]:
    del args, env, mode, include_optional
    return [
        warn(
            task,
            "task implementation",
            "Iron Mind dependency checks are pending source-contract implementation.",
            str(cwd),
        )
    ]
