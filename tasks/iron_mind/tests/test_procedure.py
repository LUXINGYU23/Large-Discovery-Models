"""Structural tests for the stable Iron Mind task adapter."""

from __future__ import annotations

import json
from pathlib import Path

from tasks.iron_mind.ldm_task import procedure


TASK_ROOT = Path(__file__).resolve().parents[1]


def test_manifest_declares_the_task_dependency_hook() -> None:
    manifest = json.loads((TASK_ROOT / "task.json").read_text(encoding="utf-8"))

    assert manifest.get("dependency_checker") == (
        "tasks.iron_mind.ldm_task.dependencies:check_dependencies"
    )


def test_procedure_is_a_thin_core_workflow_adapter() -> None:
    source = Path(procedure.__file__).read_text(encoding="utf-8")

    assert "from tasks.iron_mind.core import workflow as _workflow" in source
    assert "return _workflow.parse_args(argv)" in source
    assert "return _workflow.describe_ldm_task(*args, **kwargs)" in source
    assert "return _workflow.main(argv)" in source


def test_dependency_adapter_delegates_to_core() -> None:
    adapter_path = TASK_ROOT / "ldm_task" / "dependencies.py"

    assert adapter_path.is_file()
    source = adapter_path.read_text(encoding="utf-8")
    assert "from tasks.iron_mind.core.dependencies import check_task_dependencies" in source
    assert "return check_task_dependencies(" in source


def test_scaffold_mock_engine_path_is_removed() -> None:
    assert not (TASK_ROOT / "core" / "mock_engine.py").exists()
