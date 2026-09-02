"""Architecture guardrails for the standalone NucleoBench task."""

from __future__ import annotations

import ast
from pathlib import Path

TASK_ROOT = Path(__file__).resolve().parents[1]


def test_core_does_not_import_another_task() -> None:
    foreign_imports = []
    for path in sorted((TASK_ROOT / "core").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            foreign_imports.extend(
                (path.name, module)
                for module in _imported_modules(node)
                if module.startswith("tasks.")
                and not module.startswith("tasks.nucleobench")
            )
    assert foreign_imports == []


def _imported_modules(node: ast.AST) -> tuple[str, ...]:
    if isinstance(node, ast.Import):
        return tuple(alias.name for alias in node.names)
    if isinstance(node, ast.ImportFrom) and node.module:
        return (node.module,)
    return ()
