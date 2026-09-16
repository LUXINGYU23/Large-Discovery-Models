"""Manifest-driven task discovery and registration validation."""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from ldm_tts.registration.experiment import (
    EXPERIMENT_CONTRACT_NAME,
    ExperimentContractError,
    load_experiment_contract,
)
from ldm_tts.repository import resolve_repository_root


REPO_ROOT = resolve_repository_root(source_file=Path(__file__))
TASK_MANIFEST_NAME = "task.json"
TASK_MANIFEST_SCHEMA_VERSION = 1
TASK_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
TASK_ADAPTER_FILES = frozenset({"__init__.py", "dependencies.py", "procedure.py"})
HOOK_PATTERN = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_.]*:[A-Za-z_][A-Za-z0-9_]*$"
)


class TaskRegistrationError(ValueError):
    """Raised when a task manifest cannot be registered safely."""


@dataclass(frozen=True)
class TaskDefinition:
    """Stable task identity and its conventional repository entry points."""

    task_id: str
    description: str
    relative_root: Path
    module: str
    manifest_path: Path
    dependency_checker: str | None = None
    experiment_contract_path: Path | None = None
    pilot_evaluation: dict = field(default_factory=dict)


@dataclass(frozen=True)
class TaskValidationIssue:
    """One actionable task-layout validation result."""

    level: str
    message: str
    path: Path


def discover_task_definitions(
    repository_root: Path = REPO_ROOT,
) -> dict[str, TaskDefinition]:
    """Discover registered task manifests under ``<repository_root>/tasks``."""

    repository_root = Path(repository_root).resolve()
    tasks_root = repository_root / "tasks"
    if not tasks_root.is_dir():
        raise TaskRegistrationError(f"Tasks directory does not exist: {tasks_root}")

    definitions: dict[str, TaskDefinition] = {}
    for manifest_path in sorted(tasks_root.glob(f"*/{TASK_MANIFEST_NAME}")):
        definition = load_task_manifest(manifest_path, repository_root=repository_root)
        if definition.task_id in definitions:
            previous = definitions[definition.task_id].manifest_path
            raise TaskRegistrationError(
                f"Duplicate task ID {definition.task_id!r}: {previous} and {manifest_path}"
            )
        definitions[definition.task_id] = definition
    return definitions


def load_task_manifest(
    manifest_path: Path,
    *,
    repository_root: Path = REPO_ROOT,
) -> TaskDefinition:
    """Parse and validate one versioned task manifest."""

    manifest_path = Path(manifest_path).resolve()
    repository_root = Path(repository_root).resolve()
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise TaskRegistrationError(f"Task manifest does not exist: {manifest_path}") from exc
    except json.JSONDecodeError as exc:
        raise TaskRegistrationError(
            f"Invalid JSON in {manifest_path}: line {exc.lineno}, column {exc.colno}"
        ) from exc
    if not isinstance(data, dict):
        raise TaskRegistrationError(f"Task manifest must contain a JSON object: {manifest_path}")

    allowed_keys = {
        "schema_version",
        "task_id",
        "description",
        "dependency_checker",
        "pilot_evaluation",
    }
    unknown = sorted(str(key) for key in data if key not in allowed_keys)
    if unknown:
        raise TaskRegistrationError(
            f"Unknown task manifest field(s) in {manifest_path}: {', '.join(unknown)}"
        )

    schema_version = data.get("schema_version")
    if schema_version != TASK_MANIFEST_SCHEMA_VERSION:
        raise TaskRegistrationError(
            f"Unsupported schema_version {schema_version!r} in {manifest_path}; "
            f"expected {TASK_MANIFEST_SCHEMA_VERSION}."
        )

    task_id = str(data.get("task_id") or "").strip()
    if not TASK_ID_PATTERN.fullmatch(task_id):
        raise TaskRegistrationError(
            f"Invalid task_id {task_id!r} in {manifest_path}; use lowercase Python "
            "identifiers containing letters, digits, and underscores."
        )
    if manifest_path.parent.name != task_id:
        raise TaskRegistrationError(
            f"Task ID {task_id!r} must match its directory name "
            f"{manifest_path.parent.name!r}."
        )

    description = str(data.get("description") or "").strip()
    if not description:
        raise TaskRegistrationError(f"Missing non-empty description in {manifest_path}")

    dependency_checker = data.get("dependency_checker")
    if dependency_checker is not None:
        dependency_checker = str(dependency_checker).strip()
        if not HOOK_PATTERN.fullmatch(dependency_checker):
            raise TaskRegistrationError(
                f"Invalid dependency_checker {dependency_checker!r} in {manifest_path}; "
                "expected 'python.module:function'."
            )

    pilot = data.get("pilot_evaluation", {})
    if not isinstance(pilot, dict) or set(pilot) - {"methods", "submission_adapter"}:
        raise TaskRegistrationError("pilot_evaluation requires methods and/or submission_adapter")
    methods = pilot.get("methods", {})
    if not isinstance(methods, dict) or any(
        not TASK_ID_PATTERN.fullmatch(name) or mode not in {"none", "openai", "callable"}
        for name, mode in methods.items()
    ):
        raise TaskRegistrationError("pilot methods must map method names to proposal modes")
    hook = pilot.get("submission_adapter")
    if hook is not None and (not isinstance(hook, str) or not HOOK_PATTERN.fullmatch(hook)):
        raise TaskRegistrationError("pilot submission_adapter must be 'python.module:function'")
    relative_root = Path("tasks") / task_id
    contract_path = relative_root / EXPERIMENT_CONTRACT_NAME
    return TaskDefinition(
        task_id=task_id,
        description=description,
        relative_root=relative_root,
        module=f"tasks.{task_id}.ldm_task.procedure",
        manifest_path=relative_root / TASK_MANIFEST_NAME,
        dependency_checker=dependency_checker,
        pilot_evaluation=pilot,
        experiment_contract_path=(
            contract_path if (repository_root / contract_path).is_file() else None
        ),
    )


def validate_task_layout(
    definition: TaskDefinition,
    *,
    repository_root: Path = REPO_ROOT,
) -> list[TaskValidationIssue]:
    """Validate the conventional files and callable surface of one task."""

    root = Path(repository_root).resolve() / definition.relative_root
    issues: list[TaskValidationIssue] = []
    required_files = (
        root / TASK_MANIFEST_NAME,
        root / "README.md",
        root / "pyproject.toml",
        root / "__init__.py",
        root / "ldm_task" / "__init__.py",
        root / "ldm_task" / "procedure.py",
        root / "core" / "__init__.py",
        root / "resources" / "README.md",
    )
    for path in required_files:
        if not path.is_file():
            issues.append(TaskValidationIssue("error", "Required task file is missing.", path))

    procedure_path = root / "ldm_task" / "procedure.py"
    if procedure_path.is_file():
        try:
            tree = ast.parse(procedure_path.read_text(encoding="utf-8"), filename=str(procedure_path))
        except SyntaxError as exc:
            issues.append(
                TaskValidationIssue(
                    "error",
                    f"Procedure module has invalid Python syntax: {exc.msg} (line {exc.lineno}).",
                    procedure_path,
                )
            )
        else:
            functions = {
                node.name
                for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            if "main" not in functions:
                issues.append(
                    TaskValidationIssue(
                        "error",
                        "Procedure must define main(argv) as the runner interface.",
                        procedure_path,
                    )
                )
            for recommended in ("parse_args", "describe_ldm_task"):
                if recommended not in functions:
                    issues.append(
                        TaskValidationIssue(
                            "warning",
                            f"Procedure should define {recommended} for consistency and inspection.",
                            procedure_path,
                        )
                    )

    tests_dir = root / "tests"
    if not tests_dir.is_dir():
        issues.append(
            TaskValidationIssue(
                "error",
                "Task-local tests directory is missing.",
                tests_dir,
            )
        )

    adapter_dir = root / "ldm_task"
    if adapter_dir.is_dir():
        unexpected_adapter_files = sorted(
            path
            for path in adapter_dir.iterdir()
            if path.is_file() and path.name not in TASK_ADAPTER_FILES
        )
        for path in unexpected_adapter_files:
            issues.append(
                TaskValidationIssue(
                    "error",
                    "Task implementation files belong in core/, resources/, or scripts/; "
                    "ldm_task/ is reserved for the shared-runner adapter.",
                    path,
                )
            )

    config_dir = Path(repository_root).resolve() / "config" / definition.task_id
    if not config_dir.is_dir():
        issues.append(
            TaskValidationIssue(
                "warning",
                "No config directory was found for this task.",
                config_dir,
            )
        )
    if definition.experiment_contract_path is not None:
        contract_path = Path(repository_root).resolve() / definition.experiment_contract_path
        try:
            contract = load_experiment_contract(contract_path)
        except ExperimentContractError as exc:
            issues.append(TaskValidationIssue("error", str(exc), contract_path))
        else:
            if contract.task_id != definition.task_id:
                issues.append(
                    TaskValidationIssue(
                        "error",
                        f"Experiment contract task_id {contract.task_id!r} does not match "
                        f"registered task {definition.task_id!r}.",
                        contract_path,
                    )
                )
    return issues


def get_task_definition(task_id: str) -> TaskDefinition:
    """Return a registered task or raise a user-facing lookup error."""

    if TASK_DISCOVERY_ERROR is not None:
        raise TaskRegistrationError(str(TASK_DISCOVERY_ERROR))
    try:
        return TASK_DEFINITIONS[task_id]
    except KeyError as exc:
        raise KeyError(
            f"Unknown task {task_id!r}; expected one of {sorted(TASK_DEFINITIONS)}"
        ) from exc


try:
    TASK_DEFINITIONS = discover_task_definitions()
    TASK_DISCOVERY_ERROR: TaskRegistrationError | None = None
except TaskRegistrationError as exc:
    TASK_DEFINITIONS = {}
    TASK_DISCOVERY_ERROR = exc

REPOSITORY_RELATIVE_PREFIXES = tuple(
    sorted(
        {
            "config",
            "data",
            "ldm_tts",
            "scripts",
            "skills",
            "tasks",
        }
    )
)


__all__ = [
    "REPOSITORY_RELATIVE_PREFIXES",
    "TASK_DEFINITIONS",
    "TASK_DISCOVERY_ERROR",
    "TASK_MANIFEST_NAME",
    "TASK_MANIFEST_SCHEMA_VERSION",
    "TASK_ADAPTER_FILES",
    "TaskDefinition",
    "TaskRegistrationError",
    "TaskValidationIssue",
    "discover_task_definitions",
    "get_task_definition",
    "load_task_manifest",
    "validate_task_layout",
]
