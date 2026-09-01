"""Dependency checks for the currently registered NucleoBench contract."""

from __future__ import annotations

from importlib import metadata
from pathlib import Path
from typing import Any

from ldm_tts.registration.dependencies import (
    DependencyCheck,
    fail,
    ok,
    plan_check_context,
)
from ldm_tts.registration.experiment import (
    ExperimentContractError,
    load_experiment_contract,
)
from tasks.nucleobench.core.cases import CaseCatalogError, load_case_catalog
from tasks.nucleobench.core.constants import TASK_ID, UPSTREAM_PACKAGE_VERSION

TASK_ROOT = Path(__file__).resolve().parents[1]


def check_task_dependencies(
    plan: dict[str, Any], *, include_optional: bool = True
) -> list[DependencyCheck]:
    """Verify the tracked contract and the pinned runtime required by real runs."""

    del include_optional
    task, _args, _env, _cwd, mode = plan_check_context(plan)
    task = task or TASK_ID
    try:
        load_case_catalog()
        load_experiment_contract(TASK_ROOT / "experiment.json")
    except (CaseCatalogError, ExperimentContractError, OSError) as exc:
        return [fail(task, "tracked contract", str(exc))]
    checks = [ok(task, "tracked contract", "Pinned registration resources are valid.")]
    if mode == "mock":
        return checks
    try:
        installed = metadata.version("nucleobench")
    except metadata.PackageNotFoundError:
        checks.append(
            fail(
                task,
                "official runtime",
                "Install the task's 'official' extra before a real campaign.",
            )
        )
    else:
        checks.append(
            ok(task, "official runtime", f"nucleobench=={installed} is installed.")
            if installed == UPSTREAM_PACKAGE_VERSION
            else fail(
                task,
                "official runtime",
                "Installed NucleoBench does not match the pinned source.",
                f"expected={UPSTREAM_PACKAGE_VERSION} actual={installed}",
            )
        )
    return checks


__all__ = ["check_task_dependencies"]
