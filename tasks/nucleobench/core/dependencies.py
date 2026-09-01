"""Dependency checks for the currently registered NucleoBench contract."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ldm_tts.registration.dependencies import DependencyCheck, fail, ok
from ldm_tts.registration.experiment import ExperimentContractError, load_experiment_contract

from tasks.nucleobench.core.cases import CaseCatalogError, load_case_catalog
from tasks.nucleobench.core.constants import TASK_ID


TASK_ROOT = Path(__file__).resolve().parents[1]


def check_task_dependencies(
    plan: dict[str, Any], *, include_optional: bool = True
) -> list[DependencyCheck]:
    """Verify the tracked contract used by registration and dry-run inspection."""

    del include_optional
    task = str(plan.get("task") or TASK_ID)
    try:
        load_case_catalog()
        load_experiment_contract(TASK_ROOT / "experiment.json")
    except (CaseCatalogError, ExperimentContractError, OSError) as exc:
        return [fail(task, "tracked contract", str(exc))]
    return [ok(task, "tracked contract", "Pinned registration resources are valid.")]


__all__ = ["check_task_dependencies"]
