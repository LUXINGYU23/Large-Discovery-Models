"""Dependency-check adapter for Iron Mind."""

from __future__ import annotations

from typing import Any

from ldm_tts.registration.dependencies import DependencyCheck, plan_check_context
from tasks.iron_mind.core.dependencies import check_task_dependencies


def check_dependencies(
    plan: dict[str, Any], *, include_optional: bool = True
) -> list[DependencyCheck]:
    task, args, env, cwd, mode = plan_check_context(plan)
    return check_task_dependencies(
        task,
        args,
        env,
        cwd,
        mode=mode,
        include_optional=include_optional,
        contract_profile=str(plan.get("contract_profile", "")),
    )
