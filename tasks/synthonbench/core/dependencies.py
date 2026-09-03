"""Dependency checks for source-pinned SynthonBench campaigns."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from ldm_tts.registration.dependencies import (
    DependencyCheck,
    arg_value,
    check_llm_settings,
    fail,
    ok,
    plan_check_context,
)
from tasks.synthonbench.core.constants import (
    OFFICIAL_PACKAGE_VERSION,
    OFFICIAL_SOURCE_COMMIT,
)
from tasks.synthonbench.core.data import validate_prepared_data
from tasks.synthonbench.core.provider import (
    API_KEY_ENV_NAMES,
    BASE_URL_ENV_NAMES,
    MODEL_ENV_NAMES,
)


def check_task_dependencies(plan: dict[str, Any], *, include_optional: bool = True) -> list[DependencyCheck]:
    """Validate imports for the official example and pinned real-track assets."""

    del include_optional
    task, args, env, _cwd, mode = plan_check_context(plan)
    checks = [_package_check(task, "synthonbench"), _package_check(task, "rdkit")]
    if mode == "mock" or bool(args.get("mock")):
        return checks
    checks.extend(_prepared_data_checks(task, args))
    if arg_value(args, "search-method", default="ldm") == "bo":
        checks.append(ok(task, "proposal provider", "Pure BO does not use a model endpoint."))
    else:
        checks.extend(_provider_checks(task, args, env))
    return checks


def _package_check(task: str, module: str) -> DependencyCheck:
    if importlib.util.find_spec(module) is None:
        return fail(task, module, f"Missing task dependency {module!r}.")
    if module != "synthonbench":
        return ok(task, module, "Task dependency is importable.")
    import synthonbench

    if synthonbench.__version__ != OFFICIAL_PACKAGE_VERSION:
        return fail(task, module, "Installed version does not match the source-pinned task contract.",
                    f"expected={OFFICIAL_PACKAGE_VERSION} actual={synthonbench.__version__}")
    return ok(task, module, "Source-pinned SynthonBench package is importable.", OFFICIAL_SOURCE_COMMIT)


def _prepared_data_checks(task: str, args: dict[str, Any]) -> list[DependencyCheck]:
    data_dir = arg_value(args, "data-dir")
    source_dir = arg_value(args, "source-dir")
    scale = arg_value(args, "scale", default="1M")
    oracle_kind = arg_value(args, "oracle-kind", default="surrogate")
    if not data_dir or not source_dir:
        return [fail(task, "prepared official data", "Real campaigns require --data-dir and --source-dir.")]
    try:
        validate_prepared_data(Path(data_dir), Path(source_dir), scale, oracle_kind)
    except (OSError, ValueError) as exc:
        return [fail(task, "prepared official data", str(exc))]
    return [ok(task, "prepared official data", "Pinned source checkout and released files are ready.")]


def _provider_checks(task: str, args: dict[str, Any], env: dict[str, str]) -> list[DependencyCheck]:
    checks = check_llm_settings(
        task,
        args,
        env,
        url_arg="llm-url",
        model_arg="llm-model-name",
        api_arg="api-key",
        url_env=BASE_URL_ENV_NAMES,
        model_env=MODEL_ENV_NAMES,
        api_env=API_KEY_ENV_NAMES,
        required=True,
    )
    if arg_value(args, "search-method", default="ldm") not in {
        "ldm_harness",
        "harness",
    }:
        return checks

    raw_key_path = arg_value(args, "harness-api-key-file")
    if not raw_key_path:
        return checks

    checks = [check for check in checks if check.name != "LLM API key"]
    key_path = Path(str(raw_key_path)).expanduser()
    try:
        configured = key_path.is_file() and bool(
            key_path.read_text(encoding="utf-8").strip()
        )
    except OSError as exc:
        return checks + [
            fail(task, "Harness API key file", "Cannot read key file.", str(exc))
        ]
    if not configured:
        return checks + [
            fail(
                task,
                "Harness API key file",
                "Key file must exist and contain a non-empty API key.",
                str(key_path),
            )
        ]
    return checks + [
        ok(
            task,
            "Harness API key file",
            "Harness API key file is configured.",
            str(key_path),
        )
    ]


__all__ = ["check_task_dependencies"]
