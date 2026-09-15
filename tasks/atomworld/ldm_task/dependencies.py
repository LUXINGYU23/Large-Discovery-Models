"""Lightweight dependency checks without importing scientific packages."""

from __future__ import annotations
import importlib.util
import shutil
from pathlib import Path
from ldm_tts.registration.dependencies import (
    arg_value,
    configured_value,
    fail,
    first_env,
    mask_secret,
    ok,
    skip,
    plan_check_context,
)
from tasks.atomworld.core.data import DEFAULT_UPSTREAM


def check_dependencies(plan, *, include_optional=True):
    task, args, env, cwd, mode = plan_check_context(plan)
    mock = bool(args.get("mock") or mode == "mock")
    harness = args.get("search-method", "llm") in {"harness", "blind_harness_compiled"}
    if mock:
        checks = [
            ok(
                task,
                "mock",
                "Deterministic fixture replaces only the dataset and judge.",
            )
        ]
        if not harness:
            checks.append(
                ok(
                    task,
                    "proposal_provider",
                    "Mock direct proposals use the deterministic fixture and no service.",
                )
            )
            return checks
    else:
        checks = []
        for module in ("pymatgen", "numpy"):
            present = importlib.util.find_spec(module) is not None
            checks.append(
                (ok if present else fail)(
                    task,
                    module,
                    "Available" if present else "Install the task environment",
                )
            )
        raw = (
            args.get("upstream-root")
            or env.get("ATOMWORLD_UPSTREAM_ROOT")
            or str(DEFAULT_UPSTREAM)
        )
        upstream = Path(raw)
        if not upstream.is_absolute():
            upstream = cwd / upstream
        checks.append(
            (ok if (upstream / "src/atomworld/evaluate.py").is_file() else fail)(
                task, "upstream evaluator", "Pinned local AtomWorld evaluator required"
            )
        )
        data = Path(args.get("data-dir", ""))
        if not data.is_absolute():
            data = cwd / data
        checks.append(
            (
                ok
                if all(
                    (data / name).is_file()
                    for name in ("public.jsonl", "private.jsonl", "manifest.json")
                )
                else fail
            )(task, "dataset", "Prepared public/private dataset with manifest required")
        )
    if args.get("proposal-format") == "operations":
        present = importlib.util.find_spec("ase") is not None
        checks.append(
            (ok if present else fail)(
                task, "ase", "ASE required by bounded geometry tools"
            )
        )
    if args.get("search-method", "llm") in {"harness", "blind_harness_compiled"}:
        checks.append(
            (ok if shutil.which("docker") else fail)(
                task,
                "Harness Docker client",
                "Shared Pi sidecar requires Docker and a Linux KVM host; build its task guest image first",
            )
        )
    if harness or not mock:
        provider_settings = (
            (
                "LLM URL",
                "llm-url",
                ("LLM_BASE_URL", "LDM_LLM_URL", "OPENAI_BASE_URL"),
                "LLM base URL is configured.",
            ),
            (
                "LLM model",
                "llm-model-name",
                ("LLM_MODEL_NAME", "LDM_LLM_MODEL", "OPENAI_MODEL"),
                "LLM model name is configured.",
            ),
            (
                "LLM API key",
                "llm-api-key",
                ("LLM_API_KEY", "LDM_LLM_API_KEY", "OPENAI_API_KEY"),
                "LLM API key value is configured.",
            ),
        )
        for name, option, keys, message in provider_settings:
            value = configured_value(arg_value(args, option) or first_env(env, keys))
            detail = mask_secret(value) if name == "LLM API key" and value else value
            checks.append(
                (ok if value else fail)(
                    task,
                    name,
                    message if value else f"Set --{option} or one of {', '.join(keys)}.",
                    detail,
                )
            )
    checks.append(
        skip(
            task,
            "GPU",
            "Official evaluator runs on CPU; model can use a separate endpoint",
        )
    )
    return checks
