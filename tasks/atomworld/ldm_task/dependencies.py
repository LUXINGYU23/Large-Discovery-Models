"""Lightweight dependency checks without importing scientific packages."""

from __future__ import annotations
import importlib.util
import shutil
from pathlib import Path
from ldm_tts.registration.dependencies import fail, ok, skip, plan_check_context
from tasks.atomworld.core.data import DEFAULT_UPSTREAM


def check_dependencies(plan, *, include_optional=True):
    task, args, env, cwd, mode = plan_check_context(plan)
    if args.get("mock") or mode == "mock":
        return [
            ok(
                task,
                "mock",
                "Deterministic fixture needs no external dataset, service or GPU",
            )
        ]
    checks = []
    for module in ("pymatgen", "numpy"):
        present = importlib.util.find_spec(module) is not None
        checks.append(
            (ok if present else fail)(
                task, module, "Available" if present else "Install the task environment"
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
    for name, keys in (
        ("LLM URL", ("LLM_BASE_URL", "LDM_LLM_URL", "OPENAI_BASE_URL")),
        ("LLM model", ("LLM_MODEL_NAME", "LDM_LLM_MODEL", "OPENAI_MODEL")),
    ):
        option = "llm-url" if name == "LLM URL" else "llm-model-name"
        configured = args.get(option) or any(env.get(key) for key in keys)
        checks.append(
            (ok if configured else fail)(
                task, name, "Configured" if configured else f"Set {keys[0]}"
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
