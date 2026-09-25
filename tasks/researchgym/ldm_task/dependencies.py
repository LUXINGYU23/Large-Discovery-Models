"""Read-only dependency checks; heavyweight imports stay out of module import."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from ldm_tts.registration.dependencies import (
    check_cuda_visibility, check_llm_settings, fail, ok, parse_device_ids, plan_check_context, skip, warn,
)

HARNESS_METHODS = {"harness", "ldm_harness", "ldm_harness_compiled"}
CASE_IMPORTS = {
    "time_series_explanation": ("torch", "captum", "numpy"),
    "continual_learning": ("torch", "timm", "torchvision", "numpy"),
    "cross_modal_retrieval": ("torch", "torch.distributed.run", "transformers", "ruamel.yaml"),
    "improving_replay_buffers": ("torch", "dm_control", "numpy"),
}


def check_dependencies(plan, *, include_optional=True):
    task, args, env, cwd, mode = plan_check_context(plan)
    method = args.get("search-method", "ldm")
    checks = []
    if method == "bo":
        return [fail(task, "search method", "bo is not supported for ResearchGym program search.")]
    harness = method in HARNESS_METHODS
    if harness:
        checks.extend(_harness_checks(task, args, env))
    mock = bool(args.get("mock")) or mode == "mock"
    if mock:
        checks.append(ok(task, "mock", "Synthetic proposals and analytic scores need no dataset, GPU or endpoint."))
        return checks
    checks.extend(check_llm_settings(task, args, env, url_arg="llm-url", model_arg="llm-model", api_arg="llm-api-key",
                                     url_env=("LLM_BASE_URL", "LDM_LLM_URL"), model_env=("LLM_MODEL_NAME", "LDM_LLM_MODEL"),
                                     api_env=("LLM_API_KEY", "LDM_LLM_API_KEY", "OPENAI_API_KEY"), required=True))
    from tasks.researchgym.core.cases import load_case
    from tasks.researchgym.core.source import SourceMismatchError, UpstreamCase

    case = load_case(args.get("case", "time_series_explanation"))
    root = Path(str(args.get("upstream-root") or env.get("RESEARCHGYM_ROOT") or Path.home() / "ResearchGym")).expanduser()
    data_root = args.get("case-data-root")
    try:
        info = UpstreamCase(root, case, data_root=Path(data_root) if data_root else None).verify()
        checks.append(ok(task, "upstream source", f"{info['verified_files']} files match ResearchGym {info['commit'][:12]}.",
                         str(root)))
    except (SourceMismatchError, FileNotFoundError) as exc:
        checks.append(fail(task, "upstream source", str(exc)))
    if case.case_id == "time_series_explanation":
        checks.append(_tse_checkpoints(task, case, root, data_root))
    python = str(args.get("case-python") or env.get("RESEARCHGYM_CASE_PYTHON") or "")
    if not python:
        checks.append(fail(task, "case python", "Set --case-python to the case environment interpreter."))
    elif not Path(python).exists() and not shutil.which(python):
        checks.append(fail(task, "case python", f"Interpreter not found: {python}"))
    else:
        missing = []
        for module in CASE_IMPORTS[case.case_id]:
            probe = subprocess.run([python, "-c", f"import importlib.util,sys; sys.exit(importlib.util.find_spec({module!r}) is None)"],
                                   capture_output=True, text=True, timeout=120)
            if probe.returncode:
                missing.append(module)
        checks.append(fail(task, "case imports", "Missing in case environment: " + ", ".join(missing))
                      if missing else ok(task, "case imports", "Case environment imports resolve.", python))
    devices = str(args.get("devices") or env.get("RESEARCHGYM_DEVICES") or "")
    if devices:
        checks.append(check_cuda_visibility(task, "devices", requested_device="cuda", env=env,
                                            requested_devices=parse_device_ids(devices)))
    else:
        checks.append(warn(task, "devices", "No --devices; case jobs run without CUDA_VISIBLE_DEVICES assignment."))
    return checks


def _tse_checkpoints(task, case, root, data_root):
    base = Path(data_root) if data_root else root / "tasks/test/time-series-explanation"
    matrix = case.raw["jobs"]["matrix"]
    missing = [f"{d}/state_classifier_{f}_42_no_imputation" for d in matrix["dataset"] for f in matrix["fold"]
               if not (base / "model" / d / f"state_classifier_{f}_42_no_imputation").exists()]
    if missing:
        return fail(task, "classifier checkpoints", f"{len(missing)} trained state classifiers are missing "
                    f"(first: {missing[0]}); train them with the upstream real/main.py --train True first.")
    return ok(task, "classifier checkpoints", "All state classifiers for 5 datasets x 5 folds exist.")


def _harness_checks(task, args, env):
    checks = []
    if args.get("harness-command-json"):
        return [ok(task, "harness sidecar", "Mock fake sidecar command configured.")]
    docker = shutil.which("docker")
    checks.append(ok(task, "docker", "Docker CLI found.", docker) if docker
                  else fail(task, "docker", "Harness methods need Docker to start the Pi sidecar."))
    checks.append(ok(task, "kvm", "/dev/kvm is present.") if os.path.exists("/dev/kvm")
                  else warn(task, "kvm", "/dev/kvm is absent on this host; the sidecar host needs KVM."))
    cache = args.get("harness-cache-dir")
    checks.append(ok(task, "guest cache", "Harness cache directory configured.", str(cache)) if cache
                  else skip(task, "guest cache", "Using ~/.cache/ldm-gondolin; build the researchgym guest there first."))
    return checks
