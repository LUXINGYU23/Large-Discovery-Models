import os
from importlib.util import find_spec
from pathlib import Path

from ldm_tts.registration.dependencies import fail, ok, plan_check_context


def check_dependencies(plan, *, include_optional=True):
    task, args, env, cwd, _ = plan_check_context(plan)
    checks = []

    def check(name, condition, explanation):
        checks.append((ok if condition else fail)(task, name, explanation))

    if args.get("mock"):
        check("Offline fixtures", True, "Deterministic generator and mock oracle")
        return checks
    root = Path(args.get("root", "/Aidata/shared/ldm/antibody-poc"))
    method = args.get("method", "ldm_harness")
    if method != "antbo" or not args.get("initialization-run"):
        for name in ("rfd3", "proteinmpnn"):
            check(name, os.access(root / "env" / name / "bin/python", os.X_OK), f"Model environment under {root}/env")
    if method == "ldm_harness":
        check("Antibody GP", find_spec("gpytorch") and find_spec("torch"), "Install the task's gp extra")
    if args.get("oracle") == "absolut":
        base = args.get("absolut-path") or env.get("ABSOLUT_PATH")
        check("Absolut", base and (Path(base) / "src/bin/Absolut").is_file(), "External repertoire executable")
        from tasks.antibody.core.absolut_oracle import AbsolutOracle, TARGETS
        target = args.get("antigen", "1ADQ_A")
        if base and TARGETS.get(target, target) in TARGETS.values():
            resources = args.get("absolut-resources") or [base]
            if isinstance(resources, str):
                resources = [resources]
            try:
                AbsolutOracle(base, resources, Path(cwd) / "runs", int(args.get("absolut-timeout", 600))).preflight(target, require_cache=True)
                check("Antigen resources", True, target)
            except Exception as exc:
                check("Antigen resources", False, str(exc))
    if method in ("ldm_harness", "harness"):
        check("Pi endpoint", env.get("LLM_BASE_URL") and env.get("LLM_MODEL_NAME") and
              (env.get("LLM_API_KEY") or env.get("OPENAI_API_KEY")), "Responses endpoint, model and API key")
    if args.get("enable-boltzgen"):
        suffix = {"HER2": "HER2", "PD-1": "PD1"}.get(args.get("antigen"))
        check("BoltzGen antigen", suffix is not None, "Explicit HER2 or PD-1 host configuration")
        for key in ("BOLTZGEN_PYTHON", "BOLTZGEN_SCAFFOLD", "BOLTZGEN_CHECKPOINT", "BOLTZGEN_INVERSE_FOLD_CHECKPOINT", "BOLTZGEN_MOLDIR",
                    "BOLTZGEN_SPEC_" + (suffix or "UNCONFIGURED")):
            path = env.get(key)
            check(key, path and Path(path).is_file() and (key != "BOLTZGEN_PYTHON" or os.access(path, os.X_OK)),
                  "Required only when the optional BoltzGen tool is enabled")
    if method == "antbo":
        check("Native AntBO", (Path(env.get("ANTBO_HOME", "")) / "bo/optimizer.py").is_file(),
              "ANTBO_HOME must point to upstream HEBO/AntBO")
    return checks
