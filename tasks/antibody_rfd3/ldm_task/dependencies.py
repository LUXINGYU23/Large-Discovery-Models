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
        check("Generator fixtures", not args.get("initialization-run"), "Deterministic generator fixtures")
    root = Path(args.get("root", "/Aidata/shared/ldm/antibody-poc"))
    method = args.get("method", "ldm_harness")
    if not args.get("mock") and (method != "antbo" or not args.get("initialization-run")):
        for name in ("rfd3", "proteinmpnn"):
            check(name, os.access(root / "env" / name / "bin/python", os.X_OK), f"Model environment under {root}/env")
    if method == "ldm_harness":
        check("Antibody GP", find_spec("gpytorch") and find_spec("torch"), "Install the task's gp extra")
    if args.get("oracle") == "absolut":
        base = args.get("absolut-path") or env.get("ABSOLUT_PATH")
        check("Absolut", base and (Path(base) / "src/bin/Absolut").is_file(), "External repertoire executable")
    if method in ("ldm_harness", "harness"):
        check("Pi endpoint", env.get("LLM_BASE_URL") and env.get("LLM_MODEL_NAME") and
              (env.get("LLM_API_KEY") or env.get("OPENAI_API_KEY")), "Responses endpoint, model and API key")
    if method == "antbo":
        check("Native AntBO", (Path(env.get("ANTBO_HOME", "")) / "bo/optimizer.py").is_file(),
              "ANTBO_HOME must point to upstream HEBO/AntBO")
    return checks
