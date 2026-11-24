import os
import subprocess
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
        binary = base and os.access(Path(base) / "src/bin/Absolut", os.X_OK)
        if method == "ldm_harness":
            check("Absolut", binary and (Path(base) / "assets.json").is_file(),
                  "Executable and immutable cache manifest")
        else:
            check("Absolut", binary, "External repertoire executable")
            from tasks.antibody.core.absolut_oracle import AbsolutOracle, TARGETS
            target = args.get("antigen", "1ADQ_A")
            if binary and TARGETS.get(target, target) in TARGETS.values():
                resources = args.get("absolut-resources") or [base]
                if isinstance(resources, str):
                    resources = [resources]
                try:
                    AbsolutOracle(base, resources, Path(cwd) / "runs",
                                  int(args.get("absolut-timeout", 600))).preflight(target, require_cache=True)
                except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
                    check("Antigen resources", False, str(exc))
                else:
                    check("Antigen resources", True, target)
    if method in ("ldm_harness", "harness"):
        check("Pi endpoint", env.get("LLM_BASE_URL") and env.get("LLM_MODEL_NAME") and
              (env.get("LLM_API_KEY") or env.get("OPENAI_API_KEY")), "Responses endpoint, model and API key")
    if args.get("enable-protein-tools") and not args.get("mock"):
        from tasks.antibody.tools.structure_design.bootstrap import MPNN_COMMIT
        checkpoint = args.get("solublempnn-checkpoint") or root / f"archive/models/proteinmpnn/{MPNN_COMMIT}/soluble/v_48_020.pt"
        check("SolubleMPNN weights", Path(checkpoint).is_file() and Path(str(checkpoint) + ".source.json").is_file(),
              "Soluble weights and source record; never reuse the vanilla override")
        script = env.get("IPSAE_SCRIPT")
        check("ipSAE script", script and Path(script).is_file(), "Configure IPSAE_SCRIPT for auxiliary interface scoring")
    if args.get("enable-boltzgen") and not args.get("mock"):
        suffix = {"HER2": "HER2", "1N8Z_C": "HER2", "PD-1": "PD1", "5GGR_Y": "PD1"}.get(args.get("antigen"))
        check("BoltzGen antigen", suffix is not None, "Configured HER2 or PD-1 antigen")
        for key in ("BOLTZGEN_PYTHON", "BOLTZGEN_SCAFFOLD", "BOLTZGEN_CHECKPOINT", "BOLTZGEN_INVERSE_FOLD_CHECKPOINT", "BOLTZGEN_MOLDIR", "BOLTZGEN_SPEC_" + (suffix or "UNCONFIGURED")):
            path = env.get(key)
            check(key, path and Path(path).is_file() and (key != "BOLTZGEN_PYTHON" or os.access(path, os.X_OK)),
                  "Required only for the enabled native BoltzGen tool")
    if method == "antbo":
        check("Native AntBO", (Path(env.get("ANTBO_HOME", "")) / "bo/optimizer.py").is_file(),
              "ANTBO_HOME must point to upstream HEBO/AntBO")
    return checks
