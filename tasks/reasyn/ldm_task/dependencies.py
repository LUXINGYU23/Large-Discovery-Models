"""Read-only dependency inventory, with all heavyweight imports kept lazy."""

from __future__ import annotations
import importlib.util
import shutil
from pathlib import Path
from ldm_tts.registration.dependencies import (
    configured_value, ok, fail, warn, plan_check_context,
)


HARNESS_METHODS = {"harness", "ldm_harness", "ldm_harness_compiled"}


def check_dependencies(plan, *, include_optional=True):
    task, args, env, cwd, mode = plan_check_context(plan)
    mock = bool(args.get("mock")) or mode == "mock"
    if mock:
        checks = [
            ok(
                task,
                "mock",
                "Deterministic fixture uses no checkpoint, GPU, oracle or endpoint.",
            )
        ]
        if args.get("search-method") in HARNESS_METHODS or args.get("proposal-mode") == "harness":
            checks[0] = ok(task, "mock", "Projection and oracle values are synthetic; Harness still uses its configured provider.")
            checks.extend(_provider_checks(task, args, env, cwd))
        return checks
    root = (
        Path(
            args.get("upstream-root")
            or env.get("REASYN_ROOT")
            or Path(__file__).resolve().parents[4] / "ReaSyn-reasyn_v2"
        )
        .expanduser()
        .resolve()
    )
    benchmark = args.get("benchmark", "reconstruction")
    dataset = args.get("dataset", "zinc250k")
    checkpoints = str(
        args.get("model-paths")
        or env.get("REASYN_MODEL_PATHS")
        or "data/trained_model/nv-reasyn-ar-166m-v2.ckpt,data/trained_model/nv-reasyn-eb-174m-v2.ckpt"
    ).split(",")
    paths = {
        "upstream": root / "reasyn/sampler/sampler.py",
        "fingerprint_index": args.get(
            "fpindex", "data/processed/comp_2048/fpindex.pkl"
        ),
        "reaction_matrix": args.get(
            "rxn-matrix", "data/processed/comp_2048/matrix.pkl"
        ),
    }
    paths.update({f"checkpoint_{i}": p.strip() for i, p in enumerate(checkpoints)})
    if benchmark == "reconstruction" and dataset == "zinc250k":
        paths["extra_zinc_index"] = args.get(
            "additional-fpindex", "data/processed/zinc250k_2048/fpindex.pkl"
        )
    if benchmark == "reconstruction" and not args.get("target-smiles"):
        paths["targets"] = (
            args.get("targets-file")
            or "data/"
            + {
                "zinc250k": "test_zinc250k.txt",
                "enamine": "enamine_smiles_1k.txt",
                "chembl": "chembl_filtered_1k.txt",
                "custom": "targets.txt",
            }[dataset]
        )
    checks = []
    for name, value in paths.items():
        path = Path(value)
        path = path if path.is_absolute() else root / path
        checks.append(
            (ok if path.is_file() else fail)(
                task,
                name,
                "Available" if path.is_file() else "Missing required asset",
                str(path),
            )
        )
    if len(checkpoints) != 2:
        checks.append(
            fail(
                task,
                "checkpoint_pair",
                "Exactly AR then EB checkpoint paths are required.",
            )
        )
    checks.extend(_provider_checks(task, args, env, cwd))
    for module in ("rdkit", "tdc"):
        present = importlib.util.find_spec(module) is not None
        checks.append(
            (ok if present else fail)(
                task,
                module,
                "Import discoverable"
                if present
                else "Install the task chemistry extra; real scoring requires this module.",
            )
        )
    if include_optional:
        for module in ("torch", "omegaconf", "einops"):
            checks.append(
                (ok if importlib.util.find_spec(module) else warn)(
                    task,
                    module,
                    "Import discoverable"
                    if importlib.util.find_spec(module)
                    else "Not installed in current interpreter; REASYN_PYTHON may select a dedicated upstream environment.",
                )
            )
        checks.append(
            warn(
                task,
                "qualification",
                "Real model/evaluator qualification remains draft until official assets and a real seed run are verified.",
            )
        )
    return checks


def _provider_checks(task, args, env, cwd):
    checks = []
    method = args.get("search-method", "auto")
    proposal = args.get("proposal-mode", "auto")
    harness = method in HARNESS_METHODS or proposal == "harness"
    needs_pool = (method == "bo" and proposal != "baseline") or (
        args.get("initialization-mode") == "shared_start"
        and args.get("benchmark", "reconstruction") == "tdc"
    )
    if needs_pool:
        raw_pool = configured_value(str(args.get("bo-targets-file") or env.get("REASYN_BO_TARGETS") or ""))
        pool = Path(raw_pool).expanduser() if raw_pool else None
        if pool is not None and not pool.is_absolute():
            pool = cwd / pool
        checks.append(
            (ok if pool is not None and pool.is_file() else fail)(
                task,
                "bo_target_pool",
                "Score-blind molecular target pool is available."
                if pool is not None and pool.is_file()
                else "Real BO and shared TDC initialization require --bo-targets-file or REASYN_BO_TARGETS with a score-blind SMILES pool.",
                str(pool) if pool is not None else "",
            )
        )
    if method == "bo" or proposal == "baseline":
        checks.append(ok(task, "proposal_provider", "Local query generation does not use a model endpoint."))
    elif harness or proposal not in ("baseline", "mock"):
        url = configured_value(
            str(
            args.get("llm-url")
            or env.get("LLM_BASE_URL")
            or env.get("LDM_LLM_URL")
            or env.get("OPENAI_BASE_URL")
            or ""
            )
        )
        model = configured_value(str(
            args.get("llm-model")
            or env.get("LLM_MODEL_NAME")
            or env.get("LDM_LLM_MODEL")
            or ""
        ))
        checks.append(
            (ok if url and model else fail)(
                task,
                "proposal_endpoint",
                "Endpoint and model configured; runtime provider preflight still required."
                if url and model
                else "Set LLM_BASE_URL and LLM_MODEL_NAME; credentials only via LLM_API_KEY.",
            )
        )
    if harness:
        checks.append(
            (ok if shutil.which("docker") else fail)(
                task,
                "harness_container_runtime",
                "Docker CLI is available; runtime still verifies the daemon and sidecar image."
                if shutil.which("docker")
                else "Install Docker and build the shared ldm-pi-harness sidecar image.",
            )
        )
        resource_root = Path(__file__).resolve().parents[1] / "resources/harness"
        resources = [
            "profiles/molecular_research/AGENTS.md",
            "skills/molecular-design/SKILL.md",
            "tools/molecular_research.mjs",
            "image/guest-image.json",
            "image/Dockerfile",
            "image/lock/requirements.lock",
        ]
        if method == "ldm_harness_compiled":
            resources += [
                "profiles/policy_architect/AGENTS.md",
                "skills/compile-ldm-policy/SKILL.md",
            ]
        missing = [str(resource_root / path) for path in resources if not (resource_root / path).is_file()]
        checks.append(
            (fail if missing else ok)(
                task,
                "harness_resources",
                "Missing packaged task research resources." if missing else "Task roles, skills, research tools and guest image recipe are packaged.",
                ", ".join(missing),
            )
        )
        if args.get("harness-mcp-config"):
            config = Path(str(args["harness-mcp-config"])).expanduser()
            config = config if config.is_absolute() else cwd / config
            checks.append(
                (ok if config.is_file() else fail)(
                    task, "harness_mcp_config",
                    "Configured MCP file is available." if config.is_file() else "Configured MCP file is missing.",
                    str(config),
                )
            )
    return checks
