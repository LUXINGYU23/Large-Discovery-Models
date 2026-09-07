"""Read-only dependency inventory, with all heavyweight imports kept lazy."""

from __future__ import annotations
import importlib.util
from pathlib import Path
from ldm_tts.registration.dependencies import ok, fail, warn, plan_check_context


def check_dependencies(plan, *, include_optional=True):
    task, args, env, cwd, mode = plan_check_context(plan)
    mock = bool(args.get("mock")) or mode == "mock"
    if mock:
        return [
            ok(
                task,
                "mock",
                "Deterministic fixture uses no checkpoint, GPU, oracle or endpoint.",
            )
        ]
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
    proposal = args.get("proposal-mode", "openai")
    if proposal not in ("baseline", "mock"):
        url = (
            args.get("llm-url")
            or env.get("LLM_BASE_URL")
            or env.get("LDM_LLM_URL")
            or env.get("OPENAI_BASE_URL")
        )
        model = (
            args.get("llm-model")
            or env.get("LLM_MODEL_NAME")
            or env.get("LDM_LLM_MODEL")
        )
        checks.append(
            (ok if url and model else fail)(
                task,
                "proposal_endpoint",
                "Endpoint and model configured; runtime Chat Completions preflight still required."
                if url and model
                else "Set LLM_BASE_URL and LLM_MODEL_NAME; credentials only via LLM_API_KEY.",
            )
        )
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
