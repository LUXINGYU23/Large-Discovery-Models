"""Pinned assets and scientific identity for ReaSyn campaign recovery."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ldm_tts.harness.protocol import file_sha256
from ldm_tts.harness.ucb_diagnostics import DIAGNOSTICS_PATH
from ldm_tts.transport.openai import generation_body

TASK_ROOT = Path(__file__).resolve().parents[1]


def _directory_hash(root):
    root = Path(root).resolve()
    entries = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        entries.append({"path": path.relative_to(root).as_posix(), "sha256": file_sha256(path)})
    return hashlib.sha256(json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def harness_component_digests(args):
    if args.proposal_mode != "harness":
        return {}
    digests = {
        "resource_tree": _directory_hash(TASK_ROOT / "resources/harness"),
        "harness_mcp_config": None,
    }
    if args.harness_mcp_config:
        digests["harness_mcp_config"] = file_sha256(args.harness_mcp_config)
    if args.search_method == "ldm_harness_compiled":
        digests["policy_diagnostics"] = file_sha256(DIAGNOSTICS_PATH)
    return digests


def scientific_identity(args, configuration, target, contract):
    identity_keys = (
        "benchmark",
        "mock",
        "oracle",
        "seed",
        "proposal_mode",
        "num_cycles",
        "search_width",
        "exhaustiveness",
        "num_editflow_samples",
        "model_paths",
        "fpindex",
        "rxn_matrix",
        "additional_fpindex",
        "reservoir_size",
        "evaluations_per_round",
        "max_oracle_calls",
        "llm_model",
        "llm_max_tokens",
        "llm_wire_api", "llm_reasoning", "llm_temperature", "llm_extra_body_json",
        "max_results",
        "projection_time_limit",
        "projection_timeout",
        "device",
        "evaluator_python",
        "gp_history_limit",
        "acquisition_beta", "acquisition_alpha", "acquisition_eta", "acquisition_z_clip",
        "search_method", "proposal_batch_size", "bo_pool_size", "max_replenishment_batches",
        "recovery_attempts", "projection_retry_targets", "initialization_mode",
        "proposal_recovery_seed_span",
        "harness_sessions", "harness_thinking", "harness_tool_budget",
        "harness_research_strategy", "harness_research_strategy_identity",
        "harness_sidecar_image", "policy_runner_image", "harness_mcp_config",
        "harness_wall_time_seconds", "harness_response_timeout",
    )
    scientific = {k: configuration[k] for k in identity_keys}
    scientific["resolved_generation_body"] = generation_body(
        wire_api=args.llm_wire_api, reasoning=args.llm_reasoning,
        extra_body=json.loads(args.llm_extra_body_json),
    )
    scientific["bo_targets"] = getattr(args, "bo_targets", [])
    scientific["original_target"] = target
    scientific["source_archive_digest"] = contract.benchmark["source_commit"]
    scientific["asset_digests"] = args.asset_digests
    scientific["harness_component_digests"] = configuration["harness_component_digests"]
    if args.benchmark == "reconstruction":
        scientific["q0_identity_version"] = "canonical_query_group_trial_allocation_v1"
    return scientific


def check_assets(args):
    args.asset_digests = {}
    if args.mock:
        return
    paths = [
        args.upstream_root / "reasyn/sampler/sampler.py",
        args.fpindex,
        args.rxn_matrix,
        *args.model_paths,
    ]
    if args.additional_fpindex:
        paths.append(args.additional_fpindex)
    missing = [str(p) for p in paths if not p.is_file()]
    if len(args.model_paths) != 2:
        raise ValueError("exactly two checkpoints required: AR then EB")
    if missing:
        raise FileNotFoundError("Missing ReaSyn assets: " + ", ".join(missing))

    # Hash once before all target campaigns. Same-path asset replacement cannot
    # silently reuse cached products or resume an incompatible scientific run.
    for path in sorted(set(paths) | set((args.upstream_root / "reasyn").rglob("*.py"))):
        args.asset_digests[str(path)] = file_sha256(path)
    pinned = json.loads((TASK_ROOT / "resources/source_provenance.json").read_text())[
        "files"
    ]
    prefix = "ReaSyn-reasyn_v2/"
    for name, expected in pinned.items():
        if name.startswith(prefix) and name.endswith((".py", ".yml")):
            path = args.upstream_root / name[len(prefix) :]
            actual = file_sha256(path)
            if actual != expected:
                raise ValueError(
                    "Supplied upstream source differs from pinned archive: " + name
                )
    args.asset_digests["adapter_projection_worker"] = file_sha256(
        TASK_ROOT / "core/projection_worker.py"
    )
