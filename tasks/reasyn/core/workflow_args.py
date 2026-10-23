"""ReaSyn CLI resolution and serialized campaign configuration."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

from ldm_tts.harness.research_strategy import configure_research_strategy
from ldm_tts.transport.openai import WIRE_APIS, REASONING_LEVELS, generation_body
from .metrics import ORACLES, WIDTH_ONE

TASK_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = TASK_ROOT.parents[1]


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="LDM-guided frozen ReaSyn reconstruction and released 13-oracle TDC optimization."
    )
    p.add_argument("--mock", action="store_true")
    p.add_argument(
        "--benchmark", choices=("reconstruction", "tdc"), default="reconstruction"
    )
    p.add_argument(
        "--proposal-mode",
        choices=("auto", "mock", "callable", "none", "openai", "baseline", "harness"),
        default="auto",
    )
    p.add_argument("--iterations", type=int, default=2)
    p.add_argument("--reservoir-size", "--proposal-samples", dest="reservoir_size", type=int, default=4)
    p.add_argument("--search-method", choices=("auto", "baseline", "ldm", "bo", "llm", "harness", "ldm_harness", "ldm_harness_compiled"), default="auto")
    p.add_argument("--proposal-batch-size", type=int, default=2)
    p.add_argument("--bo-pool-size", type=int)
    p.add_argument("--bo-targets-file", type=Path, default=os.environ.get("REASYN_BO_TARGETS"))
    p.add_argument("--max-replenishment-batches", type=int, default=4)
    p.add_argument("--recovery-attempts", type=int, default=3)
    p.add_argument("--projection-retry-targets", type=int)
    p.add_argument("--initialization-mode", choices=("none", "shared_start"), default="none")
    p.add_argument("--run-name", default="")
    p.add_argument("--evaluations-per-round", type=int, default=1)
    p.add_argument("--max-oracle-calls", type=int, default=10000)
    p.add_argument("--oracle", choices=ORACLES, default="jnk3")
    p.add_argument("--seed", "--campaign-index", dest="seed", type=int, default=0)
    p.add_argument(
        "--upstream-root",
        type=Path,
        default=Path(
            os.environ.get("REASYN_ROOT", str(REPO_ROOT.parent / "ReaSyn-reasyn_v2"))
        ),
    )
    p.add_argument("--model-paths", default=os.environ.get("REASYN_MODEL_PATHS", ""))
    p.add_argument("--fpindex", type=Path)
    p.add_argument("--rxn-matrix", type=Path)
    p.add_argument("--additional-fpindex", type=Path)
    p.add_argument(
        "--dataset",
        choices=("zinc250k", "enamine", "chembl", "custom"),
        default="zinc250k",
    )
    p.add_argument("--targets-file", type=Path)
    p.add_argument("--target-smiles", default="")
    p.add_argument("--target-limit", type=int, default=1)
    p.add_argument("--search-width", type=int, default=0)
    p.add_argument("--exhaustiveness", type=int, default=4)
    p.add_argument("--num-cycles", type=int, default=1)
    p.add_argument("--num-editflow-samples", type=int, default=4)
    p.add_argument("--max-results", type=int, default=100)
    p.add_argument("--projection-time-limit", type=int, default=1000)
    p.add_argument("--projection-timeout", type=int, default=3600,
        help="Single-target process allowance including loading; each additional serial target adds projection-time-limit seconds.")
    p.add_argument(
        "--evaluator-python", default=os.environ.get("REASYN_PYTHON", sys.executable)
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--acquisition-beta", type=float, default=1.0)
    p.add_argument("--acquisition-alpha", type=float, default=1.0)
    p.add_argument("--acquisition-eta", type=float, default=1.0)
    p.add_argument("--acquisition-z-clip", type=float, default=5.0)
    p.add_argument("--harness-sessions", type=int, default=4)
    p.add_argument("--harness-sidecar-image", default="ldm-pi-harness:latest")
    p.add_argument("--harness-docker-host", default="")
    p.add_argument("--harness-container-user", default="auto")
    p.add_argument("--harness-cache-dir", type=Path)
    p.add_argument("--harness-thinking", choices=("off", "minimal", "low", "medium", "high", "xhigh", "max"))
    p.add_argument("--harness-wall-time-seconds", type=int, default=1800)
    p.add_argument("--harness-response-timeout", type=int, default=2100)
    p.add_argument("--harness-mcp-config", type=Path)
    p.add_argument("--harness-tool-budget", action="append", default=[])
    p.add_argument("--policy-runner-image", default="ldm-pi-harness:latest")
    p.add_argument("--gp-history-limit", type=int, default=256)
    p.add_argument(
        "--llm-url",
        default=os.environ.get(
            "LLM_BASE_URL",
            os.environ.get("LDM_LLM_URL", os.environ.get("OPENAI_BASE_URL", "")),
        ),
    )
    p.add_argument(
        "--llm-model",
        default=os.environ.get("LLM_MODEL_NAME", os.environ.get("LDM_LLM_MODEL", "")),
    )
    p.add_argument("--llm-max-tokens", type=int, default=4096)
    p.add_argument("--llm-wire-api", choices=WIRE_APIS, default="responses")
    p.add_argument("--llm-reasoning", choices=REASONING_LEVELS)
    p.add_argument("--llm-temperature", type=float, default=0.7)
    p.add_argument("--llm-extra-body-json", default="{}")
    p.add_argument("--out-dir", type=Path, default=Path("runs/reasyn"))
    p.add_argument("--resume-from", type=Path)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--harness-research-strategy-config", type=Path)
    args = p.parse_args(argv)
    if args.search_method == "auto":
        args.search_method = "baseline" if args.proposal_mode == "baseline" else "ldm"
    if args.proposal_mode == "callable":
        args.proposal_mode = "mock"
    if args.proposal_mode == "none":
        args.proposal_mode = "auto"
    harness_method = args.search_method in ("harness", "ldm_harness", "ldm_harness_compiled")
    if args.llm_reasoning is not None and args.harness_thinking is not None and args.llm_reasoning != args.harness_thinking:
        p.error("llm-reasoning and harness-thinking must agree")
    args.llm_reasoning = args.llm_reasoning or args.harness_thinking or "off"
    args.harness_thinking = "off" if args.llm_reasoning == "none" else args.llm_reasoning
    try:
        extra = json.loads(args.llm_extra_body_json)
        if not isinstance(extra, dict):
            raise ValueError("llm-extra-body-json must be an object")
        generation_body(wire_api=args.llm_wire_api, reasoning=args.llm_reasoning, extra_body=extra)
        args.llm_extra_body_json = json.dumps(extra, sort_keys=True)
        if not math.isfinite(args.llm_temperature) or not 0 <= args.llm_temperature <= 2:
            raise ValueError("llm-temperature must be finite and between 0 and 2")
    except ValueError as exc:
        p.error(str(exc))
    if harness_method and args.llm_wire_api != "responses":
        p.error("Harness requires llm-wire-api responses")
    if harness_method:
        if args.proposal_mode not in ("auto", "harness"):
            p.error("Harness methods require proposal-mode auto or harness; mock controls the evaluator only")
        args.proposal_mode = "harness"
    elif args.proposal_mode == "harness":
        p.error("proposal-mode harness requires a Harness search-method")
    if args.bo_pool_size is None:
        args.bo_pool_size = args.reservoir_size
    if args.projection_retry_targets is None:
        args.projection_retry_targets = args.reservoir_size * args.recovery_attempts
    if args.run_name and (Path(args.run_name).name != args.run_name or args.run_name in (".", "..")):
        p.error("run-name must be a single directory name")
    args.upstream_root = args.upstream_root.expanduser().resolve()
    if args.harness_cache_dir:
        args.harness_cache_dir = args.harness_cache_dir.expanduser().resolve()
    if args.harness_mcp_config:
        args.harness_mcp_config = args.harness_mcp_config.expanduser().resolve()
    if args.proposal_mode == "auto":
        args.proposal_mode = "mock" if args.mock else "openai"
    if args.proposal_mode == "mock" and not args.mock:
        p.error("mock proposals require --mock; they are not scientific results")
    if args.proposal_mode == "baseline" and args.benchmark == "tdc":
        p.error("use upstream optimize_tdc.py for Graph GA baseline")
    if not args.search_width:
        args.search_width = (
            (1 if args.oracle in WIDTH_ONE else 2) if args.benchmark == "tdc" else 8
        )
    raw = (
        args.model_paths
        or "data/trained_model/nv-reasyn-ar-166m-v2.ckpt,data/trained_model/nv-reasyn-eb-174m-v2.ckpt"
    )

    def upstream_path(value):
        value = Path(value).expanduser()
        return (
            value.resolve()
            if value.is_absolute()
            else (args.upstream_root / value).resolve()
        )

    args.model_paths = [upstream_path(v.strip()) for v in raw.split(",")]
    args.fpindex = upstream_path(args.fpindex or "data/processed/comp_2048/fpindex.pkl")
    args.rxn_matrix = upstream_path(
        args.rxn_matrix or "data/processed/comp_2048/matrix.pkl"
    )
    if args.targets_file:
        args.targets_file = upstream_path(args.targets_file)
    if args.additional_fpindex:
        args.additional_fpindex = upstream_path(args.additional_fpindex)
    elif args.benchmark == "reconstruction" and args.dataset == "zinc250k":
        args.additional_fpindex = upstream_path(
            "data/processed/zinc250k_2048/fpindex.pkl"
        )
    if (
        args.iterations < 1
        or args.reservoir_size < 1
        or not 1 <= args.evaluations_per_round <= args.reservoir_size
    ):
        p.error("positive iterations and 1 <= evaluation batch <= reservoir required")
    if not 1 <= args.max_oracle_calls <= 10000:
        p.error("oracle call cap must be in [1,10000]")
    if any(
        v < 1
        for v in (
            args.search_width,
            args.exhaustiveness,
            args.num_cycles,
            args.num_editflow_samples,
            args.max_results,
            args.target_limit,
            args.gp_history_limit,
            args.projection_time_limit,
            args.projection_timeout,
        )
    ):
        p.error("search, dataset and timing limits must be positive")
    if args.seed < 0 or any(not math.isfinite(v) or v < 0 for v in (args.acquisition_beta, args.acquisition_alpha, args.acquisition_eta)):
        p.error("seed and acquisition parameters must be finite and nonnegative")
    if (args.proposal_batch_size < 1 or args.bo_pool_size < args.evaluations_per_round
            or args.max_replenishment_batches < 0 or args.recovery_attempts < 0
            or args.projection_retry_targets < 0 or args.harness_sessions < 1
            or args.harness_wall_time_seconds < 1 or args.harness_response_timeout < 1
            or not math.isfinite(args.acquisition_z_clip) or args.acquisition_z_clip <= 0):
        p.error("Invalid sampling, recovery, or Harness limits")
    try:
        configure_research_strategy(args, enabled=args.search_method in ("ldm_harness", "ldm_harness_compiled"))
        if args._research_strategy.requires_main and args.harness_sessions != 4:
            raise ValueError("ReaSyn research trees require four Harness sessions")
    except ValueError as exc:
        p.error(str(exc))
    return args


def jsonable_args(args):
    def convert(v):
        if isinstance(v, Path):
            return str(v)
        if isinstance(v, list):
            return [convert(x) for x in v]
        return v

    return {k: convert(v) for k, v in vars(args).items() if k != "provider_api_key" and not k.startswith("_")}
