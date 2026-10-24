"""One shared LDM campaign over a fixed schedule of benchmark questions and revisions."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

from ldm_tts.transport.openai import WIRE_APIS, REASONING_LEVELS, generation_body
from .data import DEFAULT_UPSTREAM
from .methods import LDM_METHODS, HARNESS_METHODS


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument(
        "--upstream-root",
        type=Path,
        default=Path(os.environ.get("ATOMWORLD_UPSTREAM_ROOT", DEFAULT_UPSTREAM)),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("runs/atomworld"))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument(
        "--proposal-mode",
        choices=("mock", "real", "live", "none", "callable", "openai"),
        default=None,
    )
    parser.add_argument("--attempts-per-sample", type=int, default=1)
    parser.add_argument(
        "--search-method",
        choices=("llm", "harness", "harness_public_audit", *LDM_METHODS),
        default="ldm_harness_compiled",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Pilot alias for scheduled attempts per sample",
    )
    parser.add_argument("--campaign-index", type=int, default=0)
    parser.add_argument("--proposal-samples", type=int, default=8)
    parser.add_argument("--bo-pool-size", type=int, default=32)
    parser.add_argument("--acquisition-alpha", type=float, default=1.0)
    parser.add_argument("--acquisition-eta", type=float, default=1.0)
    parser.add_argument("--ucb-beta", type=float, default=1.0)
    parser.add_argument("--z-clip", type=float, default=5.0)
    parser.add_argument("--gp-history-limit", type=int, default=128)
    parser.add_argument("--gp-lengthscale", type=float, default=1.5)
    parser.add_argument("--gp-noise", type=float, default=1e-4)
    parser.add_argument(
        "--initialization-mode",
        choices=("shared_start", "independent"),
        default="shared_start",
    )
    parser.add_argument("--service-retry-allowance", type=int, default=3)
    parser.add_argument(
        "--harness-profile", nargs="+", action="extend", default=None
    )
    parser.add_argument("--harness-sidecar-image", default="ldm-pi-harness:latest")
    parser.add_argument(
        "--harness-cache-dir", type=Path, default=Path.home() / ".cache/ldm-gondolin"
    )
    parser.add_argument("--harness-docker-host", default="")
    parser.add_argument("--harness-container-user", default=None)
    parser.add_argument(
        "--harness-thinking",
        choices=("off", "minimal", "low", "medium", "high", "xhigh", "max"),
        default=None,
    )
    parser.add_argument("--harness-wall-time-seconds", type=int, default=1800)
    parser.add_argument("--harness-response-timeout", type=float, default=2100)
    parser.add_argument("--harness-recovery-seconds", type=float, default=0)
    parser.add_argument("--harness-max-submission-attempts", type=int, default=3)
    parser.add_argument(
        "--harness-tool-budget",
        nargs="+",
        action="extend",
        default=None,
    )
    default_harness_tool_budgets = [
        "web_search=8",
        "web_fetch=16",
        "resolve-library-id=4",
        "query-docs=8",
        "bash=64",
    ]
    parser.add_argument("--harness-mcp-config", type=Path, default=None)
    parser.add_argument("--policy-runner-image", default="ldm-pi-harness:latest")
    parser.add_argument(
        "--limit", type=int, default=0, help="0 evaluates every prepared sample"
    )
    parser.add_argument(
        "--sample-id",
        default=None,
        help="Select one exact public sample ID for a pilot case",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--proposal-format", choices=("cif", "operations"), default="cif"
    )
    parser.add_argument(
        "--tools-root",
        type=Path,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--llm-url",
        default=os.environ.get(
            "LLM_BASE_URL",
            os.environ.get("LDM_LLM_URL", os.environ.get("OPENAI_BASE_URL", "")),
        ),
    )
    parser.add_argument(
        "--llm-model-name",
        default=os.environ.get(
            "LLM_MODEL_NAME",
            os.environ.get("LDM_LLM_MODEL", os.environ.get("OPENAI_MODEL", "")),
        ),
    )
    parser.add_argument("--llm-max-tokens", type=int, default=8192)
    parser.add_argument("--llm-timeout", type=float, default=180.0)
    parser.add_argument("--llm-temperature", type=float, default=0.0)
    parser.add_argument("--llm-wire-api", choices=WIRE_APIS, default="responses")
    parser.add_argument("--llm-reasoning", choices=REASONING_LEVELS)
    parser.add_argument(
        "--llm-extra-body-json",
        default="{}",
        help="Provider reasoning controls; never put credentials here",
    )
    parser.add_argument("--harness-research-strategy-config", type=Path)
    args = parser.parse_args(argv)
    from ldm_tts.harness.research_strategy import configure_research_strategy
    try:
        configure_research_strategy(args, enabled=args.search_method in HARNESS_METHODS)
    except ValueError as exc:
        parser.error(str(exc))
    if args.llm_reasoning is not None and args.harness_thinking is not None and args.llm_reasoning != args.harness_thinking:
        parser.error("llm-reasoning and harness-thinking must agree")
    args.llm_reasoning = args.llm_reasoning or args.harness_thinking or "off"
    args.harness_thinking = "off" if args.llm_reasoning == "none" else args.llm_reasoning
    if args.search_method in HARNESS_METHODS and args.llm_wire_api != "responses":
        parser.error("Harness requires llm-wire-api responses")
    if not math.isfinite(args.llm_temperature) or not 0 <= args.llm_temperature <= 2:
        parser.error("llm-temperature must be finite and between 0 and 2")
    # The shared config runner repeats list-valued flags. Preserve all entries
    # both in that form and in a grouped, interactive command-line invocation.
    if args.harness_profile is None:
        from tasks.atomworld.core.harness import PROFILE_IDS
        args.harness_profile = list(PROFILE_IDS)
    if args.harness_tool_budget is None:
        args.harness_tool_budget = default_harness_tool_budgets
    if args.proposal_mode in {"mock", "callable"}:
        args.mock = True
    if args.resume_from is not None:
        args.out_dir, args.resume = args.resume_from, True
    elif args.run_name is not None:
        if Path(args.run_name).name != args.run_name or args.run_name in {".", ".."}:
            parser.error("run-name must be a single safe path component")
        args.out_dir = args.out_dir / args.run_name
    if args.iterations is not None:
        args.attempts_per_sample = args.iterations
    if args.attempts_per_sample < 1 or args.limit < 0:
        parser.error("attempts-per-sample must be positive and limit nonnegative")
    if args.campaign_index < 0 or args.service_retry_allowance < 0:
        parser.error("campaign-index and service-retry-allowance must be nonnegative")
    if min(args.proposal_samples, args.bo_pool_size, args.gp_history_limit) < 1:
        parser.error("LDM proposal, BO pool and history counts must be positive")
    if any(not math.isfinite(v) or v < 0 for v in (args.acquisition_alpha, args.acquisition_eta, args.ucb_beta)):
        parser.error("LDM alpha, eta and beta must be finite and nonnegative")
    if any(not math.isfinite(v) or v <= 0 for v in (args.z_clip, args.gp_lengthscale, args.gp_noise)):
        parser.error("LDM scales and noise must be finite and positive")
    if args.search_method in LDM_METHODS and args.proposal_format != "cif":
        parser.error("LDM optimizes complete CIF proposals")
    if args.tools_root is not None:
        parser.error(
            "--tools-root is obsolete: bounded geometry tools now ship inside this task"
        )
    if args.search_method in HARNESS_METHODS:
        from ldm_tts.harness import parse_tool_call_budgets
        from tasks.atomworld.core.harness import harness_profiles

        try:
            harness_profiles(args.harness_profile)
            if args._research_strategy.requires_main and len(args.harness_profile) != 4:
                raise ValueError("AtomWorld research trees require four Harness profiles")
            if "policy_architect" in args.harness_profile:
                raise ValueError(
                    "policy_architect is reserved for the independent policy session"
                )
            parse_tool_call_budgets(
                args.harness_tool_budget,
                excluded_tools=("submit_answer", "submit_optimization_policy"),
            )
        except ValueError as exc:
            parser.error(str(exc))
        if (
            min(
                args.harness_wall_time_seconds,
                args.harness_response_timeout,
                args.harness_max_submission_attempts,
            )
            <= 0
            or args.harness_recovery_seconds < 0
        ):
            parser.error(
                "Harness deadlines and submission count must be positive; recovery nonnegative"
            )
        if args.proposal_format != "cif":
            parser.error(
                "Harness submits CIF answers; execute geometry operations inside its guest"
            )
    if not args.mock and args.data_dir is None:
        parser.error(
            "--data-dir is required for real runs; prepare the released data first"
        )
    try:
        extra = json.loads(args.llm_extra_body_json)
    except ValueError:
        parser.error("llm-extra-body-json must be a JSON object")
    if not isinstance(extra, dict):
        parser.error("llm-extra-body-json must be a JSON object")
    try:
        generation_body(wire_api=args.llm_wire_api, reasoning=args.llm_reasoning, extra_body=extra)
        args.llm_extra_body_json = json.dumps(extra, sort_keys=True)
    except ValueError as exc:
        parser.error(str(exc))
    if any(
        any(
            word in key.lower()
            for word in ("token_key", "api_key", "authorization", "password", "secret")
        )
        for key in extra
    ):
        parser.error(
            "Credentials must be supplied through environment variables, not extra-body JSON"
        )
    if args.mock and args.proposal_format != "cif":
        parser.error(
            "Mock fixture uses CIF output; operations mode is verified with the real local geometry tool"
        )
    return args
