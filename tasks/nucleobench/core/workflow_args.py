"""CLI normalization, validation and configuration identity for NucleoBench."""

from __future__ import annotations

import argparse
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ldm_tts.contracts import LDMTaskSpec
from ldm_tts.harness import POLICY_CAPABILITIES, parse_tool_call_budgets
from ldm_tts.harness.container import resolve_container_user
from ldm_tts.harness.pi import DEFAULT_NETWORK_TOOL_BUDGETS
from ldm_tts.harness.research_strategy import configure_research_strategy
from ldm_tts.harness.research_tree import CoverageForks
from ldm_tts.transport.openai import WIRE_APIS
from tasks.nucleobench.core import harness_runtime, provider as provider_config
from tasks.nucleobench.core.cases import get_case
from tasks.nucleobench.core.constants import (
    COMPILED_POLICY_METHOD, DEFAULT_LDM_ALPHA, DEFAULT_LDM_ETA,
    PARALLEL_HARNESS_METHODS, PERSISTENT_HARNESS_METHODS, SEARCH_METHODS,
)
from tasks.nucleobench.core.digests import canonical_json_sha256
from tasks.nucleobench.core.hamming_gp import HammingGPUCBConfig
from tasks.nucleobench.core.harness import AVAILABLE_HARNESS_PROFILE_IDS, HARNESS_PROFILE_IDS
from tasks.nucleobench.core.mock import MOCK_CASE
from tasks.nucleobench.core.proposals import DEFAULT_PROPOSAL_MAX_WORKERS, direct_request_limit

EXECUTION_PROFILES = ("qualification", "pilot_evaluation", "official_benchmark")
TERMINATION_KINDS = ("rounds", "wall_time")
INITIALIZATION_MODES = ("shared_start", "official_start")
DEFAULT_HARDWARE_PROFILE = "n1-highmem-16-cpu"
LLM_REASONING_LEVELS = (
    "off",
    "none",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
    "max",
)
HARNESS_THINKING_LEVELS = tuple(
    level for level in LLM_REASONING_LEVELS if level != "none"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Config-driven NucleoBench workflow using the shared LDM engine.")
    parser.add_argument("--case-id", default="malinois_k562")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument(
        "--execution-profile", choices=EXECUTION_PROFILES, default="qualification"
    )
    parser.add_argument(
        "--termination-kind", choices=TERMINATION_KINDS, default="rounds"
    )
    parser.add_argument(
        "--initialization-mode", choices=INITIALIZATION_MODES, default="shared_start"
    )
    parser.add_argument(
        "--search-method", choices=tuple(sorted(SEARCH_METHODS)), default="ldm"
    )
    parser.add_argument("--proposal-mode", choices=("none", "openai"))
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--max-seconds", type=int)
    parser.add_argument("--reservoir-size", type=int, default=4)
    parser.add_argument("--evaluations-per-round", type=int, default=1)
    parser.add_argument("--oracle-batch-size", type=int)
    parser.add_argument("--proposal-samples", type=int)
    parser.add_argument("--bo-pool-size", type=int)
    parser.add_argument("--proposal-candidates-per-request", type=int)
    parser.add_argument(
        "--proposal-max-workers", type=int, default=DEFAULT_PROPOSAL_MAX_WORKERS
    )
    parser.add_argument("--campaign-index", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--prepared-dir", type=Path)
    parser.add_argument("--expected-start-set-sha256")
    parser.add_argument("--hardware-profile", default=DEFAULT_HARDWARE_PROFILE)
    parser.add_argument("--acquisition-beta", type=float, default=1.0)
    parser.add_argument("--gp-noise-variance", type=float, default=1.0e-4)
    parser.add_argument("--gp-jitter", type=float, default=1.0e-8)
    parser.add_argument("--gp-target-std-floor", type=float, default=0.1)
    parser.add_argument("--gp-min-history-for-fit", type=int, default=4)
    parser.add_argument("--gp-max-observations", type=int, default=256)
    parser.add_argument("--gp-global-best-observations", type=int, default=64)
    parser.add_argument("--alpha", type=float, default=DEFAULT_LDM_ALPHA)
    parser.add_argument("--eta", type=float, default=DEFAULT_LDM_ETA)
    parser.add_argument("--z-clip", type=float, default=5.0)
    parser.add_argument("--llm-url")
    parser.add_argument("--llm-model-name")
    parser.add_argument("--llm-wire-api", choices=WIRE_APIS, default="chat_completions")
    parser.add_argument("--llm-reasoning", choices=LLM_REASONING_LEVELS, default="off")
    parser.add_argument("--llm-timeout", type=float, default=120.0)
    parser.add_argument("--llm-max-tokens", type=int, default=262_144)
    parser.add_argument("--llm-temperature", type=float, default=0.7)
    parser.add_argument("--llm-extra-body-json", default="{}")
    parser.add_argument("--api-key-file", type=Path)
    parser.add_argument("--harness-sidecar-image", default="ldm-pi-harness:latest")
    parser.add_argument("--harness-candidates-per-session", type=int)
    parser.add_argument("--harness-unique-candidates", action="store_true")
    parser.add_argument(
        "--harness-profile", action="append", choices=AVAILABLE_HARNESS_PROFILE_IDS
    )
    parser.add_argument(
        "--harness-thinking", choices=HARNESS_THINKING_LEVELS, default="max"
    )
    parser.add_argument("--harness-mcp-config", type=Path)
    parser.add_argument("--harness-sol-pi-config", type=Path)
    parser.add_argument("--harness-research-strategy-config", type=Path,
                        help="JSON strategy configuration for independent sessions or configurable research forks")
    parser.add_argument("--harness-cache-dir", type=Path)
    parser.add_argument("--harness-docker-host")
    parser.add_argument("--harness-container-user")
    parser.add_argument("--harness-response-timeout", type=float, default=2100.0)
    parser.add_argument("--harness-wall-time-seconds", type=int, default=1800)
    parser.add_argument("--harness-tool-budget", action="append", metavar="NAME=COUNT")
    parser.add_argument("--harness-surrogate-query", action="store_true")
    parser.add_argument(
        "--policy-capability",
        action="append",
        choices=tuple(sorted(POLICY_CAPABILITIES)),
    )
    parser.add_argument("--policy-tool-budget", action="append", metavar="NAME=COUNT")
    parser.add_argument("--policy-max-submission-attempts", type=int)
    parser.add_argument(
        "--no-harness-context7",
        action="store_false",
        dest="harness_context7",
        default=True,
    )
    parser.add_argument("--out-dir", type=Path, default=Path("runs"))
    parser.add_argument("--run-name")
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        _apply_derived_args(args)
    except ValueError as exc:
        parser.error(str(exc))
    _validate_args(args, parser)
    return args


def _apply_derived_args(args: argparse.Namespace) -> None:
    strategy = configure_research_strategy(
        args, enabled=args.search_method in PARALLEL_HARNESS_METHODS and not args.mock,
        measurement_supported=True,
    )
    args.harness_main_research = strategy.requires_main
    if args.iterations is None and args.termination_kind == "rounds":
        args.iterations = 2
    if args.oracle_batch_size is None:
        case = MOCK_CASE if args.mock else get_case(args.case_id)
        args.oracle_batch_size = (
            4 if case.model_family == "enformer" else args.evaluations_per_round
        )
    if args.proposal_mode is None:
        args.proposal_mode = (
            "openai" if args.search_method in {"ldm", "llm"} else "none"
        )
    if args.harness_profile is None:
        args.harness_profile = list(HARNESS_PROFILE_IDS)
    if args.harness_candidates_per_session is None:
        args.harness_candidates_per_session = args.evaluations_per_round
    if args.bo_pool_size is None:
        args.bo_pool_size = 3 * args.evaluations_per_round
    if args.proposal_samples is None and args.search_method in PARALLEL_HARNESS_METHODS:
        args.proposal_samples = (
            len(args.harness_profile) * args.harness_candidates_per_session
        )
    elif args.proposal_samples is None:
        args.proposal_samples = args.evaluations_per_round * (
            1 if args.search_method in {"llm", "harness"} else 4
        )
    if args.proposal_candidates_per_request is None:
        args.proposal_candidates_per_request = (
            args.evaluations_per_round if args.search_method == "ldm" else 1
        )
    args.proposal_request_limit = (
        direct_request_limit(args.search_method, args.evaluations_per_round)
        if args.search_method in {"ldm", "llm"}
        else 0
    )
    if (
        args.search_method in PERSISTENT_HARNESS_METHODS
        and args.harness_container_user is None
    ):
        container_user = resolve_container_user(None, args.harness_docker_host)
        if container_user:
            args.harness_container_user = container_user
    args.initialization_evaluations = 1
    if args.search_method == COMPILED_POLICY_METHOD:
        if args.policy_tool_budget is None:
            args.policy_tool_budget = list(DEFAULT_NETWORK_TOOL_BUDGETS)
        if args.policy_capability is None:
            args.policy_capability = sorted(POLICY_CAPABILITIES)
        if args.policy_max_submission_attempts is None:
            args.policy_max_submission_attempts = 3


def _validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    strategy = args._research_strategy
    if strategy.measurement_policy is not None and (
        args.mock or args.search_method not in {"ldm", *PARALLEL_HARNESS_METHODS}
    ):
        parser.error("measurement coverage requires real acquisition-tilted selection")
    if isinstance(strategy.fork_policy, CoverageForks) and len(harness_runtime.proposal_profiles(args)) < 3:
        parser.error("coverage forks require at least three research profiles")
    if args.harness_main_research and (
        args.search_method not in PARALLEL_HARNESS_METHODS or args.harness_sol_pi_config is not None
        or args.termination_kind != "rounds" or args.mock
        or args.iterations is None or args.iterations < 2
    ):
        parser.error("fork strategies require real parallel Harness with at least one active round and without SoL-Pi")
    if args.harness_surrogate_query and args.search_method not in PERSISTENT_HARNESS_METHODS:
        parser.error("--harness-surrogate-query requires a persistent Harness method")
    positive_counts = (
        "reservoir_size",
        "evaluations_per_round",
        "oracle_batch_size",
        "proposal_samples",
        "bo_pool_size",
        "proposal_candidates_per_request",
        "proposal_max_workers",
        "gp_min_history_for_fit",
        "gp_max_observations",
        "gp_global_best_observations",
        "llm_max_tokens",
        "harness_candidates_per_session",
        "harness_wall_time_seconds",
    )
    if any(getattr(args, name) < 1 for name in positive_counts):
        parser.error(
            "proposal, GP, token, Harness, and evaluation counts must be positive"
        )
    if args.campaign_index < 0 or args.start_index < 0:
        parser.error("--campaign-index and --start-index must be non-negative")
    finite_positive = (
        "llm_timeout",
        "harness_response_timeout",
        "gp_noise_variance",
        "gp_jitter",
        "gp_target_std_floor",
        "z_clip",
    )
    if any(
        not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0
        for name in finite_positive
    ):
        parser.error("timeouts, GP scales, and --z-clip must be finite and positive")
    for name in ("acquisition_beta", "alpha", "eta"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and non-negative")
    if not math.isfinite(args.llm_temperature) or not 0 <= args.llm_temperature <= 2:
        parser.error("--llm-temperature must be finite and between 0 and 2")
    if args.gp_global_best_observations > args.gp_max_observations:
        parser.error(
            "--gp-global-best-observations cannot exceed --gp-max-observations"
        )
    if args.termination_kind == "rounds":
        if (
            args.iterations is None
            or args.iterations < 1
            or args.max_seconds is not None
        ):
            parser.error(
                "round termination requires positive --iterations and prohibits "
                "--max-seconds"
            )
    elif (
        args.iterations is not None or args.max_seconds is None or args.max_seconds < 1
    ):
        parser.error(
            "wall-time termination requires positive --max-seconds and prohibits "
            "--iterations"
        )
    if args.execution_profile == "pilot_evaluation" and (
        args.termination_kind != "rounds"
        or args.initialization_mode != "shared_start"
    ):
        parser.error(
            "pilot_evaluation requires round termination and shared_start initialization"
        )
    if args.execution_profile == "official_benchmark" and (
        args.termination_kind != "wall_time"
        or args.initialization_mode != "official_start"
    ):
        parser.error("official_benchmark requires wall_time and official_start")
    if args.execution_profile == "qualification" and args.termination_kind != "rounds":
        parser.error("qualification runs use round termination")
    expected_samples = args.evaluations_per_round * (
        1 if args.search_method in {"llm", "harness"} else 4
    )
    if args.search_method in PARALLEL_HARNESS_METHODS:
        expected_samples = len(args.harness_profile) * args.harness_candidates_per_session
    if args.search_method not in PARALLEL_HARNESS_METHODS and (
        tuple(args.harness_profile) != HARNESS_PROFILE_IDS or args.harness_unique_candidates
    ):
        parser.error("Custom Harness sampling settings require a parallel Harness method")
    if args.search_method == "bo" and args.proposal_samples < args.evaluations_per_round:
        parser.error("BO proposal samples must cover the evaluation batch")
    if args.search_method != "bo" and args.proposal_samples != expected_samples:
        parser.error(
            f"--proposal-samples must equal {expected_samples} for {args.search_method}"
        )
    if args.search_method == "ldm" or args.search_method in PARALLEL_HARNESS_METHODS:
        if not args.evaluations_per_round <= args.bo_pool_size < args.proposal_samples:
            parser.error("LDM requires evaluation batch <= BO pool < proposal samples")
    elif args.bo_pool_size != 3 * args.evaluations_per_round:
        parser.error("--bo-pool-size requires an LDM method")
    expected_request_size = (
        args.evaluations_per_round if args.search_method == "ldm" else 1
    )
    if (
        args.search_method in {"ldm", "llm"}
        and args.proposal_candidates_per_request != expected_request_size
    ):
        parser.error(
            "direct LDM uses one B-sized request per lineage; direct LLM uses one "
            "candidate per request"
        )
    if args.search_method in {"ldm", "llm"}:
        if args.proposal_mode != "openai" and not args.mock:
            parser.error(
                f"--search-method={args.search_method} requires --proposal-mode=openai"
            )
    elif args.proposal_mode != "none":
        parser.error(
            f"--search-method={args.search_method} requires --proposal-mode=none"
        )
    if args.search_method == "harness" and (
        args.harness_candidates_per_session != args.evaluations_per_round
    ):
        parser.error("Direct Harness candidates per session must equal evaluations per round")
    if args.harness_tool_budget is None:
        args.harness_tool_budget = [
            value
            for value in DEFAULT_NETWORK_TOOL_BUDGETS
            if args.harness_context7
            or not value.startswith(("resolve-library-id=", "query-docs="))
        ]
    try:
        provider_config.parse_extra_body(args.llm_extra_body_json)
        budgets = parse_tool_call_budgets(
            args.harness_tool_budget,
            excluded_tools=("submit_candidates",),
        )
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))
    if not args.harness_context7 and {
        "resolve-library-id",
        "query-docs",
    } & set(budgets):
        parser.error("Context7 tools cannot have budgets when Context7 is disabled")
    if args.search_method == COMPILED_POLICY_METHOD:
        try:
            policy_budgets = parse_tool_call_budgets(
                args.policy_tool_budget,
                excluded_tools=("submit_optimization_policy",),
            )
        except ValueError as exc:
            parser.error(str(exc))
        if args.policy_max_submission_attempts < 1:
            parser.error("--policy-max-submission-attempts must be positive")
        if len(args.policy_capability) != len(set(args.policy_capability)):
            parser.error("--policy-capability values must be unique")
        if not args.harness_context7 and {
            "resolve-library-id",
            "query-docs",
        } & set(policy_budgets):
            parser.error("Context7 policy tools cannot have budgets when disabled")
    elif any(
        value is not None
        for value in (
            args.policy_capability,
            args.policy_tool_budget,
            args.policy_max_submission_attempts,
        )
    ):
        parser.error("Policy options require --search-method=ldm_harness_compiled")
    if args.mock:
        if args.case_id != MOCK_CASE.case_id:
            parser.error(f"mock execution requires --case-id {MOCK_CASE.case_id}")
        if args.iterations is None or args.iterations < 2:
            parser.error("mock execution requires at least two rounds")


def gp_config(args: argparse.Namespace) -> HammingGPUCBConfig:
    return HammingGPUCBConfig(
        beta=args.acquisition_beta,
        noise_variance=args.gp_noise_variance,
        jitter=args.gp_jitter,
        target_std_floor=args.gp_target_std_floor,
        min_history_for_fit=args.gp_min_history_for_fit,
        max_observations=args.gp_max_observations,
        global_best_observations=args.gp_global_best_observations,
    )


def campaign_budget(
    args: argparse.Namespace,
    profile_budget: Mapping[str, int | float] | None,
) -> dict[str, int | float]:
    if args.termination_kind == "wall_time":
        return {"initialization_evaluations": 1, **dict(profile_budget or {})}
    assert args.iterations is not None
    active_rounds = max(0, args.iterations - 1)
    evaluated = active_rounds * args.evaluations_per_round
    candidate_limit = active_rounds * args.proposal_samples
    limits: dict[str, int | float] = {
        "outer_iterations": args.iterations,
        "initialization_evaluations": 1,
        "valid_search_candidates": candidate_limit,
        "selected_candidates": evaluated,
        "external_evaluations": evaluated,
        "expensive_evaluation_attempts": evaluated,
        "successful_evaluations": evaluated,
    }
    if args.search_method in {"ldm", "llm"}:
        request_limit = active_rounds * args.proposal_request_limit
        limits.update(proposal_attempts=request_limit, llm_requests=request_limit)
    elif args.search_method in PARALLEL_HARNESS_METHODS:
        turns = active_rounds * len(args.harness_profile)
        limits.update(proposal_attempts=turns, harness_turns=turns)
        if args.harness_main_research:
            limits["harness_turns"] += active_rounds
            limits["main_research_turns"] = active_rounds
        if args.search_method == COMPILED_POLICY_METHOD:
            limits["policy_harness_turns"] = active_rounds
    elif args.search_method == "harness":
        limits.update(proposal_attempts=active_rounds, harness_turns=active_rounds)
    elif args.search_method == "bo":
        limits.update(proposal_attempts=0, llm_requests=0)
    return {**limits, **dict(profile_budget or {})}


def method_preset_sha256(
    args: argparse.Namespace,
    task_spec: LDMTaskSpec,
) -> str:
    payload = {
        "task_spec": task_spec.to_dict(),
        "proposal": {
            "mode": args.proposal_mode,
            "samples": args.proposal_samples,
            "candidates_per_request": args.proposal_candidates_per_request,
            "max_workers": args.proposal_max_workers,
            "request_limit": args.proposal_request_limit,
        },
        "provider": {
            "base_url": args.llm_url,
            "model": args.llm_model_name,
            "wire_api": args.llm_wire_api,
            "reasoning": args.llm_reasoning,
            "temperature": args.llm_temperature,
            "max_tokens": args.llm_max_tokens,
            "extra_body": provider_config.parse_extra_body(args.llm_extra_body_json),
        },
        "harness": harness_runtime.describe(args),
        "compiled_policy": (
            {
                "capabilities": args.policy_capability,
                "tool_call_budgets": parse_tool_call_budgets(
                    args.policy_tool_budget,
                    excluded_tools=("submit_optimization_policy",),
                ),
                "max_submission_attempts": args.policy_max_submission_attempts,
            }
            if args.search_method == COMPILED_POLICY_METHOD
            else None
        ),
    }
    return canonical_json_sha256(payload)


def jsonable_args(args: argparse.Namespace) -> dict[str, Any]:
    result = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
        if key != "api_key_file" and not key.startswith("_")
    }
    if args.search_method != COMPILED_POLICY_METHOD:
        for key in (
            "policy_capability",
            "policy_tool_budget",
            "policy_max_submission_attempts",
        ):
            result.pop(key, None)
    return result
