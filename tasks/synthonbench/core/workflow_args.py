"""CLI parsing and invariant validation for SynthonBench campaigns."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from tasks.synthonbench.core.catalog import REACTION_ALLOCATIONS
from tasks.synthonbench.core.constants import (
    DEFAULT_ACQUISITION_ETA,
    DEFAULT_BO_POOL_SIZE,
    DEFAULT_BO_SEARCH_SAMPLES,
    DEFAULT_FINGERPRINT_BITS,
    DEFAULT_GP_KERNEL_JITTER,
    DEFAULT_GP_LANDMARKS,
    DEFAULT_GP_MEAN_STD,
    DEFAULT_GP_OBSERVATION_NOISE_STD,
    DEFAULT_GP_REACTION_WEIGHT,
    DEFAULT_GP_SIGNAL_STD,
    DEFAULT_LLM_EXTRA_BODY_JSON,
    DEFAULT_LLM_MAX_TOKENS,
    DEFAULT_PROPOSAL_CANDIDATES_PER_REQUEST,
    DEFAULT_PROPOSAL_MAX_WORKERS,
    DEFAULT_PROPOSAL_SAMPLES,
    DEFAULT_SLATE_SIZE,
    ORACLE_KINDS,
    SCALES,
    TARGETS,
)
from tasks.synthonbench.core.prompting import DEFAULT_PROMPT_POLICY, PROMPT_POLICIES
from tasks.synthonbench.core.provider import parse_openai_extra_body_json
from tasks.synthonbench.core.harness import HARNESS_PROFILE_IDS
from tasks.synthonbench.core.search import INITIALIZATION_MODES, SEARCH_METHODS


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse one shared-runner SynthonBench task invocation."""

    parser = argparse.ArgumentParser(description="Run the SynthonBench LDM task.")
    _add_benchmark_arguments(parser)
    _add_ldm_arguments(parser)
    _add_provider_arguments(parser)
    _add_runtime_arguments(parser)
    args = parser.parse_args(argv)
    if args.proposal_candidates_per_request is None:
        args.proposal_candidates_per_request = (
            min(DEFAULT_PROPOSAL_CANDIDATES_PER_REQUEST, args.proposal_samples)
            if args.search_method == "ldm" and args.proposal_backend == "direct"
            else 1
        )
    return args


def validate_args(args: argparse.Namespace) -> None:
    """Reject arguments that would violate the official or LDM task contract."""

    _validate_counts(args)
    _validate_numbers(args)
    _validate_provider_options(args)
    _validate_mode(args)


def _add_benchmark_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--scale", choices=SCALES, default="1M")
    parser.add_argument("--target", choices=TARGETS, default="kif11")
    parser.add_argument("--oracle-kind", choices=ORACLE_KINDS, default="surrogate")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--campaign-index", type=int, default=0)


def _add_ldm_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--proposal-samples", type=int, default=DEFAULT_PROPOSAL_SAMPLES)
    parser.add_argument("--search-method", choices=SEARCH_METHODS, default="ldm")
    parser.add_argument("--initialization-mode", choices=INITIALIZATION_MODES, default="none")
    parser.add_argument("--bo-pool-size", type=int, default=DEFAULT_BO_POOL_SIZE)
    parser.add_argument("--bo-search-samples", type=int, default=DEFAULT_BO_SEARCH_SAMPLES)
    parser.add_argument("--proposal-candidates-per-request", type=int)
    parser.add_argument("--proposal-max-workers", type=int, default=DEFAULT_PROPOSAL_MAX_WORKERS)
    parser.add_argument("--evaluations-per-round", type=int, default=1)
    parser.add_argument("--slate-size", type=int, default=DEFAULT_SLATE_SIZE)
    parser.add_argument("--reaction-allocation", choices=REACTION_ALLOCATIONS, default="product_weighted")
    parser.add_argument("--fingerprint-bits", type=int, default=DEFAULT_FINGERPRINT_BITS)
    parser.add_argument("--gp-landmarks", type=int, default=DEFAULT_GP_LANDMARKS)
    parser.add_argument("--gp-kernel-jitter", type=float, default=DEFAULT_GP_KERNEL_JITTER)
    parser.add_argument("--gp-signal-std", type=float, default=DEFAULT_GP_SIGNAL_STD)
    parser.add_argument("--gp-mean-std", type=float, default=DEFAULT_GP_MEAN_STD)
    parser.add_argument("--gp-observation-noise-std", type=float, default=DEFAULT_GP_OBSERVATION_NOISE_STD)
    parser.add_argument("--gp-reaction-weight", type=float, default=DEFAULT_GP_REACTION_WEIGHT)
    parser.add_argument("--acquisition-beta", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--eta", type=float, default=DEFAULT_ACQUISITION_ETA)
    parser.add_argument("--z-clip", type=float, default=5.0)
    parser.add_argument("--prompt-policy", choices=PROMPT_POLICIES, default=DEFAULT_PROMPT_POLICY)
    parser.add_argument("--proposal-backend", choices=("direct", "harness"), default="direct")


def _add_provider_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--proposal-mode", choices=("callable", "none", "openai"), default="callable")
    parser.add_argument("--llm-url")
    parser.add_argument("--llm-model-name")
    parser.add_argument("--api-key")
    parser.add_argument("--llm-timeout", type=float, default=120.0)
    parser.add_argument("--llm-max-tokens", type=int, default=DEFAULT_LLM_MAX_TOKENS)
    parser.add_argument("--llm-temperature", type=float, default=0.7)
    parser.add_argument("--llm-json-mode", action="store_true")
    parser.add_argument(
        "--llm-extra-body-json",
        default=DEFAULT_LLM_EXTRA_BODY_JSON,
        help="Provider-specific JSON object merged into the OpenAI-compatible request body.",
    )
    parser.add_argument("--harness-image", default="ldm-pi-harness:latest")
    parser.add_argument("--harness-candidates-per-session", type=int, default=16)
    parser.add_argument(
        "--harness-thinking",
        choices=("off", "minimal", "low", "medium", "high", "xhigh", "max"),
        default="max",
    )
    parser.add_argument("--harness-api-key-file", type=Path)
    parser.add_argument("--harness-cache-dir", type=Path)
    parser.add_argument("--harness-docker-host")
    parser.add_argument("--harness-container-user")
    parser.add_argument("--harness-response-timeout", type=float, default=2100.0)
    parser.add_argument("--harness-wall-time-seconds", type=int, default=1800)
    parser.add_argument(
        "--no-harness-context7",
        action="store_false",
        dest="harness_context7",
        default=True,
    )


def _add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--out-dir", type=Path, default=Path("runs"))
    parser.add_argument("--run-name", default="")
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--audit-timeout", type=float, default=900.0)
    parser.add_argument("--dry-run", action="store_true")


def _validate_counts(args: argparse.Namespace) -> None:
    if args.iterations < 0 or args.campaign_index < 0:
        raise SystemExit("--iterations and --campaign-index must be non-negative")
    positive = ("proposal_samples", "bo_pool_size", "bo_search_samples", "proposal_candidates_per_request",
                "proposal_max_workers", "evaluations_per_round",
                "slate_size", "fingerprint_bits", "gp_landmarks", "llm_max_tokens",
                "harness_candidates_per_session", "harness_wall_time_seconds")
    if any(getattr(args, name) < 1 for name in positive):
        raise SystemExit("proposal, pool, worker, feature, and token counts must be positive")
    if args.search_method == "ldm" and args.proposal_samples <= args.bo_pool_size:
        raise SystemExit("--proposal-samples must exceed --bo-pool-size")
    if args.search_method != "llm" and args.evaluations_per_round > args.bo_pool_size:
        raise SystemExit("--evaluations-per-round cannot exceed --bo-pool-size")
    if args.proposal_backend == "direct" and args.search_method in {"ldm", "llm"}:
        breadth = (
            args.proposal_samples
            if args.search_method == "ldm"
            else args.evaluations_per_round
        )
        if breadth % args.proposal_candidates_per_request:
            raise SystemExit(
                "the direct proposal breadth must divide evenly by "
                "--proposal-candidates-per-request"
            )
    if args.proposal_backend == "harness":
        expected = len(HARNESS_PROFILE_IDS) * args.harness_candidates_per_session
        if args.proposal_samples != expected:
            raise SystemExit(
                "--proposal-samples must equal four times --harness-candidates-per-session"
            )


def _validate_numbers(args: argparse.Namespace) -> None:
    positive = (
        "gp_kernel_jitter",
        "gp_signal_std",
        "gp_observation_noise_std",
        "z_clip",
        "llm_timeout",
        "audit_timeout",
        "harness_response_timeout",
    )
    if any(not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0 for name in positive):
        raise SystemExit("GP scales, timeouts, and --z-clip must be finite and positive")
    nonnegative = ("acquisition_beta", "alpha", "eta", "gp_mean_std", "gp_reaction_weight")
    if any(not math.isfinite(getattr(args, name)) or getattr(args, name) < 0 for name in nonnegative):
        raise SystemExit("acquisition-beta, alpha, and eta must be finite and non-negative")
    if not math.isfinite(args.llm_temperature) or not 0.0 <= args.llm_temperature <= 2.0:
        raise SystemExit("--llm-temperature must be finite and between 0 and 2")


def _validate_provider_options(args: argparse.Namespace) -> None:
    try:
        parse_openai_extra_body_json(args.llm_extra_body_json)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def _validate_mode(args: argparse.Namespace) -> None:
    if args.proposal_backend == "harness":
        if args.search_method != "ldm":
            raise SystemExit("--proposal-backend=harness is available only with --search-method=ldm")
        if args.proposal_mode != "none":
            raise SystemExit("--proposal-backend=harness requires --proposal-mode=none")
        if not args.mock and (args.data_dir is None or args.source_dir is None):
            raise SystemExit("real SynthonBench campaigns require --data-dir and --source-dir")
        if args.oracle_kind == "glide" and args.scale != "1M":
            raise SystemExit("the official real-Glide oracle exists only for --scale=1M")
        return
    model_method = args.search_method in {"ldm", "llm"}
    if args.mock and model_method and args.proposal_mode not in {"callable", "openai"}:
        raise SystemExit("mock model methods require --proposal-mode=callable or openai")
    if args.mock and not model_method and args.proposal_mode != "none":
        raise SystemExit("mock BO requires --proposal-mode=none")
    if not args.mock and model_method and args.proposal_mode != "openai":
        raise SystemExit("real model methods require --proposal-mode=openai")
    if not args.mock and not model_method and args.proposal_mode != "none":
        raise SystemExit("real BO requires --proposal-mode=none")
    if not args.mock and (args.data_dir is None or args.source_dir is None):
        raise SystemExit("real SynthonBench campaigns require --data-dir and --source-dir")
    if args.oracle_kind == "glide" and args.scale != "1M":
        raise SystemExit("the official real-Glide oracle exists only for --scale=1M")


__all__ = ["parse_args", "validate_args"]
