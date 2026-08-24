"""CLI parsing and validation for the Iron Mind workflow."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from tasks.iron_mind.core.prompting import (
    DEFAULT_PROMPT_POLICY,
    PROMPT_POLICIES,
    validate_prompt_policy,
)
from tasks.iron_mind.core.proposals import DEFAULT_PROPOSAL_MAX_WORKERS
from tasks.iron_mind.core.provider import parse_openai_extra_body_json
from tasks.iron_mind.core.search import INITIALIZATION_MODES, SEARCH_METHODS


DEFAULT_PROPOSAL_SAMPLES = 64
DEFAULT_BO_POOL_SIZE = 32
DEFAULT_LLM_MAX_TOKENS = 512


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse one public Iron Mind campaign invocation."""

    parser = argparse.ArgumentParser(description="Run the Iron Mind LDM task.")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--proposal-mode", choices=("callable", "none", "openai"), default="callable")
    parser.add_argument("--search-method", choices=SEARCH_METHODS, default="ldm")
    parser.add_argument("--initialization-mode", choices=INITIALIZATION_MODES, default="none")
    parser.add_argument("--dataset-id", default="buchwald_hartwig")
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--proposal-samples", type=int, default=DEFAULT_PROPOSAL_SAMPLES)
    parser.add_argument("--bo-pool-size", type=int, default=DEFAULT_BO_POOL_SIZE)
    parser.add_argument("--proposal-max-workers", type=int, default=DEFAULT_PROPOSAL_MAX_WORKERS)
    parser.add_argument("--evaluations-per-round", type=int, default=1)
    parser.add_argument("--acquisition-beta", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--eta", type=float, default=3.0)
    parser.add_argument("--z-clip", type=float, default=5.0)
    parser.add_argument("--out-dir", type=Path, default=Path("runs"))
    parser.add_argument("--run-name", default="")
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--llm-url")
    parser.add_argument("--llm-model-name")
    parser.add_argument("--api-key")
    parser.add_argument("--llm-timeout", type=float, default=120.0)
    parser.add_argument("--llm-max-tokens", type=int, default=DEFAULT_LLM_MAX_TOKENS)
    parser.add_argument("--llm-temperature", type=float, default=0.7)
    parser.add_argument("--llm-json-mode", action="store_true")
    parser.add_argument(
        "--llm-extra-body-json",
        default="",
        help="Provider-specific JSON object merged into the OpenAI-compatible request body.",
    )
    parser.add_argument(
        "--prompt-policy",
        choices=tuple(sorted(PROMPT_POLICIES)),
        default=DEFAULT_PROMPT_POLICY,
    )
    parser.add_argument("--campaign-index", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    """Reject campaign options that violate the public task contract."""

    if args.iterations < 0:
        raise SystemExit("--iterations must be non-negative")
    if args.proposal_samples < 1:
        raise SystemExit("--proposal-samples must be positive")
    if args.bo_pool_size < 1:
        raise SystemExit("--bo-pool-size must be positive")
    if args.search_method == "ldm" and args.proposal_samples <= args.bo_pool_size:
        raise SystemExit("--proposal-samples must exceed --bo-pool-size")
    if args.proposal_max_workers < 1:
        raise SystemExit("--proposal-max-workers must be positive")
    if args.evaluations_per_round != 1:
        raise SystemExit("Iron Mind requires --evaluations-per-round=1")
    _validate_non_negative(args.acquisition_beta, "--acquisition-beta")
    _validate_non_negative(args.alpha, "--alpha")
    _validate_non_negative(args.eta, "--eta")
    if not math.isfinite(args.z_clip) or args.z_clip <= 0:
        raise SystemExit("--z-clip must be finite and positive")
    if not math.isfinite(args.llm_temperature) or not 0.0 <= args.llm_temperature <= 2.0:
        raise SystemExit("--llm-temperature must be finite and between 0 and 2")
    try:
        validate_prompt_policy(args.prompt_policy)
        parse_openai_extra_body_json(args.llm_extra_body_json)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if args.campaign_index < 0:
        raise SystemExit("--campaign-index must be non-negative")
    if args.mock and args.dataset_id != "buchwald_hartwig":
        raise SystemExit("Mock campaigns require --dataset-id=buchwald_hartwig")
    _validate_search_mode(args)
    if not args.mock and args.data_dir is None:
        raise SystemExit("Non-mock Iron Mind campaigns require --data-dir")


def _validate_non_negative(value: float, option: str) -> None:
    if not math.isfinite(value) or value < 0:
        raise SystemExit(f"{option} must be finite and non-negative")


def _validate_search_mode(args: argparse.Namespace) -> None:
    model_method = args.search_method in {"ldm", "llm"}
    if args.mock and model_method and args.proposal_mode not in {"callable", "openai"}:
        raise SystemExit("Mock model methods require --proposal-mode=callable or openai")
    if args.mock and not model_method and args.proposal_mode != "none":
        raise SystemExit("Mock BO requires --proposal-mode=none")
    if not args.mock and model_method and args.proposal_mode != "openai":
        raise SystemExit("Real model methods require --proposal-mode=openai")
    if not args.mock and not model_method and args.proposal_mode != "none":
        raise SystemExit("Real BO requires --proposal-mode=none")
