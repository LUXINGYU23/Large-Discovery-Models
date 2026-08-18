"""Small helpers kept out of the Iron Mind workflow orchestrator."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ldm_tts.engine import LDMEngineState
from ldm_tts.engine.run_store import CampaignRuntime


def derived_budget(args: Any) -> dict[str, int]:
    """Build the task budget only when no contract profile is active."""

    selected = args.iterations * args.evaluations_per_round
    return {
        "outer_iterations": args.iterations,
        "llm_requests": args.iterations if args.proposal_mode == "openai" else 0,
        "proposal_attempts": args.iterations,
        "valid_search_candidates": args.iterations * args.reservoir_size,
        "selected_candidates": selected,
        "external_evaluations": selected,
        "expensive_evaluation_attempts": selected,
        "successful_evaluations": selected,
        "benchmark_jobs": selected,
    }


def jsonable_args(args: Any) -> dict[str, Any]:
    """Return a serializable CLI snapshot without credential fields."""

    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
        if key != "api_key"
    }


def load_campaign_state(runtime: CampaignRuntime, resume: bool) -> LDMEngineState:
    """Load checkpoint state only for an explicit resume request."""

    if not resume:
        return LDMEngineState()
    checkpoint = runtime.load_checkpoint()
    return LDMEngineState() if checkpoint is None else LDMEngineState.from_checkpoint(checkpoint)


def record_seed_prior(runtime: CampaignRuntime, seed_prior: Any) -> None:
    """Persist non-budget seed provenance before a real endpoint request."""

    keys = seed_prior.blocked_canonical_keys
    if len(keys) != 1:
        raise ValueError("Iron Mind real campaigns require exactly one blocked seed key.")
    runtime.record(
        "qualification_seed_prior_loaded",
        {
            "candidate_id": seed_prior.observation.candidate_id,
            "canonical_key": keys[0],
            "excluded_from_campaign_budget": True,
        },
    )
