"""Small helpers kept out of the Iron Mind workflow orchestrator."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from ldm_tts.engine import LDMEngineState
from ldm_tts.engine.run_store import CampaignRuntime
from ldm_tts.registration.dependencies import is_local_url
from ldm_tts.transport import ProposalClient
from ldm_tts.transport.openai_http import EndpointRequestError

from tasks.iron_mind.core.provider import (
    OpenAIProviderSettings,
    resolve_openai_provider_settings,
)
from tasks.iron_mind.core.harness import HARNESS_PROFILE_IDS


def derived_budget(args: Any, *, domain_size: int) -> dict[str, int]:
    """Account for initialization and method-specific candidate generation exactly."""

    initial_rounds = int(args.initialization_mode == "shared_random" and args.iterations > 0)
    initial_evaluations = initial_rounds * args.evaluations_per_round
    search_rounds = args.iterations - initial_rounds
    selected = args.iterations * args.evaluations_per_round
    proposal_requests = _proposal_requests(args, search_rounds)
    valid = _valid_candidates(args, domain_size, initial_evaluations, search_rounds)
    budget = {
        "outer_iterations": args.iterations,
        "proposal_attempts": proposal_requests,
        "valid_search_candidates": valid,
        "selected_candidates": selected,
        "external_evaluations": selected,
        "expensive_evaluation_attempts": selected,
        "successful_evaluations": selected,
        "benchmark_jobs": selected,
    }
    if args.proposal_backend == "harness":
        budget["harness_turns"] = search_rounds * len(HARNESS_PROFILE_IDS)
    else:
        budget["llm_requests"] = proposal_requests if args.proposal_mode == "openai" else 0
    return budget


def campaign_budget(
    args: Any,
    profile_budget: Mapping[str, int | float] | None,
    *,
    domain_size: int,
) -> dict[str, int | float]:
    """Combine dynamic reservoir accounting with fixed profile limits."""

    return {**derived_budget(args, domain_size=domain_size), **dict(profile_budget or {})}


def _proposal_requests(args: Any, search_rounds: int) -> int:
    if args.search_method == "bo":
        return 0
    if args.proposal_backend == "harness":
        return search_rounds * len(HARNESS_PROFILE_IDS)
    per_round = args.proposal_samples if args.search_method == "ldm" else args.evaluations_per_round
    return search_rounds * per_round


def _valid_candidates(args: Any, domain_size: int, initial: int, search_rounds: int) -> int:
    if args.search_method == "ldm":
        return initial + search_rounds * args.proposal_samples
    if args.search_method == "llm":
        return initial + search_rounds * args.evaluations_per_round
    remaining = max(0, domain_size - initial)
    return initial + sum(
        max(0, remaining - index * args.evaluations_per_round)
        for index in range(search_rounds)
    )


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


def provider_settings(args: Any) -> OpenAIProviderSettings:
    """Resolve generic OpenAI-compatible endpoint settings for one run."""

    settings = resolve_openai_provider_settings(
        base_url=args.llm_url,
        model=args.llm_model_name,
        api_key=args.api_key,
    )
    key_file = getattr(args, "harness_api_key_file", None)
    if getattr(args, "proposal_backend", "direct") == "harness" and key_file is not None:
        api_key = Path(key_file).expanduser().read_text(encoding="utf-8").strip()
        if not api_key:
            raise ValueError("harness API key file is empty")
        settings = replace(settings, api_key=api_key)
    return settings


def preflight_endpoint(
    client: ProposalClient,
    runtime: CampaignRuntime,
    args: Any,
    payload: dict[str, Any],
    provider: OpenAIProviderSettings,
) -> bool:
    """Verify the endpoint before consuming proposal budget."""

    if not provider.base_url or not provider.model:
        return pause_endpoint(
            runtime,
            args,
            payload,
            "Set LLM_BASE_URL and LLM_MODEL_NAME for OpenAI proposal mode.",
        )
    if not provider.api_key and not is_local_url(provider.base_url):
        return pause_endpoint(
            runtime,
            args,
            payload,
            "Set LLM_API_KEY for a non-local OpenAI-compatible endpoint.",
        )
    try:
        preflight = client.preflight()  # type: ignore[attr-defined]
    except EndpointRequestError as exc:
        return pause_endpoint(runtime, args, payload, str(exc))
    runtime.record("endpoint_preflight_succeeded", preflight)
    payload["endpoint_preflight"] = preflight
    return True


def pause_endpoint(
    runtime: CampaignRuntime,
    args: Any,
    payload: dict[str, Any],
    message: str,
    *,
    phase: str = "endpoint_preflight",
) -> bool:
    """Persist a resumable endpoint failure and report its run directory."""

    runtime.pause(
        "paused_endpoint_unavailable",
        phase=phase,
        message=message,
        details={"model": args.llm_model_name},
    )
    payload.update(run_dir=str(runtime.run_dir.resolve()), status="paused_endpoint_unavailable")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return False
