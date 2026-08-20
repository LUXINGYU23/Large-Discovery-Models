"""Small helpers kept out of the Iron Mind workflow orchestrator."""

from __future__ import annotations

import json
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


def derived_budget(args: Any) -> dict[str, int]:
    """Build the task budget only when no contract profile is active."""

    selected = args.iterations * args.evaluations_per_round
    proposal_requests = args.iterations * args.proposal_samples
    return {
        "outer_iterations": args.iterations,
        "llm_requests": proposal_requests if args.proposal_mode == "openai" else 0,
        "proposal_attempts": proposal_requests,
        "valid_search_candidates": args.iterations * args.proposal_samples,
        "selected_candidates": selected,
        "external_evaluations": selected,
        "expensive_evaluation_attempts": selected,
        "successful_evaluations": selected,
        "benchmark_jobs": selected,
    }


def campaign_budget(
    args: Any,
    profile_budget: Mapping[str, int | float] | None,
) -> dict[str, int | float]:
    """Combine dynamic reservoir accounting with fixed profile limits."""

    return {**derived_budget(args), **dict(profile_budget or {})}


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

    return resolve_openai_provider_settings(
        base_url=args.llm_url,
        model=args.llm_model_name,
        api_key=args.api_key,
    )


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
