"""Runtime accounting and provider helpers for SynthonBench campaigns."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from ldm_tts.engine import LDMEngineState
from ldm_tts.engine.run_store import CampaignRuntime
from ldm_tts.registration.dependencies import is_local_url
from ldm_tts.transport.openai_http import EndpointRequestError
from tasks.synthonbench.core.provider import (
    OpenAIProviderSettings,
    resolve_openai_provider_settings,
)
from tasks.synthonbench.core.harness import HARNESS_PROFILE_IDS


def campaign_budget(args: Any, profile_budget: Mapping[str, int | float] | None) -> dict[str, int | float]:
    initial_rounds = int(args.initialization_mode == "shared_random" and args.iterations > 0)
    initial = initial_rounds * args.evaluations_per_round
    search_rounds = args.iterations - initial_rounds
    selected = args.iterations * args.evaluations_per_round
    proposal_attempts = _proposal_attempt_count(args, search_rounds)
    if args.proposal_backend == "harness":
        harness_turns = search_rounds * len(HARNESS_PROFILE_IDS)
        llm_requests = harness_turns * args.harness_provider_calls
        harness_web_calls = harness_turns * args.harness_web_calls
        harness_context7_calls = harness_turns * args.harness_context7_calls
        harness_artifact_bytes = harness_turns * args.harness_artifact_bytes
    else:
        harness_turns = harness_web_calls = harness_context7_calls = harness_artifact_bytes = 0
        llm_requests = proposal_attempts if args.proposal_mode == "openai" else 0
    dynamic = {
        "outer_iterations": args.iterations,
        "llm_requests": llm_requests,
        "proposal_attempts": proposal_attempts,
        "harness_turns": harness_turns,
        "harness_web_calls": harness_web_calls,
        "harness_context7_calls": harness_context7_calls,
        "harness_artifact_bytes": harness_artifact_bytes,
        "valid_search_candidates": _valid_candidate_count(args, initial, search_rounds),
        "selected_candidates": selected,
        "external_evaluations": selected,
        "expensive_evaluation_attempts": selected,
        "successful_evaluations": selected,
        "benchmark_jobs": selected,
    }
    return {**dynamic, **dict(profile_budget or {})}


def _proposal_attempt_count(args: Any, search_rounds: int) -> int:
    if args.search_method == "bo":
        return 0
    if args.proposal_backend == "harness":
        return search_rounds * len(HARNESS_PROFILE_IDS)
    per_round = args.proposal_samples if args.search_method == "ldm" else args.evaluations_per_round
    return search_rounds * per_round


def _valid_candidate_count(args: Any, initial: int, search_rounds: int) -> int:
    if args.search_method == "ldm":
        return initial + search_rounds * args.proposal_samples
    if args.search_method == "llm":
        return initial + search_rounds * args.evaluations_per_round
    return initial + search_rounds * args.proposal_samples


def jsonable_args(args: Any) -> dict[str, Any]:
    return {key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items() if key != "api_key"}


def load_campaign_state(runtime: CampaignRuntime, resume: bool) -> LDMEngineState:
    if not resume:
        return LDMEngineState()
    checkpoint = runtime.load_checkpoint()
    return LDMEngineState() if checkpoint is None else LDMEngineState.from_checkpoint(checkpoint)


def provider_settings(args: Any) -> OpenAIProviderSettings:
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


def preflight_endpoint(client, runtime: CampaignRuntime, args: Any, payload: dict[str, Any],
                       provider: OpenAIProviderSettings) -> bool:
    if not provider.base_url or not provider.model:
        return pause_endpoint(runtime, args, payload, "Set LLM_BASE_URL and LLM_MODEL_NAME for OpenAI proposal mode.")
    if not provider.api_key and not is_local_url(provider.base_url):
        return pause_endpoint(runtime, args, payload, "Set LLM_API_KEY for a non-local OpenAI-compatible endpoint.")
    try:
        preflight = client.preflight()
    except EndpointRequestError as exc:
        return pause_endpoint(runtime, args, payload, str(exc))
    runtime.record("endpoint_preflight_succeeded", preflight)
    payload["endpoint_preflight"] = preflight
    return True


def pause_endpoint(runtime: CampaignRuntime, args: Any, payload: dict[str, Any], message: str,
                   *, phase: str = "endpoint_preflight") -> bool:
    runtime.pause("paused_endpoint_unavailable", phase=phase, message=message,
                  details={"model": args.llm_model_name})
    payload.update(run_dir=str(runtime.run_dir.resolve()), status="paused_endpoint_unavailable")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return False


__all__ = ["campaign_budget", "jsonable_args", "load_campaign_state", "pause_endpoint", "preflight_endpoint", "provider_settings"]
