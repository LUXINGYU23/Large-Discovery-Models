"""Pi client assembly for one sample's public research workspace."""

from __future__ import annotations

import json
import os
from pathlib import Path

from ldm_tts.harness import (HarnessClient, HarnessLimits, HarnessNetworkPolicy,
    HarnessToolExtension, file_sha256, parse_tool_call_budgets)
from ldm_tts.harness.container import pi_sidecar_command
from ldm_tts.harness.mcp import load_harness_mcp_config
from ldm_tts.harness.pi import PiHarnessConfig, load_pi_guest_runtime, policy_mcp_server
from ldm_tts.transport.openai import EndpointRequestError, OpenAICompatibleProposalClient, generation_body
from .data import write_json
from .harness import RESOURCE_ROOT, TOOL_NAMES, harness_profiles, submission_contract


def provider_api_key():
    return os.environ.get(
        "LLM_API_KEY",
        os.environ.get("LDM_LLM_API_KEY", os.environ.get("OPENAI_API_KEY", "")),
    )


def preflight_provider(args):
    if not args.llm_url or not args.llm_model_name:
        raise EndpointRequestError(
            "Set LLM_BASE_URL and LLM_MODEL_NAME or the corresponding CLI options"
        )
    if not provider_api_key():
        raise EndpointRequestError(
            "Set LLM_API_KEY, LDM_LLM_API_KEY, or OPENAI_API_KEY for Harness proposals"
        )
    client = OpenAICompatibleProposalClient(
        url=args.llm_url,
        model=args.llm_model_name,
        api_key=provider_api_key(),
        timeout_seconds=args.harness_response_timeout,
        max_tokens=args.llm_max_tokens,
        temperature=args.llm_temperature,
        max_retries=0,
        wire_api="responses",
        extra_body=generation_body(wire_api=args.llm_wire_api, reasoning=args.llm_reasoning,
                                   extra_body=json.loads(args.llm_extra_body_json)),
    )
    return {"backend": "harness", "wire_api": "responses", **client.preflight()}


def create_client(args, root, sample, *, campaign_id, policy=False, profiles=None, contract=None, fork_sources=None):
    """Mount only task research resources and this sample's public artifact tree."""
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "public_task.json", sample)
    if not (root / "public_history.json").exists():
        write_json(root / "public_history.json", {"drafts": []})
    profiles = profiles or harness_profiles(
        ("policy_architect",) if policy else args.harness_profile,
        compiled=args.search_method == "ldm_harness_compiled",
    )
    if policy:
        from ldm_tts.harness import policy_submission_contract

        contract = policy_submission_contract(args.harness_max_submission_attempts)
    else:
        contract = contract or submission_contract(args.harness_max_submission_attempts)
    mcp = load_harness_mcp_config(args.harness_mcp_config)
    servers = (*mcp.servers, policy_mcp_server(**({
        "diagnostics_path": "/resources/policy_diagnostics.py",
        "diagnostics_sha256": file_sha256(RESOURCE_ROOT / "policy_diagnostics.py"),
    } if args.search_method == "ldm_harness_compiled" else {}))) if policy else mcp.servers
    return HarnessClient(
        pi_sidecar_command(
            root, args.harness_sidecar_image, cache_root=args.harness_cache_dir,
            docker_host=args.harness_docker_host, container_user=args.harness_container_user,
            environment={"LDM_ATOMWORLD_PUBLIC_ROOT": "/artifacts"},
            mounts=((RESOURCE_ROOT, "/resources", True),),
        ),
        api_key=provider_api_key(),
        named_secrets=mcp.named_secrets,
        response_timeout_seconds=args.harness_response_timeout,
        config=PiHarnessConfig(
            artifact_root=Path("/artifacts"),
            profiles=profiles,
            fork_sources=fork_sources or {},
            campaign_id=campaign_id,
            task_id="atomworld",
            case_id=sample["sample_id"],
            seed=args.campaign_index,
            submission_contract=contract,
            base_url=args.llm_url,
            model=args.llm_model_name,
            guest_runtime=load_pi_guest_runtime("atomworld", RESOURCE_ROOT / "image"),
            tool_extensions=(
                HarnessToolExtension(
                    Path("/resources/tools/public_context.mjs"),
                    file_sha256(RESOURCE_ROOT / "tools/public_context.mjs"),
                    TOOL_NAMES,
                ),
            ),
            mcp_servers=servers,
            thinking=args.harness_thinking,
            provider_request_body={
                "temperature": args.llm_temperature, "max_output_tokens": args.llm_max_tokens,
                **generation_body(wire_api=args.llm_wire_api, reasoning=args.llm_reasoning,
                                  extra_body=json.loads(args.llm_extra_body_json)),
            },
            limits=HarnessLimits(
                wall_time_seconds=args.harness_wall_time_seconds,
                tool_call_budgets=parse_tool_call_budgets(
                    args.harness_tool_budget, excluded_tools=(contract.tool_name,)
                ),
            ),
            network_policy=HarnessNetworkPolicy(
                forbidden_query_patterns=(
                    r"atomworld",
                    r"target_cif",
                    r"private\.jsonl",
                    r"benchmark.*answers?",
                )
            ),
        ),
    )
