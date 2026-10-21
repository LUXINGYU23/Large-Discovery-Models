"""Pi configuration, provider preflight and client lifetimes for ReaSyn."""

from __future__ import annotations

import json
from contextlib import ExitStack
from copy import copy
from pathlib import Path

from ldm_tts.engine.run_store import atomic_json_write
from ldm_tts.harness import (
    DockerPolicyExecutor, HarnessClient, HarnessLimits, HarnessNetworkPolicy,
    HarnessToolExtension, PolicyResearchController, file_sha256,
    load_harness_mcp_config, parse_tool_call_budgets, policy_submission_contract,
)
from ldm_tts.harness.container import pi_sidecar_command, resolve_container_user
from ldm_tts.harness.pi import PiHarnessConfig, load_pi_guest_runtime, policy_mcp_server
from ldm_tts.harness.research_runtime import ResearchCoordinator, main_research_profile, research_contract
from ldm_tts.transport.openai import OpenAICompatibleProposalClient, EndpointRequestError, generation_body
from .harness import HarnessTargetSource, RESOURCE_ROOT, profiles, submission_contract
from .optimization_policy import ReaSynPolicyAdapter

TOOL_NAMES = ("describe_reasyn_task", "get_measured_history", "check_measured_product")


def open_proposals(args, runtime, target, *, stack: ExitStack, api_key: str):
    provider_args = copy(args)
    provider_args.provider_api_key = api_key
    runtime.consume("endpoint_preflight_requests")
    preflight = preflight_provider(provider_args)
    runtime.record("endpoint_preflight_succeeded", preflight)
    root = runtime.run_dir / "harness"
    client = create_client(provider_args, root, runtime.run_id, target)
    stack.callback(client.close)
    coordinator = branch_client = None
    if args._research_strategy.requires_main:
        def account_main(counts, *, usage_key=None):
            counts = {("harness_provider_calls" if key == "llm_requests" else key): value
                      for key, value in counts.items()}
            runtime.consume_many(counts, usage_key=usage_key)

        coordinator = ResearchCoordinator(
            root, args._research_strategy, client.config.profiles,
            lambda: create_client(
                provider_args, root / "main", runtime.run_id, target,
                session_profiles=main_research_profile(root / "main", client.config.profiles[0]),
                contract=research_contract(),
            ), account=account_main,
        )
        branch_client = lambda location, sources: create_client(
            provider_args, location, runtime.run_id, target, fork_sources=sources,
        )
    else:
        client.start()
    return HarnessTargetSource(
        client, root, args, target=target, account=runtime.consume_many,
        coordinator=coordinator, branch_client=branch_client,
    )


def open_policy(args, runtime, target, encoder, *, stack: ExitStack, api_key: str):
    provider_args = copy(args)
    provider_args.provider_api_key = api_key
    root = runtime.run_dir / "policy_harness"
    client = create_client(provider_args, root, runtime.run_id, target, policy=True)
    stack.callback(client.close)
    client.start()
    adapter = ReaSynPolicyAdapter(
        encoder, benchmark=args.benchmark, alpha=args.acquisition_alpha,
        eta=args.acquisition_eta, seed=args.seed, history_limit=args.gp_history_limit,
        beta=args.acquisition_beta, z_clip=args.acquisition_z_clip,
    )
    controller = PolicyResearchController(
        client=client, adapter=adapter,
        executor=DockerPolicyExecutor(
            args.policy_runner_image, args.harness_docker_host,
            resolve_container_user(args.harness_container_user, args.harness_docker_host),
        ), root=root, account=runtime.consume_many,
        recovery_budget=lambda: args.harness_wall_time_seconds * 2,
    )
    return adapter, controller


def preflight_provider(args):
    if not args.llm_url or not args.llm_model:
        raise ValueError("Set LLM_BASE_URL and LLM_MODEL_NAME for real Harness proposals")
    if not getattr(args, "provider_api_key", ""):
        raise EndpointRequestError(
            "Set LLM_API_KEY or OPENAI_API_KEY for real Harness proposals"
        )
    client = OpenAICompatibleProposalClient(
        url=args.llm_url,
        model=args.llm_model,
        api_key=args.provider_api_key,
        timeout_seconds=args.harness_response_timeout,
        max_tokens=args.llm_max_tokens,
        max_retries=0,
        wire_api="responses",
        temperature=args.llm_temperature,
        extra_body=generation_body(wire_api=args.llm_wire_api, reasoning=args.llm_reasoning,
                                   extra_body=json.loads(args.llm_extra_body_json)),
    )
    return {"backend": "harness", "wire_api": "responses", **client.preflight()}


def create_client(args, root, campaign_id, target, *, policy=False, session_profiles=None, contract=None, fork_sources=None):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    context = {"benchmark": args.benchmark, "oracle": args.oracle if not target else None,
               "original_target": target, "mock": args.mock,
               "feedback": "Only completed task measurements are available. Projection is not an oracle score."}
    atomic_json_write(root / "context.json", context)
    history_root = root.parent / "harness" if policy else root
    history_root.mkdir(parents=True, exist_ok=True)
    if not (history_root / "measured_history.json").exists():
        atomic_json_write(history_root / "measured_history.json", {"observations": []})
    mounts = [(RESOURCE_ROOT, "/resources", True)]
    if policy:
        mounts.append((history_root, "/measured_history", True))
    command = pi_sidecar_command(
        root, args.harness_sidecar_image, cache_root=args.harness_cache_dir,
        docker_host=args.harness_docker_host, container_user=args.harness_container_user,
        environment={
            "LDM_REASYN_CONTEXT": "/artifacts/context.json",
            "LDM_REASYN_HISTORY": "/measured_history/measured_history.json" if policy else "/artifacts/measured_history.json",
        },
        mounts=mounts,
    )
    mcp = load_harness_mcp_config(args.harness_mcp_config)
    contract = contract or (policy_submission_contract() if policy else submission_contract())
    return HarnessClient(command, api_key=args.provider_api_key,
        named_secrets=mcp.named_secrets, response_timeout_seconds=args.harness_response_timeout,
        config=PiHarnessConfig(
            artifact_root=Path("/artifacts"), base_url=args.llm_url, model=args.llm_model,
            profiles=session_profiles or profiles(args.harness_sessions, policy=policy), campaign_id=campaign_id,
            fork_sources=fork_sources or {},
            task_id="reasyn", case_id=f"{args.benchmark}:{target or args.oracle}" + (":policy" if policy else ""),
            seed=args.seed, submission_contract=contract,
            guest_runtime=load_pi_guest_runtime("reasyn", RESOURCE_ROOT / "image"),
            tool_extensions=(HarnessToolExtension(Path("/resources/tools/molecular_research.mjs"),
                file_sha256(RESOURCE_ROOT / "tools/molecular_research.mjs"), TOOL_NAMES),),
            mcp_servers=(*mcp.servers, policy_mcp_server(
                diagnostics_path="/resources/policy_diagnostics.py",
                diagnostics_sha256=file_sha256(RESOURCE_ROOT / "policy_diagnostics.py"),
            )) if policy else mcp.servers,
            thinking=args.harness_thinking, context7_enabled=False,
            provider_request_body={
                "temperature": args.llm_temperature, "max_output_tokens": args.llm_max_tokens,
                **generation_body(wire_api=args.llm_wire_api, reasoning=args.llm_reasoning,
                                  extra_body=json.loads(args.llm_extra_body_json)),
            },
            limits=HarnessLimits(wall_time_seconds=args.harness_wall_time_seconds,
                tool_call_budgets=parse_tool_call_budgets(args.harness_tool_budget, excluded_tools=(contract.tool_name,))),
            network_policy=HarnessNetworkPolicy(forbidden_query_patterns=("benchmark_result", "oracle_cache", "checkpoint")),
        ))
