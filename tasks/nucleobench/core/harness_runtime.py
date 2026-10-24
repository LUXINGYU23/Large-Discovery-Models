"""NucleoBench research resources, role contracts and Pi client assembly."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from contextlib import ExitStack
from pathlib import Path
from typing import Any

from ldm_tts.engine.run_store import CampaignRuntime
from ldm_tts.engine.expansion import ReservoirExpander
from ldm_tts.harness import (
    DockerPolicyExecutor, HarnessClient, HarnessLimits, HarnessNetworkPolicy, HarnessProfile,
    PolicyResearchController,
    HarnessSubmissionContract, parse_tool_call_budgets, policy_submission_contract,
)
from ldm_tts.harness.container import pi_sidecar_command, resolve_container_user
from ldm_tts.harness.research_runtime import research_contract, main_research_profile
from ldm_tts.harness.pi import PiForkSource, PiHarnessConfig, policy_mcp_server
from ldm_tts.harness.protocol import file_sha256
from ldm_tts.harness.ucb_diagnostics import DIAGNOSTICS_PATH
from tasks.nucleobench.core.constants import (
    COMPILED_POLICY_METHOD, PARALLEL_HARNESS_METHODS, PERSISTENT_HARNESS_METHODS, TASK_ID,
)
from tasks.nucleobench.core.benchmark_clock import BenchmarkClock
from tasks.nucleobench.core.candidate import NucleoBenchCandidateDomain
from tasks.nucleobench.core.factory import build_proposal_expander
from tasks.nucleobench.core.forked_harness import ForkedResearchExpander
from tasks.nucleobench.core.hamming_gp import HammingGPUCBConfig
from tasks.nucleobench.core.harness import (
    HARNESS_FORBIDDEN_PATTERNS, HARNESS_SKILL_IDS,
    direct_harness_profile, harness_guest_runtime,
    harness_profiles, harness_submission_contract, harness_tool_extensions,
    write_harness_sequence_context,
)
from tasks.nucleobench.core.oracles.official import PreparedCase
from tasks.nucleobench.core.optimization_policy import (
    NucleoOptimizationPolicyAdapter, policy_harness_profile,
)
from tasks.nucleobench.core.policy_features import NucleoPolicyFeatureEncoder
from tasks.nucleobench.core.provider import ProviderSettings, parse_extra_body
from tasks.nucleobench.core.research import MEASURED_HISTORY_FILE

TASK_ROOT = Path(__file__).resolve().parents[1]


def open_proposals(
    args: argparse.Namespace,
    runtime: CampaignRuntime,
    provider: ProviderSettings,
    prepared: PreparedCase,
    mcp,
    *,
    stack: ExitStack,
    benchmark_clock: BenchmarkClock | None,
    gp_config: HammingGPUCBConfig,
) -> ReservoirExpander:
    profiles = proposal_profiles(args)
    if args._research_strategy.requires_main:
        (runtime.run_dir / "harness" / MEASURED_HISTORY_FILE).parent.mkdir(parents=True, exist_ok=True)
        return ForkedResearchExpander(
            root=runtime.run_dir / "harness",
            main_client=lambda: create_client(
                args, runtime, provider, prepared, main_research_profile(runtime.run_dir / "harness/main", profiles[0]), mcp,
                artifact_root=runtime.run_dir / "harness/main", contract=research_contract(),
            ),
            branch_client=lambda root, source: create_client(
                args, runtime, provider, prepared, profiles, mcp,
                artifact_root=root, fork_sources=source,
            ),
            strategy=args._research_strategy,
            expander_options={
                "domain": NucleoBenchCandidateDomain(prepared.context), "profiles": profiles,
                "candidates_per_profile": args.harness_candidates_per_session,
                "campaign_id": runtime.run_id, "first_active_round": 1,
                "attach_empirical_q0": True, "allow_repeated_occurrences": not args.harness_unique_candidates,
                "account": runtime.consume_many, "benchmark_clock": benchmark_clock,
                "surrogate_query_config": gp_config if args.harness_surrogate_query else None,
            },
        )
    client = stack.enter_context(create_client(args, runtime, provider, prepared, profiles, mcp))
    return build_proposal_expander(
        args.search_method, prepared.context,
        seed=args.campaign_index, evaluations_per_round=args.evaluations_per_round,
        harness_client=client, harness_artifact_root=runtime.run_dir / "harness",
        harness_session_profiles=profiles,
        harness_candidates_per_session=args.harness_candidates_per_session,
        harness_unique_candidates=args.harness_unique_candidates,
        research_roles=args.harness_research_roles,
        campaign_id=runtime.run_id, first_active_round=1,
        account=runtime.consume_many, benchmark_clock=benchmark_clock,
        surrogate_query_config=gp_config if args.harness_surrogate_query else None,
    )


def open_policy(
    args: argparse.Namespace,
    runtime: CampaignRuntime,
    provider: ProviderSettings,
    prepared: PreparedCase,
    mcp,
    *,
    stack: ExitStack,
    benchmark_clock: BenchmarkClock | None,
    gp_config: HammingGPUCBConfig,
) -> tuple[NucleoOptimizationPolicyAdapter, PolicyResearchController]:
    adapter = NucleoOptimizationPolicyAdapter(
        NucleoPolicyFeatureEncoder(prepared.context),
        seed=args.campaign_index, gp_config=gp_config,
        default_alpha=args.alpha, default_eta=args.eta,
        enabled_capabilities=tuple(args.policy_capability),
        proposal_sampling={
            "session_count": len(proposal_profiles(args)),
            "candidates_per_session": args.harness_candidates_per_session,
            "within_session_repeats_allowed": not args.harness_unique_candidates,
            "cross_session_agreement_allowed": True,
        },
        evaluations_per_round=args.evaluations_per_round,
        benchmark_clock=benchmark_clock,
        measured_history_path=runtime.run_dir / "harness" / MEASURED_HISTORY_FILE,
    )
    client = stack.enter_context(create_client(
        args, runtime, provider, prepared, policy_harness_profile(), mcp, policy=True,
    ))
    controller = PolicyResearchController(
        client=client, adapter=adapter,
        executor=DockerPolicyExecutor(
            image=args.harness_sidecar_image,
            docker_host=args.harness_docker_host or "",
            container_user=resolve_container_user(args.harness_container_user, args.harness_docker_host),
        ),
        root=(runtime.run_dir / "policy_harness").resolve(),
        account=runtime.consume_many,
        recovery_budget=lambda: (
            float(benchmark_clock.snapshot()["remaining_seconds"])
            if benchmark_clock is not None else 2.0 * args.harness_wall_time_seconds
        ),
    )
    return adapter, controller


def create_client(
    args: argparse.Namespace,
    runtime: CampaignRuntime,
    provider: ProviderSettings,
    prepared: PreparedCase,
    profiles: Sequence[HarnessProfile],
    mcp,
    *,
    policy: bool = False,
    artifact_root: Path | None = None,
    contract: HarnessSubmissionContract | None = None,
    fork_sources: dict[str, PiForkSource] | None = None,
) -> HarnessClient:
    artifact_root = (artifact_root or runtime.run_dir / ("policy_harness" if policy else "harness")).resolve()
    resource_root = (TASK_ROOT / "resources" / "harness").resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)
    write_harness_sequence_context(
        prepared.context,
        artifact_root / "sequence_context.json",
    )
    mounts = [(resource_root, "/resources", True)]
    surrogate_query = args.harness_surrogate_query and not policy and contract is None
    query_environment = {}
    if surrogate_query:
        mounts.append((
            TASK_ROOT / "core/hamming_posterior.py", "/task_runtime/hamming_posterior.py", True,
        ))
        query_environment["LDM_NUCLEOBENCH_SURROGATE"] = "/artifacts/surrogate"
        query_environment["PYTHONPATH"] = "/task_runtime"
    mcp_servers = mcp.servers
    if policy:
        mounts.append((DIAGNOSTICS_PATH, "/task_runtime/ucb_diagnostics.py", True))
        submission_contract = policy_submission_contract(args.policy_max_submission_attempts)
        history_path = "/measured_history/observations.json"
        mounts.append((
            (runtime.run_dir / "harness" / MEASURED_HISTORY_FILE).parent.resolve(),
            "/measured_history", True,
        ))
        for source, destination in (
            ("README.md", "task_README.md"),
            ("resources/README.md", "resources_README.md"),
            ("resources/cases/catalog.json", "case_catalog.json"),
            ("resources/upstream_contract.json", "upstream_contract.json"),
            ("resources/verification_record.json", "verification_record.json"),
        ):
            mounts.append(((TASK_ROOT / source).resolve(), f"/public/{destination}", True))
        mcp_servers = (*mcp_servers, policy_mcp_server(
            diagnostics_path="/task_runtime/ucb_diagnostics.py",
            diagnostics_sha256=file_sha256(DIAGNOSTICS_PATH),
        ))
    else:
        submission_contract = contract or harness_submission_contract(args.harness_candidates_per_session)
        history_path = "/artifacts/measured_history/observations.json"
        (artifact_root / MEASURED_HISTORY_FILE).parent.mkdir(parents=True, exist_ok=True)
    return HarnessClient(
        pi_sidecar_command(
            artifact_root,
            args.harness_sidecar_image,
            cache_root=args.harness_cache_dir,
            docker_host=args.harness_docker_host,
            container_user=args.harness_container_user,
            environment={
                "LDM_NUCLEOBENCH_CONTEXT": "/artifacts/sequence_context.json",
                "LDM_NUCLEOBENCH_HISTORY": history_path,
                **query_environment,
            },
            mounts=mounts,
        ),
        api_key=provider.api_key,
        config=PiHarnessConfig(
            artifact_root=Path("/artifacts"),
            base_url=provider.base_url,
            model=provider.model,
            profiles=profiles,
            campaign_id=runtime.run_id,
            task_id=TASK_ID,
            case_id=f"{args.case_id}:optimization_policy" if policy else args.case_id,
            seed=args.campaign_index,
            submission_contract=submission_contract,
            guest_runtime=harness_guest_runtime(),
            tool_extensions=harness_tool_extensions(surrogate_query=surrogate_query),
            mcp_servers=mcp_servers,
            thinking=args.harness_thinking,
            sol_pi=sol_pi_config(args),
            fork_sources=fork_sources or {},
            provider_request_body=parse_extra_body(args.llm_extra_body_json),
            limits=HarnessLimits(
                wall_time_seconds=args.harness_wall_time_seconds,
                tool_call_budgets=parse_tool_call_budgets(
                    args.policy_tool_budget if policy else args.harness_tool_budget,
                    excluded_tools=(submission_contract.tool_name,),
                ),
            ),
            network_policy=HarnessNetworkPolicy(
                forbidden_query_patterns=HARNESS_FORBIDDEN_PATTERNS,
            ),
            context7_enabled=args.harness_context7,
        ),
        named_secrets=mcp.named_secrets,
        response_timeout_seconds=args.harness_response_timeout,
    )


def proposal_profiles(args: argparse.Namespace):
    if args.search_method in PARALLEL_HARNESS_METHODS:
        return harness_profiles(args.harness_profile)
    if args.search_method == "harness":
        return direct_harness_profile(blank_slate=args.harness_blank_slate)
    return ()


def sol_pi_config(args: argparse.Namespace) -> dict[str, Any] | None:
    if args.harness_sol_pi_config is None:
        return None
    value = json.loads(args.harness_sol_pi_config.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("SoL-Pi configuration must be a JSON object")
    return value


def describe(args: argparse.Namespace) -> dict[str, Any] | None:
    if args.search_method not in PERSISTENT_HARNESS_METHODS:
        return None
    profiles = (
        args.harness_profile
        if args.search_method in PARALLEL_HARNESS_METHODS
        else tuple(profile.profile_id for profile in proposal_profiles(args))
    )
    description = {
        "research_mode": "blank_slate" if args.harness_blank_slate else "open_research",
        "image": args.harness_sidecar_image,
        "profile_ids": [profile.profile_id for profile in proposal_profiles(args)],
        "profile_roles": list(profiles),
        "session_count": len(profiles),
        "candidates_per_session": args.harness_candidates_per_session,
        "unique_candidates_per_session": (
            args.search_method == "harness" or args.harness_unique_candidates
        ),
        "thinking": args.harness_thinking,
        "surrogate_query": args.harness_surrogate_query,
        "wall_time_seconds": args.harness_wall_time_seconds,
        "response_timeout_seconds": args.harness_response_timeout,
        "tool_call_budgets": parse_tool_call_budgets(
            args.harness_tool_budget,
            excluded_tools=("submit_candidates",),
        ),
        "context7_enabled": args.harness_context7,
        "mcp_configured": args.harness_mcp_config is not None,
        "skills_loaded": True,
        "skill_ids": list(HARNESS_SKILL_IDS),
        "sol_pi": sol_pi_config(args),
        "main_research": args.harness_main_research,
        "research_strategy": args.harness_research_strategy_identity,
    }
    if args.search_method == COMPILED_POLICY_METHOD:
        description["policy_session"] = {
            "profile_id": "policy_architect",
            "skills_loaded": True,
            "editable_components": ["prior_mean", "alpha", "eta"],
        }
    return description
