"""NucleoBench research resources, role contracts and Pi client assembly."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ldm_tts.engine.run_store import CampaignRuntime
from ldm_tts.harness import (
    HarnessClient, HarnessLimits, HarnessNetworkPolicy, HarnessProfile,
    HarnessSubmissionContract, parse_tool_call_budgets, policy_submission_contract,
)
from ldm_tts.harness.container import pi_sidecar_command
from ldm_tts.harness.pi import PiForkSource, PiHarnessConfig, policy_mcp_server
from ldm_tts.harness.protocol import file_sha256
from ldm_tts.harness.ucb_diagnostics import DIAGNOSTICS_PATH
from tasks.nucleobench.core.constants import (
    COMPILED_POLICY_METHOD, PARALLEL_HARNESS_METHODS, PERSISTENT_HARNESS_METHODS, TASK_ID,
)
from tasks.nucleobench.core.harness import (
    DIRECT_HARNESS_PROFILE_ID, HARNESS_FORBIDDEN_PATTERNS, HARNESS_SKILL_IDS,
    direct_harness_profile, harness_guest_runtime,
    harness_profiles, harness_submission_contract, harness_tool_extensions,
    write_harness_sequence_context,
)
from tasks.nucleobench.core.oracles.official import PreparedCase
from tasks.nucleobench.core.provider import ProviderSettings, parse_extra_body
from tasks.nucleobench.core.research import MEASURED_HISTORY_FILE

TASK_ROOT = Path(__file__).resolve().parents[1]

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
        return direct_harness_profile()
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
        else (DIRECT_HARNESS_PROFILE_ID,)
    )
    description = {
        "research_mode": "open_research",
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
