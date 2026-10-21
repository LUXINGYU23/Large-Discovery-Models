"""Task resources, role contracts and Pi client assembly."""

from __future__ import annotations

from pathlib import Path

from ldm_tts.engine.run_store import CampaignRuntime
from ldm_tts.harness import (
    HarnessClient, HarnessLimits, HarnessNetworkPolicy,
    parse_tool_call_budgets, policy_submission_contract,
)
from ldm_tts.harness.container import pi_sidecar_command
from ldm_tts.harness.pi import PiHarnessConfig, policy_mcp_server
from ldm_tts.harness.protocol import file_sha256
from ldm_tts.harness.ucb_diagnostics import DIAGNOSTICS_PATH
from tasks.iron_mind.core.constants import FORBIDDEN_QUERY_PATTERNS, TASK_ID
from tasks.iron_mind.core.harness import (
    direct_harness_profile, harness_guest_runtime,
    harness_profiles, harness_submission_contract, harness_tool_extensions,
    write_harness_space_catalog,
)
from tasks.iron_mind.core.optimization_policy import policy_harness_profile
from tasks.iron_mind.core.research import MEASURED_HISTORY_FILE, write_measured_history
from tasks.iron_mind.core.search import PARALLEL_HARNESS_METHODS
from tasks.iron_mind.core.candidate import IronMindCandidateDomain

TASK_ROOT = Path(__file__).resolve().parents[1]


def create_client(
    args, runtime: CampaignRuntime, provider, table, mcp, *, policy: bool = False,
    artifact_root=None, profiles=None, contract=None, fork_sources=None,
) -> HarnessClient:
    artifact_root = Path(artifact_root or runtime.run_dir / ("policy_harness" if policy else "harness")).resolve()
    harness_resources = (TASK_ROOT / "resources" / "harness").resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)
    write_harness_space_catalog(
        IronMindCandidateDomain(table.schema, table),
        artifact_root / "reaction_space.json",
    )
    mounts = [(harness_resources, "/resources", True)]
    history_root = (
        runtime.run_dir / "harness" / MEASURED_HISTORY_FILE
    ).parent.resolve()
    if policy:
        mounts.append((DIAGNOSTICS_PATH, "/task_runtime/ucb_diagnostics.py", True))
        mounts.append((history_root, "/measured_history", True))
        mounts.extend(
            (
                ((TASK_ROOT / "README.md").resolve(), "/public/task_README.md", True),
                (
                    (TASK_ROOT / "resources" / "README.md").resolve(),
                    "/public/resources_README.md",
                    True,
                ),
                (
                    (TASK_ROOT / "resources" / "reaction_schemas.json").resolve(),
                    "/public/reaction_schemas.json",
                    True,
                ),
                (
                    (TASK_ROOT / "resources" / "upstream_contract.json").resolve(),
                    "/public/upstream_contract.json",
                    True,
                ),
            )
        )
        profiles = policy_harness_profile()
        submission_contract = policy_submission_contract(
            args.policy_max_submission_attempts
        )
        mcp_servers = (
            *mcp.servers,
            policy_mcp_server(
                diagnostics_path="/task_runtime/ucb_diagnostics.py",
                diagnostics_sha256=file_sha256(DIAGNOSTICS_PATH),
            ),
        )
        history_path = "/measured_history/observations.json"
    else:
        if not (artifact_root / MEASURED_HISTORY_FILE).exists():
            write_measured_history(artifact_root, ())
        profiles = profiles or proposal_profiles(args)
        submission_contract = contract or harness_submission_contract(
            candidates_per_profile(args)
        )
        mcp_servers = mcp.servers
        history_path = "/artifacts/measured_history/observations.json"
    config = PiHarnessConfig(
        artifact_root=Path("/artifacts"),
        base_url=provider.base_url,
        model=provider.model,
        profiles=profiles,
        fork_sources=fork_sources or {},
        campaign_id=runtime.run_id,
        task_id=TASK_ID,
        case_id=args.dataset_id + (":optimization_policy" if policy else ""),
        seed=args.campaign_index,
        submission_contract=submission_contract,
        guest_runtime=harness_guest_runtime(),
        tool_extensions=harness_tool_extensions(),
        mcp_servers=mcp_servers,
        thinking=args.harness_thinking,
        limits=HarnessLimits(
            wall_time_seconds=args.harness_wall_time_seconds,
            tool_call_budgets=parse_tool_call_budgets(
                args.policy_tool_budget if policy else args.harness_tool_budget,
                excluded_tools=(submission_contract.tool_name,),
            ),
        ),
        network_policy=HarnessNetworkPolicy(
            forbidden_query_patterns=FORBIDDEN_QUERY_PATTERNS
        ),
        context7_enabled=args.harness_context7,
    )
    return HarnessClient(
        pi_sidecar_command(
            artifact_root,
            args.harness_sidecar_image,
            cache_root=args.harness_cache_dir,
            docker_host=args.harness_docker_host,
            container_user=args.harness_container_user,
            environment={
                "LDM_IRON_MIND_CATALOG": "/artifacts/reaction_space.json",
                "LDM_IRON_MIND_HISTORY": history_path,
            },
            mounts=tuple(mounts),
        ),
        api_key=provider.api_key,
        config=config,
        named_secrets=mcp.named_secrets,
        response_timeout_seconds=args.harness_response_timeout,
    )


def proposal_profiles(args):
    if args.search_method in PARALLEL_HARNESS_METHODS:
        return harness_profiles()
    if args.search_method == "harness":
        return direct_harness_profile()
    return ()


def candidates_per_profile(args) -> int:
    if args.search_method in PARALLEL_HARNESS_METHODS:
        return args.harness_candidates_per_session
    if args.search_method == "harness":
        return args.evaluations_per_round
    return 0
