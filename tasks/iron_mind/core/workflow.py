"""Workflow assembly for one Iron Mind LDM campaign."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from ldm_tts.contracts import LDMTaskSpec
from ldm_tts.data import DataCollectionSink
from ldm_tts.engine import LDMEngineConfig
from ldm_tts.engine.run_store import CampaignRuntime, unique_run_dir
from ldm_tts.harness import (
    HarnessClient,
    HarnessLimits,
    HarnessNetworkPolicy,
    HarnessPoolConfig,
    load_harness_mcp_config,
    parse_tool_call_budgets,
)
from ldm_tts.registration.experiment import (
    load_active_experiment_contract,
    load_experiment_contract,
    snapshot_experiment_contract,
)
from ldm_tts.transport import CallableProposalClient, ProposalClient
from ldm_tts.transport.openai_http import EndpointRequestError

from tasks.iron_mind.core.data import FrozenReactionTable
from tasks.iron_mind.core.candidate import IronMindCandidateDomain
from tasks.iron_mind.core.dependencies import load_pinned_reaction_table
from tasks.iron_mind.core.constants import OBJECTIVE_NAME, TASK_ID
from tasks.iron_mind.core.factory import (
    CampaignComponentOptions,
    build_base_reaction_selector,
    build_campaign_components,
    build_reaction_selector,
)
from tasks.iron_mind.core.harness import (
    HARNESS_FORBIDDEN_PATTERNS,
    HARNESS_PROFILE_IDS,
    direct_harness_profile,
    harness_guest_runtime,
    harness_candidate_schema,
    harness_profiles as parallel_harness_profiles,
    harness_tool_extensions,
    write_harness_space_catalog,
)
from tasks.iron_mind.core.mock import (
    MOCK_SEED_ROW_COUNT,
    load_mock_table,
    mock_proposal_response,
)
from tasks.iron_mind.core.proposal_transport import build_openai_reaction_client
from tasks.iron_mind.core.provider import (
    OpenAIProviderSettings,
    parse_openai_extra_body_json,
)
from tasks.iron_mind.core.reporting import write_campaign_reports
from tasks.iron_mind.core.schema import ReactionDatasetSchema, load_reaction_schemas
from tasks.iron_mind.core.surrogate import ReactionOneHotEncoder
from tasks.iron_mind.core.search import finite_domain_size
from tasks.iron_mind.core.task_spec import (
    build_direct_acquisition,
    build_reaction_task_spec,
    disabled_surrogate,
)
from tasks.iron_mind.core.workflow_support import (
    campaign_budget,
    jsonable_args,
    load_campaign_state,
    pause_endpoint,
    preflight_endpoint,
    provider_settings,
)
from tasks.iron_mind.core.workflow_args import parse_args, validate_args

TASK_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = TASK_ROOT / "resources" / "reaction_schemas.json"
MOCK_ORACLE_PATH = TASK_ROOT / "resources" / "mock_oracle.csv"


def describe_ldm_task(args: argparse.Namespace) -> LDMTaskSpec:
    """Describe the configured reaction task before campaign assembly."""

    return _task_spec(args, _load_table(args))


def _task_spec(args: argparse.Namespace, table: FrozenReactionTable) -> LDMTaskSpec:
    schema = table.schema
    encoder = (
        None
        if args.search_method in {"llm", "harness"}
        else ReactionOneHotEncoder(schema)
    )
    selector = _selector(args, schema, encoder)
    return build_reaction_task_spec(
        schema,
        (
            selector.describe()
            if selector is not None
            else build_direct_acquisition(args.search_method)
        ),
        proposal_samples=_proposal_samples(args),
        bo_pool_size=args.bo_pool_size,
        proposal_max_workers=args.proposal_max_workers,
        prompt_policy=args.prompt_policy,
        search_method=args.search_method,
        initialization_mode=args.initialization_mode,
        surrogate=(
            encoder.describe()
            if encoder is not None
            else disabled_surrogate(args.search_method)
        ),
        domain_size=finite_domain_size(table),
        harness_profile_count=(
            len(HARNESS_PROFILE_IDS)
            if args.search_method == "ldm_harness"
            else 1
            if args.search_method == "harness"
            else 0
        ),
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validate_args(args)
    table = _load_table(args)
    task_spec = _task_spec(args, table)
    contract, profile_name = load_active_experiment_contract()
    if contract is None:
        contract = load_experiment_contract(TASK_ROOT / "experiment.json")
    payload = _run_payload(args, task_spec, contract.digest, profile_name)
    if args.dry_run:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    return _run_campaign(args, table, task_spec, contract, profile_name, payload)


def _run_campaign(
    args, table, task_spec, contract, profile_name: str, payload: dict[str, Any]
) -> int:
    provider = (
        provider_settings(args)
        if args.proposal_mode == "openai" or args.search_method in {"ldm_harness", "harness"}
        else None
    )
    if provider is not None:
        args.llm_url = provider.base_url
        args.llm_model_name = provider.model
    runtime = _open_runtime(args, table, task_spec, contract, profile_name)
    if args.search_method in {"ldm_harness", "harness"}:
        assert provider is not None
        missing = _missing_harness_provider(provider)
        if missing:
            pause_endpoint(runtime, args, payload, missing)
            return 2
        harness_client = _harness_client(args, runtime, provider, table)
        try:
            harness_client.start()
            components = _components(args, table, runtime, None, harness_client)
            return _finish_campaign(args, components, runtime, payload)
        finally:
            harness_client.close()
    client = _proposal_client(args, table, provider)
    if args.proposal_mode == "openai":
        assert provider is not None and client is not None
        if not preflight_endpoint(client, runtime, args, payload, provider):
            return 2
    components = _components(args, table, runtime, client, None)
    return _finish_campaign(args, components, runtime, payload)


def _components(args, table, runtime, client, harness_client):
    sink = DataCollectionSink.from_env(default_root=runtime.run_dir / "ldm_data")
    before_requests = (
        (lambda count: runtime.consume("llm_requests", count))
        if args.proposal_mode == "openai"
        else None
    )
    profiles = _harness_profiles(args)
    return build_campaign_components(
        CampaignComponentOptions(
            client=client,
            schema=table.schema,
            table=table,
            sink=sink,
            runtime=runtime,
            proposal_samples=_proposal_samples(args),
            bo_pool_size=args.bo_pool_size,
            search_method=args.search_method,
            initialization_mode=args.initialization_mode,
            proposal_max_workers=args.proposal_max_workers,
            before_requests=before_requests,
            acquisition_beta=args.acquisition_beta,
            acquisition_alpha=args.alpha,
            acquisition_eta=args.eta,
            acquisition_z_clip=args.z_clip,
            selection_seed=args.campaign_index,
            prompt_policy=args.prompt_policy,
            harness_client=harness_client,
            harness_profiles=profiles,
            account_harness_usage=(
                runtime.consume_many
                if args.search_method in {"ldm_harness", "harness"}
                else None
            ),
        )
    )


def _missing_harness_provider(provider) -> str:
    missing = [
        name
        for name, value in (
            ("LLM_BASE_URL", provider.base_url),
            ("LLM_MODEL_NAME", provider.model),
            ("LLM_API_KEY or --harness-api-key-file", provider.api_key),
        )
        if not value
    ]
    return "Set " + ", ".join(missing) + " for the harness backend." if missing else ""


def _harness_client(
    args,
    runtime: CampaignRuntime,
    provider: OpenAIProviderSettings,
    table: FrozenReactionTable,
) -> HarnessClient:
    mcp = load_harness_mcp_config(args.harness_mcp_config)
    artifact_root = (runtime.run_dir / "harness").resolve()
    cache_root = (
        args.harness_cache_dir.expanduser().resolve()
        if args.harness_cache_dir is not None
        else (Path.home() / ".cache" / "ldm-gondolin").resolve()
    )
    resource_root = (TASK_ROOT / "resources" / "harness").resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)
    (cache_root / "runtime-overlays").mkdir(exist_ok=True)
    domain = IronMindCandidateDomain(table.schema, table)
    write_harness_space_catalog(domain, artifact_root / "reaction_space.json")
    command = ["docker"]
    if args.harness_docker_host:
        command.extend(("--host", args.harness_docker_host))
    command.extend(("run", "--rm", "-i"))
    container_user = args.harness_container_user
    if not container_user and hasattr(os, "getuid") and hasattr(os, "getgid"):
        container_user = f"{os.getuid()}:{os.getgid()}"
    if container_user:
        command.extend(("--user", container_user))
    command.extend(
        (
            "--device",
            "/dev/kvm",
            "--env",
            "HOME=/runtime-home",
            "--env",
            "XDG_CACHE_HOME=/runtime-home/.cache",
            "--env",
            "GONDOLIN_IMAGE_STORE=/runtime-home/.cache/gondolin/images",
            "--env",
            "GONDOLIN_SESSIONS_DIR=/runtime-home/.cache/gondolin/sessions",
            "--env",
            "LDM_HARNESS_CACHE_ROOT=/runtime-home/.cache/gondolin",
            "--env",
            "TMPDIR=/runtime-home/.cache/gondolin/runtime-overlays",
            "--env",
            "LDM_IRON_MIND_CATALOG=/artifacts/reaction_space.json",
            "--mount",
            f"type=bind,src={artifact_root},dst=/artifacts",
            "--mount",
            f"type=bind,src={resource_root},dst=/resources,readonly",
            "--mount",
            f"type=bind,src={cache_root},dst=/runtime-home/.cache/gondolin",
            args.harness_sidecar_image,
        )
    )
    profiles = _harness_profiles(args)
    config = HarnessPoolConfig(
        artifact_root=Path("/artifacts"),
        base_url=provider.base_url,
        model=provider.model,
        profiles=profiles,
        campaign_id=runtime.run_id,
        task_id=TASK_ID,
        case_id=args.dataset_id,
        seed=args.campaign_index,
        candidate_schema=harness_candidate_schema(table.schema),
        guest_runtime=harness_guest_runtime(),
        tool_extensions=harness_tool_extensions(),
        mcp_servers=mcp.servers,
        thinking=args.harness_thinking,
        limits=HarnessLimits(
            wall_time_seconds=args.harness_wall_time_seconds,
            tool_call_budgets=parse_tool_call_budgets(args.harness_tool_budget),
        ),
        network_policy=HarnessNetworkPolicy(
            forbidden_query_patterns=HARNESS_FORBIDDEN_PATTERNS,
        ),
        context7_enabled=args.harness_context7,
    )
    return HarnessClient(
        command,
        api_key=provider.api_key,
        config=config,
        named_secrets=mcp.named_secrets,
        response_timeout_seconds=args.harness_response_timeout,
    )


def _finish_campaign(
    args,
    components,
    runtime: CampaignRuntime,
    payload: dict[str, Any],
) -> int:
    state = load_campaign_state(runtime, args.resume_from is not None)
    try:
        config = LDMEngineConfig(
            args.iterations,
            _reservoir_size(args, components),
            args.evaluations_per_round,
        )
        result = components.engine.run(
            config,
            state=state,
        )
    except EndpointRequestError as exc:
        pause_endpoint(
            runtime,
            args,
            payload,
            str(exc),
            phase="reservoir_expansion",
        )
        return 2
    write_campaign_reports(runtime, objective_name=OBJECTIVE_NAME)
    payload.update(engine_summary=result.summary, run_dir=str(runtime.run_dir.resolve()))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if result.summary["successful_evaluation_count"] else 1


_validate_args = validate_args


def _run_payload(
    args: argparse.Namespace,
    task_spec: LDMTaskSpec,
    contract_sha256: str,
    profile_name: str,
) -> dict[str, Any]:
    return {
        "task": TASK_ID,
        "mode": "mock" if args.mock else "real",
        "search_method": args.search_method,
        "initialization_mode": args.initialization_mode,
        "proposal_mode": args.proposal_mode,
        "dataset_id": args.dataset_id,
        "objective": OBJECTIVE_NAME,
        "contract_profile": profile_name,
        "contract_sha256": contract_sha256,
        "ldm_task_spec": task_spec.to_dict(),
    }


def _open_runtime(args, table, task_spec, contract, profile_name: str) -> CampaignRuntime:
    default_name = (
        "mock" if args.mock else f"{args.dataset_id}-campaign-{args.campaign_index:02d}"
    )
    run_dir = args.resume_from.resolve() if args.resume_from else unique_run_dir(args.out_dir / (args.run_name or default_name))
    profile = contract.profile(profile_name) if profile_name else None
    runtime = CampaignRuntime.open(
        run_dir,
        task=TASK_ID,
        config=jsonable_args(args),
        task_spec=task_spec,
        budget_limits=campaign_budget(
            args,
            None if profile is None else profile.budget,
            domain_size=finite_domain_size(table),
        ),
        contract_snapshot=contract.to_dict(),
        contract_sha256=contract.digest,
        contract_profile=profile_name,
        resume=args.resume_from is not None,
    )
    if args.resume_from is None:
        snapshot_experiment_contract(contract, run_dir, profile=profile_name)
    return runtime


def _schema_for(dataset_id: str) -> ReactionDatasetSchema:
    try:
        return load_reaction_schemas(SCHEMA_PATH)[dataset_id]
    except KeyError as exc:
        raise SystemExit(f"Unknown --dataset-id: {dataset_id}") from exc


def _load_mock_table(
    schema: ReactionDatasetSchema,
    candidate_count: int = MOCK_SEED_ROW_COUNT,
    round_count: int = 1,
    slot_seed: int = 0,
) -> FrozenReactionTable:
    return load_mock_table(
        schema,
        MOCK_ORACLE_PATH,
        candidate_count=candidate_count,
        round_count=round_count,
        slot_seed=slot_seed,
    )


def _load_table(args: argparse.Namespace) -> FrozenReactionTable:
    if args.mock:
        return _load_mock_table(
            _schema_for(args.dataset_id),
            candidate_count=args.proposal_samples * max(1, args.iterations),
            round_count=max(1, args.iterations),
            slot_seed=args.campaign_index,
        )
    assert args.data_dir is not None
    return load_pinned_reaction_table(dataset_id=args.dataset_id, data_root=args.data_dir)


def _proposal_client(
    args: argparse.Namespace,
    table: FrozenReactionTable,
    provider: OpenAIProviderSettings | None,
) -> ProposalClient | None:
    if args.proposal_mode == "callable":
        return CallableProposalClient(
            lambda request: mock_proposal_response(
                table,
                proposal_index=int(request.metadata["proposal_index"]),
                slot_focus=request.metadata.get("slot_focus"),
            )
        )
    if args.proposal_mode == "none":
        return None
    assert provider is not None
    return build_openai_reaction_client(
        base_url=provider.base_url,
        model=provider.model,
        api_key=provider.api_key,
        timeout_seconds=args.llm_timeout,
        max_tokens=args.llm_max_tokens,
        temperature=args.llm_temperature,
        json_mode=args.llm_json_mode,
        extra_body=parse_openai_extra_body_json(args.llm_extra_body_json),
    )


def _selector(args, schema: ReactionDatasetSchema, encoder: ReactionOneHotEncoder | None):
    if args.search_method in {"llm", "harness"}:
        return None
    assert encoder is not None
    if args.search_method == "bo":
        return build_base_reaction_selector(
            schema=schema,
            beta=args.acquisition_beta,
            feature_version=encoder.version,
        )
    return build_reaction_selector(
        schema=schema,
        beta=args.acquisition_beta,
        alpha=args.alpha,
        eta=args.eta,
        z_clip=args.z_clip,
        seed=args.campaign_index,
        pool_size=args.bo_pool_size,
        proposal_sample_count=args.proposal_samples,
        feature_version=encoder.version,
    )


def _reservoir_size(args, components) -> int:
    if args.search_method in {"ldm", "ldm_harness"}:
        return args.proposal_samples
    if args.search_method in {"llm", "harness"}:
        return args.evaluations_per_round
    return components.task_spec.reservoir.max_size or args.proposal_samples


def _proposal_samples(args) -> int:
    return (
        args.evaluations_per_round
        if args.search_method == "harness"
        else args.proposal_samples
    )


def _harness_profiles(args):
    if args.search_method == "ldm_harness":
        return parallel_harness_profiles(args.harness_candidates_per_session)
    if args.search_method == "harness":
        return direct_harness_profile(args.evaluations_per_round)
    return ()
