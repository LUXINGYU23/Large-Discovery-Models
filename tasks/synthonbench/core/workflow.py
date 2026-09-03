"""Canonical shared-runner workflow for the SynthonBench LDM task."""

from __future__ import annotations

import argparse
import json
from contextlib import ExitStack
from pathlib import Path
from typing import Any

from ldm_tts.contracts import LDMTaskSpec
from ldm_tts.data import DataCollectionSink
from ldm_tts.engine import LDMEngineConfig
from ldm_tts.engine.run_store import CampaignRuntime, unique_run_dir
from ldm_tts.harness import (
    DockerPolicyExecutor,
    HarnessClient,
    HarnessLimits,
    HarnessNetworkPolicy,
    HarnessPoolConfig,
    PolicyResearchController,
    load_harness_mcp_config,
    parse_tool_call_budgets,
    policy_mcp_server,
    policy_submission_contract,
)
from ldm_tts.harness.container import docker_identity_args, resolve_container_user
from ldm_tts.registration.experiment import (
    load_active_experiment_contract,
    load_experiment_contract,
    snapshot_experiment_contract,
)
from ldm_tts.transport import CallableProposalClient, ProposalClient
from ldm_tts.transport.openai_http import EndpointRequestError
from tasks.synthonbench.core.data import (
    LoadedSynthonBenchmark,
    load_mock_benchmark,
    load_official_benchmark,
)
from tasks.synthonbench.core.constants import FORBIDDEN_QUERY_PATTERNS, TASK_ID
from tasks.synthonbench.core.factory import (
    CampaignComponentOptions,
    build_base_synthon_selector,
    build_campaign_components,
    build_synthon_selector,
)
from tasks.synthonbench.core.harness import (
    HARNESS_PROFILE_IDS,
    direct_harness_profile,
    harness_guest_runtime,
    harness_profiles,
    harness_submission_contract,
    harness_tool_extensions,
    write_harness_space_catalog,
)
from tasks.synthonbench.core.nystrom_encoder import SynthonNystromEncoder
from tasks.synthonbench.core.optimization_policy import (
    SynthonOptimizationPolicyAdapter,
    policy_harness_profile,
)
from tasks.synthonbench.core.policy_features import SynthonPolicyFeatureEncoder
from tasks.synthonbench.core.mock import mock_proposal_response
from tasks.synthonbench.core.proposal_transport import build_openai_synthon_client
from tasks.synthonbench.core.provider import parse_openai_extra_body_json
from tasks.synthonbench.core.reporting import write_campaign_reports
from tasks.synthonbench.core.search import (
    COMPILED_POLICY_METHOD,
    PARALLEL_HARNESS_METHODS,
    PERSISTENT_HARNESS_METHODS,
)
from tasks.synthonbench.core.task_spec import (
    build_direct_acquisition,
    build_synthon_task_spec,
)
from tasks.synthonbench.core.workflow_args import parse_args, validate_args
from tasks.synthonbench.core.workflow_support import (
    campaign_budget,
    jsonable_args,
    load_campaign_state,
    pause_endpoint,
    preflight_endpoint,
    provider_settings,
)

TASK_ROOT = Path(__file__).resolve().parents[1]


def describe_ldm_task(args: argparse.Namespace, benchmark: LoadedSynthonBenchmark | None = None) -> LDMTaskSpec:
    """Build the declared task semantics using the same encoder and selector as runs."""

    benchmark = _load_benchmark(args) if benchmark is None else benchmark
    encoder = _encoder(args, benchmark)
    selector = _selector(args, encoder)
    return build_synthon_task_spec(
        encoder=encoder,
        acquisition=(
            selector.describe()
            if selector is not None
            else build_direct_acquisition(args.search_method)
        ),
        proposal_samples=_proposal_samples(args),
        evaluations_per_round=args.evaluations_per_round,
        bo_pool_size=args.bo_pool_size,
        bo_search_samples=args.bo_search_samples,
        proposal_candidates_per_request=args.proposal_candidates_per_request,
        proposal_max_workers=args.proposal_max_workers,
        slate_size=args.slate_size,
        reaction_allocation=args.reaction_allocation,
        prompt_policy=args.prompt_policy,
        search_method=args.search_method,
        initialization_mode=args.initialization_mode,
        harness_profile_count=(
            len(HARNESS_PROFILE_IDS)
            if args.search_method in PARALLEL_HARNESS_METHODS
            else 1
            if args.search_method == "harness"
            else 0
        ),
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validate_args(args)
    benchmark = _load_benchmark(args)
    task_spec = describe_ldm_task(args, benchmark)
    contract, profile_name = _load_contract()
    payload = _run_payload(args, benchmark, task_spec, contract.digest, profile_name)
    if args.dry_run:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    return _run_campaign(args, benchmark, task_spec, contract, profile_name, payload)


def _load_benchmark(args: argparse.Namespace) -> LoadedSynthonBenchmark:
    budget = args.iterations * args.evaluations_per_round
    if args.mock:
        return load_mock_benchmark(budget=budget, seed=args.campaign_index)
    assert args.data_dir is not None and args.source_dir is not None
    return load_official_benchmark(
        data_dir=args.data_dir.resolve(),
        source_dir=args.source_dir.resolve(),
        scale=args.scale,
        target=args.target,
        oracle_kind=args.oracle_kind,
        budget=budget,
        seed=args.campaign_index,
    )


def _load_contract():
    contract, profile_name = load_active_experiment_contract()
    return (load_experiment_contract(TASK_ROOT / "experiment.json"), profile_name) if contract is None else (contract, profile_name)


def _run_campaign(args, benchmark, task_spec, contract, profile_name: str,
                  payload: dict[str, Any]) -> int:
    provider = (
        provider_settings(args)
        if args.proposal_mode == "openai"
        or args.search_method in PERSISTENT_HARNESS_METHODS
        else None
    )
    if provider is not None:
        args.llm_url, args.llm_model_name = provider.base_url, provider.model
    runtime = _open_runtime(args, task_spec, contract, profile_name)
    if args.search_method in PERSISTENT_HARNESS_METHODS:
        assert provider is not None
        missing = _missing_harness_provider(provider)
        if missing:
            pause_endpoint(runtime, args, payload, missing)
            return 2
        mcp = load_harness_mcp_config(args.harness_mcp_config)
        with ExitStack() as stack:
            harness_client = stack.enter_context(
                _proposal_harness_client(args, runtime, provider, benchmark, mcp)
            )
            policy_controller = None
            policy_adapter = None
            if args.search_method == COMPILED_POLICY_METHOD:
                policy_adapter = SynthonOptimizationPolicyAdapter(
                    SynthonPolicyFeatureEncoder(
                        benchmark.task.space,
                        benchmark.task.allowed_reactions,
                    ),
                    target=benchmark.target,
                    seed=args.campaign_index,
                    acquisition_beta=args.acquisition_beta,
                    gp_signal_std=args.gp_signal_std,
                    gp_mean_std=args.gp_mean_std,
                    gp_observation_noise_std=args.gp_observation_noise_std,
                    gp_reaction_weight=args.gp_reaction_weight,
                    fingerprint_bits=args.fingerprint_bits,
                    gp_landmarks=args.gp_landmarks,
                    gp_kernel_jitter=args.gp_kernel_jitter,
                    default_alpha=args.alpha,
                    default_eta=args.eta,
                    enabled_capabilities=tuple(args.policy_capability),
                )
                policy_client = stack.enter_context(
                    _policy_harness_client(args, runtime, provider, benchmark, mcp)
                )
                policy_controller = PolicyResearchController(
                    client=policy_client,
                    adapter=policy_adapter,
                    executor=DockerPolicyExecutor(
                        image=args.harness_sidecar_image,
                        docker_host=args.harness_docker_host or "",
                        container_user=resolve_container_user(
                            args.harness_container_user,
                            args.harness_docker_host,
                        ),
                    ),
                    root=(runtime.run_dir / "policy_harness").resolve(),
                    account=runtime.consume_many,
                )
            components = _components(
                args,
                benchmark,
                runtime,
                None,
                harness_client,
                policy_controller,
                policy_adapter,
            )
            return _finish_campaign(args, benchmark, components, runtime, payload)
    client = _proposal_client(args, provider)
    if provider is not None and not preflight_endpoint(client, runtime, args, payload, provider):
        return 2
    components = _components(args, benchmark, runtime, client, None, None, None)
    return _finish_campaign(args, benchmark, components, runtime, payload)


def _open_runtime(args, task_spec, contract, profile_name: str) -> CampaignRuntime:
    profile = contract.profile(profile_name) if profile_name else None
    run_dir = _run_dir(args)
    runtime = CampaignRuntime.open(
        run_dir,
        task=TASK_ID,
        config=jsonable_args(args),
        task_spec=task_spec,
        budget_limits=campaign_budget(args, None if profile is None else profile.budget),
        contract_snapshot=contract.to_dict(),
        contract_sha256=contract.digest,
        contract_profile=profile_name,
        resume=args.resume_from is not None,
    )
    if args.resume_from is None:
        snapshot_experiment_contract(contract, run_dir, profile=profile_name)
    return runtime


def _run_dir(args: argparse.Namespace) -> Path:
    if args.resume_from is not None:
        return args.resume_from.resolve()
    default = "official_example" if args.mock else f"{args.oracle_kind}_{args.scale}_{args.target}_s{args.campaign_index}"
    return unique_run_dir(args.out_dir / (args.run_name or default))


def _proposal_client(args: argparse.Namespace, provider) -> ProposalClient | None:
    if args.proposal_mode == "callable":
        return CallableProposalClient(mock_proposal_response)
    if args.proposal_mode == "none":
        return None
    assert provider is not None
    return build_openai_synthon_client(
        base_url=provider.base_url,
        model=provider.model,
        api_key=provider.api_key,
        timeout_seconds=args.llm_timeout,
        max_tokens=args.llm_max_tokens,
        temperature=args.llm_temperature,
        json_mode=args.llm_json_mode,
        extra_body=parse_openai_extra_body_json(args.llm_extra_body_json),
    )


def _encoder(args, benchmark) -> SynthonNystromEncoder | None:
    if args.search_method in {"llm", "harness"}:
        return None
    return SynthonNystromEncoder(
        benchmark.task.space,
        benchmark.task.allowed_reactions,
        landmark_count=args.gp_landmarks,
        seed=args.campaign_index,
        fingerprint_bits=args.fingerprint_bits,
        kernel_jitter=args.gp_kernel_jitter,
        reaction_weight=args.gp_reaction_weight,
    )


def _selector(args, encoder: SynthonNystromEncoder | None):
    if encoder is None:
        return None
    if args.search_method == "bo":
        return build_base_synthon_selector(
            encoder=encoder,
            gp_signal_std=args.gp_signal_std,
            gp_mean_std=args.gp_mean_std,
            gp_observation_noise_std=args.gp_observation_noise_std,
            acquisition_beta=args.acquisition_beta,
        )
    return build_synthon_selector(
        encoder=encoder,
        selection_seed=args.campaign_index,
        gp_signal_std=args.gp_signal_std,
        gp_mean_std=args.gp_mean_std,
        gp_observation_noise_std=args.gp_observation_noise_std,
        acquisition_beta=args.acquisition_beta,
        alpha=args.alpha,
        eta=args.eta,
        z_clip=args.z_clip,
        bo_pool_size=args.bo_pool_size,
        proposal_samples=args.proposal_samples,
        policy_mode=args.search_method == COMPILED_POLICY_METHOD,
    )


def _reservoir_size(args) -> int:
    if args.search_method in {"llm", "harness"}:
        return args.evaluations_per_round
    return args.bo_search_samples if args.search_method == "bo" else args.proposal_samples


def _components(
    args,
    benchmark,
    runtime,
    client,
    harness_client,
    policy_controller,
    policy_adapter,
):
    sink = DataCollectionSink.from_env(default_root=runtime.run_dir / "ldm_data")
    before_requests = (
        (lambda count: runtime.consume("llm_requests", count))
        if args.proposal_mode == "openai"
        else None
    )
    profiles = _harness_profiles(args)
    return build_campaign_components(CampaignComponentOptions(
        client=client,
        official_task=benchmark.task,
        runtime=runtime,
        sink=sink,
        target=benchmark.target,
        proposal_samples=_proposal_samples(args),
        bo_pool_size=args.bo_pool_size,
        bo_search_samples=args.bo_search_samples,
        evaluations_per_round=args.evaluations_per_round,
        search_method=args.search_method,
        initialization_mode=args.initialization_mode,
        proposal_candidates_per_request=args.proposal_candidates_per_request,
        proposal_max_workers=args.proposal_max_workers,
        slate_size=args.slate_size,
        reaction_allocation=args.reaction_allocation,
        selection_seed=args.campaign_index,
        fingerprint_bits=args.fingerprint_bits,
        gp_landmarks=args.gp_landmarks,
        gp_kernel_jitter=args.gp_kernel_jitter,
        gp_signal_std=args.gp_signal_std,
        gp_mean_std=args.gp_mean_std,
        gp_observation_noise_std=args.gp_observation_noise_std,
        gp_reaction_weight=args.gp_reaction_weight,
        acquisition_beta=args.acquisition_beta,
        alpha=args.alpha,
        eta=args.eta,
        z_clip=args.z_clip,
        prompt_policy=args.prompt_policy,
        before_requests=before_requests,
        harness_client=harness_client,
        harness_profiles=profiles,
        harness_candidates_per_profile=_harness_candidates_per_profile(args),
        account_harness_usage=(
            runtime.consume_many
            if args.search_method in PERSISTENT_HARNESS_METHODS
            else None
        ),
        policy_controller=policy_controller,
        policy_adapter=policy_adapter,
    ))


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


def _proposal_harness_client(
    args,
    runtime: CampaignRuntime,
    provider,
    benchmark,
    mcp,
) -> HarnessClient:
    artifact_root = (runtime.run_dir / "harness").resolve()
    resource_root = (TASK_ROOT / "resources" / "harness").resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)
    write_harness_space_catalog(
        benchmark.task.space,
        benchmark.task.allowed_reactions,
        artifact_root / "synthon_space.json",
        reactions_path=(
            benchmark.data_dir / "spaces" / "reactions.tsv"
            if benchmark.data_dir is not None
            else None
        ),
    )
    profiles = _harness_profiles(args)
    config = HarnessPoolConfig(
        artifact_root=Path("/artifacts"),
        base_url=provider.base_url,
        model=provider.model,
        profiles=profiles,
        campaign_id=runtime.run_id,
        task_id=TASK_ID,
        case_id=f"{args.oracle_kind}:{args.scale}:{args.target}",
        seed=args.campaign_index,
        submission_contract=harness_submission_contract(
            _harness_candidates_per_profile(args)
        ),
        guest_runtime=harness_guest_runtime(),
        tool_extensions=harness_tool_extensions(),
        mcp_servers=mcp.servers,
        thinking=args.harness_thinking,
        limits=HarnessLimits(
            wall_time_seconds=args.harness_wall_time_seconds,
            tool_call_budgets=parse_tool_call_budgets(
                args.harness_tool_budget,
                excluded_tools=("submit_candidates",),
            ),
        ),
        network_policy=HarnessNetworkPolicy(
            forbidden_query_patterns=FORBIDDEN_QUERY_PATTERNS,
        ),
        context7_enabled=args.harness_context7,
    )
    return HarnessClient(
        _harness_command(
            args,
            artifact_root,
            _harness_cache_root(args),
            environment={
                "LDM_SYNTHON_SPACE_CATALOG": "/artifacts/synthon_space.json"
            },
            mounts=((resource_root, "/resources", True),),
        ),
        api_key=provider.api_key,
        config=config,
        named_secrets=mcp.named_secrets,
        response_timeout_seconds=args.harness_response_timeout,
    )


def _policy_harness_client(
    args,
    runtime: CampaignRuntime,
    provider,
    benchmark,
    mcp,
) -> HarnessClient:
    artifact_root = (runtime.run_dir / "policy_harness").resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)
    write_harness_space_catalog(
        benchmark.task.space,
        benchmark.task.allowed_reactions,
        artifact_root / "synthon_space.json",
        reactions_path=(
            benchmark.data_dir / "spaces" / "reactions.tsv"
            if benchmark.data_dir is not None
            else None
        ),
    )
    harness_resources = (TASK_ROOT / "resources" / "harness").resolve()
    mounts = (
        (harness_resources, "/resources", True),
        ((TASK_ROOT / "README.md").resolve(), "/public/task_README.md", True),
        (
            (TASK_ROOT / "resources" / "README.md").resolve(),
            "/public/resources_README.md",
            True,
        ),
        (
            (TASK_ROOT / "resources" / "upstream_contract.json").resolve(),
            "/public/upstream_contract.json",
            True,
        ),
        (
            (TASK_ROOT / "resources" / "verification_record.json").resolve(),
            "/public/verification_record.json",
            True,
        ),
    )
    config = HarnessPoolConfig(
        artifact_root=Path("/artifacts"),
        base_url=provider.base_url,
        model=provider.model,
        profiles=policy_harness_profile(),
        campaign_id=runtime.run_id,
        task_id=TASK_ID,
        case_id=f"{args.oracle_kind}:{args.scale}:{args.target}:optimization_policy",
        seed=args.campaign_index,
        submission_contract=policy_submission_contract(
            args.policy_max_submission_attempts
        ),
        guest_runtime=harness_guest_runtime(),
        tool_extensions=harness_tool_extensions(),
        mcp_servers=(*mcp.servers, policy_mcp_server()),
        thinking=args.harness_thinking,
        limits=HarnessLimits(
            wall_time_seconds=args.harness_wall_time_seconds,
            tool_call_budgets=parse_tool_call_budgets(
                args.policy_tool_budget,
                excluded_tools=("submit_optimization_policy",),
            ),
        ),
        network_policy=HarnessNetworkPolicy(
            forbidden_query_patterns=FORBIDDEN_QUERY_PATTERNS,
        ),
        context7_enabled=args.harness_context7,
    )
    return HarnessClient(
        _harness_command(
            args,
            artifact_root,
            _harness_cache_root(args),
            environment={
                "LDM_SYNTHON_SPACE_CATALOG": "/artifacts/synthon_space.json"
            },
            mounts=mounts,
        ),
        api_key=provider.api_key,
        config=config,
        named_secrets=mcp.named_secrets,
        response_timeout_seconds=args.harness_response_timeout,
    )


def _harness_cache_root(args) -> Path:
    root = (
        args.harness_cache_dir.expanduser().resolve()
        if args.harness_cache_dir is not None
        else (Path.home() / ".cache" / "ldm-gondolin").resolve()
    )
    root.mkdir(parents=True, exist_ok=True)
    return root


def _harness_command(
    args,
    artifact_root: Path,
    cache_root: Path,
    *,
    environment: dict[str, str],
    mounts: tuple[tuple[Path, str, bool], ...],
) -> list[str]:
    cache_root.mkdir(parents=True, exist_ok=True)
    (cache_root / "runtime-overlays").mkdir(exist_ok=True)
    command = ["docker"]
    if args.harness_docker_host:
        command.extend(("--host", args.harness_docker_host))
    command.extend(("run", "--rm", "-i"))
    container_user = resolve_container_user(
        args.harness_container_user,
        args.harness_docker_host,
    )
    command.extend(docker_identity_args(container_user, args.harness_docker_host))
    command.extend(("--device", "/dev/kvm"))
    runtime_environment = {
        "HOME": "/runtime-home",
        "XDG_CACHE_HOME": "/runtime-home/.cache",
        "GONDOLIN_IMAGE_STORE": "/runtime-home/.cache/gondolin/images",
        "GONDOLIN_SESSIONS_DIR": "/runtime-home/.cache/gondolin/sessions",
        "LDM_HARNESS_CACHE_ROOT": "/runtime-home/.cache/gondolin",
        "TMPDIR": "/runtime-home/.cache/gondolin/runtime-overlays",
        **environment,
    }
    for name, value in sorted(runtime_environment.items()):
        command.extend(("--env", f"{name}={value}"))
    command.extend((
        "--mount",
        f"type=bind,src={artifact_root},dst=/artifacts",
        "--mount",
        f"type=bind,src={cache_root},dst=/runtime-home/.cache/gondolin",
    ))
    for source, destination, readonly in mounts:
        suffix = ",readonly" if readonly else ""
        command.extend((
            "--mount",
            f"type=bind,src={source},dst={destination}{suffix}",
        ))
    command.append(args.harness_sidecar_image)
    return command


def _finish_campaign(args, benchmark, components, runtime: CampaignRuntime,
                     payload: dict[str, Any]) -> int:
    state = load_campaign_state(runtime, args.resume_from is not None)
    components.evaluator.restore_observations(state.observations)
    try:
        result = components.engine.run(
            LDMEngineConfig(
                args.iterations,
                _reservoir_size(args),
                args.evaluations_per_round,
            ),
            state=state,
        )
    except EndpointRequestError as exc:
        pause_endpoint(runtime, args, payload, str(exc), phase="reservoir_expansion")
        return 2
    report = write_campaign_reports(runtime, result, components.evaluator, benchmark,
                                    audit_timeout_seconds=args.audit_timeout)
    payload.update(engine_summary=result.summary, result=report, run_dir=str(runtime.run_dir.resolve()))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if result.summary["successful_evaluation_count"] else 1


def _proposal_samples(args) -> int:
    return (
        args.evaluations_per_round
        if args.search_method == "harness"
        else args.proposal_samples
    )


def _harness_profiles(args):
    if args.search_method in PARALLEL_HARNESS_METHODS:
        return harness_profiles()
    if args.search_method == "harness":
        return direct_harness_profile()
    return ()


def _harness_candidates_per_profile(args) -> int:
    if args.search_method in PARALLEL_HARNESS_METHODS:
        return args.harness_candidates_per_session
    if args.search_method == "harness":
        return args.evaluations_per_round
    return 0


def _run_payload(args, benchmark, task_spec, contract_sha256: str, profile_name: str) -> dict[str, Any]:
    return {
        "task": TASK_ID,
        "mode": benchmark.mode,
        "scale": benchmark.scale,
        "target": benchmark.target,
        "oracle_kind": benchmark.oracle_kind,
        "proposal_mode": args.proposal_mode,
        "search_method": args.search_method,
        "initialization_mode": args.initialization_mode,
        "contract_profile": profile_name,
        "contract_sha256": contract_sha256,
        "ldm_task_spec": task_spec.to_dict(),
    }


__all__ = ["describe_ldm_task", "main", "parse_args"]
