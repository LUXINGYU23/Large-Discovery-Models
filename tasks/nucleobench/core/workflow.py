"""Config-driven NucleoBench workflow using the shared LDM engine."""

from __future__ import annotations

import argparse
import json
import math
from argparse import Namespace
from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack
from pathlib import Path
from typing import Any

from tasks.nucleobench.core import harness_runtime, provider as provider_config

from ldm_tts.contracts import LDMTaskSpec
from ldm_tts.data import DataCollectionSink
from ldm_tts.engine import LDMEngine, LDMEngineConfig, LDMEngineState
from ldm_tts.engine.reporting import (
    build_campaign_result,
    build_trajectory_rows,
    load_successful_observations,
    read_json_object,
    write_trajectory_csv,
)
from ldm_tts.engine.run_store import CampaignRuntime, atomic_json_write, unique_run_dir
from ldm_tts.harness import (
    DockerPolicyExecutor,
    HarnessClient,
    HarnessError,
    HarnessProfile,
    PolicyResearchController,
    load_harness_mcp_config,
)
from ldm_tts.harness.container import resolve_container_user
from ldm_tts.optimization.records import AcquisitionSelector, SurrogateEncoder
from ldm_tts.registration.experiment import (
    ExperimentContract,
    load_active_experiment_contract,
    load_experiment_contract,
    snapshot_experiment_contract,
)
from ldm_tts.transport import ProposalClient
from ldm_tts.transport.openai_http import EndpointRequestError
from tasks.nucleobench.core.benchmark_clock import BenchmarkClock
from tasks.nucleobench.core.research import MEASURED_HISTORY_FILE, write_measured_history
from tasks.nucleobench.core.candidate import NucleoBenchCandidateDomain
from tasks.nucleobench.core.cases import NucleoBenchCase, get_case
from tasks.nucleobench.core.constants import (
    COMPILED_POLICY_METHOD,
    PERSISTENT_HARNESS_METHODS,
    TASK_ID,
    UPSTREAM_COMMIT,
)
from tasks.nucleobench.core.designer import (
    NucleoBenchDesigner,
    initialize_designer_state,
)
from tasks.nucleobench.core.digests import file_digest
from tasks.nucleobench.core.evaluator import NucleoBenchEvaluator
from tasks.nucleobench.core.factory import (
    build_proposal_expander,
    build_surrogate_components,
)
from tasks.nucleobench.core.forked_harness import (
    ForkedResearchExpander, main_research_profile,
)
from ldm_tts.harness.research_runtime import research_contract, write_fork_provenance
from tasks.nucleobench.core.mock import (
    MOCK_CASE,
    MOCK_CONTEXT,
    build_mock_expander,
    build_mock_task_spec,
    mock_energies,
)
from tasks.nucleobench.core.optimization_policy import (
    NucleoOptimizationPolicyAdapter,
    policy_harness_profile,
)
from tasks.nucleobench.core.policy_features import NucleoPolicyFeatureEncoder
from tasks.nucleobench.core.oracles.official import (
    PreparedCase,
    load_official_case,
    load_prepared_case,
)
from tasks.nucleobench.core.proposals import build_openai_mutation_client
from tasks.nucleobench.core.reporting import (
    inventory_official_outputs,
    write_campaign_reports,
)
from tasks.nucleobench.core.task_spec import build_task_spec
from tasks.nucleobench.core.workflow_args import (
    campaign_budget, gp_config, jsonable_args, method_preset_sha256, parse_args,
)

TASK_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_PROPOSALS_PER_ROUND = 1


def describe_ldm_task(args: argparse.Namespace) -> LDMTaskSpec:
    return (
        build_mock_task_spec()
        if args.mock
        else build_task_spec(
            get_case(args.case_id),
            search_method=args.search_method,
            evaluations_per_round=args.evaluations_per_round,
            proposal_samples=args.proposal_samples,
            proposal_max_workers=args.proposal_max_workers,
            harness_profile_count=len(args.harness_profile),
            harness_candidates_per_session=args.harness_candidates_per_session,
            harness_unique_candidates=args.harness_unique_candidates,
            bo_pool_size=args.bo_pool_size,
        )
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    case = MOCK_CASE if args.mock else get_case(args.case_id)
    task_spec = describe_ldm_task(args)
    contract, profile_name = _load_contract()
    if (
        not args.dry_run
        and not args.mock
        and args.execution_profile in {"pilot_evaluation", "official_benchmark"}
        and (contract.qualification != "qualified" or case.state != "qualified")
    ):
        raise SystemExit(
            f"NucleoBench case {case.case_id!r} is not qualified for pilot or "
            "official execution. Run the case qualification gates first."
        )
    provider = (
        provider_config.resolve_provider_settings(args)
        if not args.mock and args.search_method != "bo"
        else None
    )
    payload = _run_payload(
        args, case, task_spec, contract.digest, profile_name, provider
    )
    if args.dry_run:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if args.mock:
        return _run_mock(args, contract, profile_name, payload)
    return _run_real(args, case, contract, profile_name, provider, payload)


def _load_contract() -> tuple[ExperimentContract, str]:
    contract, profile_name = load_active_experiment_contract()
    return (
        (load_experiment_contract(TASK_ROOT / "experiment.json"), profile_name)
        if contract is None
        else (contract, profile_name)
    )


def _run_payload(
    args: argparse.Namespace,
    case: NucleoBenchCase,
    task_spec: LDMTaskSpec,
    contract_sha256: str,
    profile_name: str,
    provider: provider_config.ProviderSettings | None,
) -> dict[str, Any]:
    return {
        "task": TASK_ID,
        "mode": "mock" if args.mock else "real",
        "case_id": args.case_id,
        "case": case.to_dict(),
        "contract_profile": profile_name,
        "contract_sha256": contract_sha256,
        "ldm_task_spec": task_spec.to_dict(),
        "proposal_mode": args.proposal_mode,
        "search_method": args.search_method,
        "execution": _execution_summary(args),
        "proposal_provider": (
            {"required": False}
            if provider is None
            else {
                "required": True,
                "configured": bool(
                    provider.base_url and provider.model and provider.api_key
                ),
                "wire_api": (
                    "responses"
                    if args.search_method in PERSISTENT_HARNESS_METHODS
                    else args.llm_wire_api
                ),
                "reasoning": (
                    args.harness_thinking
                    if args.search_method in PERSISTENT_HARNESS_METHODS
                    else args.llm_reasoning
                ),
            }
        ),
        "harness": harness_runtime.describe(args),
    }


def _execution_summary(args: argparse.Namespace) -> dict[str, Any]:
    total_rounds = args.iterations if args.termination_kind == "rounds" else None
    return {
        "execution_profile": args.execution_profile,
        "termination_kind": args.termination_kind,
        "total_rounds": total_rounds,
        "active_optimization_rounds": (
            None if total_rounds is None else max(0, total_rounds - 1)
        ),
        "initialization_evaluations": 1,
        "benchmark_comparable": args.execution_profile == "official_benchmark" and args.resume_from is None,
        "hardware_profile": args.hardware_profile,
        "oracle_batch_size": args.oracle_batch_size,
    }


def _run_mock(
    args: argparse.Namespace,
    contract: ExperimentContract,
    profile_name: str,
    payload: dict[str, Any],
) -> int:
    run_dir = unique_run_dir(args.out_dir / (args.run_name or "mock"))
    task_spec = build_mock_task_spec()
    assert args.iterations is not None
    evaluation_limit = args.iterations * args.evaluations_per_round
    runtime = CampaignRuntime.open(
        run_dir,
        task=TASK_ID,
        config=jsonable_args(args),
        task_spec=task_spec,
        budget_limits={
            "outer_iterations": args.iterations,
            "valid_search_candidates": 1 + (args.iterations - 1) * args.reservoir_size,
            "selected_candidates": evaluation_limit,
            "external_evaluations": evaluation_limit,
            "expensive_evaluation_attempts": evaluation_limit,
        },
        contract_snapshot=contract.to_dict(),
        contract_sha256=contract.digest,
        contract_profile=profile_name,
    )
    snapshot_experiment_contract(contract, run_dir, profile=profile_name)
    sink = DataCollectionSink.from_env(default_root=run_dir / "ldm_data")
    engine = LDMEngine(
        task_spec=task_spec,
        expander=build_mock_expander(),
        candidate_domain=NucleoBenchCandidateDomain(MOCK_CONTEXT, sink=sink),
        evaluator=NucleoBenchEvaluator(MOCK_CONTEXT, mock_energies),
        runtime=runtime,
    )
    result = engine.run(
        LDMEngineConfig(
            iterations=args.iterations,
            reservoir_size=args.reservoir_size,
            evaluations_per_round=args.evaluations_per_round,
        )
    )
    observations = load_successful_observations(run_dir / "checkpoint.json")
    trajectory = build_trajectory_rows(
        observations,
        objective_name="utility",
        direction="maximize",
    )
    write_trajectory_csv(
        run_dir / "trajectory.csv",
        trajectory,
        fieldnames=("evaluation", "round", "candidate_id", "utility", "best_utility"),
    )
    report = {
        **build_campaign_result(
            run_dir, objective_name="utility", direction="maximize"
        ),
        "mode": "mock",
        "case_id": MOCK_CASE.case_id,
        "engine_summary": result.summary,
        "artifacts": {"trajectory": "trajectory.csv", "summary": "summary.json"},
    }
    atomic_json_write(run_dir / "result.json", report)
    payload.update(run_dir=str(run_dir.resolve()), result=report)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if result.summary["successful_evaluation_count"] else 1


def _run_real(
    args: argparse.Namespace,
    case: NucleoBenchCase,
    contract: ExperimentContract,
    profile_name: str,
    provider: provider_config.ProviderSettings | None,
    payload: dict[str, Any],
) -> int:
    if args.source_dir is None or args.prepared_dir is None:
        raise SystemExit(
            "Real NucleoBench execution requires --source-dir and --prepared-dir."
        )
    expected_digest = (args.expected_start_set_sha256 or "").strip()
    if len(expected_digest) != 64:
        raise SystemExit(
            "Real NucleoBench execution requires --expected-start-set-sha256."
        )
    if (
        args.execution_profile == "official_benchmark"
        and args.max_seconds != case.max_seconds
    ):
        raise SystemExit(
            f"Official {case.case_id} execution requires "
            f"--max-seconds={case.max_seconds}."
        )
    expected_profile = {
        "pilot_evaluation": "pilot_evaluation",
        "official_benchmark": "official_benchmark",
    }.get(args.execution_profile)
    if expected_profile is not None and profile_name != expected_profile:
        raise SystemExit(
            f"{args.execution_profile} requires contract profile {expected_profile!r}."
        )

    prepared = load_prepared_case(
        args.prepared_dir,
        case_id=case.case_id,
        start_index=args.start_index,
    )
    if prepared.context.start_set_digest != expected_digest:
        raise SystemExit(
            "Prepared start-set digest does not match the configured digest."
        )
    if provider is not None:
        args.llm_url = provider.base_url
        args.llm_model_name = provider.model
    encoder, selector = build_surrogate_components(
        args.search_method,
        prepared.context,
        evaluations_per_round=args.evaluations_per_round,
        proposal_samples=args.proposal_samples,
        bo_pool_size=args.bo_pool_size,
        seed=args.campaign_index,
        gp_config=gp_config(args),
        alpha=args.alpha,
        eta=args.eta,
        z_clip=args.z_clip,
        measurement_policy=args._research_strategy.measurement_policy,
    )
    task_spec = build_task_spec(
        prepared.context.case,
        search_method=args.search_method,
        evaluations_per_round=args.evaluations_per_round,
        proposal_samples=args.proposal_samples,
        proposal_max_workers=args.proposal_max_workers,
        harness_profile_count=len(args.harness_profile),
        harness_candidates_per_session=args.harness_candidates_per_session,
        harness_unique_candidates=args.harness_unique_candidates,
        bo_pool_size=args.bo_pool_size,
        acquisition=None if selector is None else selector.describe(),
        surrogate=None if encoder is None else encoder.describe(),
    )
    payload["ldm_task_spec"] = task_spec.to_dict()
    runtime = _open_runtime(args, task_spec, contract, profile_name)
    benchmark_clock = (
        BenchmarkClock(args.max_seconds) if args.termination_kind == "wall_time" else None
    )
    if benchmark_clock is not None and args.resume_from is not None:
        benchmark_clock = BenchmarkClock.resume(args.max_seconds, runtime.events())
        if benchmark_clock.remaining_before_start <= 0:
            raise ValueError("The original wall-time budget is exhausted")
    client: ProposalClient | None = None
    harness_client: HarnessClient | None = None
    forked_expander = None
    stack = ExitStack()
    profiles = harness_runtime.proposal_profiles(args)
    if args.search_method != "bo":
        assert provider is not None
        missing = _missing_provider(provider)
        if missing:
            return _pause_endpoint(runtime, args, payload, missing)
    if args.search_method in {"ldm", "llm"}:
        assert provider is not None
        client = _proposal_client(args, provider)
        try:
            preflight = client.preflight()
        except EndpointRequestError as exc:
            return _pause_endpoint(runtime, args, payload, str(exc))
        runtime.record("endpoint_preflight_succeeded", preflight)
        payload["endpoint_preflight"] = preflight
    elif args.search_method in PERSISTENT_HARNESS_METHODS:
        assert provider is not None
        try:
            mcp = load_harness_mcp_config(args.harness_mcp_config)
            strategy = args._research_strategy
            if strategy.requires_main:
                (runtime.run_dir / "harness" / MEASURED_HISTORY_FILE).parent.mkdir(parents=True, exist_ok=True)
                forked_expander = ForkedResearchExpander(
                    root=runtime.run_dir / "harness",
                    main_client=lambda: harness_runtime.create_client(
                        args, runtime, provider, prepared, main_research_profile(), mcp,
                        artifact_root=runtime.run_dir / "harness/main", contract=research_contract(),
                    ),
                    branch_client=lambda root, source: harness_runtime.create_client(
                        args, runtime, provider, prepared, profiles, mcp,
                        artifact_root=root, fork_sources=source,
                    ),
                    strategy=strategy,
                    expander_options={
                        "domain": NucleoBenchCandidateDomain(prepared.context), "profiles": profiles,
                        "candidates_per_profile": args.harness_candidates_per_session,
                        "campaign_id": runtime.run_id, "first_active_round": 1,
                        "attach_empirical_q0": True, "allow_repeated_occurrences": not args.harness_unique_candidates,
                        "account": runtime.consume_many, "benchmark_clock": benchmark_clock,
                        "surrogate_query_config": gp_config(args) if args.harness_surrogate_query else None,
                    },
                )
            else:
                harness_client = stack.enter_context(
                    harness_runtime.create_client(args, runtime, provider, prepared, profiles, mcp)
                )
            if args.search_method == COMPILED_POLICY_METHOD:
                policy_adapter = NucleoOptimizationPolicyAdapter(
                    NucleoPolicyFeatureEncoder(prepared.context),
                    seed=args.campaign_index,
                    gp_config=gp_config(args),
                    default_alpha=args.alpha,
                    default_eta=args.eta,
                    enabled_capabilities=tuple(args.policy_capability),
                    proposal_sampling={
                        "session_count": len(profiles),
                        "candidates_per_session": args.harness_candidates_per_session,
                        "within_session_repeats_allowed": not args.harness_unique_candidates,
                        "cross_session_agreement_allowed": True,
                    },
                    evaluations_per_round=args.evaluations_per_round,
                    benchmark_clock=benchmark_clock,
                    measured_history_path=runtime.run_dir / "harness" / MEASURED_HISTORY_FILE,
                )
                policy_client = stack.enter_context(
                    harness_runtime.create_client(
                        args, runtime, provider, prepared, policy_harness_profile(), mcp,
                        policy=True,
                    )
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
                    recovery_budget=lambda: (
                        float(benchmark_clock.snapshot()["remaining_seconds"])
                        if benchmark_clock is not None else 2.0 * args.harness_wall_time_seconds
                    ),
                )
                encoder, selector = build_surrogate_components(
                    args.search_method,
                    prepared.context,
                    evaluations_per_round=args.evaluations_per_round,
                    proposal_samples=args.proposal_samples,
                    bo_pool_size=args.bo_pool_size,
                    seed=args.campaign_index,
                    gp_config=gp_config(args),
                    alpha=args.alpha,
                    eta=args.eta,
                    z_clip=args.z_clip,
                    policy_controller=policy_controller,
                    policy_adapter=policy_adapter,
                    measurement_policy=args._research_strategy.measurement_policy,
                )
        except HarnessError as exc:
            stack.close()
            return _pause_endpoint(runtime, args, payload, str(exc))

    try:
        official = load_official_case(args.source_dir, prepared, runtime)
        state = _campaign_state(runtime, args.resume_from is not None)
        evaluator = NucleoBenchEvaluator(
            prepared.context, official.model, batch_size=args.oracle_batch_size
        )
        if not state.observations:
            runtime.consume("outer_iterations")
            state = initialize_designer_state(prepared.context, evaluator, runtime)
        engine = _real_engine(
            args,
            prepared,
            task_spec,
            evaluator,
            runtime,
            encoder,
            selector,
            profiles,
            client,
            harness_client,
            benchmark_clock,
            forked_expander,
        )
        designer = NucleoBenchDesigner(
            engine=engine,
            state=state,
            context=prepared.context,
            reservoir_size=args.proposal_samples,
            evaluations_per_step=args.evaluations_per_round,
        )
        method_digest = method_preset_sha256(args, task_spec)
        oracle_manifest = _oracle_manifest(args, prepared)
        expected_active_steps = None
        termination_args: dict[str, int] | None
        if args.termination_kind == "rounds":
            assert args.iterations is not None
            remaining_steps = args.iterations - state.next_round
            if remaining_steps < 0:
                raise ValueError(
                    "checkpoint already exceeds the configured round limit"
                )
            if remaining_steps == 0:
                report = _finish_completed_resume(runtime)
                termination_args = None
            else:
                expected_active_steps = args.iterations - 1
                termination_args = {"max_number_of_rounds": remaining_steps}
        else:
            assert args.max_seconds is not None
            assert benchmark_clock is not None
            termination_args = {"max_seconds": math.ceil(benchmark_clock.remaining_before_start)}

        if termination_args is not None:
            all_args = build_official_runner_args(
                official.parsed_args_type,
                model_name=case.model_name,
                optimization_name=f"ldm_tts_{args.search_method}",
                start_sequence=prepared.context.start_sequence,
                positions_to_mutate=prepared.context.editable_positions,
                output_path=_official_output_dir(runtime, state.next_round),
                proposals_per_round=OFFICIAL_PROPOSALS_PER_ROUND,
                model_init_args=prepared.model_init_args,
                optimizer_init_args=_optimizer_manifest(args),
                **termination_args,
            )
            report = run_official_driver(
                run_loop=official.run_loop,
                model=(designer.measured_sample_energies
                       if args.termination_kind == "rounds" else official.model),
                designer=designer,
                all_args=all_args,
                runtime=runtime,
                execution=lambda active_steps: _execution_record(
                    args, prepared, method_digest, active_steps
                ),
                oracle_manifest=oracle_manifest,
                expected_active_steps=expected_active_steps,
                benchmark_clock=benchmark_clock,
            )
        if harness_client is not None or forked_expander is not None:
            write_measured_history(runtime.run_dir / "harness", designer.state.observations, prepared.context)
        if forked_expander is not None:
            write_fork_provenance(runtime.run_dir / "harness", compiled_policy=args.search_method == COMPILED_POLICY_METHOD)
    except Exception as exc:
        status = json.loads(runtime.status.path.read_text(encoding="utf-8"))
        if status.get("status") != "failed":
            runtime.fail(exc)
        raise
    finally:
        stack.close()

    payload.update(run_dir=str(runtime.run_dir.resolve()), result=report)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if report["evaluation_count"] else 1


def _real_engine(
    args: argparse.Namespace,
    prepared: PreparedCase,
    task_spec: LDMTaskSpec,
    evaluator: NucleoBenchEvaluator,
    runtime: CampaignRuntime,
    encoder: SurrogateEncoder | None,
    selector: AcquisitionSelector | None,
    profiles: Sequence[HarnessProfile],
    client: ProposalClient | None,
    harness_client: HarnessClient | None,
    benchmark_clock: BenchmarkClock | None = None,
    forked_expander: ForkedResearchExpander | None = None,
) -> LDMEngine:
    expander = forked_expander or build_proposal_expander(
        args.search_method,
        prepared.context,
        seed=args.campaign_index,
        evaluations_per_round=args.evaluations_per_round,
        client=client,
        harness_client=harness_client,
        harness_artifact_root=runtime.run_dir / "harness",
        harness_session_profiles=profiles,
        harness_candidates_per_session=args.harness_candidates_per_session,
        harness_unique_candidates=args.harness_unique_candidates,
        benchmark_clock=benchmark_clock,
        surrogate_query_config=gp_config(args) if args.harness_surrogate_query else None,
        campaign_id=runtime.run_id,
        first_active_round=1,
        max_workers=args.proposal_max_workers,
        before_requests=(
            (lambda count: runtime.consume("llm_requests", count))
            if args.search_method in {"ldm", "llm"}
            else None
        ),
        account=(
            runtime.consume_many
            if args.search_method in PERSISTENT_HARNESS_METHODS
            else None
        ),
    )
    domain = NucleoBenchCandidateDomain(
        prepared.context,
        sink=DataCollectionSink.from_env(default_root=runtime.run_dir / "ldm_data"),
    )
    return LDMEngine(
        task_spec=task_spec,
        expander=expander,
        candidate_domain=domain,
        evaluator=evaluator,
        runtime=runtime,
        selector=selector,
        surrogate_encoder=encoder,
    )


def _open_runtime(
    args: argparse.Namespace,
    task_spec: LDMTaskSpec,
    contract: ExperimentContract,
    profile_name: str,
) -> CampaignRuntime:
    run_dir = (
        args.resume_from.resolve()
        if args.resume_from is not None
        else unique_run_dir(args.out_dir / (args.run_name or _default_run_name(args)))
    )
    profile = contract.profile(profile_name) if profile_name else None
    runtime = CampaignRuntime.open(
        run_dir,
        task=TASK_ID,
        config=jsonable_args(args),
        task_spec=task_spec,
        budget_limits=campaign_budget(
            args, None if profile is None else profile.budget
        ),
        contract_snapshot=contract.to_dict(),
        contract_sha256=contract.digest,
        contract_profile=profile_name,
        resume=args.resume_from is not None,
    )
    if args.resume_from is None:
        snapshot_experiment_contract(contract, run_dir, profile=profile_name)
    return runtime


def _campaign_state(runtime: CampaignRuntime, resume: bool) -> LDMEngineState:
    if not resume:
        return LDMEngineState()
    checkpoint = runtime.load_checkpoint()
    return (
        LDMEngineState()
        if checkpoint is None
        else LDMEngineState.from_checkpoint(checkpoint)
    )


def _finish_completed_resume(runtime: CampaignRuntime) -> dict[str, Any]:
    summary = read_json_object(runtime.run_dir / "summary.json")
    result = read_json_object(runtime.run_dir / "result.json")
    runtime.finish(summary)
    return result


def _default_run_name(args: argparse.Namespace) -> str:
    return (
        f"{args.execution_profile}_{args.case_id}_{args.search_method}_"
        f"start_{args.start_index}_seed_{args.campaign_index}"
    )


def _official_output_dir(runtime: CampaignRuntime, next_round: int) -> Path:
    return runtime.run_dir / "official" / f"attempt_{next_round:06d}"


def _proposal_client(
    args: argparse.Namespace,
    provider: provider_config.ProviderSettings,
) -> Any:
    return build_openai_mutation_client(
        base_url=provider.base_url,
        model=provider.model,
        api_key=provider.api_key,
        wire_api=args.llm_wire_api,
        timeout_seconds=args.llm_timeout,
        max_tokens=args.llm_max_tokens,
        temperature=args.llm_temperature,
        reasoning=args.llm_reasoning,
        extra_body=provider_config.parse_extra_body(args.llm_extra_body_json),
    )


def _missing_provider(provider: provider_config.ProviderSettings) -> str:
    missing = [
        name
        for name, value in (
            ("LLM_BASE_URL", provider.base_url),
            ("LLM_MODEL_NAME", provider.model),
            ("LLM_API_KEY or --api-key-file", provider.api_key),
        )
        if not value
    ]
    return "Set " + ", ".join(missing) + "." if missing else ""


def _pause_endpoint(
    runtime: CampaignRuntime,
    args: argparse.Namespace,
    payload: dict[str, Any],
    message: str,
) -> int:
    runtime.pause(
        "paused_endpoint_unavailable",
        phase="endpoint_preflight",
        message=message,
        details={"model": args.llm_model_name},
    )
    payload.update(
        run_dir=str(runtime.run_dir.resolve()),
        status="paused_endpoint_unavailable",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 2


def _execution_record(
    args: argparse.Namespace,
    prepared: PreparedCase,
    method_digest: str,
    active_steps: int,
) -> dict[str, Any]:
    return {
        "execution_profile": args.execution_profile,
        "termination_kind": args.termination_kind,
        "total_rounds": active_steps + 1,
        "active_optimization_rounds": active_steps,
        "initialization_evaluations": 1,
        "benchmark_comparable": args.execution_profile == "official_benchmark" and args.resume_from is None,
        "case_id": args.case_id,
        "start_index": args.start_index,
        "start_set_digest": prepared.context.start_set_digest,
        "optimization_seed": args.campaign_index,
        "hardware_profile": args.hardware_profile,
        "search_method": args.search_method,
        "method_preset_sha256": method_digest,
        "max_seconds": args.max_seconds,
    }


def _oracle_manifest(
    args: argparse.Namespace,
    prepared: PreparedCase,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "benchmark": {
            "name": "NucleoBench",
            "source_commit": UPSTREAM_COMMIT,
        },
        "case": prepared.context.case.to_dict(),
        "paired_start": {
            "start_set_sha256": prepared.context.start_set_digest,
            "start_index": prepared.context.start_index,
        },
        "oracle": {
            "model_name": prepared.context.case.model_name,
            "model_artifact_sha256": file_digest(prepared.model_artifact),
            "model_init_args": prepared.model_init_args,
            "inference_batch_size": args.oracle_batch_size,
        },
        "official_runner": {
            "proposals_per_round": OFFICIAL_PROPOSALS_PER_ROUND,
            "termination_kind": args.termination_kind,
            "max_seconds": args.max_seconds,
        },
        "hardware_profile": args.hardware_profile,
    }


def _optimizer_manifest(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "search_method": args.search_method,
        "evaluations_per_step": args.evaluations_per_round,
        "proposal_samples": args.proposal_samples,
        "seed": args.campaign_index,
    }


def build_official_runner_args(
    parsed_args_type: type,
    *,
    model_name: str,
    optimization_name: str,
    start_sequence: str,
    positions_to_mutate: Sequence[int] | None,
    output_path: Path,
    proposals_per_round: int,
    max_seconds: int | None = None,
    max_number_of_rounds: int | None = None,
    model_init_args: Mapping[str, Any] | None = None,
    optimizer_init_args: Mapping[str, Any] | None = None,
) -> Any:
    if (max_seconds is None) == (max_number_of_rounds is None):
        raise ValueError("official runner requires exactly one termination mode")
    if max_seconds is not None and max_seconds < 1:
        raise ValueError("max_seconds must be positive")
    if max_number_of_rounds is not None and max_number_of_rounds < 1:
        raise ValueError("max_number_of_rounds must be positive")
    if proposals_per_round < 1:
        raise ValueError("proposals_per_round must be positive")
    main_args = Namespace(
        model=model_name,
        optimization=optimization_name,
        start_sequence=start_sequence,
        positions_to_mutate=(
            None if positions_to_mutate is None else list(positions_to_mutate)
        ),
        output_path=str(Path(output_path)),
        optimization_steps_per_output=1,
        proposals_per_round=proposals_per_round,
        max_seconds=max_seconds,
        max_number_of_rounds=max_number_of_rounds,
        ignore_errors=False,
        ignore_empty_cmd_args=False,
    )
    return parsed_args_type(
        main_args=main_args,
        model_init_args=Namespace(**dict(model_init_args or {})),
        opt_init_args=Namespace(**dict(optimizer_init_args or {})),
    )


def run_official_driver(
    *,
    run_loop: Callable[..., Any],
    model: Callable[[Sequence[str]], Any],
    designer: NucleoBenchDesigner,
    all_args: Any,
    runtime: CampaignRuntime,
    execution: Mapping[str, Any] | Callable[[int], Mapping[str, Any]],
    oracle_manifest: Mapping[str, Any],
    expected_active_steps: int | None = None,
    benchmark_clock: BenchmarkClock | None = None,
) -> dict[str, Any]:
    try:
        if benchmark_clock is not None:
            # Start just before the unchanged runner, conservatively including entry overhead.
            runtime.record(
                "benchmark_clock_resumed" if benchmark_clock.elapsed_before_start else "benchmark_clock_started",
                benchmark_clock.start(),
            )
        run_loop(model=model, opt=designer, all_args=all_args, ignore_errors=False)
        if (
            expected_active_steps is not None
            and designer.active_steps != expected_active_steps
        ):
            raise RuntimeError(
                "official runner completed a different number of active steps: "
                f"{designer.active_steps} != {expected_active_steps}"
            )
        official_outputs = inventory_official_outputs(
            Path(all_args.main_args.output_path),
            runtime.run_dir,
        )
    except Exception as exc:
        status = json.loads(runtime.status.path.read_text(encoding="utf-8"))
        if status.get("status") != "failed":
            runtime.fail(exc)
        raise

    summary = {
        **designer.summary(),
        "stop_reason": "official_driver_finished",
        "official_output_count": len(official_outputs),
    }
    runtime.finish(summary)
    resolved_execution = (
        execution(designer.active_steps) if callable(execution) else execution
    )
    return write_campaign_reports(
        runtime,
        execution=resolved_execution,
        oracle_manifest=oracle_manifest,
        official_outputs=official_outputs,
    )


__all__ = [
    "build_official_runner_args",
    "describe_ldm_task",
    "main",
    "parse_args",
    "run_official_driver",
]
