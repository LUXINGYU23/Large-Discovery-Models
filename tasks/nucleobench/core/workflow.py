"""Config-driven NucleoBench workflow using the shared LDM engine."""

from __future__ import annotations

import argparse
import json
import math
import os
from argparse import Namespace
from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ldm_tts.harness.pi import DEFAULT_NETWORK_TOOL_BUDGETS, PiHarnessConfig, policy_mcp_server
from ldm_tts.contracts import LDMTaskSpec
from ldm_tts.harness.protocol import file_sha256
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
    POLICY_CAPABILITIES,
    DockerPolicyExecutor,
    HarnessClient,
    HarnessError,
    HarnessLimits,
    HarnessNetworkPolicy,
    HarnessProfile,
    PolicyResearchController,
    load_harness_mcp_config,
    parse_tool_call_budgets,
    policy_submission_contract,
)
from ldm_tts.harness.container import docker_identity_args, resolve_container_user
from ldm_tts.optimization.records import AcquisitionSelector, SurrogateEncoder
from ldm_tts.registration.experiment import (
    ExperimentContract,
    load_active_experiment_contract,
    load_experiment_contract,
    snapshot_experiment_contract,
)
from ldm_tts.transport import ProposalClient
from ldm_tts.transport.openai import WIRE_APIS
from ldm_tts.transport.openai_http import EndpointRequestError
from tasks.nucleobench.core.benchmark_clock import BenchmarkClock
from tasks.nucleobench.core.research import MEASURED_HISTORY_FILE, write_measured_history
from tasks.nucleobench.core.candidate import NucleoBenchCandidateDomain
from tasks.nucleobench.core.cases import NucleoBenchCase, get_case
from tasks.nucleobench.core.constants import (
    COMPILED_POLICY_METHOD,
    DEFAULT_LDM_ALPHA,
    DEFAULT_LDM_ETA,
    PARALLEL_HARNESS_METHODS,
    PERSISTENT_HARNESS_METHODS,
    SEARCH_METHODS,
    TASK_ID,
    UPSTREAM_COMMIT,
)
from tasks.nucleobench.core.designer import (
    NucleoBenchDesigner,
    initialize_designer_state,
)
from tasks.nucleobench.core.digests import canonical_json_sha256, file_digest
from tasks.nucleobench.core.evaluator import NucleoBenchEvaluator
from tasks.nucleobench.core.factory import (
    build_proposal_expander,
    build_surrogate_components,
)
from tasks.nucleobench.core.hamming_gp import HammingGPUCBConfig
from tasks.nucleobench.core.harness import (
    AVAILABLE_HARNESS_PROFILE_IDS,
    DIRECT_HARNESS_PROFILE_ID,
    HARNESS_FORBIDDEN_PATTERNS,
    HARNESS_PROFILE_IDS,
    HARNESS_SKILL_IDS,
    direct_harness_profile,
    harness_guest_runtime,
    harness_profiles,
    harness_submission_contract,
    harness_tool_extensions,
    write_harness_sequence_context,
)
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
from tasks.nucleobench.core.proposals import (
    DEFAULT_PROPOSAL_MAX_WORKERS,
    build_openai_mutation_client,
    direct_request_limit,
)
from tasks.nucleobench.core.reporting import (
    inventory_official_outputs,
    write_campaign_reports,
)
from tasks.nucleobench.core.task_spec import build_task_spec

TASK_ROOT = Path(__file__).resolve().parents[1]
EXECUTION_PROFILES = ("qualification", "pilot_evaluation", "official_benchmark")
TERMINATION_KINDS = ("rounds", "wall_time")
INITIALIZATION_MODES = ("shared_start", "official_start")
OFFICIAL_PROPOSALS_PER_ROUND = 1
DEFAULT_HARDWARE_PROFILE = "n1-highmem-16-cpu"
LLM_REASONING_LEVELS = (
    "off",
    "none",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
    "max",
)
HARNESS_THINKING_LEVELS = tuple(
    level for level in LLM_REASONING_LEVELS if level != "none"
)


@dataclass(frozen=True)
class ProviderSettings:
    base_url: str
    model: str
    api_key: str


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", default="malinois_k562")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument(
        "--execution-profile", choices=EXECUTION_PROFILES, default="qualification"
    )
    parser.add_argument(
        "--termination-kind", choices=TERMINATION_KINDS, default="rounds"
    )
    parser.add_argument(
        "--initialization-mode", choices=INITIALIZATION_MODES, default="shared_start"
    )
    parser.add_argument(
        "--search-method", choices=tuple(sorted(SEARCH_METHODS)), default="ldm"
    )
    parser.add_argument("--proposal-mode", choices=("none", "openai"))
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--max-seconds", type=int)
    parser.add_argument("--reservoir-size", type=int, default=4)
    parser.add_argument("--evaluations-per-round", type=int, default=1)
    parser.add_argument("--oracle-batch-size", type=int)
    parser.add_argument("--proposal-samples", type=int)
    parser.add_argument("--bo-pool-size", type=int)
    parser.add_argument("--proposal-candidates-per-request", type=int)
    parser.add_argument(
        "--proposal-max-workers", type=int, default=DEFAULT_PROPOSAL_MAX_WORKERS
    )
    parser.add_argument("--campaign-index", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--prepared-dir", type=Path)
    parser.add_argument("--expected-start-set-sha256")
    parser.add_argument("--hardware-profile", default=DEFAULT_HARDWARE_PROFILE)
    parser.add_argument("--acquisition-beta", type=float, default=1.0)
    parser.add_argument("--gp-noise-variance", type=float, default=1.0e-4)
    parser.add_argument("--gp-jitter", type=float, default=1.0e-8)
    parser.add_argument("--gp-target-std-floor", type=float, default=0.1)
    parser.add_argument("--gp-min-history-for-fit", type=int, default=4)
    parser.add_argument("--gp-max-observations", type=int, default=256)
    parser.add_argument("--gp-global-best-observations", type=int, default=64)
    parser.add_argument("--alpha", type=float, default=DEFAULT_LDM_ALPHA)
    parser.add_argument("--eta", type=float, default=DEFAULT_LDM_ETA)
    parser.add_argument("--z-clip", type=float, default=5.0)
    parser.add_argument("--llm-url")
    parser.add_argument("--llm-model-name")
    parser.add_argument("--llm-wire-api", choices=WIRE_APIS, default="chat_completions")
    parser.add_argument("--llm-reasoning", choices=LLM_REASONING_LEVELS, default="off")
    parser.add_argument("--llm-timeout", type=float, default=120.0)
    parser.add_argument("--llm-max-tokens", type=int, default=262_144)
    parser.add_argument("--llm-temperature", type=float, default=0.7)
    parser.add_argument("--llm-extra-body-json", default="{}")
    parser.add_argument("--api-key-file", type=Path)
    parser.add_argument("--harness-sidecar-image", default="ldm-pi-harness:latest")
    parser.add_argument("--harness-candidates-per-session", type=int)
    parser.add_argument("--harness-unique-candidates", action="store_true")
    parser.add_argument(
        "--harness-profile", action="append", choices=AVAILABLE_HARNESS_PROFILE_IDS
    )
    parser.add_argument(
        "--harness-thinking", choices=HARNESS_THINKING_LEVELS, default="max"
    )
    parser.add_argument("--harness-mcp-config", type=Path)
    parser.add_argument("--harness-cache-dir", type=Path)
    parser.add_argument("--harness-docker-host")
    parser.add_argument("--harness-container-user")
    parser.add_argument("--harness-response-timeout", type=float, default=2100.0)
    parser.add_argument("--harness-wall-time-seconds", type=int, default=1800)
    parser.add_argument("--harness-tool-budget", action="append", metavar="NAME=COUNT")
    parser.add_argument(
        "--policy-capability",
        action="append",
        choices=tuple(sorted(POLICY_CAPABILITIES)),
    )
    parser.add_argument("--policy-tool-budget", action="append", metavar="NAME=COUNT")
    parser.add_argument("--policy-max-submission-attempts", type=int)
    parser.add_argument(
        "--no-harness-context7",
        action="store_false",
        dest="harness_context7",
        default=True,
    )
    parser.add_argument("--out-dir", type=Path, default=Path("runs"))
    parser.add_argument("--run-name")
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    _apply_derived_args(args)
    _validate_args(args, parser)
    return args


def _apply_derived_args(args: argparse.Namespace) -> None:
    if args.iterations is None and args.termination_kind == "rounds":
        args.iterations = 2
    if args.oracle_batch_size is None:
        case = MOCK_CASE if args.mock else get_case(args.case_id)
        args.oracle_batch_size = (
            4 if case.model_family == "enformer" else args.evaluations_per_round
        )
    if args.proposal_mode is None:
        args.proposal_mode = (
            "openai" if args.search_method in {"ldm", "llm"} else "none"
        )
    if args.harness_profile is None:
        args.harness_profile = list(HARNESS_PROFILE_IDS)
    if args.harness_candidates_per_session is None:
        args.harness_candidates_per_session = args.evaluations_per_round
    if args.bo_pool_size is None:
        args.bo_pool_size = 3 * args.evaluations_per_round
    if args.proposal_samples is None and args.search_method in PARALLEL_HARNESS_METHODS:
        args.proposal_samples = (
            len(args.harness_profile) * args.harness_candidates_per_session
        )
    elif args.proposal_samples is None:
        args.proposal_samples = args.evaluations_per_round * (
            1 if args.search_method in {"llm", "harness"} else 4
        )
    if args.proposal_candidates_per_request is None:
        args.proposal_candidates_per_request = (
            args.evaluations_per_round if args.search_method == "ldm" else 1
        )
    args.proposal_request_limit = (
        direct_request_limit(args.search_method, args.evaluations_per_round)
        if args.search_method in {"ldm", "llm"}
        else 0
    )
    if (
        args.search_method in PERSISTENT_HARNESS_METHODS
        and args.harness_container_user is None
    ):
        container_user = resolve_container_user(None, args.harness_docker_host)
        if container_user:
            args.harness_container_user = container_user
    args.initialization_evaluations = 1
    if args.search_method == COMPILED_POLICY_METHOD:
        if args.policy_tool_budget is None:
            args.policy_tool_budget = list(DEFAULT_NETWORK_TOOL_BUDGETS)
        if args.policy_capability is None:
            args.policy_capability = sorted(POLICY_CAPABILITIES)
        if args.policy_max_submission_attempts is None:
            args.policy_max_submission_attempts = 3


def _validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    positive_counts = (
        "reservoir_size",
        "evaluations_per_round",
        "oracle_batch_size",
        "proposal_samples",
        "bo_pool_size",
        "proposal_candidates_per_request",
        "proposal_max_workers",
        "gp_min_history_for_fit",
        "gp_max_observations",
        "gp_global_best_observations",
        "llm_max_tokens",
        "harness_candidates_per_session",
        "harness_wall_time_seconds",
    )
    if any(getattr(args, name) < 1 for name in positive_counts):
        parser.error(
            "proposal, GP, token, Harness, and evaluation counts must be positive"
        )
    if args.campaign_index < 0 or args.start_index < 0:
        parser.error("--campaign-index and --start-index must be non-negative")
    finite_positive = (
        "llm_timeout",
        "harness_response_timeout",
        "gp_noise_variance",
        "gp_jitter",
        "gp_target_std_floor",
        "z_clip",
    )
    if any(
        not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0
        for name in finite_positive
    ):
        parser.error("timeouts, GP scales, and --z-clip must be finite and positive")
    for name in ("acquisition_beta", "alpha", "eta"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and non-negative")
    if not math.isfinite(args.llm_temperature) or not 0 <= args.llm_temperature <= 2:
        parser.error("--llm-temperature must be finite and between 0 and 2")
    if args.gp_global_best_observations > args.gp_max_observations:
        parser.error(
            "--gp-global-best-observations cannot exceed --gp-max-observations"
        )
    if args.termination_kind == "rounds":
        if (
            args.iterations is None
            or args.iterations < 1
            or args.max_seconds is not None
        ):
            parser.error(
                "round termination requires positive --iterations and prohibits "
                "--max-seconds"
            )
    elif (
        args.iterations is not None or args.max_seconds is None or args.max_seconds < 1
    ):
        parser.error(
            "wall-time termination requires positive --max-seconds and prohibits "
            "--iterations"
        )
    if args.execution_profile == "pilot_evaluation" and (
        args.termination_kind != "rounds"
        or args.iterations != 12
        or args.initialization_mode != "shared_start"
    ):
        parser.error(
            "pilot_evaluation requires 12 rounds and shared_start initialization"
        )
    if args.execution_profile == "official_benchmark" and (
        args.termination_kind != "wall_time"
        or args.initialization_mode != "official_start"
    ):
        parser.error("official_benchmark requires wall_time and official_start")
    if args.execution_profile == "qualification" and args.termination_kind != "rounds":
        parser.error("qualification runs use round termination")
    expected_samples = args.evaluations_per_round * (
        1 if args.search_method in {"llm", "harness"} else 4
    )
    if args.search_method in PARALLEL_HARNESS_METHODS:
        expected_samples = len(args.harness_profile) * args.harness_candidates_per_session
    if args.search_method not in PARALLEL_HARNESS_METHODS and (
        tuple(args.harness_profile) != HARNESS_PROFILE_IDS or args.harness_unique_candidates
    ):
        parser.error("Custom Harness sampling settings require a parallel Harness method")
    if args.proposal_samples != expected_samples:
        parser.error(
            f"--proposal-samples must equal {expected_samples} for {args.search_method}"
        )
    if args.search_method == "ldm" or args.search_method in PARALLEL_HARNESS_METHODS:
        if not args.evaluations_per_round <= args.bo_pool_size < args.proposal_samples:
            parser.error("LDM requires evaluation batch <= BO pool < proposal samples")
    elif args.bo_pool_size != 3 * args.evaluations_per_round:
        parser.error("--bo-pool-size requires an LDM method")
    expected_request_size = (
        args.evaluations_per_round if args.search_method == "ldm" else 1
    )
    if (
        args.search_method in {"ldm", "llm"}
        and args.proposal_candidates_per_request != expected_request_size
    ):
        parser.error(
            "direct LDM uses one B-sized request per lineage; direct LLM uses one "
            "candidate per request"
        )
    if args.search_method in {"ldm", "llm"}:
        if args.proposal_mode != "openai" and not args.mock:
            parser.error(
                f"--search-method={args.search_method} requires --proposal-mode=openai"
            )
    elif args.proposal_mode != "none":
        parser.error(
            f"--search-method={args.search_method} requires --proposal-mode=none"
        )
    if args.search_method == "harness" and (
        args.harness_candidates_per_session != args.evaluations_per_round
    ):
        parser.error("Direct Harness candidates per session must equal evaluations per round")
    if args.harness_tool_budget is None:
        args.harness_tool_budget = [
            value
            for value in DEFAULT_NETWORK_TOOL_BUDGETS
            if args.harness_context7
            or not value.startswith(("resolve-library-id=", "query-docs="))
        ]
    try:
        _parse_extra_body(args.llm_extra_body_json)
        budgets = parse_tool_call_budgets(
            args.harness_tool_budget,
            excluded_tools=("submit_candidates",),
        )
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))
    if not args.harness_context7 and {
        "resolve-library-id",
        "query-docs",
    } & set(budgets):
        parser.error("Context7 tools cannot have budgets when Context7 is disabled")
    if args.search_method == COMPILED_POLICY_METHOD:
        try:
            policy_budgets = parse_tool_call_budgets(
                args.policy_tool_budget,
                excluded_tools=("submit_optimization_policy",),
            )
        except ValueError as exc:
            parser.error(str(exc))
        if args.policy_max_submission_attempts < 1:
            parser.error("--policy-max-submission-attempts must be positive")
        if len(args.policy_capability) != len(set(args.policy_capability)):
            parser.error("--policy-capability values must be unique")
        if not args.harness_context7 and {
            "resolve-library-id",
            "query-docs",
        } & set(policy_budgets):
            parser.error("Context7 policy tools cannot have budgets when disabled")
    elif any(
        value is not None
        for value in (
            args.policy_capability,
            args.policy_tool_budget,
            args.policy_max_submission_attempts,
        )
    ):
        parser.error("Policy options require --search-method=ldm_harness_compiled")
    if args.mock:
        if args.case_id != MOCK_CASE.case_id:
            parser.error(f"mock execution requires --case-id {MOCK_CASE.case_id}")
        if args.iterations is None or args.iterations < 2:
            parser.error("mock execution requires at least two rounds")


def describe_ldm_task(args: argparse.Namespace) -> LDMTaskSpec:
    return (
        build_mock_task_spec()
        if args.mock
        else build_task_spec(
            get_case(args.case_id),
            search_method=args.search_method,
            evaluations_per_round=args.evaluations_per_round,
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
        resolve_provider_settings(args)
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
    provider: ProviderSettings | None,
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
        "harness": _harness_description(args),
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
        config=_jsonable_args(args),
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
    provider: ProviderSettings | None,
    payload: dict[str, Any],
) -> int:
    if args.source_dir is None or args.prepared_dir is None:
        raise SystemExit(
            "Real NucleoBench execution requires --source-dir and --prepared-dir."
        )
    expected_digest = _configured(args.expected_start_set_sha256)
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
        gp_config=_gp_config(args),
        alpha=args.alpha,
        eta=args.eta,
        z_clip=args.z_clip,
    )
    task_spec = build_task_spec(
        prepared.context.case,
        search_method=args.search_method,
        evaluations_per_round=args.evaluations_per_round,
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
    stack = ExitStack()
    profiles = _harness_profiles(args)
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
            harness_client = stack.enter_context(
                _harness_client(args, runtime, provider, prepared, profiles, mcp)
            )
            if args.search_method == COMPILED_POLICY_METHOD:
                policy_adapter = NucleoOptimizationPolicyAdapter(
                    NucleoPolicyFeatureEncoder(prepared.context),
                    seed=args.campaign_index,
                    gp_config=_gp_config(args),
                    default_alpha=args.alpha,
                    default_eta=args.eta,
                    enabled_capabilities=tuple(args.policy_capability),
                    proposal_sampling={
                        "session_count": len(harness_client.config.profiles),
                        "candidates_per_session": args.harness_candidates_per_session,
                        "within_session_repeats_allowed": not args.harness_unique_candidates,
                        "cross_session_agreement_allowed": True,
                    },
                    evaluations_per_round=args.evaluations_per_round,
                    benchmark_clock=benchmark_clock,
                    measured_history_path=runtime.run_dir / "harness" / MEASURED_HISTORY_FILE,
                )
                policy_client = stack.enter_context(
                    _harness_client(
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
                        if benchmark_clock is not None else float(args.harness_wall_time_seconds)
                    ),
                )
                encoder, selector = build_surrogate_components(
                    args.search_method,
                    prepared.context,
                    evaluations_per_round=args.evaluations_per_round,
                    proposal_samples=args.proposal_samples,
                    bo_pool_size=args.bo_pool_size,
                    seed=args.campaign_index,
                    gp_config=_gp_config(args),
                    alpha=args.alpha,
                    eta=args.eta,
                    z_clip=args.z_clip,
                    policy_controller=policy_controller,
                    policy_adapter=policy_adapter,
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
        )
        designer = NucleoBenchDesigner(
            engine=engine,
            state=state,
            context=prepared.context,
            reservoir_size=args.proposal_samples,
            evaluations_per_step=args.evaluations_per_round,
        )
        method_digest = _method_preset_sha256(args, task_spec)
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
                model=official.model,
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
        if harness_client is not None:
            write_measured_history(runtime.run_dir / "harness", designer.state.observations, prepared.context)
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
) -> LDMEngine:
    expander = build_proposal_expander(
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


def _gp_config(args: argparse.Namespace) -> HammingGPUCBConfig:
    return HammingGPUCBConfig(
        beta=args.acquisition_beta,
        noise_variance=args.gp_noise_variance,
        jitter=args.gp_jitter,
        target_std_floor=args.gp_target_std_floor,
        min_history_for_fit=args.gp_min_history_for_fit,
        max_observations=args.gp_max_observations,
        global_best_observations=args.gp_global_best_observations,
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
        config=_jsonable_args(args),
        task_spec=task_spec,
        budget_limits=_campaign_budget(
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


def _campaign_budget(
    args: argparse.Namespace,
    profile_budget: Mapping[str, int | float] | None,
) -> dict[str, int | float]:
    if args.termination_kind == "wall_time":
        return {"initialization_evaluations": 1, **dict(profile_budget or {})}
    assert args.iterations is not None
    active_rounds = max(0, args.iterations - 1)
    evaluated = active_rounds * args.evaluations_per_round
    candidate_limit = active_rounds * args.proposal_samples
    limits: dict[str, int | float] = {
        "outer_iterations": args.iterations,
        "initialization_evaluations": 1,
        "valid_search_candidates": candidate_limit,
        "selected_candidates": evaluated,
        "external_evaluations": evaluated,
        "expensive_evaluation_attempts": evaluated,
        "successful_evaluations": evaluated,
    }
    if args.search_method in {"ldm", "llm"}:
        request_limit = active_rounds * args.proposal_request_limit
        limits.update(proposal_attempts=request_limit, llm_requests=request_limit)
    elif args.search_method in PARALLEL_HARNESS_METHODS:
        turns = active_rounds * len(args.harness_profile)
        limits.update(proposal_attempts=turns, harness_turns=turns)
        if args.search_method == COMPILED_POLICY_METHOD:
            limits["policy_harness_turns"] = active_rounds
    elif args.search_method == "harness":
        limits.update(proposal_attempts=active_rounds, harness_turns=active_rounds)
    elif args.search_method == "bo":
        limits.update(proposal_attempts=0, llm_requests=0)
    return {**limits, **dict(profile_budget or {})}


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
    provider: ProviderSettings,
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
        extra_body=_parse_extra_body(args.llm_extra_body_json),
    )


def _harness_profiles(args: argparse.Namespace):
    if args.search_method in PARALLEL_HARNESS_METHODS:
        return harness_profiles(args.harness_profile)
    if args.search_method == "harness":
        return direct_harness_profile()
    return ()


def _harness_client(
    args: argparse.Namespace,
    runtime: CampaignRuntime,
    provider: ProviderSettings,
    prepared: PreparedCase,
    profiles: Sequence[HarnessProfile],
    mcp,
    *,
    policy: bool = False,
) -> HarnessClient:
    artifact_root = (runtime.run_dir / ("policy_harness" if policy else "harness")).resolve()
    resource_root = (TASK_ROOT / "resources" / "harness").resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)
    write_harness_sequence_context(
        prepared.context,
        artifact_root / "sequence_context.json",
    )
    mounts = [(resource_root, "/resources", True)]
    mcp_servers = mcp.servers
    if policy:
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
            diagnostics_path="/resources/policy_diagnostics.py",
            diagnostics_sha256=file_sha256(resource_root / "policy_diagnostics.py"),
        ))
    else:
        submission_contract = harness_submission_contract(args.harness_candidates_per_session)
        history_path = "/artifacts/measured_history/observations.json"
        (artifact_root / MEASURED_HISTORY_FILE).parent.mkdir(parents=True, exist_ok=True)
    return HarnessClient(
        _harness_command(
            args,
            artifact_root,
            _harness_cache_root(args),
            environment={
                "LDM_NUCLEOBENCH_CONTEXT": "/artifacts/sequence_context.json",
                "LDM_NUCLEOBENCH_HISTORY": history_path,
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
                forbidden_query_patterns=HARNESS_FORBIDDEN_PATTERNS,
            ),
            context7_enabled=args.harness_context7,
        ),
        named_secrets=mcp.named_secrets,
        response_timeout_seconds=args.harness_response_timeout,
    )


def _harness_cache_root(args: argparse.Namespace) -> Path:
    root = (
        args.harness_cache_dir.expanduser().resolve()
        if args.harness_cache_dir is not None
        else (Path.home() / ".cache" / "ldm-gondolin").resolve()
    )
    root.mkdir(parents=True, exist_ok=True)
    return root


def _harness_command(
    args: argparse.Namespace,
    artifact_root: Path,
    cache_root: Path,
    *,
    environment: Mapping[str, str],
    mounts: Sequence[tuple[Path, str, bool]],
) -> list[str]:
    (cache_root / "runtime-overlays").mkdir(parents=True, exist_ok=True)
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
    command.extend(
        (
            "--mount",
            f"type=bind,src={artifact_root},dst=/artifacts",
            "--mount",
            f"type=bind,src={cache_root},dst=/runtime-home/.cache/gondolin",
        )
    )
    for source, destination, readonly in mounts:
        suffix = ",readonly" if readonly else ""
        command.extend(
            ("--mount", f"type=bind,src={source},dst={destination}{suffix}")
        )
    command.append(args.harness_sidecar_image)
    return command


def _missing_provider(provider: ProviderSettings) -> str:
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


def _method_preset_sha256(
    args: argparse.Namespace,
    task_spec: LDMTaskSpec,
) -> str:
    payload = {
        "task_spec": task_spec.to_dict(),
        "proposal": {
            "mode": args.proposal_mode,
            "samples": args.proposal_samples,
            "candidates_per_request": args.proposal_candidates_per_request,
            "max_workers": args.proposal_max_workers,
            "request_limit": args.proposal_request_limit,
        },
        "provider": {
            "base_url": args.llm_url,
            "model": args.llm_model_name,
            "wire_api": args.llm_wire_api,
            "reasoning": args.llm_reasoning,
            "temperature": args.llm_temperature,
            "max_tokens": args.llm_max_tokens,
            "extra_body": _parse_extra_body(args.llm_extra_body_json),
        },
        "harness": _harness_description(args),
        "compiled_policy": (
            {
                "capabilities": args.policy_capability,
                "tool_call_budgets": parse_tool_call_budgets(
                    args.policy_tool_budget,
                    excluded_tools=("submit_optimization_policy",),
                ),
                "max_submission_attempts": args.policy_max_submission_attempts,
            }
            if args.search_method == COMPILED_POLICY_METHOD
            else None
        ),
    }
    return canonical_json_sha256(payload)


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


def _jsonable_args(args: argparse.Namespace) -> dict[str, Any]:
    result = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
        if key != "api_key_file"
    }
    if args.search_method != COMPILED_POLICY_METHOD:
        for key in (
            "policy_capability",
            "policy_tool_budget",
            "policy_max_submission_attempts",
        ):
            result.pop(key, None)
    return result


def _harness_description(args: argparse.Namespace) -> dict[str, Any] | None:
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
        "profile_ids": [profile.profile_id for profile in _harness_profiles(args)],
        "profile_roles": list(profiles),
        "session_count": len(profiles),
        "candidates_per_session": args.harness_candidates_per_session,
        "unique_candidates_per_session": (
            args.search_method == "harness" or args.harness_unique_candidates
        ),
        "thinking": args.harness_thinking,
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
    }
    if args.search_method == COMPILED_POLICY_METHOD:
        description["policy_session"] = {
            "profile_id": "policy_architect",
            "skills_loaded": True,
            "editable_components": ["prior_mean", "alpha", "eta"],
        }
    return description


def resolve_provider_settings(
    args: argparse.Namespace,
    *,
    environ: Mapping[str, str] | None = None,
) -> ProviderSettings:
    environment = os.environ if environ is None else environ
    api_key = ""
    if args.api_key_file is not None:
        key_path = args.api_key_file.expanduser()
        if key_path.is_file():
            api_key = key_path.read_text(encoding="utf-8").strip()
            if not api_key:
                raise ValueError("API key file is empty")
        elif not args.dry_run:
            raise FileNotFoundError(f"API key file does not exist: {key_path}")
    if not api_key:
        api_key = _first_configured(environment, "LLM_API_KEY", "OPENAI_API_KEY")
    return ProviderSettings(
        base_url=_configured(args.llm_url)
        or _first_configured(environment, "LLM_BASE_URL"),
        model=_configured(args.llm_model_name)
        or _first_configured(environment, "LLM_MODEL_NAME"),
        api_key=api_key,
    )


def _parse_extra_body(raw: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"--llm-extra-body-json is not valid JSON: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise TypeError("--llm-extra-body-json must decode to a JSON object")
    return payload


def _configured(value: str | None) -> str:
    return "" if value is None else str(value).strip()


def _first_configured(environment: Mapping[str, str], *names: str) -> str:
    for name in names:
        value = _configured(environment.get(name))
        if value:
            return value
    return ""


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
    "resolve_provider_settings",
    "run_official_driver",
]
