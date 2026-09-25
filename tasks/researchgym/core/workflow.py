"""ResearchGym campaign assembly through ldm_tts.campaign.run_campaign."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from copy import copy
from pathlib import Path

from ldm_tts.campaign import CampaignBudget, CampaignRecipe, CampaignRequest, run_campaign
from ldm_tts.contracts import (
    AcquisitionSpec, CandidateDomainSpec, LDMTaskSpec, ObjectiveSpec, ProposalSearchSpec, ReservoirExpansionSpec,
    ReservoirSpec, ResponseSpaceSpec, SurrogateSpaceSpec,
)
from ldm_tts.data import DataCollectionSink
from ldm_tts.engine.expansion import CallableReservoirExpander, ExpansionResult, InitialRoundReservoirExpander
from ldm_tts.engine.run_store import BudgetExceededError, atomic_json_write, unique_run_dir
from ldm_tts.engine.runtime import consume_proposal_attempts
from ldm_tts.harness import DockerPolicyExecutor, HarnessError, PolicyResearchController
from ldm_tts.registration.experiment import (
    load_active_experiment_contract, load_experiment_contract, snapshot_experiment_contract,
)
from ldm_tts.transport.openai import (
    REASONING_LEVELS, WIRE_APIS, EndpointRequestError, OpenAICompatibleProposalClient, generation_body,
)

from .candidate import FEATURE_VERSION, ProgramDomain, ProgramEncoder
from .cases import CASE_IDS, TASK_ROOT, load_case
from .clock import CampaignClock, WallClockExhausted
from .evaluator import DevicesUnavailable, MockProgramEvaluator, ResearchGymEvaluator
from .harness import HarnessProgramSource, create_client, write_context
from .optimization_policy import ResearchGymPolicyAdapter
from .proposals import LDM_METHODS, DirectProgramSource, MockProposalClient, ProgramExpander, ProposalExhausted, seed_proposals
from .selection import GPSettings, ProgramLDMSelector
from .source import UpstreamCase
from .usage import ApiBudgetExhausted, ApiPricing, account_pool_tokens

METHODS = ("llm", "ldm", "harness", "ldm_harness", "ldm_harness_compiled", "bo")
HARNESS_METHODS = ("harness", "ldm_harness", "ldm_harness_compiled")
USAGE_COUNTERS = (
    "llm_failed_requests", "llm_usage_unknown", "llm_input_tokens", "llm_output_tokens",
    "harness_provider_calls", "harness_tool_calls", "harness_artifact_bytes", "harness_validation_submissions",
    "harness_usage_unknown", "harness_input_tokens", "harness_output_tokens", "api_cost_usd", "proposal_attempts",
    "policy_provider_requests", "policy_tool_calls", "policy_validation_submissions", "policy_artifact_bytes",
    "policy_wall_time_seconds", "policy_input_tokens", "policy_output_tokens", "policy_usage_unknown",
)
SECRET_ENV = re.compile(r"(API_KEY|SECRET|PASSWORD)", re.IGNORECASE)
COMPLETED_STOPS = {"iteration_budget", "observation_target", "successful_evaluation_target"}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="LDM search over ResearchGym method programs graded by official graders.")
    p.add_argument("--mock", action="store_true")
    p.add_argument("--case", choices=CASE_IDS, default="time_series_explanation")
    p.add_argument("--search-method", choices=METHODS, default="ldm")
    p.add_argument("--proposal-mode", choices=("auto", "openai", "mock", "callable", "none", "harness"), default="auto")
    p.add_argument("--initialization-mode", choices=("none", "shared_start"), default="none")
    p.add_argument("--iterations", type=int, default=4)
    p.add_argument("--proposal-samples", "--reservoir-size", dest="proposal_samples", type=int, default=8)
    p.add_argument("--proposal-batch-size", type=int, default=2)
    p.add_argument("--bo-pool-size", type=int)
    p.add_argument("--evaluations-per-round", type=int, default=1)
    p.add_argument("--max-repair-requests", type=int, default=2)
    p.add_argument("--max-refill-collections", type=int, default=2,
                   help="Extra proposal collections when distinct programs fall below the evaluation batch.")
    p.add_argument("--recovery-attempts", type=int, default=3)
    p.add_argument("--acquisition-beta", type=float, default=1.0)
    p.add_argument("--acquisition-alpha", type=float, default=1.0)
    p.add_argument("--acquisition-eta", type=float, default=1.0)
    p.add_argument("--acquisition-z-clip", type=float, default=5.0)
    p.add_argument("--gp-lengthscale", type=float, default=0.5)
    p.add_argument("--gp-noise", type=float, default=0.05)
    p.add_argument("--gp-history-limit", type=int, default=128)
    p.add_argument("--seed", "--campaign-index", dest="seed", type=int, default=0)
    p.add_argument("--upstream-root", type=Path, default=Path(os.environ.get("RESEARCHGYM_ROOT", str(Path.home() / "ResearchGym"))))
    p.add_argument("--case-data-root", type=Path, default=None,
                   help="Directory holding the case's data/model/weights links; defaults to the upstream case directory.")
    p.add_argument("--case-python", default=os.environ.get("RESEARCHGYM_CASE_PYTHON", ""))
    p.add_argument("--devices", default=os.environ.get("RESEARCHGYM_DEVICES", ""))
    p.add_argument("--min-free-mib", type=int, default=0)
    p.add_argument("--evaluation-timeout", type=float)
    p.add_argument("--grader-timeout", type=float)
    p.add_argument("--port-base", type=int, default=29600)
    p.add_argument("--campaign-hours", type=float, default=0.0, help="Total campaign wall time; 0 disables the clock.")
    p.add_argument("--harness-sessions", type=int, default=4)
    p.add_argument("--harness-sidecar-image", default="ldm-pi-harness:latest")
    p.add_argument("--harness-docker-host", default="")
    p.add_argument("--harness-container-user", default=None,
                   help="Override the sidecar UID:GID; by default the current user plus the /dev/kvm group.")
    p.add_argument("--harness-cache-dir", type=Path)
    p.add_argument("--harness-thinking", choices=("off", "minimal", "low", "medium", "high", "xhigh", "max"))
    p.add_argument("--harness-wall-time-seconds", type=int, default=1800)
    p.add_argument("--harness-response-timeout", type=int, default=2400)
    p.add_argument("--harness-submission-attempts", type=int, default=4)
    p.add_argument("--harness-mcp-config", type=Path)
    p.add_argument("--harness-context7", action="store_true")
    p.add_argument("--harness-tool-budget", action="append", default=[])
    p.add_argument("--harness-command-json", default="", help="Mock only: protocol-faithful fake sidecar command.")
    p.add_argument("--policy-tool-budget", action="append", default=[])
    p.add_argument("--policy-runner-image", default="ldm-pi-harness:latest")
    p.add_argument("--policy-submission-attempts", type=int, default=3)
    p.add_argument("--llm-url", default=os.environ.get("LLM_BASE_URL", os.environ.get("LDM_LLM_URL", "")))
    p.add_argument("--llm-model", default=os.environ.get("LLM_MODEL_NAME", os.environ.get("LDM_LLM_MODEL", "")))
    p.add_argument("--llm-max-tokens", type=int, default=16384)
    p.add_argument("--llm-wire-api", choices=WIRE_APIS, default="responses")
    p.add_argument("--llm-reasoning", choices=REASONING_LEVELS)
    p.add_argument("--llm-temperature", type=float, default=0.7)
    p.add_argument("--llm-extra-body-json", default="{}")
    p.add_argument("--llm-timeout", type=float, default=600.0)
    p.add_argument("--llm-transport-retries", type=int, default=2)
    p.add_argument("--llm-input-usd-per-mtok", type=float)
    p.add_argument("--llm-output-usd-per-mtok", type=float)
    p.add_argument("--api-budget-usd", type=float)
    p.add_argument("--out-dir", type=Path, default=Path("runs/researchgym"))
    p.add_argument("--run-name", default="")
    p.add_argument("--resume-from", type=Path)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    if args.search_method == "bo":
        p.error("bo is not supported for ResearchGym program search: no score-blind, LLM-free generator of valid "
                "method programs is defined for these cases. Use llm, ldm, harness, ldm_harness or ldm_harness_compiled.")
    harness = args.search_method in HARNESS_METHODS
    args.proposal_mode = {"callable": "mock", "none": "auto"}.get(args.proposal_mode, args.proposal_mode)
    if args.proposal_mode == "auto":
        args.proposal_mode = "harness" if harness else ("mock" if args.mock else "openai")
    if harness != (args.proposal_mode == "harness"):
        p.error("Harness methods require proposal-mode harness, and proposal-mode harness requires a Harness method")
    if args.proposal_mode == "mock" and not args.mock:
        p.error("mock proposals require --mock; they are not scientific results")
    if args.harness_command_json and not args.mock:
        p.error("harness-command-json is a mock-only fake sidecar hook")
    if args.search_method == "harness" and args.harness_sessions != 1:
        p.error("the harness method uses exactly one persistent session (set harness-sessions 1)")
    if args.llm_reasoning is not None and args.harness_thinking is not None and args.llm_reasoning != args.harness_thinking:
        p.error("llm-reasoning and harness-thinking must agree")
    args.llm_reasoning = args.llm_reasoning or args.harness_thinking or "off"
    args.harness_thinking = "off" if args.llm_reasoning == "none" else args.llm_reasoning
    if harness and args.llm_wire_api != "responses":
        p.error("the Pi Harness requires llm-wire-api responses")
    try:
        extra = json.loads(args.llm_extra_body_json)
        if not isinstance(extra, dict):
            raise ValueError("llm-extra-body-json must be an object")
        generation_body(wire_api=args.llm_wire_api, reasoning=args.llm_reasoning, extra_body=extra)
        args.llm_extra_body_json = json.dumps(extra, sort_keys=True)
    except ValueError as exc:
        p.error(str(exc))
    if args.search_method not in LDM_METHODS:
        # Direct-batch methods have no reservoir or BO pool; shared pilot bases may still set them.
        args.proposal_samples = args.bo_pool_size = args.evaluations_per_round
    if args.bo_pool_size is None:
        args.bo_pool_size = args.proposal_samples
    positive = (args.iterations, args.proposal_samples, args.proposal_batch_size, args.evaluations_per_round,
                args.harness_sessions, args.harness_wall_time_seconds, args.harness_response_timeout,
                args.harness_submission_attempts, args.policy_submission_attempts, args.gp_history_limit)
    if any(v < 1 for v in positive) or args.max_repair_requests < 0 or args.recovery_attempts < 0 \
            or args.max_refill_collections < 0 \
            or args.llm_transport_retries < 0 or args.seed < 0:
        p.error("counts, limits and sessions must be positive; repair, recovery and retry counts nonnegative")
    if not 1 <= args.evaluations_per_round <= args.bo_pool_size <= args.proposal_samples:
        p.error("require 1 <= evaluations-per-round <= bo-pool-size <= proposal-samples")
    if any(not math.isfinite(v) or v < 0 for v in (args.acquisition_beta, args.acquisition_alpha, args.acquisition_eta,
                                                    args.campaign_hours)) \
            or not args.acquisition_z_clip > 0 or not args.gp_lengthscale > 0 or not args.gp_noise > 0:
        p.error("acquisition, GP and clock parameters must be finite and in range")
    if (args.api_budget_usd is not None) and (args.llm_input_usd_per_mtok is None or args.llm_output_usd_per_mtok is None):
        p.error("api-budget-usd requires llm-input-usd-per-mtok and llm-output-usd-per-mtok prices")
    if args.run_name and (Path(args.run_name).name != args.run_name or args.run_name in (".", "..")):
        p.error("run-name must be a single directory name")
    case = load_case(args.case)
    args.evaluation_timeout = args.evaluation_timeout or case.raw["evaluation_timeout_seconds"]
    args.grader_timeout = args.grader_timeout or case.raw["grader_timeout_seconds"]
    args.devices = tuple(d.strip() for d in str(args.devices).split(",") if d.strip())
    args.upstream_root = args.upstream_root.expanduser().resolve()
    for name in ("case_data_root", "harness_cache_dir", "harness_mcp_config"):
        if getattr(args, name):
            setattr(args, name, getattr(args, name).expanduser().resolve())
    return args


def objective_name(args, case):
    return "mock_score" if args.mock else case.metric["name"]


def describe_ldm_task(args):
    case = load_case(args.case)
    ldm = args.search_method in LDM_METHODS
    objective = objective_name(args, case)
    return LDMTaskSpec(
        task="researchgym",
        candidate_domain=CandidateDomainSpec(
            name=f"ResearchGym {case.case_id} method programs", kind="python_program", dimension=None,
            representation=f"Complete Python module defining top-level {case.entry['symbol']} "
                           f"installed at {case.candidate_path}",
            constraints={"case": case.case_id, "mock": args.mock, **case.rules,
                         "forbidden_import_prefixes": case.raw["forbidden_import_prefixes"]},
        ),
        objectives=(ObjectiveSpec(objective, "maximize",
                                  "Deterministic analytic mock score" if args.mock else case.metric["aggregate"]),),
        response_spaces=(ResponseSpaceSpec(
            name="candidate_programs_json", output_kind="json",
            schema={"type": "object", "required": ["candidates"], "additionalProperties": False, "properties": {
                "candidates": {"type": "array", "items": {"type": "object", "required": ["program", "change_summary",
                               "rationale"], "additionalProperties": False, "properties": {
                    "program": {"type": "string"}, "change_summary": {"type": "string"}, "rationale": {"type": "string"},
                    "comparison_candidate_ids": {"type": "array", "items": {"type": "string"}}}}}}},
            parser="tasks.researchgym.core.proposals:validate_rows",
        ),),
        acquisition=AcquisitionSpec(
            name="ldm_rbf_gp_ucb" if ldm else "direct_submission_order", objective_names=(objective,),
            score_direction="sample" if ldm else "maximize",
            selection_rule="q0^alpha * exp(eta * robust_z(UCB)) sampled without replacement from the BO pool"
                           if ldm else "Evaluate the accepted submission in stable order",
            parameters={"alpha": args.acquisition_alpha, "eta": args.acquisition_eta, "beta": args.acquisition_beta,
                        "z_clip": args.acquisition_z_clip, "bo_pool_size": args.bo_pool_size,
                        "compiled_policy": args.search_method == "ldm_harness_compiled"} if ldm else {},
        ),
        reservoir=ReservoirSpec(
            name="program_occurrence_reservoir",
            expansions=(ReservoirExpansionSpec(
                name="persistent_research_sessions" if args.proposal_mode == "harness" else "direct_program_requests",
                action_kind="emit_candidate", response_space="candidate_programs_json", produces_candidates=True,
                description="Independent Harness sessions submit annotated candidates.json files"
                            if args.proposal_mode == "harness" else "Independent direct requests return program JSON",
            ),),
            candidate_validator="tasks.researchgym.core.candidate:ProgramDomain",
            deduplication_key="SHA-256 of the docstring-free AST dump",
            max_size=args.proposal_samples,
        ),
        surrogate=ProgramEncoder(case).describe() if ldm else SurrogateSpaceSpec(
            kind="none", representation="No surrogate; accepted submission order", dimension_policy="none"),
        proposal_search=ProposalSearchSpec(
            name="independent_occurrences_empirical_q0" if ldm else args.search_method,
            breadth=args.proposal_samples,
            evaluation_policy="ldm_sampled_official_grader" if ldm else "direct_official_grader",
        ),
        metadata={"case": case.case_id, "search_method": args.search_method, "proposal_mode": args.proposal_mode,
                  "protocol": case.protocol, "mock": args.mock, "feature_version": FEATURE_VERSION if ldm else None,
                  "policy_capabilities": ["prior_mean@1", "ldm_weights@1"]
                  if args.search_method == "ldm_harness_compiled" else [],
                  "unsupported_methods": {"bo": "no score-blind LLM-free program generator"}},
    )


SCIENTIFIC_KEYS = (
    "mock", "case", "search_method", "proposal_mode", "initialization_mode", "proposal_samples", "proposal_batch_size",
    "bo_pool_size", "evaluations_per_round", "max_repair_requests", "max_refill_collections", "acquisition_beta", "acquisition_alpha",
    "acquisition_eta", "acquisition_z_clip", "gp_lengthscale", "gp_noise", "gp_history_limit", "seed",
    "evaluation_timeout", "grader_timeout", "campaign_hours", "harness_sessions", "harness_wall_time_seconds",
    "harness_submission_attempts", "harness_tool_budget", "policy_tool_budget", "policy_submission_attempts",
    "harness_thinking", "harness_context7", "llm_model", "llm_max_tokens", "llm_wire_api", "llm_reasoning",
    "llm_temperature", "llm_extra_body_json", "llm_input_usd_per_mtok", "llm_output_usd_per_mtok", "api_budget_usd",
)


def _jsonable(args):
    return {k: (str(v) if isinstance(v, Path) else list(v) if isinstance(v, tuple) else v)
            for k, v in vars(args).items() if k != "provider_api_key"}


def jobs_per_evaluation(args, case) -> int:
    return 1 if args.mock else len(case.jobs("ldm"))


def _campaign_config(args):
    """Run config plus the proposal-counting declaration read by pilot integrity checks."""
    ldm = args.search_method in LDM_METHODS
    if args.proposal_mode == "harness":
        per_request = math.ceil(args.proposal_samples / min(args.harness_sessions, args.proposal_samples))
        extra = args.max_refill_collections * min(args.harness_sessions, args.evaluations_per_round) if ldm else 0
    else:
        per_request = args.proposal_batch_size if ldm else args.evaluations_per_round
        minibatches = math.ceil(args.proposal_samples / per_request)
        refill_requests = args.max_refill_collections * math.ceil(args.evaluations_per_round / per_request) if ldm else 0
        extra = minibatches * args.max_repair_requests + refill_requests * (1 + args.max_repair_requests)
    return {**_jsonable(args), "proposal_counting": "bounded_minibatches",
            "proposal_candidates_per_request": per_request, "max_replenishment_batches": extra}


def _provider_api_key():
    return os.environ.get("LLM_API_KEY", os.environ.get("LDM_LLM_API_KEY", os.environ.get("OPENAI_API_KEY", "")))


def _direct_client(args, *, timeout=None, wire_api=None):
    wire = wire_api or args.llm_wire_api
    return OpenAICompatibleProposalClient(
        url=args.llm_url, model=args.llm_model, api_key=_provider_api_key(), max_tokens=args.llm_max_tokens,
        max_retries=0, wire_api=wire, temperature=args.llm_temperature, timeout_seconds=timeout or args.llm_timeout,
        extra_body=generation_body(wire_api=wire, reasoning=args.llm_reasoning, extra_body=json.loads(args.llm_extra_body_json)),
    )


def _evaluator_env():
    """Candidate processes never inherit provider credentials."""
    return {k: v for k, v in os.environ.items() if not SECRET_ENV.search(k)}


def _proposal_facts(args):
    sessions = args.harness_sessions if args.proposal_mode == "harness" else None
    occurrences = args.proposal_samples if args.search_method in LDM_METHODS else args.evaluations_per_round
    return {"search_method": args.search_method, "proposal_mode": args.proposal_mode,
            "occurrences_per_round": occurrences, "harness_sessions": sessions,
            "direct_request_batch": args.proposal_batch_size if args.proposal_mode != "harness" else None,
            "evaluation_batch": args.evaluations_per_round, "bo_pool_size": args.bo_pool_size,
            "within_submission_duplicates": "forbidden", "cross_request_duplicates": "counted in empirical q0",
            "historical_repeats": "only authoritative measured candidates are excluded"}


def main(argv=None):
    args = parse_args(argv)
    case = load_case(args.case)
    spec = describe_ldm_task(args)
    contract, profile = load_active_experiment_contract()
    contract = contract or load_experiment_contract(TASK_ROOT / "experiment.json")
    if args.dry_run:
        print(json.dumps({"task": "researchgym", "contract_profile": profile, "configuration": _jsonable(args),
                          "ldm_task_spec": spec.to_dict()}, indent=2))
        return 0
    if args.resume_from:
        run_dir = args.resume_from.resolve()
    elif args.run_name:
        run_dir = (args.out_dir / args.run_name).resolve()
        if run_dir.exists():
            raise FileExistsError("named run directory already exists; use --resume-from")
    else:
        run_dir = unique_run_dir(args.out_dir.resolve())
    resume = args.resume_from is not None and (run_dir / "campaign.json").exists()
    run_dir.mkdir(parents=True, exist_ok=True)
    result, code = run(args, case, spec, contract, profile, run_dir, resume=resume)
    print(json.dumps({"task": "researchgym", "case": case.case_id, "search_method": args.search_method,
                      "mock": args.mock, "run_dir": str(run_dir), "status": result.get("status"),
                      "exit_code": code}, indent=2))
    return code


def run(args, case, spec, contract, profile, run_dir, *, resume):
    upstream, source_info, references = None, {"mock": True}, {}
    if not args.mock:
        if not args.case_python:
            raise ValueError("set --case-python (or RESEARCHGYM_CASE_PYTHON) to the case environment interpreter")
        upstream = UpstreamCase(args.upstream_root, case, data_root=args.case_data_root)
        source_info = upstream.verify()
        references = upstream.reference_sources()
    scientific = {k: _jsonable(args)[k] for k in SCIENTIFIC_KEYS} | {"source": source_info,
                                                                    "contract_sha256": contract.digest}
    identity_path = run_dir / "scientific_identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != scientific:
        raise ValueError("resume scientific configuration mismatch; start a new run directory")
    atomic_json_write(identity_path, scientific)
    snapshot_experiment_contract(contract, run_dir, profile=profile)
    previous = json.loads((run_dir / "status.json").read_text()) if resume and (run_dir / "status.json").exists() else {}
    clock = CampaignClock.open(run_dir / "benchmark_clock.json",
                               args.campaign_hours * 3600 if args.campaign_hours else None, resume=resume)
    clock.start()
    pricing = ApiPricing(args.llm_input_usd_per_mtok, args.llm_output_usd_per_mtok, args.api_budget_usd)
    domain, encoder = ProgramDomain(case), ProgramEncoder(case)
    ldm = args.search_method in LDM_METHODS
    if args.mock:
        evaluator = MockProgramEvaluator(case=case, run_dir=run_dir, encoder=encoder, clock=clock)
    else:
        evaluator = ResearchGymEvaluator(
            case=case, upstream=upstream, run_dir=run_dir, python=args.case_python, devices=args.devices,
            min_free_mib=args.min_free_mib, evaluation_timeout=args.evaluation_timeout,
            grader_timeout=args.grader_timeout, clock=clock, port_base=args.port_base, base_env=_evaluator_env())
    sink = DataCollectionSink.from_env(default_root=run_dir / "ldm_data")
    harness_clients, holder = [], {}
    if args.proposal_mode == "harness":
        source = None  # Created after preflight inside the runtime hook.
    else:
        client = MockProposalClient(case, seed=args.seed) if args.proposal_mode == "mock" else None
        source = DirectProgramSource(args=args, case=case, client=client, run_dir=run_dir, pricing=pricing, sink=sink)
    expander = ProgramExpander(args=args, case=case, domain=domain, source=source, run_dir=run_dir)
    gp = GPSettings(lengthscale=args.gp_lengthscale, noise=args.gp_noise, beta=args.acquisition_beta,
                    history_limit=args.gp_history_limit,
                    target_scale_floor=0.05 if args.mock else case.metric["gp_target_scale_floor"])
    selector = ProgramLDMSelector(gp, objective_name=objective_name(args, case), alpha=args.acquisition_alpha,
                                  eta=args.acquisition_eta, z_clip=args.acquisition_z_clip, seed=args.seed,
                                  pool_size=args.bo_pool_size) if ldm else None
    campaign_expander = expander
    rounds = args.iterations
    cap = rounds * args.evaluations_per_round
    if args.initialization_mode == "shared_start":
        campaign_expander = InitialRoundReservoirExpander(
            initializer=CallableReservoirExpander(lambda request: ExpansionResult(
                proposals=seed_proposals(case), metadata={"phase": "released_baseline_seed"})),
            search_expander=expander, initial_reservoir_size=1)
        cap = 1 + (rounds - 1) * args.evaluations_per_round
    direct = args.proposal_mode in ("openai", "mock")
    harness = args.proposal_mode == "harness"
    online = not args.mock and (direct or harness)
    total_rounds = rounds + args.recovery_attempts
    minibatches = math.ceil(args.proposal_samples / (args.proposal_batch_size if ldm else args.proposal_samples))
    minibatches += args.max_refill_collections * math.ceil(args.evaluations_per_round / args.proposal_batch_size) if ldm else 0
    request_cap = total_rounds * minibatches * (args.max_repair_requests + 1) * (args.llm_transport_retries + 2)
    previous_preflights = 0
    if resume and (run_dir / "budget.json").exists():
        previous_preflights = json.loads((run_dir / "budget.json").read_text())["counters"].get("endpoint_preflight_requests", 0)
    budget = CampaignBudget(
        rounds=rounds, reservoir_size=args.proposal_samples, batch_size=args.evaluations_per_round,
        target_observations=cap, max_evaluation_attempts=cap,
        extra_limits={
            "llm_requests": request_cap if direct else 0,
            "proposal_request_attempts": request_cap if direct else 0,
            "harness_turns": total_rounds * args.harness_sessions * (2 + args.max_refill_collections) if harness else 0,
            "policy_harness_turns": total_rounds if args.search_method == "ldm_harness_compiled" else 0,
            "recovery_attempts": args.recovery_attempts,
            "endpoint_preflight_requests": previous_preflights + 1 if online else 0,
            # One evaluation runs the case's whole job matrix, not a single job.
            "benchmark_jobs": cap * jobs_per_evaluation(args, case),
        },
    )

    def runtime_hook(runtime):
        holder["runtime"] = runtime
        runtime.consume_many({name: 0 for name in USAGE_COUNTERS})
        runtime.record("benchmark_clock_segment", clock.snapshot())
        recovery_pass = int((previous.get("details") or {}).get("proposal_recovery_pass", 0))
        if resume and str(previous.get("status", "")).startswith("paused"):
            runtime.consume("recovery_attempts")
            if previous.get("status") == "paused_proposal_exhausted":
                recovery_pass += 1
        expander.recovery_pass = recovery_pass
        if direct:
            if args.proposal_mode == "openai":
                if not args.llm_url or not args.llm_model:
                    raise ValueError("set LLM_BASE_URL and LLM_MODEL_NAME for real proposals")
                source.client = _direct_client(args)
                runtime.consume("endpoint_preflight_requests")
                runtime.record("endpoint_preflight_succeeded", source.client.preflight())
            source.runtime, source.clock = runtime, clock
            return
        provider = copy(args)
        provider.provider_api_key = "synthetic-fixture-key" if args.mock else _provider_api_key()
        command = json.loads(args.harness_command_json) if args.harness_command_json else None
        if not args.mock:
            if not args.llm_url or not args.llm_model or not provider.provider_api_key:
                raise EndpointRequestError("set LLM_BASE_URL, LLM_MODEL_NAME and LLM_API_KEY for Harness sessions")
            runtime.consume("endpoint_preflight_requests")
            probe = _direct_client(args, timeout=args.harness_response_timeout, wire_api="responses")
            runtime.record("endpoint_preflight_succeeded", {"backend": "harness", **probe.preflight()})
        facts = _proposal_facts(args)
        write_context(run_dir / "harness", case, references, facts)
        client = create_client(provider, run_dir / "harness", runtime.run_id, case, command=command)
        harness_clients.append(client)
        client.start()
        expander.source = HarnessProgramSource(client, run_dir / "harness", args, case, pricing=pricing)
        expander.source.runtime, expander.source.clock = runtime, clock
        if args.search_method == "ldm_harness_compiled":
            write_context(run_dir / "policy_harness", case, references, facts)
            policy_client = create_client(provider, run_dir / "policy_harness", runtime.run_id, case, policy=True,
                                          command=command)
            harness_clients.append(policy_client)
            policy_client.start()
            adapter = ResearchGymPolicyAdapter(case_id=case.case_id, metric=objective_name(args, case),
                                               alpha=args.acquisition_alpha, eta=args.acquisition_eta, seed=args.seed,
                                               beta=args.acquisition_beta, z_clip=args.acquisition_z_clip,
                                               proposal_facts=facts)
            from ldm_tts.harness.container import resolve_container_user
            selector.policy_adapter = adapter
            selector.policy_controller = PolicyResearchController(
                client=policy_client, adapter=adapter, root=run_dir / "policy_harness", account=runtime.consume_many,
                executor=DockerPolicyExecutor(args.policy_runner_image, args.harness_docker_host,
                                              resolve_container_user(args.harness_container_user, args.harness_docker_host)),
                recovery_budget=lambda: min(args.harness_wall_time_seconds * 2, clock.remaining()))
            selector.policy_hooks = (
                lambda: pricing.check(runtime),
                lambda: account_pool_tokens(runtime, pricing, run_dir / "policy_harness", "policy"))

    status, code = "completed", 0
    try:
        result = run_campaign(
            CampaignRequest(run_dir=run_dir, budget=budget, config=_campaign_config(args), resume=resume,
                            contract_sha256=contract.digest, contract_profile=profile, runtime_hook=runtime_hook,
                            context={"case": case.case_id, "mock": args.mock},
                            artifact_projector=lambda runtime, engine: _report(args, case, runtime, clock, source_info,
                                                                               engine.stop_reason)),
            CampaignRecipe(task_spec=spec, expander=campaign_expander, candidate_domain=domain, evaluator=evaluator,
                           surrogate_encoder=encoder if ldm else None, selector=selector),
        )
        stop = result.engine.stop_reason
        if stop not in COMPLETED_STOPS:
            status, code = "stopped_" + stop, 1
            result.runtime.status.update("stopped", phase=stop, budget=result.runtime.budget)
        report = result.projected
    except WallClockExhausted as exc:
        status, code = "stopped_wall_clock", 0
        report = _pause(holder, args, case, clock, source_info, "stopped", "wall_clock_exhausted", str(exc), expander)
    except (EndpointRequestError, HarnessError) as exc:
        status, code = "paused_endpoint_unavailable", 2
        report = _pause(holder, args, case, clock, source_info, status, "proposal", str(exc), expander)
    except DevicesUnavailable as exc:
        status, code = "paused_devices_unavailable", 2
        report = _pause(holder, args, case, clock, source_info, status, "evaluation_preparation", str(exc), expander)
    except ProposalExhausted as exc:
        runtime = holder.get("runtime")
        if runtime is not None:
            runtime.record("proposal_repair_exhausted", exc.metadata)
            consume_proposal_attempts(runtime, exc.attempts)
        status, code = "paused_proposal_exhausted", 1
        report = _pause(holder, args, case, clock, source_info, status, "proposal_repair", str(exc), expander)
    except ApiBudgetExhausted as exc:
        status, code = "stopped_api_budget", 1
        report = _pause(holder, args, case, clock, source_info, "stopped", "api_budget", str(exc), expander)
    except BudgetExceededError as exc:
        status, code = "paused_resource_budget", 1
        report = _pause(holder, args, case, clock, source_info, status, "resource_budget", str(exc), expander)
    finally:
        for client in reversed(harness_clients):
            client.close()
        clock.stop()
    report = dict(report or {})
    report["status"] = status
    atomic_json_write(run_dir / "result.json", report)
    return report, code


def _pause(holder, args, case, clock, source_info, status, phase, message, expander):
    runtime = holder.get("runtime")
    if runtime is None:
        return {"status": status, "message": message}
    details = {"proposal_recovery_pass": expander.recovery_pass}
    if status == "stopped":
        runtime.status.update("stopped", phase=phase, message=message, budget=runtime.budget, details=details)
    else:
        runtime.pause(status, phase=phase, message=message + " Resolve the cause, then resume this directory.",
                      details=details)
    return _report(args, case, runtime, clock, source_info, phase)


def _report(args, case, runtime, clock, source_info, stop_reason):
    run_dir = runtime.run_dir
    checkpoint = runtime.load_checkpoint() or {}
    observations = checkpoint.get("observations", [])
    objective = objective_name(args, case)
    events = runtime.events()
    atomic_json_write(run_dir / "search_manifest.json", {
        "rounds": [e for e in events if e.get("event_type") in ("reservoir_expanded", "reservoir_built")]})
    atomic_json_write(run_dir / "selection_record.json", {
        "selections": [e for e in events if e.get("event_type") == "candidates_selected"]})
    if args.proposal_mode == "harness":
        pools = ["harness"] + (["policy_harness"] if args.search_method == "ldm_harness_compiled" else [])
        atomic_json_write(run_dir / "harness_provenance.json", {"schema_version": 1, "pools": {
            pool: [f"{pool}/manifest.json"] for pool in pools}})
    rows, best, best_value = [], None, None
    for index, item in enumerate(observations):
        evaluation = item["evaluation"]
        value = evaluation["metrics"].get(objective) if evaluation["status"] == "succeeded" else None
        if value is not None and (best_value is None or value > best_value):
            best_value, best = value, item
        rows.append({"evaluation": index + 1, "round": item.get("round_idx"), "candidate_id": item["candidate"]["candidate_id"],
                     "status": evaluation["status"], "score": "" if value is None else value,
                     "best_so_far": "" if best_value is None else best_value, "error": evaluation.get("error", "")[:200]})
    _write_csv(run_dir / "evaluations.csv", rows)
    _write_csv(run_dir / "trajectory.csv", [
        {"evaluation": r["evaluation"], "round": r["round"], "candidate_id": r["candidate_id"], "score": r["score"]}
        for r in rows if r["status"] == "succeeded"])
    if best is not None:
        (run_dir / "best_program.py").write_text(best["candidate"]["payload"]["program"])
    report = {
        "task": "researchgym", "case": case.case_id, "search_method": args.search_method, "mock": args.mock,
        "qualification": "draft", "metric": objective, "protocol": None if args.mock else case.protocol,
        "comparability": "synthetic mock fixture" if args.mock else
        "official grader on a declared sub-protocol; not comparable to paper tables",
        "stop_reason": stop_reason, "evaluation_attempts": len(rows),
        "successful_evaluations": sum(r["status"] == "succeeded" for r in rows),
        "best": None if best is None else {"candidate_id": best["candidate"]["candidate_id"], "score": best_value,
                                            "round": best.get("round_idx"), "program": "best_program.py"},
        "source": source_info, "benchmark_clock": clock.snapshot(),
        "budget": json.loads((run_dir / "budget.json").read_text()),
    }
    atomic_json_write(run_dir / "result.json", report)
    return report


def _write_csv(path, rows):
    fields = list(rows[0]) if rows else ["evaluation", "round", "candidate_id", "score"]
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
