"""ReaSyn Science Benchmark assembly through ldm_tts.campaign.run_campaign."""

from __future__ import annotations

import json
import math
import os
from contextlib import ExitStack
from copy import copy
from pathlib import Path

from ldm_tts.campaign import (
    CampaignBudget,
    CampaignRecipe,
    CampaignRequest,
    run_campaign,
)
from ldm_tts.data import DataCollectionSink
from ldm_tts.engine.runtime import consume_proposal_attempts
from ldm_tts.engine.run_store import atomic_json_write, unique_run_dir, BudgetExceededError
from ldm_tts.engine.reporting import load_successful_observations
from ldm_tts.harness import HarnessError
from ldm_tts.registration.experiment import (
    load_active_experiment_contract,
    load_experiment_contract,
    snapshot_experiment_contract,
)
from ldm_tts.transport.openai import (
    OpenAICompatibleProposalClient,
    EndpointRequestError,
    generation_body,
)
from . import harness_runtime
from .candidate import ReaSynDomain
from .chemistry import canonicalize
from .evaluator import ReconstructionEvaluator, TDCOracleEvaluator
from .metrics import reconstruction_metrics
from .projector import Projector, ProjectionInterruptedError
from .proposals import DEFAULT_PROPOSAL_RECOVERY_SEED_SPAN, ReaSynExpander, ProposalExhausted
from .selection import TanimotoGPSelector, AcquisitionTiltedSelector
from .surrogate import MoleculeEncoder
from .provenance import check_assets, harness_component_digests, scientific_identity
from .reporting import write_campaign_report
from .task_spec import describe_ldm_task
from .workflow_args import jsonable_args, parse_args

TASK_ROOT = Path(__file__).resolve().parents[1]


def _provider_api_key():
    return os.environ.get(
        "LLM_API_KEY",
        os.environ.get("LDM_LLM_API_KEY", os.environ.get("OPENAI_API_KEY", "")),
    )


def _targets(args):
    if args.benchmark == "tdc":
        return [""]
    if args.target_smiles:
        return [canonicalize(args.target_smiles, mock=args.mock, stereo=False)]
    if args.mock:
        return ["CCO"]
    names = {
        "zinc250k": "test_zinc250k.txt",
        "enamine": "enamine_smiles_1k.txt",
        "chembl": "chembl_filtered_1k.txt",
    }
    path = args.targets_file or args.upstream_root / "data" / names.get(
        args.dataset, "targets.txt"
    )
    lines = [
        s.strip().split(",")[0] for s in path.read_text().splitlines() if s.strip()
    ]
    if lines and lines[0].lower() == "smiles":
        lines = lines[1:]
    if len(lines) < args.target_limit:
        raise ValueError(
            f"target-limit requests {args.target_limit} targets, but {path} contains "
            f"only {len(lines)}; use a complete dataset or an explicit smaller subset"
        )
    targets = [canonicalize(s, stereo=False) for s in lines[: args.target_limit]]
    if not targets or len(set(targets)) != len(targets):
        raise ValueError("requested targets must be nonempty and unique")
    return targets


def _status_recovery_pass(status_payload):
    details = status_payload.get("details", {}) if isinstance(status_payload, dict) else {}
    value = details.get("proposal_recovery_pass") if isinstance(details, dict) else None
    if value is None:
        return 0
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("status proposal_recovery_pass must be a nonnegative integer")
    return value


def _proposal_recovery_details(expander):
    value = getattr(expander, "recovery_pass", 0)
    if value:
        return {"proposal_recovery_pass": value}
    return None


def _proposal_recovery_seed_span(args, run_dir, *, resume):
    config_path = Path(run_dir) / "config.json"
    if resume and config_path.exists():
        original = json.loads(config_path.read_text())
        value = original.get("proposal_recovery_seed_span", original.get("iterations"))
    else:
        value = max(
            DEFAULT_PROPOSAL_RECOVERY_SEED_SPAN,
            args.iterations,
            args.max_oracle_calls,
        )
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("proposal_recovery_seed_span must be a positive integer")
    return value


def main(argv=None):
    args = parse_args(argv)
    spec = describe_ldm_task(args)
    contract, profile = load_active_experiment_contract()
    contract = contract or load_experiment_contract(TASK_ROOT / "experiment.json")
    if args.dry_run:
        print(
            json.dumps(
                {
                    "task": "reasyn",
                    "contract_profile": profile,
                    "configuration": jsonable_args(args),
                    "ldm_task_spec": spec.to_dict(),
                },
                indent=2,
            )
        )
        return 0
    check_assets(args)
    if args.search_method == "bo" or args.initialization_mode == "shared_start":
        from .chemistry import MOCK_SMILES
        if args.mock:
            args.bo_targets = list(MOCK_SMILES)
        elif args.bo_targets_file:
            args.bo_targets = [canonicalize(line.strip()) for line in args.bo_targets_file.expanduser().read_text().splitlines() if line.strip()]
        elif args.benchmark == "reconstruction" and args.proposal_mode == "baseline":
            args.bo_targets = []
        else:
            raise ValueError("Real BO/shared_start requires a score-blind --bo-targets-file pool")
        if args.bo_targets_file and not args.bo_targets:
            raise ValueError("BO target pool is empty")
    targets = _targets(args)
    root = (
        args.resume_from.resolve()
        if args.resume_from
        else ((args.out_dir / args.run_name).resolve() if args.run_name else unique_run_dir(args.out_dir.resolve()))
    )
    if args.run_name and not args.resume_from and root.exists():
        raise FileExistsError("Named run directory already exists; use --resume-from")
    root.mkdir(parents=True, exist_ok=True)
    identity = {
        "benchmark": args.benchmark,
        "oracle": args.oracle if args.benchmark == "tdc" else None,
        "targets": targets,
        "mock": args.mock,
    }
    manifest = root / "targets_manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != identity:
        raise ValueError("resume target manifest mismatch")
    atomic_json_write(manifest, identity)
    outputs = []
    exit_code = 0
    for index, target in enumerate(targets):
        run_dir = root if len(targets) == 1 else root / f"target-{index:05d}"
        result, code = _run_one(
            args,
            spec,
            contract,
            profile,
            run_dir,
            target,
            resume=args.resume_from is not None
            and (run_dir / "campaign.json").exists(),
        )
        outputs.append(result)
        exit_code = max(exit_code, code)
        if code == 2:
            break  # Endpoint pause must not burn through other targets.
    if args.benchmark == "reconstruction":
        all_rows = []
        for item in outputs:
            all_rows.extend(item.get("reconstruction_rows", []))
        aggregate = reconstruction_metrics(targets, all_rows, mock=args.mock)
        aggregate.update(
            task="reasyn",
            benchmark="reconstruction",
            mock=args.mock,
            qualification="draft",
            completed_campaigns=len(outputs),
            requested_campaigns=len(targets),
            comparison="LDM adaptation; projection budget/config must match baseline",
        )
        atomic_json_write(root / "benchmark_result.json", aggregate)
    print(
        json.dumps(
            {
                "task": "reasyn",
                "benchmark": args.benchmark,
                "mock": args.mock,
                "run_dir": str(root),
                "campaigns": len(outputs),
                "exit_code": exit_code,
            },
            indent=2,
        )
    )
    return exit_code


def _run_one(args, spec, contract, profile, run_dir, target, *, resume):
    run_dir.mkdir(parents=True, exist_ok=True)
    previous_status_payload = (
        json.loads((run_dir / "status.json").read_text())
        if resume and (run_dir / "status.json").exists()
        else {}
    )
    previous_status = previous_status_payload.get("status", "")
    previous_proposal_recovery_pass = _status_recovery_pass(previous_status_payload)
    # Preserve scientific config across resume; only iterations may extend.
    configuration = jsonable_args(args)
    proposal_recovery_seed_span = _proposal_recovery_seed_span(args, run_dir, resume=resume)
    args.proposal_recovery_seed_span = proposal_recovery_seed_span
    configuration.update(proposal_samples=args.reservoir_size,
        proposal_candidates_per_request=args.proposal_batch_size,
        proposal_counting="bounded_minibatches",
        proposal_recovery_seed_span=proposal_recovery_seed_span,
        harness_component_digests=harness_component_digests(args))
    scientific = scientific_identity(args, configuration, target, contract)
    identity_path = run_dir / "scientific_identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != scientific:
        raise ValueError("resume scientific configuration mismatch")
    atomic_json_write(identity_path, scientific)
    snapshot_experiment_contract(contract, run_dir, profile=profile)
    projector = Projector(args, run_dir)
    sink = DataCollectionSink.from_env(default_root=run_dir / "ldm_data")
    client = None
    if args.proposal_mode == "harness" and (not args.llm_url or not args.llm_model):
        raise ValueError("Set LLM_BASE_URL and LLM_MODEL_NAME for real Harness proposals")
    if args.proposal_mode == "openai" and args.search_method != "bo":
        if not args.llm_url or not args.llm_model:
            raise ValueError(
                "Set LLM_BASE_URL and LLM_MODEL_NAME for real LDM proposals"
            )
        client = OpenAICompatibleProposalClient(
            url=args.llm_url,
            model=args.llm_model,
            api_key=_provider_api_key(),
            max_tokens=args.llm_max_tokens,
            max_retries=0,
            wire_api=args.llm_wire_api,
            temperature=args.llm_temperature,
            extra_body=generation_body(wire_api=args.llm_wire_api, reasoning=args.llm_reasoning,
                                       extra_body=json.loads(args.llm_extra_body_json)),
        )
    expander = ReaSynExpander(args, projector, sink, target=target, client=client)
    evaluator = (
        ReconstructionEvaluator(args, projector, target)
        if target
        else TDCOracleEvaluator(args, run_dir)
    )
    encoder = MoleculeEncoder(mock=args.mock)
    selector = TanimotoGPSelector(
        objective_name=spec.objectives[0].name,
        beta=args.acquisition_beta,
        feature_version=encoder.version,
        history_limit=args.gp_history_limit,
    )
    cap = args.iterations * args.evaluations_per_round
    if not target:
        cap = min(cap, args.max_oracle_calls)
    checkpoint_path = run_dir / "checkpoint.json"
    resuming_complete = (
        resume
        and checkpoint_path.exists()
        and len(load_successful_observations(checkpoint_path)) >= cap
    )
    batches_per_round = math.ceil(args.reservoir_size / args.proposal_batch_size) + args.max_replenishment_batches
    request_cap = (args.iterations + args.recovery_attempts) * batches_per_round
    proposal_cap = args.reservoir_size + args.max_replenishment_batches * args.proposal_batch_size
    projection_rounds = args.iterations if target else args.iterations + args.recovery_attempts
    projection_cap = (cap if target else projection_rounds * proposal_cap) + args.projection_retry_targets
    uses_harness = args.proposal_mode == "harness"
    provider_preflight_required = client is not None or uses_harness
    extra = {
        "llm_requests": request_cap if client else 0,
        "proposal_request_attempts": request_cap if (client or uses_harness) else 0,
        "proposal_attempts": request_cap if (client or uses_harness) else 0,
        "recovery_attempts": args.recovery_attempts,
        "harness_turns": request_cap * (args.harness_sessions + int(args._research_strategy.requires_main)) if uses_harness else 0,
        "main_research_turns": request_cap if args._research_strategy.requires_main else 0,
        "oracle_calls": args.max_oracle_calls if not target else 0,
        "projection_targets": projection_cap,
        "projection_cycle_allowance": projection_cap * args.num_cycles,
        "endpoint_preflight_requests": (
            json.loads((run_dir / "budget.json").read_text())
            .get("counters", {})
            .get("endpoint_preflight_requests", 0)
            + (0 if resuming_complete else 1)
            if resume and (run_dir / "budget.json").exists()
            else 1
        )
        if provider_preflight_required
        else 0,
    }
    budget = CampaignBudget(
        rounds=args.iterations,
        reservoir_size=args.reservoir_size,
        batch_size=args.evaluations_per_round,
        target_observations=cap,
        max_evaluation_attempts=cap,
        extra_limits=extra,
    )
    if args.search_method.startswith("ldm"):
        selector = AcquisitionTiltedSelector(selector, alpha=args.acquisition_alpha,
            eta=args.acquisition_eta, z_clip=args.acquisition_z_clip, seed=args.seed,
            pool_size=args.bo_pool_size)
    elif args.search_method in ("llm", "harness", "baseline"):
        selector, encoder = None, None
    holder = {}
    search_expander = expander
    if args.initialization_mode == "shared_start":
        initial_args = copy(args)
        initial_args.search_method = "bo"
        initial_args.proposal_mode = "mock" if args.mock else "baseline"
        initializer = ReaSynExpander(initial_args, projector, sink, target=target)
        class PairedStartExpander:
            def expand(self, request):
                if request.round_idx == 0:
                    from dataclasses import replace
                    return replace(initializer.expand(request), selection_mode="reservoir_order")
                return search_expander.expand(request)
        expander = PairedStartExpander()

    def runtime_hook(runtime):
        holder["runtime"] = runtime
        if resuming_complete:
            return  # Completed campaigns can be reopened without live services.
        projector.before_project = lambda count: runtime.consume_many(
            {
                "projection_targets": count,
                "projection_cycle_allowance": count * args.num_cycles,
            }
        )
        proposal_recovery_pass = 0
        if resume and (previous_status.startswith("paused") or previous_status == "failed"):
            consumed = runtime.consume("recovery_attempts")
            if previous_status == "paused_proposal_exhausted":
                proposal_recovery_pass = int(consumed)
                runtime.record(
                    "proposal_recovery_pass_started",
                    {"proposal_recovery_pass": proposal_recovery_pass},
                )
            else:
                proposal_recovery_pass = previous_proposal_recovery_pass
        search_expander.recovery_pass = proposal_recovery_pass
        def before_request():
            amounts = {"proposal_request_attempts": 1}
            if client:
                amounts["llm_requests"] = 1
            runtime.consume_many(amounts)
        search_expander.before_request = before_request
        if uses_harness:
            api_key = _provider_api_key()
            search_expander.client = harness_runtime.open_proposals(
                args, runtime, target, stack=stack, api_key=api_key,
            )
            if args.search_method == "ldm_harness_compiled":
                selector.policy_adapter, selector.policy_controller = harness_runtime.open_policy(
                    args, runtime, target, encoder, stack=stack, api_key=api_key,
                )

        if not target:
            evaluator.before_oracle = lambda: runtime.consume("oracle_calls")
        if client:
            # Preflight is separately reported and runs on every new/resumed process.
            runtime.consume("endpoint_preflight_requests")
            preflight = client.preflight()
            runtime.record("endpoint_preflight_succeeded", preflight)

    with ExitStack() as stack:
        try:
            result = run_campaign(
                CampaignRequest(
                    run_dir=run_dir,
                    budget=budget,
                    config=configuration,
                    resume=resume,
                    contract_sha256=contract.digest,
                    contract_profile=profile,
                    runtime_hook=runtime_hook,
                    context={
                        "original_target": target,
                        "oracle": args.oracle,
                        "mock": args.mock,
                    },
                    artifact_projector=lambda runtime, engine: write_campaign_report(
                        args, runtime, target
                    ),
                ),
                CampaignRecipe(
                    task_spec=spec,
                    expander=expander,
                    candidate_domain=ReaSynDomain(args.benchmark, mock=args.mock),
                    evaluator=evaluator,
                    surrogate_encoder=encoder,
                    selector=selector,
                ),
            )
        except (EndpointRequestError, HarnessError, ProjectionInterruptedError) as exc:
            runtime = holder.get("runtime")
            status = "paused_projection_interrupted" if isinstance(exc, ProjectionInterruptedError) else "paused_endpoint_unavailable"
            if runtime:
                runtime.pause(status, phase="proposal_or_projection",
                    message="Resolve the interrupted service, then resume this directory within the separate recovery allowance.",
                    details=_proposal_recovery_details(search_expander))
                report = write_campaign_report(args, runtime, target)
                report["status"] = status
                atomic_json_write(run_dir / "result.json", report)
            else:
                report = {"status": status}
            return report, 2
        except ProposalExhausted as exc:
            runtime = holder.get("runtime")
            runtime.record("proposal_replenishment_exhausted", exc.metadata)
            if exc.attempts:
                consume_proposal_attempts(runtime, exc.attempts)
            runtime.pause(
                "paused_proposal_exhausted",
                phase="proposal_replenishment",
                message=str(exc),
                details=_proposal_recovery_details(search_expander),
            )
            report = write_campaign_report(args, runtime, target)
            report.update(status="paused_proposal_exhausted", proposal_diagnostics=exc.metadata)
            atomic_json_write(run_dir / "result.json", report)
            return report, 1
        except BudgetExceededError as exc:
            runtime = holder.get("runtime")
            runtime.pause(
                "paused_resource_budget",
                phase="resource_budget",
                message=str(exc),
                details=_proposal_recovery_details(search_expander),
            )
            report = write_campaign_report(args, runtime, target)
            report["status"] = "paused_resource_budget"
            atomic_json_write(run_dir / "result.json", report)
            return report, 1
    complete = result.engine.summary["successful_evaluation_count"] == cap
    if not complete:
        result.runtime.status.update("stopped", phase="incomplete_scientific_budget", budget=result.runtime.budget)
    return result.projected, 0 if complete else 1
