"""Copper v2 wiring into the shared Campaign, research tree and LDM engine."""
import argparse
import json
from contextlib import ExitStack
from pathlib import Path
from ldm_tts.campaign import CampaignRequest, CampaignBudget
from ldm_tts.contracts import RawProposal
from ldm_tts.contracts.evaluation import EvaluationPaused
from ldm_tts.engine import LDMEngine, LDMEngineState, InitialRoundReservoirExpander
from ldm_tts.engine.expansion import CallableReservoirExpander, ExpansionResult
from ldm_tts.engine.run_store import atomic_json_write
from ldm_tts.harness import HarnessError
from ldm_tts.harness.research_strategy import configure_research_strategy
from ..core.candidate import seal
from ..core.execution import Execution, digest, docker, source_identity
from ..core.harness_runtime import open_search
from ..core.encoding import CodeEncoder
from ..core.surrogate import LossSelector
from .public_catalog import load_catalog, catalog_digest
from .service_client import ServiceEvaluator
from .budget import ScientificSlots, fully_accounted
from .domain import CheckedDomain
from .interface_checks import InterfaceChecks
from .research import prepare_resources, PublicTrainingExport
from .task_spec import describe_task


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Copper v2 cross-case fusion LDM")
    p.add_argument("--public-catalog", type=Path, required=True)
    p.add_argument("--evaluation-socket", type=Path, required=True)
    p.add_argument("--evaluation-contract", required=True)
    p.add_argument("--evaluation-service-uid", type=int, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--embedding-model-path", type=Path, required=True)
    p.add_argument("--embedding-device", default="cuda:0")
    p.add_argument("--api-key-file", type=Path)
    p.add_argument("--iterations", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--candidates-per-researcher", type=int, default=8)
    p.add_argument("--evaluations-per-round", type=int, default=16)
    p.add_argument("--solver-seconds", type=int, default=120)
    p.add_argument("--harness-seconds", type=int, default=900)
    p.add_argument("--evaluator-image", default="ldm-copper-baselines:20261024")
    p.add_argument("--sidecar-image", default="ldm-pi-harness:copper-20261023")
    p.add_argument("--harness-cache-dir", type=Path, required=True)
    p.add_argument("--harness-research-strategy-config", type=Path)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    a.encoding = "fusion"
    a.search_method = "ldm_harness"
    a.llm_url, a.llm_model, a.thinking = "https://api.deepseek.com", "deepseek-flash", "low"
    a.mock = False
    if 1+a.iterations*a.evaluations_per_round > 321 or a.seed < 0 or a.iterations < 0 or a.candidates_per_researcher < 1 or not 1 <= a.evaluations_per_round <= 4*a.candidates_per_researcher:
        p.error("invalid candidate/evaluation budget")
    if a.solver_seconds < 1 or a.harness_seconds < 1:
        p.error("invalid wall budget")
    configure_research_strategy(a, enabled=True, measurement_supported=False)
    return a


def main(argv=None):
    args = parse_args(argv)
    catalog = load_catalog(args.public_catalog)
    if catalog["split"] != "train":
        raise ValueError("algorithm discovery accepts only the training suite")
    if catalog["stage"] == "formal":
        if (args.iterations,args.evaluations_per_round,args.candidates_per_researcher,args.solver_seconds)!=(20,16,8,120) or args.seed not in (42,43,3407):
            raise ValueError("formal campaign differs from preregistered budget or seeds")
    suite_digest = catalog_digest(args.public_catalog)
    args.case_id = "training-suite-" + suite_digest[:12]
    spec = describe_task(suite_digest)
    if args.dry_run:
        print(json.dumps(spec.to_dict()))
        return 0
    root = args.out_dir.resolve(); root.mkdir(parents=True, exist_ok=True)
    args._resources = prepare_resources(root)
    args._prepare_public = PublicTrainingExport(args.public_catalog)
    config = {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()
              if not k.startswith("_") and k not in {"api_key_file", "resume", "out_dir", "dry_run"}}
    config.update(implementation_sha256=source_identity(), suite_digest=suite_digest)
    for key in ("evaluator_image", "sidecar_image"):
        config[key] = docker("image", "inspect", getattr(args,key), "--format", "{{.Id}}").stdout.strip()
    args.sidecar_image = config["sidecar_image"]
    evaluations = 1 + args.iterations*args.evaluations_per_round
    proposals = 4*args.candidates_per_researcher
    main_turns = args.iterations if args._research_strategy.requires_main else 0
    budget = CampaignBudget(rounds=args.iterations+1,reservoir_size=proposals,batch_size=args.evaluations_per_round,
        target_observations=evaluations,max_evaluation_attempts=evaluations,extra_limits=dict(
            algorithm_feedback_slots=evaluations,
            interface_jobs=6*(3*4*args.candidates_per_researcher**2*args.iterations+1),
            harness_turns=4*args.iterations+main_turns,main_research_turns=main_turns))
    runtime = CampaignRequest(run_dir=root,budget=budget,config=config,resume=args.resume).open_runtime(spec)
    try:
        with ExitStack() as stack:
            execution = Execution(root/"jobs",config["evaluator_image"],runtime.consume_many)
            slots = ScientificSlots(runtime.consume_many)
            checks = InterfaceChecks(root,execution,slots)
            evaluator = ServiceEvaluator(root,args.evaluation_socket,args.evaluation_contract,
                service_uid=args.evaluation_service_uid,slots=slots)
            evaluator.verify_contract(suite_digest,args.solver_seconds,evaluations)
            search = open_search(args,root,{"case_id":args.case_id},stack,checks,runtime)
            seed = seal(root,b"from tasks.copper_inversion.v2.baselines import initialize\ndef solve(problem,budget,seed):\n    return initialize(problem)\n")
            def initialize(_):
                checks(seed)
                return ExpansionResult(proposals=(RawProposal(seed,"fixed_public_seed"),),selection_mode="reservoir_order")
            expander = InitialRoundReservoirExpander(initializer=CallableReservoirExpander(initialize),
                search_expander=search,initial_reservoir_size=1)
            state = LDMEngineState()
            if args.resume:
                checkpoint = runtime.load_checkpoint()
                if checkpoint is not None: state = LDMEngineState.from_checkpoint(checkpoint)
            engine = LDMEngine(task_spec=spec,expander=expander,candidate_domain=CheckedDomain(root,checks),
                evaluator=evaluator,runtime=runtime,surrogate_encoder=CodeEncoder(root,"fusion",args.embedding_model_path,args.embedding_device),
                selector=LossSelector("fusion"))
            try:
                outcome = engine.run(budget.engine_config(),state=state)
                stop_reason = outcome.stop_reason
            except EvaluationPaused as exc:
                if exc.status != "paused_budget" or not fully_accounted(runtime.budget,state.observations):
                    raise
                stop_reason = "scientific_source_budget"
            observations = state.observations
            successful = [o for o in observations if o.evaluation.succeeded]
            best = min(successful,key=lambda o:o.metrics["loss_total"],default=None)
            report = dict(stage=catalog["stage"],suite_digest=suite_digest,encoding="fusion",observations=len(observations),
                best=best.to_dict() if best else None,stop_reason=stop_reason,
                charged_sources=runtime.budget.counters.get("algorithm_feedback_slots",0),
                interface_rejected_sources=runtime.budget.counters.get("interface_rejected_sources",0),
                rounds_with_observations=max((o.round_idx for o in observations),default=0),
                completed_full_rounds=sum(sum(o.round_idx==r for o in observations)==args.evaluations_per_round
                                          for r in range(1,args.iterations+1)))
            atomic_json_write(root/"result.json",report)
            if best: (root/"best_solver.py").write_bytes((root/best.candidate.payload["source_path"]).read_bytes())
            print(json.dumps(dict(run_dir=str(root),observations=len(observations),best_iou=best.metrics["mean_iou"] if best else None)))
            complete = fully_accounted(runtime.budget,observations)
            if complete:
                runtime.finish(report,status="completed")
            return 0 if best and complete else 1
    except HarnessError as exc:
        runtime.pause("paused_harness",phase="harness",message=str(exc))
        raise
    except EvaluationPaused as exc:
        runtime.status.update("paused_outcome_unknown",phase="evaluation",details={"error":str(exc)},budget=runtime.budget)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
