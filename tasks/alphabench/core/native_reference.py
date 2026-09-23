"""Official native search on one shared runtime, separate from the LDM engine."""

from hashlib import sha256
import json
import math
from pathlib import Path
from uuid import uuid4

from ldm_tts.contracts import Candidate, RawProposal
from ldm_tts.contracts.evaluation import EvaluationPaused
from ldm_tts.engine.run_store import BudgetExceededError, CampaignRuntime
from .candidate import FactorDomain
from .collection import AcceptedActions
from .gateway import OracleGateway
from .initialization import alpha158_seeds
from .native_benchmark import load_benchmark
from .native_evaluation import NativeEvaluator, NativeSearchBudgetExhausted
from .native_generator import NativeGenerator, SOURCES
from .native_runtime import NativeRuntime
from .native_source import SOURCE_HASHES, load_algorithms, verified_sources
from .receipts import Receipts


def prepare_native(protocol, source_root):
    if protocol.objective != "rank_ic":
        raise ValueError("the pinned native algorithms optimize rank_ic; another search objective is unsupported")
    if source_root is None:
        raise ValueError("native execution requires --upstream-root with the pinned AlphaBench source")
    verified_sources(source_root, SOURCES)
    if protocol.profile != "ldm_matched_v1":
        from .source_profiles import prepare_source
        return prepare_source(protocol, source_root)
    verified_sources(Path(source_root) / "searcher/algo", SOURCE_HASHES)
    parameters = protocol.native_parameters
    required = {"oracle_workers", "enable_reason", "accept_threshold"}
    required |= {"workers"} if protocol.method != "alphabench_ea" else {
        "N", "mutation_rate", "crossover_rate", "pool_size", "seeds_top_k"}
    if protocol.method == "alphabench_tot":
        required |= {"N", "top_k"}
    if set(parameters) != required:
        raise ValueError("native_parameters must explicitly contain exactly: " + ", ".join(sorted(required)))
    for name, value in parameters.items():
        if name == "enable_reason":
            if type(value) is not bool:
                raise ValueError("enable_reason must be a boolean")
        elif name in {"accept_threshold", "mutation_rate", "crossover_rate"}:
            if type(value) not in (float, int) or not math.isfinite(value):
                raise ValueError(name + " must be finite")
            if name != "accept_threshold" and not 0 <= value <= 1:
                raise ValueError(name + " must be between zero and one")
        elif type(value) is not int or value < 1:
            raise ValueError(name + " must be a positive integer")
    return {"algorithm": {**{key: value for key, value in parameters.items() if key != "oracle_workers"},
        "rounds": protocol.rounds, "model": protocol.model, "temperature": protocol.temperature},
        "oracle_workers": parameters["oracle_workers"]}


def committed_native_state(runtime, method):
    states = [{"boundary": event["payload"]["id"], **event["payload"]["output"]}
              for event in runtime.events()
              if event["event_type"] == "native_boundary" and event["payload"]["kind"] == "state"]
    if method == "alphabench_cot":
        chains = {item["boundary"].rsplit("/", 1)[0]: item["state"]["chain"]
                  for item in states if item["kind"] == "cot.chain"}
        pool = [{key: row[key] for key in ("name", "expression", "metrics")}
                for chain in chains.values() for row in chain if row.get("expression")]
    elif method == "alphabench_tot":
        pool, seen = [], set()
        for item in states:
            if item["kind"] != "tot.node":
                continue
            for candidate in item["state"]["candidates"]:
                expression = candidate["expression"]
                if expression not in seen:
                    seen.add(expression)
                    pool.append({"name": candidate["name"], "expression": expression,
                        "metrics": item["state"]["evaluations"].get(candidate["name"], {})})
    else:
        pools = [item["state"]["pool"] for item in states if item["kind"] == "ea.pool"]
        pool = pools[-1] if pools else []
    return {"states": states, "native_final_pool": pool}


def run_native(protocol, args, spec, client, run_dir, initial_observations, config, contract_identity):
    resume = (run_dir / "campaign.json").exists()
    runtime = CampaignRuntime.open(run_dir, task="alphabench", run_id=None if resume else "alphabench-" + uuid4().hex,
        resume=resume, config={"mock": args.mock, "protocol": protocol.to_dict()}, task_spec=spec,
        **contract_identity,
        budget_limits={**protocol.budgets, "external_evaluations": protocol.evaluations,
            "expensive_evaluation_attempts": protocol.evaluations, "selected_candidates": protocol.evaluations})
    try:
        runtime.consume_many({key: 0 for key in runtime.budget.limits}, usage_key="task:initialize-counters")
        gateway = OracleGateway(protocol, runtime, endpoint=args.oracle_url, mock=args.mock)
        gateway.preflight()
        stages = Receipts(run_dir / "private/stages")
        stages.accept("native_reference_contract", {"config": config, "protocol": protocol.identity,
            "adapter_sha256": sha256(Path(__file__).read_bytes()).hexdigest()})
        scheduler = NativeRuntime(runtime, gateway.host)
        evaluator = NativeEvaluator(scheduler, gateway, initial_observations, workers=config["oracle_workers"])
        generator = NativeGenerator(protocol, runtime, client, gateway, AcceptedActions(run_dir), args.upstream_root)
        prefix = "mock_" if args.mock else ""
        seeds = [{"name": item.candidate.payload["name"], "expression": item.candidate.payload["expression"],
                  "metrics": {key.removeprefix(prefix): value for key, value in item.metrics.items()}}
                 for item in initial_observations if item.evaluation.succeeded and item.metrics]
        if not seeds and (protocol.profile != "ldm_matched_v1" or protocol.method != "alphabench_ea"):
            raise EvaluationPaused("native entry requires at least one successful initial seed",
                                   status="paused_no_valid_seeds")
        if protocol.profile == "upstream_benchmark_v1" and len(seeds) != config["initialization"]["baseline_count"]:
            raise EvaluationPaused("example entry requires all 42 measured Alpha158 baseline factors",
                                   status="paused_incomplete_initialization")
        result = stages.load("native_algorithm_result")
        with load_algorithms(args.upstream_root, run_dir / "private", scheduler) as module:
            if result is None and evaluator.stopped is None:
                if protocol.profile == "upstream_benchmark_v1":
                    baseline = alpha158_seeds(args.upstream_root, ("kbar", "rolling", "price"))
                    initial = {item.candidate_id: item for item in initial_observations if item.evaluation.succeeded}
                    if len(initial) != len(baseline):
                        raise EvaluationPaused("example baseline contains duplicate or failed factor identities",
                                               status="paused_incomplete_initialization")

                    def initialization_fn(identity, factors, *, market):
                        if factors != baseline or market != protocol.market:
                            raise ValueError("example baseline differs from the frozen source and market")
                        rows = []
                        for factor in factors:
                            candidate = FactorDomain(protocol.backend).admit(RawProposal(factor, "initialization"))
                            observation = initial.get(candidate.candidate_id) if isinstance(candidate, Candidate) else None
                            if observation is None:
                                raise EvaluationPaused("example baseline lacks a measured factor",
                                                       status="paused_incomplete_initialization")
                            metrics = {key.removeprefix(prefix): value for key, value in observation.evaluation.metrics.items()}
                            rows.append({**factor, "metrics": metrics})
                        return rows

                    native_config = json.loads(json.dumps(config["source_resolution"]["declared"]))
                    native_config["model"]["name"] = protocol.model
                    native_config["market"] = protocol.market
                    native_config["save_dir"] = str(run_dir / "private/native_output")
                    for name in ("cot", "tot", "ea"):
                        native_config[name]["enable"] = name == protocol.method.removeprefix("alphabench_")
                    algorithm = load_benchmark(args.upstream_root, run_dir / "private", scheduler, module,
                        initialization_fn=scheduler.callback("initialization_fn", initialization_fn),
                        search_fn=scheduler.callback("search_fn", generator), **evaluator.callbacks())
                    execute = lambda: algorithm(native_config)
                else:
                    algorithm = module.create_algo(protocol.method.removeprefix("alphabench_"), config["algorithm"],
                        search_fn=scheduler.callback("search_fn", generator), **evaluator.callbacks())
                    execute = lambda: algorithm.run(seeds, str(run_dir / "private/native_output"))
                try:
                    result = scheduler.run(execute)
                except NativeSearchBudgetExhausted:
                    pass
                else:
                    stages.accept("native_algorithm_result", result)
        settlement = evaluator.settle()
        settlement.update(generator.settle())
        state = committed_native_state(runtime, protocol.method)
        if result is not None and protocol.profile != "upstream_benchmark_v1":
            state["native_final_pool"] = result["final_pool"]
        execution = {"kind": "native_reference", "engine_native": False, "config": config,
            "algorithm_result": result, "algorithm_completed": result is not None,
            "stop_reason": "algorithm_completed" if result is not None else "search_batch_budget_exhausted",
            "remaining_search_allowance": runtime.budget.remaining("expensive_evaluation_attempts"),
            **state, **settlement}
        stages.accept("native_search_complete", execution)
        observations = initial_observations + evaluator.observations()
        runtime.checkpoint({"observations": [item.to_dict() for item in observations], "native": execution})
        return runtime, gateway, observations, execution
    except (EvaluationPaused, BudgetExceededError) as exc:
        pause = exc if isinstance(exc, EvaluationPaused) else EvaluationPaused(str(exc), status="paused_budget")
        runtime.pause(pause.status, phase="native_search", message=str(pause))
        if pause is exc:
            raise
        raise pause from exc
    except Exception as exc:
        runtime.fail(exc)
        raise
