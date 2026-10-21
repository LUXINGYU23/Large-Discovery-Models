"""Assemble the shared campaign and its private post-search stages."""

from contextlib import ExitStack
from dataclasses import replace
import json
import os
from pathlib import Path
from uuid import uuid4

from ldm_tts.campaign import CampaignBudget, CampaignRequest
from ldm_tts.contracts.evaluation import EvaluationPaused
from ldm_tts.engine.expansion import CallableReservoirExpander
from ldm_tts.engine import LDMEngine, LDMEngineState
from ldm_tts.engine.run_store import BudgetExceededError, CampaignRuntime, atomic_json_write, unique_run_dir
from ldm_tts.transport import CallableProposalClient
from ldm_tts.transport.openai import OpenAICompatibleProposalClient
from ldm_tts.registration.experiment import load_active_experiment_contract, snapshot_experiment_contract
from .candidate import FactorDomain
from .collection import AcceptedActions
from .data import verify_data_manifest
from .finalization import finalize, verify_report_artifacts
from .gateway import FactorEvaluator, OracleGateway
from .generator import DirectExpander, Generator
from .harness_runtime import open_proposals, open_policy
from .initialization import freeze_seed_source, run_initialization
from .native_reference import prepare_native, run_native
from .protocol import LDM_METHODS, digest
from .policy import CompiledFactorSelector
from .selection import FactorEncoder, FactorSelector


def mock_response(request, native_count=None):
    count = native_count if native_count is not None else json.loads(request.messages[1]["content"])["required"]
    offset = int(digest(request.messages)[:8], 16) % 10000
    return json.dumps({"generated" if native_count is not None else "candidates": [{"name": f"mock_{index}", "expression": f"Mean($close,{offset + index + 1})"}
                                     for index in range(count)]})


def make_client(protocol, mock, native_config=None, *, api_key=None):
    if mock:
        parameters = native_config["algorithm"] if native_config else protocol.native_parameters
        count = max(1, protocol.cold_seed_count, parameters.get("N", 1)) if protocol.method.startswith("alphabench_") else None
        return CallableProposalClient(lambda request: mock_response(request, count))
    key = api_key or os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise ValueError("DEEPSEEK_API_KEY is required in the Host environment")
    body = {"reasoning": {"effort": protocol.reasoning_effort}, "store": False}
    if protocol.method.startswith("alphabench_"):
        body["text"] = {"format": {"type": "json_object"}}
    return OpenAICompatibleProposalClient(url=protocol.endpoint, model=protocol.model, api_key=key,
        wire_api=protocol.wire_api, extra_body=body, temperature=protocol.temperature,
        max_retries=0, max_tokens=protocol.max_model_tokens, timeout_seconds=protocol.request_timeout)


def provider_key(args):
    key = args.api_key_file.read_text(encoding="utf-8").strip() if args.api_key_file else os.environ.get("DEEPSEEK_API_KEY", "")
    if not key:
        raise ValueError("DeepSeek key is required from --api-key-file or DEEPSEEK_API_KEY")
    return key


def bind_runner_contract(run_dir, protocol):
    path = run_dir / "experiment_contract.json"
    saved = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    if saved is not None:
        metadata = saved["snapshot"]
        if digest({key: value for key, value in saved.items() if key != "snapshot"}) != metadata["sha256"]:
            raise ValueError("runner contract snapshot integrity failure")
        expected = saved["evaluation"]["settings"]["protocol_digests"].get(metadata["profile"])
        if metadata["profile"] and expected != protocol.identity:
            raise ValueError("protocol differs from the frozen runner profile")
    contract, profile = load_active_experiment_contract()
    if contract is None:
        return {"contract_sha256": saved["snapshot"]["sha256"] if saved else protocol.identity,
                "contract_profile": saved["snapshot"]["profile"] if saved else protocol.profile}
    if contract.task_id != "alphabench":
        raise ValueError("active runner contract belongs to another task")
    if profile:
        selected = contract.profile(profile)
        if (selected.budget["search_evaluations"] != protocol.evaluations or
                contract.evaluation["settings"]["protocol_digests"].get(profile) != protocol.identity):
            raise ValueError("protocol differs from the locked runner profile")
    if saved is not None:
        if (saved["snapshot"]["sha256"], saved["snapshot"]["profile"]) != (contract.digest, profile):
            raise ValueError("runner contract changed across resume")
    else:
        snapshot_experiment_contract(contract, run_dir, profile=profile)
    return {"contract_sha256": contract.digest, "contract_profile": profile}


def run(args, protocol, spec):
    data_manifest = None
    if not args.mock:
        if not args.protocol_file or not args.data_manifest:
            raise ValueError("real runs require a frozen --protocol-file and verified --data-manifest")
        data_manifest = verify_data_manifest(args.data_manifest, protocol)
    native = protocol.method.startswith("alphabench_")
    if not native and protocol.profile != "ldm_matched_v1":
        raise ValueError("source profiles require the actual native algorithm entry")
    if protocol.method not in {"llm", "ldm", "harness", "ldm_harness", "ldm_harness_compiled"} and not native:
        raise ValueError("the selected method requires its native/Harness adapter; direct execution is forbidden")
    run_dir = args.resume_run or args.out_dir or unique_run_dir(Path(__file__).resolve().parents[1] / "runs" / ("mock" if args.mock else "campaign"))
    run_dir.mkdir(parents=True, exist_ok=True)
    protocol_path = run_dir / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text(encoding="utf-8")) != protocol.to_dict():
        raise ValueError("resume protocol identity mismatch")
    if protocol_path.exists() and not args.resume_run:
        raise ValueError("existing run requires --resume-run")
    atomic_json_write(protocol_path, protocol.to_dict())
    if data_manifest is not None:
        snapshot = run_dir / "data_manifest.json"
        if snapshot.exists():
            if json.loads(snapshot.read_text(encoding="utf-8")) != data_manifest:
                raise ValueError("run data audit differs from the verified manifest")
        else:
            atomic_json_write(snapshot, data_manifest)
    contract_identity = bind_runner_contract(run_dir, protocol)
    native_config = prepare_native(protocol, args.upstream_root) if native and protocol.profile != "ldm_matched_v1" else None
    if protocol.profile in {"upstream_searcher_v1", "upstream_benchmark_v1"}:
        if args.initialization_bundle or args.import_pool:
            raise ValueError("source initialization must come from its pinned entry")
        if protocol.profile == "upstream_searcher_v1":
            seed_file = native_config["initialization"]["seed_file"]
            if seed_file:
                expected = args.upstream_root / seed_file
                if args.seed_file and args.seed_file.read_bytes() != expected.read_bytes():
                    raise ValueError("seed file differs from the pinned source pool")
                args.seed_file = expected
            elif args.seed_file:
                raise ValueError("source cold initialization cannot be replaced with a seed file")
        elif args.seed_file:
            raise ValueError("example Alpha158 initialization cannot be replaced with a seed file")
        from .receipts import Receipts
        Receipts(run_dir / "private/stages").accept("source_entry_contract", native_config)
    if args.resume_run and (run_dir / "report_manifest.json").exists():
        verify_report_artifacts(run_dir, protocol)
        status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))["status"]
        if status != "completed":
            if status != "running":
                raise ValueError("published report cannot finish from a non-running status")
        freeze_seed_source(protocol, args, run_dir / "initialization")
        AcceptedActions(run_dir).export()
        if status != "completed":
            report = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
            CampaignRuntime.open(run_dir, task="alphabench", resume=True).finish(report)
        print(json.dumps({"task": "alphabench", "run_dir": str(run_dir.resolve()), "status": "completed", "replayed": True}))
        return 0
    if args.resume_run and (run_dir / "result.json").exists() and json.loads(
            (run_dir / "status.json").read_text(encoding="utf-8"))["status"] == "completed":
        raise ValueError("completed run lacks its report artifact manifest")
    if native and native_config is None:
        native_config = prepare_native(protocol, args.upstream_root)
    harness_method = protocol.method in {"harness", "ldm_harness", "ldm_harness_compiled"}
    key = provider_key(args) if harness_method or not args.mock else None
    client = make_client(protocol, args.mock, native_config, api_key=key)
    try:
        initial_observations, initial_pool, initial_budget = run_initialization(protocol, args, spec, client, run_dir)
    except EvaluationPaused as exc:
        print(json.dumps({"task": "alphabench", "phase": "initialization", "status": exc.status, "reason": str(exc)}))
        return 2
    initial_observations = [replace(item, round_idx=None) for item in initial_observations]
    collection = AcceptedActions(run_dir)
    compiled_selector = (CompiledFactorSelector(protocol, mock=args.mock,
        initial_candidate_ids=(item.candidate_id for item in initial_observations), collection=collection)
        if protocol.method == "ldm_harness_compiled" else None)
    runtime = None
    stack = ExitStack()

    try:
        if native:
            runtime, gateway, observations, execution = run_native(
                protocol, args, spec, client, run_dir, initial_observations, native_config, contract_identity)
        else:
            budget = CampaignBudget(rounds=protocol.rounds, reservoir_size=protocol.sessions * protocol.candidates_per_session if protocol.method in LDM_METHODS else protocol.batch_size,
                batch_size=protocol.batch_size, target_observations=len(initial_observations) + protocol.evaluations,
                max_evaluation_attempts=protocol.evaluations, extra_limits=protocol.budgets)
            request = CampaignRequest(run_dir, budget, config={"mock": args.mock, "protocol": protocol.to_dict()},
                resume=(run_dir / "campaign.json").exists(), **contract_identity, finalize_runtime=False,
                run_id=None if (run_dir / "campaign.json").exists() else "alphabench-" + uuid4().hex)
            runtime = request.open_runtime(spec)
            runtime.consume_many({key: 0 for key in protocol.budgets}, usage_key="task:initialize-counters")
            runtime.consume_many({"external_evaluations": 0, "expensive_evaluation_attempts": 0,
                                  "selected_candidates": 0}, usage_key="task:search-counters")
            gateway = OracleGateway(protocol, runtime, endpoint=args.oracle_url, mock=args.mock)
            gateway.preflight()
            if harness_method:
                expander = open_proposals(protocol, gateway, collection, api_key=key,
                    sidecar_image=args.harness_sidecar_image, stack=stack)
                if compiled_selector:
                    open_policy(protocol, gateway, compiled_selector, expander.meter, api_key=key,
                        sidecar_image=args.harness_sidecar_image, stack=stack)
            else:
                expander = DirectExpander(Generator(protocol, runtime, client, gateway, collection))
            checkpoint = runtime.load_checkpoint()
            state = LDMEngineState.from_checkpoint(checkpoint) if checkpoint else LDMEngineState(observations=initial_observations)
            engine = LDMEngine(task_spec=spec, runtime=runtime,
                expander=CallableReservoirExpander(lambda expansion: gateway.host.run(lambda: expander.expand(expansion))),
                candidate_domain=FactorDomain(protocol.backend), evaluator=FactorEvaluator(gateway),
                selector=(compiled_selector or FactorSelector(("mock_" if args.mock else "") + protocol.objective,
                    seed=protocol.random_seed, pool_size=protocol.bo_pool_size)) if protocol.method in LDM_METHODS else None,
                surrogate_encoder=FactorEncoder() if protocol.method in LDM_METHODS else None)
            result = engine.run(budget.engine_config(), state=state, finalize_runtime=False)
            observations = result.state.observations
            # Engine rounds_run is per invocation; the report records cumulative rounds across resume.
            execution = {"kind": "shared_engine", "summary": {
                **result.summary, "rounds_run": result.state.next_round}}
        if harness_method and expander.coordinator is not None:
            from ldm_tts.harness.research_runtime import write_fork_provenance
            write_fork_provenance(run_dir / "harness", compiled_policy=protocol.method == "ldm_harness_compiled")
        report = finalize(protocol, runtime, gateway, observations, initial_pool, initial_budget,
                          execution=execution, data_manifest=data_manifest)
        collection.export()
        runtime.finish(report)
        print(json.dumps({"task": "alphabench", "run_dir": str(run_dir.resolve()),
                          "qualification": report["qualification"], "complete_t3": report["complete_t3"]}, indent=2))
        return 0
    except (EvaluationPaused, BudgetExceededError) as exc:
        status = exc.status if isinstance(exc, EvaluationPaused) else "paused_budget"
        if runtime is not None:
            runtime.pause(status, phase="reconciliation", message=str(exc))
        print(json.dumps({"task": "alphabench", "status": status, "reason": str(exc)}))
        return 2
    finally:
        stack.close()
