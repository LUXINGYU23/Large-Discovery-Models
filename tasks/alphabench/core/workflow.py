"""Assemble the shared campaign and its private post-search stages."""

from dataclasses import replace
import json
import os
from pathlib import Path
from uuid import uuid4

from ldm_tts.campaign import CampaignBudget, CampaignRecipe, CampaignRequest, run_campaign
from ldm_tts.contracts.evaluation import EvaluationPaused
from ldm_tts.engine.expansion import CallableReservoirExpander
from ldm_tts.engine import LDMEngineState
from ldm_tts.engine.run_store import BudgetExceededError, atomic_json_write, unique_run_dir
from ldm_tts.transport import CallableProposalClient
from ldm_tts.transport.openai import OpenAICompatibleProposalClient
from ldm_tts.registration.experiment import load_active_experiment_contract, snapshot_experiment_contract
from .candidate import FactorDomain
from .collection import AcceptedActions
from .finalization import finalize
from .gateway import FactorEvaluator, OracleGateway
from .generator import DirectExpander, Generator
from .initialization import freeze_seed_source, run_initialization
from .native_reference import prepare_native, run_native
from .protocol import LDM_METHODS, digest, verify_data_manifest
from .selection import FactorEncoder, FactorSelector


def mock_response(request, native_count=None):
    count = native_count if native_count is not None else json.loads(request.messages[1]["content"])["required"]
    offset = int(digest(request.messages)[:8], 16) % 10000
    return json.dumps({"generated" if native_count is not None else "candidates": [{"name": f"mock_{index}", "expression": f"Mean($close,{offset + index + 1})"}
                                     for index in range(count)]})


def make_client(protocol, mock):
    if mock:
        count = max(1, protocol.cold_seed_count, protocol.native_parameters.get("N", 1)) if protocol.method.startswith("alphabench_") else None
        return CallableProposalClient(lambda request: mock_response(request, count))
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise ValueError("DEEPSEEK_API_KEY is required in the Host environment")
    body = {"reasoning": {"effort": protocol.reasoning_effort}, "store": False}
    if protocol.method.startswith("alphabench_"):
        body["text"] = {"format": {"type": "json_object"}}
    return OpenAICompatibleProposalClient(url=protocol.endpoint, model=protocol.model, api_key=key,
        wire_api=protocol.wire_api, extra_body=body, temperature=protocol.temperature,
        max_retries=0, max_tokens=protocol.max_model_tokens, timeout_seconds=protocol.request_timeout)


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
    if not args.mock:
        if protocol.profile != "ldm_matched_v1":
            raise ValueError("native source profiles require their resolved parameter adapter")
        if not args.protocol_file or not args.data_manifest:
            raise ValueError("real runs require a frozen --protocol-file and verified --data-manifest")
        verify_data_manifest(args.data_manifest, protocol)
    native = protocol.method.startswith("alphabench_")
    if protocol.method not in {"llm", "ldm"} and not native:
        raise ValueError("the selected method requires its native/Harness adapter; direct execution is forbidden")
    run_dir = args.resume_run or args.out_dir or unique_run_dir(Path(__file__).resolve().parents[1] / "runs" / ("mock" if args.mock else "campaign"))
    run_dir.mkdir(parents=True, exist_ok=True)
    protocol_path = run_dir / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text(encoding="utf-8")) != protocol.to_dict():
        raise ValueError("resume protocol identity mismatch")
    if protocol_path.exists() and not args.resume_run:
        raise ValueError("existing run requires --resume-run")
    atomic_json_write(protocol_path, protocol.to_dict())
    contract_identity = bind_runner_contract(run_dir, protocol)
    if args.resume_run and (run_dir / "result.json").exists() and json.loads((run_dir / "status.json").read_text(encoding="utf-8"))["status"] == "completed":
        freeze_seed_source(protocol, args, run_dir / "initialization")
        AcceptedActions(run_dir).export()
        print(json.dumps({"task": "alphabench", "run_dir": str(run_dir.resolve()), "status": "completed", "replayed": True}))
        return 0
    native_config = prepare_native(protocol, args.upstream_root) if native else None
    client = make_client(protocol, args.mock)
    try:
        initial_observations, initial_pool, initial_budget = run_initialization(protocol, args, spec, client, run_dir)
    except EvaluationPaused as exc:
        print(json.dumps({"task": "alphabench", "phase": "initialization", "status": exc.status, "reason": str(exc)}))
        return 2
    initial_observations = [replace(item, round_idx=None) for item in initial_observations]
    collection = AcceptedActions(run_dir)
    objects = {}

    def configure(runtime):
        objects["runtime"] = runtime
        runtime.consume_many({key: 0 for key in protocol.budgets}, usage_key="task:initialize-counters")
        runtime.consume_many({"external_evaluations": 0, "expensive_evaluation_attempts": 0,
                              "selected_candidates": 0}, usage_key="task:search-counters")
        gateway = OracleGateway(protocol, runtime, endpoint=args.oracle_url, mock=args.mock)
        gateway.preflight()
        generator = Generator(protocol, runtime, client, gateway, collection)
        objects.update(gateway=gateway, expander=DirectExpander(generator), evaluator=FactorEvaluator(gateway))

    class Evaluator:
        def evaluate(self, candidate):
            return objects["evaluator"].evaluate(candidate)

        def evaluation_attempt_usage_key(self, candidate):
            return objects["evaluator"].evaluation_attempt_usage_key(candidate)

    try:
        if native:
            runtime, gateway, observations, execution = run_native(
                protocol, args, spec, client, run_dir, initial_observations, native_config, contract_identity)
            objects.update(runtime=runtime, gateway=gateway)
        else:
            budget = CampaignBudget(rounds=protocol.rounds, reservoir_size=protocol.sessions * protocol.candidates_per_session if protocol.method in LDM_METHODS else protocol.batch_size,
                batch_size=protocol.batch_size, target_observations=len(initial_observations) + protocol.evaluations,
                max_evaluation_attempts=protocol.evaluations, extra_limits=protocol.budgets)
            result = run_campaign(CampaignRequest(run_dir, budget, config={"mock": args.mock, "protocol": protocol.to_dict()},
                resume=(run_dir / "campaign.json").exists(), runtime_hook=configure, **contract_identity, finalize_runtime=False,
                run_id=None if (run_dir / "campaign.json").exists() else "alphabench-" + uuid4().hex,
                state_factory=lambda runtime: LDMEngineState.from_checkpoint(runtime.load_checkpoint()) if runtime.load_checkpoint() else LDMEngineState(observations=initial_observations)),
                CampaignRecipe(spec, CallableReservoirExpander(lambda request: objects["gateway"].host.run(lambda: objects["expander"].expand(request))),
                    FactorDomain(protocol.backend), Evaluator(),
                    selector=FactorSelector(("mock_" if args.mock else "") + protocol.objective, seed=protocol.random_seed) if protocol.method in LDM_METHODS else None,
                    surrogate_encoder=FactorEncoder() if protocol.method in LDM_METHODS else None))
            observations, execution = result.engine.state.observations, {"kind": "shared_engine", "summary": result.engine.summary}
        report = finalize(protocol, objects["runtime"], objects["gateway"], observations, initial_pool, initial_budget,
                          execution=execution)
        collection.export()
        print(json.dumps({"task": "alphabench", "run_dir": str(run_dir.resolve()),
                          "qualification": report["qualification"], "complete_t3": report["complete_t3"]}, indent=2))
        return 0
    except (EvaluationPaused, BudgetExceededError) as exc:
        status = exc.status if isinstance(exc, EvaluationPaused) else "paused_budget"
        if "runtime" in objects:
            objects["runtime"].pause(status, phase="reconciliation", message=str(exc))
        print(json.dumps({"task": "alphabench", "status": status, "reason": str(exc)}))
        return 2
