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
from ldm_tts.engine.run_store import atomic_json_write, unique_run_dir
from ldm_tts.transport import CallableProposalClient
from ldm_tts.transport.openai import OpenAICompatibleProposalClient
from .candidate import FactorDomain
from .collection import AcceptedActions
from .finalization import finalize
from .gateway import FactorEvaluator, OracleGateway
from .generator import DirectExpander, Generator
from .initialization import freeze_seed_source, run_initialization
from .protocol import LDM_METHODS, digest, verify_data_manifest
from .selection import FactorEncoder, FactorSelector


def mock_response(request):
    count = json.loads(request.messages[1]["content"])["required"]
    offset = int(digest(request.messages)[:8], 16) % 10000
    return json.dumps({"candidates": [{"name": f"mock_{index}", "expression": f"Mean($close,{offset + index + 1})"}
                                     for index in range(count)]})


def make_client(protocol, mock):
    if mock:
        return CallableProposalClient(mock_response)
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise ValueError("DEEPSEEK_API_KEY is required in the Host environment")
    return OpenAICompatibleProposalClient(url=protocol.endpoint, model=protocol.model, api_key=key,
        wire_api=protocol.wire_api, extra_body={"reasoning": {"effort": protocol.reasoning_effort}, "store": False},
        max_retries=0, max_tokens=protocol.max_model_tokens, timeout_seconds=protocol.request_timeout)


def run(args, protocol, spec):
    if not args.mock:
        if protocol.profile != "ldm_matched_v1":
            raise ValueError("native source profiles require their resolved parameter adapter")
        if not args.protocol_file or not args.data_manifest:
            raise ValueError("real runs require a frozen --protocol-file and verified --data-manifest")
        verify_data_manifest(args.data_manifest, protocol)
    if protocol.method not in {"llm", "ldm"}:
        raise ValueError("the selected method requires its native/Harness adapter; direct execution is forbidden")
    run_dir = args.resume_run or args.out_dir or unique_run_dir(Path(__file__).resolve().parents[1] / "runs" / ("mock" if args.mock else "campaign"))
    run_dir.mkdir(parents=True, exist_ok=True)
    protocol_path = run_dir / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text(encoding="utf-8")) != protocol.to_dict():
        raise ValueError("resume protocol identity mismatch")
    if protocol_path.exists() and not args.resume_run:
        raise ValueError("existing run requires --resume-run")
    atomic_json_write(protocol_path, protocol.to_dict())
    if args.resume_run and (run_dir / "result.json").exists() and json.loads((run_dir / "status.json").read_text(encoding="utf-8"))["status"] == "completed":
        freeze_seed_source(protocol, args, run_dir / "initialization")
        AcceptedActions(run_dir).export()
        print(json.dumps({"task": "alphabench", "run_dir": str(run_dir.resolve()), "status": "completed", "replayed": True}))
        return 0
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

    budget = CampaignBudget(rounds=protocol.rounds, reservoir_size=protocol.sessions * protocol.candidates_per_session if protocol.method in LDM_METHODS else protocol.batch_size,
        batch_size=protocol.batch_size, target_observations=len(initial_observations) + protocol.evaluations,
        max_evaluation_attempts=protocol.evaluations, extra_limits=protocol.budgets)
    try:
        result = run_campaign(CampaignRequest(run_dir, budget, config={"mock": args.mock, "protocol": protocol.to_dict()},
            resume=(run_dir / "campaign.json").exists(), runtime_hook=configure, contract_sha256=protocol.identity, finalize_runtime=False,
            run_id=None if (run_dir / "campaign.json").exists() else "alphabench-" + uuid4().hex,
            state_factory=lambda runtime: LDMEngineState.from_checkpoint(runtime.load_checkpoint()) if runtime.load_checkpoint() else LDMEngineState(observations=initial_observations)),
            CampaignRecipe(spec, CallableReservoirExpander(lambda request: objects["gateway"].host.run(lambda: objects["expander"].expand(request))),
                FactorDomain(protocol.backend), Evaluator(),
                selector=FactorSelector(("mock_" if args.mock else "") + protocol.objective, seed=protocol.random_seed) if protocol.method in LDM_METHODS else None,
                surrogate_encoder=FactorEncoder() if protocol.method in LDM_METHODS else None))
        report = finalize(protocol, result.runtime, objects["gateway"], result.engine, initial_pool, initial_budget)
        collection.export()
        print(json.dumps({"task": "alphabench", "run_dir": str(run_dir.resolve()),
                          "qualification": report["qualification"], "complete_t3": report["complete_t3"]}, indent=2))
        return 0
    except EvaluationPaused as exc:
        if "runtime" in objects:
            objects["runtime"].status.update(exc.status, phase="reconciliation", budget=objects["runtime"].budget)
        print(json.dumps({"task": "alphabench", "status": exc.status, "reason": str(exc)}))
        return 2
