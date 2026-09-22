"""Seed sources and imports stay distinct from same-run recovery."""

import importlib.util
import json
from pathlib import Path

from ldm_tts.contracts import RawProposal
from ldm_tts.engine.run_store import atomic_json_write
from .candidate import FactorDomain
from .protocol import digest


def load_seeds(path):
    path = Path(path)
    if path.suffix == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            raw = [{"name": name, "expression": expr} for name, expr in raw.items()]
    elif path.suffix == ".jsonl":
        raw = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        raw = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.strip().startswith("#")]
    if not isinstance(raw, list):
        raise ValueError("seed source must contain a list or name-to-expression mapping")
    result = []
    for index, item in enumerate(raw):
        if isinstance(item, str): item = {"expression": item}
        if not isinstance(item, dict): raise ValueError("invalid seed entry")
        expression = item.get("expression", item.get("qlib_expression_default"))
        if not isinstance(expression, str) or not expression.strip(): raise ValueError("seed expression is missing")
        result.append({"name": item.get("name", f"seed_{index}"), "expression": expression})
    return result


def alpha158_seeds(upstream_root, groups=("kbar", "rolling")):
    path = Path(upstream_root) / "factors/lib/alpha158/__init__.py"
    spec = importlib.util.spec_from_file_location("alphabench_alpha158", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    module.FACTOR_DIR = str(path.parent)
    module.COMPILE_FILE = str(path.parent / "qlib_compile_product.json")
    _, compiled = module.load_factors_alpha158(exclude_var="vwap", collection=list(groups))
    return [{"name": name, "expression": item["qlib_expression_default"]} for name, item in compiled.items()]


def seed_payloads(protocol, generator, *, upstream_root=None, seed_file=None, import_pool=None):
    if protocol.init_mode == "cold":
        generated, _ = generator.generate(identity="initialization", count=protocol.cold_seed_count,
                                          instruction="Generate the initial factor pool.")
        seeds = generated["candidates"]
    elif protocol.init_mode == "alpha158":
        if not upstream_root: raise ValueError("alpha158 requires the pinned AlphaBench root")
        seeds = alpha158_seeds(upstream_root, protocol.alpha158_groups)
    elif protocol.init_mode == "file":
        if not seed_file: raise ValueError("file initialization requires --seed-file")
        seeds = load_seeds(seed_file)
    else:
        if not import_pool: raise ValueError("pool initialization requires --import-pool")
        imported = json.loads(Path(import_pool).read_text(encoding="utf-8"))
        seeds = imported["final_pool"]
        if not isinstance(seeds, list): raise ValueError("final_pool must be a list")
    return seeds


def run_initialization(protocol, args, spec, client, run_dir):
    from ldm_tts.campaign import CampaignBudget, CampaignRecipe, CampaignRequest, run_campaign
    from ldm_tts.contracts import EvaluationResult
    from ldm_tts.contracts.evaluation import EVALUATION_ATTEMPT_RECEIPT_KEY
    from ldm_tts.engine.expansion import CallableReservoirExpander, ExpansionResult
    from ldm_tts.engine import LDMEngineState
    from .collection import AcceptedActions
    from .gateway import OracleGateway
    from .generator import Generator
    from .receipts import Receipts

    directory = run_dir / "initialization"
    manifest_path = directory / "seed_manifest.json"
    if manifest_path.exists() and json.loads((directory / "status.json").read_text(encoding="utf-8"))["status"] == "completed":
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["protocol"] != protocol.identity:
            raise ValueError("initialization protocol mismatch")
        state = LDMEngineState.from_checkpoint({"observations": manifest["observations"]})
        return state.observations, manifest["pool"], json.loads((directory / "budget.json").read_text(encoding="utf-8"))
    objects = {}
    collection = AcceptedActions(directory)

    def configure(runtime):
        objects["runtime"] = runtime
        gateway = OracleGateway(protocol, runtime, endpoint=args.oracle_url, mock=args.mock)
        gateway.preflight()
        objects.update(gateway=gateway, generator=Generator(protocol, runtime, client, gateway, collection))

    def expand(request):
        seeds = seed_payloads(protocol, objects["generator"], upstream_root=args.upstream_root,
                              seed_file=args.seed_file, import_pool=args.import_pool)
        if len(seeds) > protocol.budgets["initialization_evaluations"]:
            raise ValueError("initialization budget cannot evaluate the complete seed source")
        Receipts(directory / "manifests").accept("seeds", {"protocol": protocol.identity, "seed_digest": digest(seeds), "seeds": seeds})
        return ExpansionResult(proposals=tuple(RawProposal(seed, "initialization", {"attempt_position": index})
                                               for index, seed in enumerate(seeds)), selection_mode="reservoir_order")

    class Evaluator:
        def evaluation_attempt_usage_key(self, candidate):
            return digest(objects["gateway"].identity("initialization", candidate.metadata["attempt_position"], candidate))

        def evaluate(self, candidate):
            position = candidate.metadata["attempt_position"]
            raw = objects["gateway"].evaluate(candidate, phase="initialization", position=position)
            if raw["success"]:
                objects["gateway"].evaluate(candidate, phase="validation", position=["initialization", position])
            prefix = "mock_" if args.mock else ""
            return EvaluationResult(candidate.candidate_id, "succeeded" if raw["success"] else "invalid",
                metrics={prefix + key: value for key, value in raw.get("metrics", {}).items() if value is not None},
                metadata={EVALUATION_ATTEMPT_RECEIPT_KEY: self.evaluation_attempt_usage_key(candidate)},
                error=raw.get("error", ""))

    limit = protocol.budgets["initialization_evaluations"]
    result = run_campaign(CampaignRequest(directory, CampaignBudget(rounds=1, reservoir_size=limit,
            batch_size=limit, max_evaluation_attempts=limit, extra_limits=protocol.budgets),
            resume=(directory / "campaign.json").exists(), runtime_hook=configure,
            config={"phase": "initialization", "protocol": protocol.to_dict()}, finalize_runtime=False),
        CampaignRecipe(spec, CallableReservoirExpander(expand), FactorDomain(protocol.backend), Evaluator()))
    pool = []
    for observation in result.engine.state.observations:
        candidate = observation.candidate
        position = ["initialization", candidate.metadata["attempt_position"]]
        receipt = objects["gateway"].receipts.load(objects["gateway"].identity("validation", position, candidate))
        pool.append({**candidate.payload, "candidate_id": candidate.candidate_id, "success": observation.evaluation.succeeded,
                     "validation_position": position, "validation": receipt["response"] if receipt else None})
    atomic_json_write(directory / "seed_manifest.json", {"protocol": protocol.identity, "pool": pool,
                       "observations": [item.to_dict() for item in result.engine.state.observations]})
    collection.export()
    result.runtime.finish({**result.engine.summary, "phase": "initialization", "seed_count": len(pool)})
    return result.engine.state.observations, pool, result.runtime.budget.snapshot()
