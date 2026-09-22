"""Seed sources and imports stay distinct from same-run recovery."""

import importlib.util
import hashlib
import json
import math
from pathlib import Path
from uuid import uuid4

from ldm_tts.contracts import Candidate, RawProposal
from ldm_tts.engine.run_store import atomic_json_write
from .candidate import FactorDomain
from .protocol import digest


def parse_seeds(text, suffix, *, pool=False):
    if suffix == ".json":
        raw = json.loads(text)
        if pool and isinstance(raw, dict):
            raw = raw["final_pool"]
        if isinstance(raw, dict):
            raw = [{"name": name, "expression": expr} for name, expr in raw.items()]
    elif suffix == ".jsonl":
        raw = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        raw = [line.strip() for line in text.splitlines() if line.strip() and not line.strip().startswith("#")]
    if not isinstance(raw, list):
        raise ValueError("seed source must contain a list or name-to-expression mapping")
    result = []
    for index, item in enumerate(raw):
        if isinstance(item, str): item = {"expression": item}
        if not isinstance(item, dict): item = {}
        expression = item.get("expression", item.get("qlib_expression_default"))
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


def freeze_seed_source(protocol, args, directory):
    """Freeze imported content privately; only names and expressions become proposals."""
    path = directory / "private/seed_source.json"
    existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    if existing and existing["protocol"] != protocol.identity:
        raise ValueError("initialization source protocol mismatch")
    if existing and existing["mode"] == "shared_bundle":
        if args.seed_file or args.import_pool:
            raise ValueError("cannot replace a shared seed bundle with another source")
        if args.initialization_bundle:
            from .seed_bundle import verify_seed_bundle
            if (str(args.initialization_bundle.resolve()) != existing["directory"]
                or verify_seed_bundle(args.initialization_bundle, protocol, args.mock)["digest"] != existing["bundle_digest"]):
                raise ValueError("shared seed bundle source changed")
        return existing
    if existing and args.initialization_bundle:
        raise ValueError("cannot replace initialization with a shared seed bundle")
    supplied = args.seed_file if protocol.init_mode == "file" else args.import_pool if protocol.init_mode == "import_pool" else None
    if existing and supplied is None and not (protocol.init_mode == "alpha158" and args.upstream_root):
        return existing
    source = {"protocol": protocol.identity, "mode": protocol.init_mode}
    if protocol.init_mode in {"file", "import_pool"}:
        if supplied is None:
            raise ValueError("file/pool initialization requires its source path")
        content = supplied.read_bytes()
        text = content.decode("utf-8-sig")
        source.update(file_sha256=hashlib.sha256(content).hexdigest(), format=supplied.suffix.lower(),
                      source_text=text, seeds=parse_seeds(text, supplied.suffix.lower(), pool=protocol.init_mode == "import_pool"))
        if protocol.init_mode == "import_pool":
            source["prior_outcome_exposure"] = "external pool; prior holdout exposure is unknown; all expressions are re-evaluated"
    elif protocol.init_mode == "alpha158":
        if not args.upstream_root:
            raise ValueError("alpha158 requires the pinned AlphaBench root")
        directory_path = args.upstream_root / "factors/lib/alpha158"
        files = [directory_path / name for name in ("__init__.py", "qlib_compile_product.json", *(group + ".json" for group in protocol.alpha158_groups))]
        source.update(groups=list(protocol.alpha158_groups),
                      files={file.name: hashlib.sha256(file.read_bytes()).hexdigest() for file in files},
                      seeds=alpha158_seeds(args.upstream_root, protocol.alpha158_groups))
    if existing is not None and existing != source:
        raise ValueError("initialization source changed; use a new run")
    if existing is None:
        atomic_json_write(path, source)
    return source


def seed_admissions(seeds, backend):
    domain, seen, rows = FactorDomain(backend), {}, []
    for index, payload in enumerate(seeds):
        candidate = domain.admit(RawProposal(payload, "initialization"))
        row = {"position": index, "role": "initialization", "payload": payload}
        if not isinstance(candidate, Candidate):
            row.update(status="rejected", reason=candidate.reason, message=candidate.message)
        elif candidate.candidate_id in seen:
            row.update(status="duplicate", candidate_id=candidate.candidate_id, duplicate_of=seen[candidate.candidate_id])
        else:
            seen[candidate.candidate_id] = index
            row.update(status="admitted", candidate_id=candidate.candidate_id)
        rows.append(row)
    return rows


def run_initialization(protocol, args, spec, client, run_dir):
    from ldm_tts.campaign import CampaignBudget, CampaignRecipe, CampaignRequest, run_campaign
    from ldm_tts.contracts import EvaluationResult
    from ldm_tts.contracts.evaluation import EVALUATION_ATTEMPT_RECEIPT_KEY, EvaluationPaused
    from ldm_tts.engine.expansion import CallableReservoirExpander, ExpansionResult
    from ldm_tts.engine import LDMEngineState
    from .collection import AcceptedActions
    from .gateway import OracleGateway
    from .generator import Generator
    from .receipts import Receipts

    directory = run_dir / "initialization"
    source_path = directory / "private/seed_source.json"
    saved_source = json.loads(source_path.read_text(encoding="utf-8")) if source_path.exists() else None
    if args.initialization_bundle or (saved_source and saved_source["mode"] == "shared_bundle"):
        from .seed_bundle import import_seed_bundle
        return import_seed_bundle(protocol, args, directory)
    source = freeze_seed_source(protocol, args, directory)
    manifest_path = directory / "seed_manifest.json"
    if manifest_path.exists() and json.loads((directory / "status.json").read_text(encoding="utf-8"))["status"] == "completed":
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["protocol"] != protocol.identity or manifest["source_digest"] != digest(source):
            raise ValueError("initialization protocol mismatch")
        state = LDMEngineState.from_checkpoint({"observations": manifest["observations"]})
        return state.observations, manifest["pool"], json.loads((directory / "budget.json").read_text(encoding="utf-8"))
    objects = {}
    collection = AcceptedActions(directory)

    def configure(runtime):
        try:
            gateway = OracleGateway(protocol, runtime, endpoint=args.oracle_url, mock=args.mock)
            gateway.preflight()
            objects["gateway"] = gateway
            receipt_store = Receipts(directory / "manifests")
            seed_record = receipt_store.load("seeds")
            if seed_record is None:
                if protocol.init_mode == "cold":
                    if protocol.cold_seed_count > protocol.budgets["initialization_evaluations"]:
                        raise ValueError("cold seed count exceeds the initialization evaluation budget")
                    seeds = []
                    if protocol.cold_seed_count:
                        if protocol.method.startswith("alphabench_"):
                            from .native_generator import NativeGenerator
                            from .native_runtime import NativeRuntime
                            native = NativeGenerator(protocol, runtime, client, gateway, collection, args.upstream_root)
                            scheduler = NativeRuntime(runtime, gateway.host)
                            seeds = scheduler.run(lambda: scheduler.callback("cold_start", native.cold_start)())
                        else:
                            seeds = Generator(protocol, runtime, client, gateway, collection).generate(
                                identity="initialization", count=protocol.cold_seed_count,
                                instruction="Generate the initial factor pool.")[0]["candidates"]
                else:
                    seeds = source["seeds"]
                seed_record = {"source_digest": digest(source), "seeds": seeds, "admissions": seed_admissions(seeds, protocol.backend)}
                receipt_store.accept("seeds", seed_record)
            if seed_record["source_digest"] != digest(source):
                raise ValueError("seed source differs from frozen initialization")
            admitted = sum(row["status"] == "admitted" for row in seed_record["admissions"])
            if admitted > protocol.budgets["initialization_evaluations"]:
                raise ValueError("initialization budget cannot evaluate the complete admitted seed source")
            objects["seed_record"] = seed_record
        except EvaluationPaused as exc:
            runtime.pause(exc.status, phase="initialization", message=str(exc))
            raise
        except Exception as exc:
            runtime.fail(exc)
            raise

    def expand(request):
        seeds = objects["seed_record"]["seeds"]
        return ExpansionResult(proposals=tuple(RawProposal(seed, "initialization", {"attempt_position": index})
                                               for index, seed in enumerate(seeds)), selection_mode="reservoir_order",
                               schema_update={"initialization_source_count": len(seeds)})

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
                metrics={prefix + key: value for key, value in raw.get("metrics", {}).items() if type(value) in (int, float) and math.isfinite(value)},
                metadata={EVALUATION_ATTEMPT_RECEIPT_KEY: self.evaluation_attempt_usage_key(candidate)},
                error=raw.get("error", ""))

    limit = protocol.budgets["initialization_evaluations"]
    result = run_campaign(CampaignRequest(directory, CampaignBudget(rounds=1, reservoir_size=max(1, limit),
            batch_size=max(1, limit), max_evaluation_attempts=limit, extra_limits=protocol.budgets),
            resume=(directory / "campaign.json").exists(), runtime_hook=configure,
            run_id=None if (directory / "campaign.json").exists() else "initialization-" + uuid4().hex,
            config={"phase": "initialization", "protocol": protocol.to_dict()}, finalize_runtime=False),
        CampaignRecipe(spec, CallableReservoirExpander(expand), FactorDomain(protocol.backend), Evaluator()))
    pool = []
    for observation in result.engine.state.observations:
        candidate = observation.candidate
        position = ["initialization", candidate.metadata["attempt_position"]]
        receipt = objects["gateway"].receipts.load(objects["gateway"].identity("validation", position, candidate))
        pool.append({**candidate.payload, "candidate_id": candidate.candidate_id, "success": observation.evaluation.succeeded,
                     "validation_position": position, "validation": receipt["response"] if receipt else None})
    observations = [item.to_dict() for item in result.engine.state.observations]
    atomic_json_write(directory / "seed_manifest.json", {"protocol": protocol.identity, "pool": pool,
                       "protocol_contract": protocol.to_dict(), "mock": args.mock,
                       "source_digest": digest(source), "admissions": objects["seed_record"]["admissions"],
                       "observations": observations, "public_information_digest": digest(observations)})
    collection.export()
    result.runtime.finish({**result.engine.summary, "phase": "initialization", "seed_count": len(pool)})
    return result.engine.state.observations, pool, result.runtime.budget.snapshot()
