"""Verified reuse of one completed initialization by matched methods."""

import hashlib
import json
import math
from pathlib import Path
from uuid import uuid4

from ldm_tts.contracts import Candidate, RawProposal
from ldm_tts.engine import LDMEngineState
from ldm_tts.engine.run_store import CampaignRuntime, atomic_json_write
from .candidate import FactorDomain
from .protocol import T3Protocol, digest


def initialization_contract(protocol):
    search_only = {"method", "rounds", "evaluations", "batch_size", "sessions", "candidates_per_session",
                   "factor_select_n", "stock_topk", "stock_n_drop", "assay_portfolio", "budgets"}
    return {key: value for key, value in protocol.to_dict().items() if key not in search_only}


def verify_seed_bundle(directory, protocol, mock):
    from .initialization import seed_admissions

    directory = Path(directory)
    inventory = {}
    def read(relative):
        body = (directory / relative).read_bytes()
        inventory[relative] = hashlib.sha256(body).hexdigest()
        return json.loads(body)

    if protocol.profile != "ldm_matched_v1":
        raise ValueError("shared seed bundles require the matched protocol")
    manifest = read("seed_manifest.json")
    source_protocol = T3Protocol(**manifest["protocol_contract"])
    if (manifest["protocol"] != source_protocol.identity or manifest["mock"] != mock
        or initialization_contract(source_protocol) != initialization_contract(protocol)):
        raise ValueError("seed bundle scientific/model contract mismatch")
    status, campaign = read("status.json"), read("campaign.json")
    if status["status"] != "completed" or status["run_id"] != campaign["run_id"]:
        raise ValueError("seed bundle initialization is not complete")
    source = read("private/seed_source.json")
    if source["protocol"] != source_protocol.identity or digest(source) != manifest["source_digest"]:
        raise ValueError("seed bundle source identity mismatch")
    journal = read("manifests/" + digest("seeds") + ".json")
    admissions = seed_admissions(journal["seeds"], protocol.backend)
    if (journal["source_digest"] != manifest["source_digest"] or journal["admissions"] != admissions
        or manifest["admissions"] != admissions):
        raise ValueError("seed bundle admission journal mismatch")
    observations, pool = manifest["observations"], manifest["pool"]
    admitted = [row for row in admissions if row["status"] == "admitted"]
    if (manifest["public_information_digest"] != digest(observations)
        or read("checkpoint.json")["state"]["observations"] != observations
        or len(pool) != len(observations) or len(admitted) != len(observations)):
        raise ValueError("seed bundle observations are incomplete")
    budget = read("budget.json")
    if budget["counters"]["initialization_evaluations"] != len(observations):
        raise ValueError("seed bundle creation accounting mismatch")
    domain = FactorDomain(protocol.backend)
    prefix = "mock_" if mock else ""
    for observation, row, admission in zip(observations, pool, admitted):
        candidate = Candidate(**observation["candidate"])
        canonical = domain.admit(RawProposal(admission["payload"], "initialization"))
        if (not isinstance(canonical, Candidate) or candidate.payload != canonical.payload
            or candidate.candidate_id != canonical.candidate_id or candidate.canonical_key != canonical.canonical_key
            or candidate.source != "initialization" or row["candidate_id"] != candidate.candidate_id
            or candidate.metadata["attempt_position"] != admission["position"]):
            raise ValueError("seed bundle candidate identity mismatch")
        responses = {}
        for phase in ("initialization", "validation") if observation["evaluation"]["status"] == "succeeded" else ("initialization",):
            position = admission["position"] if phase == "initialization" else ["initialization", admission["position"]]
            identity = {"run": campaign["run_id"], "protocol": source_protocol.identity, "phase": phase,
                        "position": position, "candidate": candidate.candidate_id}
            receipt = read("private/oracle/" + digest(identity) + ".json")
            if receipt["state"] != "completed" or receipt["identity"] != identity:
                raise ValueError("seed bundle oracle receipt is incomplete")
            request, response = receipt["request"], receipt["response"]
            if (receipt["request_digest"] != digest(request) or receipt["response_digest"] != digest(response)
                or request["request_id"] != digest(identity) or response["request_id"] != request["request_id"]
                or request["protocol"] != source_protocol.to_dict() or request["phase"] != phase
                or request["operation"] != "evaluate" or request["fast"] is not True
                or (request["start"], request["end"]) != source_protocol.interval("search" if phase == "initialization" else phase)
                or request["expression"] != candidate.payload["expression"] or request["dialect"] != candidate.payload["dialect"]
                or response.get("mock", False) != mock):
                raise ValueError("seed bundle oracle receipt contract mismatch")
            responses[phase] = response
        measured = responses["initialization"]
        metrics = {prefix + key: value for key, value in measured.get("metrics", {}).items()
                   if type(value) in (int, float) and math.isfinite(value)}
        expected_pool = {**candidate.payload, "candidate_id": candidate.candidate_id, "success": measured["success"],
                         "validation_position": ["initialization", admission["position"]], "validation": responses.get("validation")}
        if (row != expected_pool or observation["evaluation"]["metrics"] != metrics
            or observation["evaluation"]["status"] != ("succeeded" if measured["success"] else "invalid")):
            raise ValueError("seed bundle observations differ from oracle receipts")
    generation_steps = []
    for path in sorted((directory / "generation").glob("*.json")):
        record = read(path.relative_to(directory).as_posix())
        generation_steps.append({key: record[key] for key in ("attempt_count", "complete")})
    return {"manifest": manifest, "budget": budget, "inventory": inventory, "generation_steps": generation_steps,
            "digest": digest(inventory), "public_information_digest": digest(observations), "run_id": campaign["run_id"]}


def import_seed_bundle(protocol, args, directory):
    source_path, manifest_path = directory / "private/seed_source.json", directory / "seed_manifest.json"
    source = json.loads(source_path.read_text(encoding="utf-8")) if source_path.exists() else None
    if args.seed_file or args.import_pool:
        raise ValueError("choose a shared seed bundle or an unevaluated seed source")
    if source and (source["protocol"] != protocol.identity or source["mode"] != "shared_bundle"):
        raise ValueError("cannot replace an initialization with a different seed bundle")
    origin = args.initialization_bundle.resolve() if args.initialization_bundle else Path(source["directory"])
    if source and str(origin) != source["directory"]:
        raise ValueError("shared seed bundle source changed")
    status_path = directory / "status.json"
    if manifest_path.exists() and json.loads(status_path.read_text(encoding="utf-8"))["status"] == "completed":
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["protocol"] != protocol.identity or manifest["source_digest"] != digest(source):
            raise ValueError("imported initialization identity mismatch")
        if args.initialization_bundle and verify_seed_bundle(origin, protocol, args.mock)["digest"] != source["bundle_digest"]:
            raise ValueError("shared seed bundle content changed")
        return LDMEngineState.from_checkpoint({"observations": manifest["observations"]}).observations, manifest["pool"], json.loads((directory / "budget.json").read_text())
    bundle = verify_seed_bundle(origin, protocol, args.mock)
    frozen = {"protocol": protocol.identity, "mode": "shared_bundle", "directory": str(origin), "bundle_digest": bundle["digest"]}
    if source and source != frozen:
        raise ValueError("shared seed bundle content changed during recovery")
    atomic_json_write(source_path, frozen)
    runtime = CampaignRuntime.open(directory, task="alphabench", contract_sha256=protocol.identity,
        resume=(directory / "campaign.json").exists(),
        run_id=None if (directory / "campaign.json").exists() else "initialization-" + uuid4().hex,
        config={"phase": "shared_initialization", "protocol": protocol.to_dict()},
        budget_limits={key: 0 for key in bundle["budget"]["limits"]})
    runtime.budget.metadata["shared_seed_creation"] = {"run_id": bundle["run_id"], "budget": bundle["budget"],
        "bundle_digest": bundle["digest"], "public_information_digest": bundle["public_information_digest"],
        "generation_steps": bundle["generation_steps"]}
    runtime.consume_many({key: 0 for key in bundle["budget"]["counters"]}, usage_key="seed-bundle:import")
    manifest = {**bundle["manifest"], "protocol": protocol.identity, "protocol_contract": protocol.to_dict(),
                "source_digest": digest(frozen), "bundle_digest": bundle["digest"]}
    atomic_json_write(directory / "private/seed_bundle_inventory.json", bundle["inventory"])
    atomic_json_write(manifest_path, manifest)
    runtime.record("shared_seeds_imported", {"bundle_digest": bundle["digest"], "source_run_id": bundle["run_id"]},
                   event_key="seed-bundle:imported")
    runtime.checkpoint({"observations": manifest["observations"]})
    runtime.finish({"phase": "initialization", "seed_count": len(manifest["pool"]), "imported": True})
    return LDMEngineState.from_checkpoint({"observations": manifest["observations"]}).observations, manifest["pool"], runtime.budget.snapshot()
