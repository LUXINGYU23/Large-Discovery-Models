"""Private validation ranking and immutable test selection precede every test call."""

import csv
from collections import Counter
import json
import math

from ldm_tts.contracts import Candidate, RawProposal
from ldm_tts.engine.run_store import atomic_json_write
from .candidate import FactorDomain
from .protocol import digest
from .receipts import Receipts
from .reporting import daily_metrics, generation_costs, search_metrics, signal_diversity, structure_diversity
from .quality import audit_quality


def finalize(protocol, runtime, gateway, engine_result, initial_pool, initial_budget):
    private = runtime.run_dir / "private"
    stages = Receipts(private / "stages")
    runtime.status.update("running", phase="validation_complete", budget=runtime.budget)
    domain = FactorDomain(protocol.backend)
    pool = {}
    for row in initial_pool:
        if row["success"]:
            candidate = domain.admit(RawProposal(row, "initialization"))
            pool[candidate.candidate_id] = (candidate, row["validation_position"], "initialization", row["validation"])
    for observation in engine_result.state.observations:
        if observation.evaluation.succeeded and observation.candidate.source != "initialization":
            candidate = observation.candidate
            pool[candidate.candidate_id] = (candidate, candidate.metadata["attempt_position"], "search", None)
    ranked = []
    for candidate, position, source, raw in pool.values():
        if raw is None:
            receipt = gateway.receipts.load(gateway.identity("validation", position, candidate))
            if receipt is None or receipt["state"] != "completed":
                raise ValueError("validation must complete before freezing test selection")
            raw = receipt["response"]
        value = raw.get("metrics", {}).get(protocol.objective)
        if not raw["success"] or type(value) not in (int, float) or not math.isfinite(value):
            continue
        ranked.append({"candidate": candidate.to_dict(), "validation_score": value, "source": source})
    ranked.sort(key=lambda row: (-row["validation_score"], row["candidate"]["candidate_id"]))
    selection = {"protocol": protocol.identity, "direction": protocol.direction,
                 "stock_topk": protocol.stock_topk, "stock_n_drop": protocol.stock_n_drop,
                 "selected": ranked[:protocol.factor_select_n]}
    stages.accept("selection_frozen", selection)
    atomic_json_write(runtime.run_dir / "selection_frozen.json", selection)
    runtime.status.update("running", phase="test", budget=runtime.budget)
    tested = []
    for index, row in enumerate(selection["selected"]):
        candidate = Candidate(**row["candidate"])
        raw = gateway.evaluate(candidate, phase="test", position=[digest(selection), index], fast=False)
        if raw["success"] and not isinstance(raw.get("portfolio"), dict):
            raise ValueError("full T3 test requires portfolio outputs")
        tested.append({"candidate": candidate.payload, "candidate_id": candidate.candidate_id,
                       "raw": raw, "derived": daily_metrics(raw.get("daily", []), direction=protocol.direction)})
    stages.accept("test", {"selection_digest": digest(selection), "results": tested})
    runtime.status.update("running", phase="independent_analysis", budget=runtime.budget)
    combination = gateway.combine([Candidate(**row["candidate"]) for row in selection["selected"]],
                                  selection_digest=digest(selection)) if selection["selected"] else None
    quality = audit_quality(protocol, runtime, gateway)
    search_raw = [gateway.receipts.load(gateway.identity("search", item.candidate.metadata["attempt_position"], item.candidate))["response"]
                  for item in engine_result.state.observations if item.candidate.source != "initialization"]
    expressions = [item[0].payload["expression"] for item in pool.values()]
    generation = [json.loads(path.read_text(encoding="utf-8")) for path in sorted((runtime.run_dir / "generation").glob("*.json"))]
    initial_generation = [json.loads(path.read_text(encoding="utf-8")) for path in sorted((runtime.run_dir / "initialization/generation").glob("*.json"))]
    prefix = "mock_" if gateway.mock else ""
    initial_raw = [{"success": item.evaluation.succeeded,
                    "metrics": {key.removeprefix(prefix): value for key, value in item.evaluation.metrics.items()}}
                   for item in engine_result.state.observations if item.candidate.source == "initialization"]
    seed_manifest = json.loads((runtime.run_dir / "initialization/seed_manifest.json").read_text(encoding="utf-8"))
    shared_creation = initial_budget.get("metadata", {}).get("shared_seed_creation")
    creation_steps = shared_creation["generation_steps"] if shared_creation else initial_generation
    report = {"task": "alphabench", "mock": gateway.mock, "protocol": protocol.to_dict(), "protocol_digest": protocol.identity,
              "method": protocol.method, "initialization": {"seed_count": len(initial_pool), "budget": initial_budget,
                  "source_digest": seed_manifest["source_digest"], "source_records": len(seed_manifest["admissions"]),
                  "public_information_digest": seed_manifest["public_information_digest"],
                  "source_dispositions": dict(Counter(row["status"] for row in seed_manifest["admissions"])),
                  "generation_cost": generation_costs(initial_generation)},
              "search": {**search_metrics(search_raw), "generation_cost": generation_costs(generation)}, "test": tested,
              "including_initialization": {**search_metrics(initial_raw + search_raw),
                  "generation_cost": generation_costs(creation_steps + generation),
                  "initialization_cost_basis": "shared_source_creation" if shared_creation else "current_run"},
              "independent_combination": combination, "quality_audit": quality,
              "final_pool": [item[0].payload for item in pool.values()],
              "structure_diversity": structure_diversity(expressions),
              "test_signal_diversity": signal_diversity([row["raw"].get("scores", []) for row in tested]),
              "budget": runtime.budget.snapshot(), "engine": engine_result.summary,
              "qualification": "mock_verified" if gateway.mock else "unqualified"}
    # Mandatory analysis capability is checked before claiming a complete report.
    report["complete_t3"] = False
    report["pending_capabilities"] = ["data_qualification", "native_algorithm_recovery", "assay_market_qualification",
                                      "persistent_harness", "compiled_policy", "native_profiles",
                                      "full_metrics_and_quality_coverage", "crash_and_guest_isolation", "full_matrix_qualification"]
    atomic_json_write(runtime.run_dir / "result.json", report)
    with (runtime.run_dir / "trajectory.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("search_attempt", "candidate_id", "status", "objective")); writer.writeheader()
        for index, item in enumerate(row for row in engine_result.state.observations if row.candidate.source != "initialization"):
            writer.writerow({"search_attempt": index+1, "candidate_id": item.candidate_id,
                             "status": item.evaluation.status,
                             "objective": item.evaluation.metrics.get(("mock_" if gateway.mock else "")+protocol.objective)})
    runtime.finish(report)
    return report
