"""Private validation ranking and immutable test selection precede every test call."""

import csv
from collections import Counter
import json
import math

from ldm_tts.contracts import Candidate, RawProposal
from ldm_tts.contracts.evaluation import EvaluationPaused
from ldm_tts.engine.run_store import atomic_json_write
from .candidate import FactorDomain
from .protocol import digest
from .receipts import Receipts
from .reporting import daily_metrics, generation_costs, search_metrics, signal_diversity, structure_diversity
from .quality import audit_quality


def finalize(protocol, runtime, gateway, observations, initial_pool, initial_budget, *, execution):
    private = runtime.run_dir / "private"
    stages = Receipts(private / "stages")
    runtime.status.update("running", phase="validation_complete", budget=runtime.budget)
    domain = FactorDomain(protocol.backend)
    native_pool = protocol.profile in {"upstream_searcher_v1", "upstream_benchmark_v1"}
    if native_pool and not execution["algorithm_completed"]:
        raise EvaluationPaused("source entry has not completed its configured algorithm", status="paused_native_search")
    pool = {}
    for row in initial_pool:
        if row["success"]:
            candidate = domain.admit(RawProposal(row, "initialization"))
            pool[candidate.candidate_id] = (candidate, row["validation_position"], "initialization", row["validation"])
    for observation in observations:
        if (observation.evaluation.succeeded or native_pool) and observation.candidate.source != "initialization":
            candidate = observation.candidate
            pool[candidate.candidate_id] = (candidate, candidate.metadata["attempt_position"], "search", None)
    pool_rows = list(pool.values())
    if native_pool:
        pool_rows = []
        for row in execution["native_final_pool"]:
            candidate = domain.admit(RawProposal(row, "native_final_pool"))
            if not isinstance(candidate, Candidate) or candidate.candidate_id not in pool:
                raise ValueError("native final pool contains a factor without a measured receipt")
            pool_rows.append((candidate, *pool[candidate.candidate_id][1:]))
    ranked, validation_scores = [], []
    for candidate, position, source, raw in pool_rows:
        if raw is None:
            receipt = gateway.receipts.load(gateway.identity("validation", position, candidate))
            if receipt is None or receipt["state"] != "completed":
                raise ValueError("validation must complete before freezing test selection")
            raw = receipt["response"]
        validation_scores.append({"candidate_id": candidate.candidate_id, "scores": raw.get("scores", [])})
        value = raw.get("metrics", {}).get(protocol.validation_metric)
        if not raw["success"] or type(value) not in (int, float) or not math.isfinite(value):
            if native_pool:
                raise EvaluationPaused("native final pool lacks complete finite private validation; search values cannot replace it",
                                       status="paused_incomplete_validation")
            continue
        ranked.append({"candidate": candidate.to_dict(), "validation_score": value, "source": source})
    ranked.sort(key=(lambda row: -row["validation_score"]) if native_pool else
                (lambda row: (-row["validation_score"], row["candidate"]["candidate_id"])))
    selection = {"protocol": protocol.identity, "direction": protocol.direction, "validation_metric": protocol.validation_metric,
                 "stock_topk": protocol.stock_topk, "stock_n_drop": protocol.stock_n_drop,
                 "selected": ranked[:protocol.factor_select_n]}
    stages.accept("selection_frozen", selection)
    atomic_json_write(runtime.run_dir / "selection_frozen.json", selection)
    runtime.status.update("running", phase="test", budget=runtime.budget)
    tested = []
    for index, row in enumerate(selection["selected"]):
        candidate = Candidate(**row["candidate"])
        raw = gateway.evaluate(candidate, phase="test", position=[digest(selection), index], fast=False)
        tested.append({"candidate": candidate.payload, "candidate_id": candidate.candidate_id,
                       "raw": raw, "derived": daily_metrics(raw.get("daily", []), direction=protocol.direction)})
    stages.accept("test", {"selection_digest": digest(selection), "results": tested})
    runtime.status.update("running", phase="independent_analysis", budget=runtime.budget)
    combination = gateway.combine([Candidate(**row["candidate"]) for row in selection["selected"]],
                                  selection_digest=digest(selection)) if selection["selected"] else None
    quality = audit_quality(protocol, runtime, gateway)
    search_raw = [gateway.receipts.load(gateway.identity("search", item.candidate.metadata["attempt_position"], item.candidate))["response"]
                  for item in observations if item.candidate.source != "initialization"]
    expressions = [item[0].payload["expression"] for item in pool_rows]
    generation = [json.loads(path.read_text(encoding="utf-8")) for path in sorted((runtime.run_dir / "generation").glob("*.json"))]
    initial_generation = [json.loads(path.read_text(encoding="utf-8")) for path in sorted((runtime.run_dir / "initialization/generation").glob("*.json"))]
    prefix = "mock_" if gateway.mock else ""
    initial_raw = [{"success": item.evaluation.succeeded,
                    "metrics": {key.removeprefix(prefix): value for key, value in item.evaluation.metrics.items()}}
                   for item in observations if item.candidate.source == "initialization"]
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
              "final_pool": [item[0].payload for item in pool_rows],
              "diversity": {
                  "final_pool": {"phase": "validation", "interval": protocol.interval("validation"),
                      "scope": "native algorithm final pool" if native_pool else "all successful measured initialization and search candidates",
                      "structure": structure_diversity(expressions), "signal": signal_diversity(validation_scores)},
                  "test_selection": {"phase": "test", "interval": protocol.interval("test"),
                      "selection_digest": digest(selection),
                      "structure": structure_diversity([row["candidate"]["expression"] for row in tested]),
                      "signal": signal_diversity([{"candidate_id": row["candidate_id"], "scores": row["raw"].get("scores", [])} for row in tested])}},
              "budget": runtime.budget.snapshot(), "execution": execution,
              "qualification": "mock_verified" if gateway.mock else "unqualified"}
    # Mandatory analysis capability is checked before claiming a complete report.
    report["complete_t3"] = False
    report["pending_capabilities"] = ["data_qualification", "native_algorithm_recovery", "assay_market_qualification",
                                      "persistent_harness", "compiled_policy", "assay_portfolio_endpoint",
                                      "full_metrics_and_quality_coverage", "crash_and_guest_isolation", "full_matrix_qualification"]
    atomic_json_write(runtime.run_dir / "result.json", report)
    with (runtime.run_dir / "trajectory.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("search_attempt", "candidate_id", "status", "objective")); writer.writeheader()
        for index, item in enumerate(row for row in observations if row.candidate.source != "initialization"):
            writer.writerow({"search_attempt": index+1, "candidate_id": item.candidate_id,
                             "status": item.evaluation.status,
                             "objective": item.evaluation.metrics.get(("mock_" if gateway.mock else "")+protocol.objective)})
    runtime.finish(report)
    return report
