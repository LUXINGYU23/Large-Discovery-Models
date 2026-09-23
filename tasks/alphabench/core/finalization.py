"""Private validation ranking and immutable test selection precede every test call."""

import csv
from collections import Counter
import hashlib
import json
import math

from ldm_tts.contracts import Candidate, RawProposal
from ldm_tts.contracts.evaluation import EvaluationPaused
from ldm_tts.engine.run_store import atomic_json_write
from .candidate import FactorDomain
from .protocol import digest
from .receipts import Receipts
from .reporting import daily_metrics, ea_update_metrics, generation_costs, render_report, search_metrics, signal_diversity, structure_diversity
from .quality import audit_quality


def verify_report_artifacts(run_dir, protocol):
    manifest = json.loads((run_dir / "report_manifest.json").read_text(encoding="utf-8"))
    if manifest["protocol_digest"] != protocol.identity or set(manifest["files"]) != {
            "result.json", "report.md", "trajectory.csv"}:
        raise ValueError("report artifact manifest differs from the frozen protocol")
    for name, expected in manifest["files"].items():
        if hashlib.sha256((run_dir / name).read_bytes()).hexdigest() != expected:
            raise ValueError("completed report artifact differs from its manifest: " + name)
    report = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    if report["protocol_digest"] != protocol.identity or digest(report["protocol"]) != protocol.identity:
        raise ValueError("completed result differs from the frozen protocol")
    if (json.loads((run_dir / "budget.json").read_text(encoding="utf-8")) != report["budget"] or
            json.loads((run_dir / "initialization/budget.json").read_text(encoding="utf-8")) !=
            report["initialization"]["budget"]):
        raise ValueError("completed result differs from the persisted budget ledgers")


def finalize(protocol, runtime, gateway, observations, initial_pool, initial_budget, *, execution):
    private = runtime.run_dir / "private"
    stages = Receipts(private / "stages")
    runtime.status.update("running", phase="validation_complete", budget=runtime.budget)
    domain = FactorDomain(protocol.backend)
    native = protocol.method.startswith("alphabench_")
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
    sidecar_usage = [record["sidecar_usage"] for record in generation
                     if record.get("attempt_kind") == "harness_provider_calls"]
    if protocol.method == "ldm_harness_compiled":
        sidecar_usage.extend(json.loads(path.read_text(encoding="utf-8")).get("harness_turn", {}).get("usage")
            for path in sorted((runtime.run_dir / "policy_harness/rounds").glob("round_*/result.json")))
    initial_generation = [json.loads(path.read_text(encoding="utf-8")) for path in sorted((runtime.run_dir / "initialization/generation").glob("*.json"))]
    prefix = "mock_" if gateway.mock else ""
    initial_raw = [{"success": item.evaluation.succeeded,
                    "metrics": {key.removeprefix(prefix): value for key, value in item.evaluation.metrics.items()}}
                   for item in observations if item.candidate.source == "initialization"]
    seed_manifest = json.loads((runtime.run_dir / "initialization/seed_manifest.json").read_text(encoding="utf-8"))
    shared_creation = initial_budget.get("metadata", {}).get("shared_seed_creation")
    creation_steps = shared_creation["generation_steps"] if shared_creation else initial_generation
    budget = runtime.budget.snapshot()
    expected_sidecar_turns = (budget["counters"].get("harness_turns", 0) + budget["counters"].get("policy_turns", 0)
                              if protocol.method in {"harness", "ldm_harness", "ldm_harness_compiled"} else 0)
    tool_usage_complete = (len(sidecar_usage) == expected_sidecar_turns and
                           all(isinstance(item, dict) and isinstance(item.get("toolCalls"), dict)
                               and all(type(count) is int and count >= 0 for count in item["toolCalls"].values())
                               for item in sidecar_usage))
    tool_calls = Counter()
    if tool_usage_complete:
        for usage in sidecar_usage:
            tool_calls.update(usage["toolCalls"])
    costs = {"scope": "entire run including initialization, proposal and policy",
             "model_requests": (initial_budget["counters"].get("model_requests", 0) +
                                budget["counters"].get("model_requests", 0)),
             "model_requests_by_stage": {"initialization": initial_budget["counters"].get("model_requests", 0),
                                         "search_and_policy": budget["counters"].get("model_requests", 0)},
             "tool_calls": sum(tool_calls.values()) if tool_usage_complete else None,
             "tool_calls_by_name": dict(sorted(tool_calls.items())) if tool_usage_complete else None,
             "tool_usage_reason": None if tool_usage_complete else "sidecar_usage_incomplete"}
    report = {"task": "alphabench", "mock": gateway.mock, "protocol": protocol.to_dict(), "protocol_digest": protocol.identity,
              "method": protocol.method, "initialization": {"seed_count": len(initial_pool), "budget": initial_budget,
                  "source_digest": seed_manifest["source_digest"], "source_records": len(seed_manifest["admissions"]),
                  "public_information_digest": seed_manifest["public_information_digest"],
                  "source_dispositions": dict(Counter(row["status"] for row in seed_manifest["admissions"])),
                  "generation_cost": generation_costs(initial_generation)},
              "search": {**search_metrics(search_raw), "generation_cost": generation_costs(generation),
                         "ea_update": ea_update_metrics(execution["states"], planned_rounds=protocol.rounds)
                         if protocol.method == "alphabench_ea" else {"applicable": False, "reason": "not_ea"}},
              "test": tested,
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
              "budget": budget, "costs": costs, "execution": execution,
              "qualification": "mock_verified" if gateway.mock else "unqualified"}
    selected_count = len(selection["selected"])
    test_success = sum(row["raw"]["success"] and isinstance(row["raw"].get("portfolio"), dict) for row in tested)
    combined = combination is not None and combination["success"] and isinstance(combination.get("portfolio"), dict)
    search_complete = execution["algorithm_completed"] if native else execution["summary"]["stop_reason"] == "observation_target"
    report["completeness"] = {
        "initialization": {"status": "complete", "seed_count": len(initial_pool)},
        "search": {"status": "complete" if search_complete else "partial",
                   "attempts": len(search_raw), "planned_max_attempts": protocol.evaluations,
                   "reason": None if search_complete else execution["stop_reason"] if native else execution["summary"]["stop_reason"]},
        "validation": {"status": "complete" if pool_rows and len(ranked) == len(pool_rows) else "partial" if pool_rows else "unavailable",
                       "attempts": len(pool_rows), "finite_successes": len(ranked),
                       "reason": "invalid_validation_score" if pool_rows and len(ranked) < len(pool_rows) else
                                 "no_final_pool" if not pool_rows else None},
        "selection_frozen": {"status": "complete", "selection_digest": digest(selection),
                             "selected": selected_count},
        "test": {"status": "complete" if selected_count and test_success == selected_count else
                 "partial" if selected_count else "unavailable",
                 "selected": selected_count, "successful_portfolios": test_success,
                 "reason": "failed_test_or_portfolio" if selected_count and test_success < selected_count else
                           "no_selection" if not selected_count else None},
        "independent_combination": {"status": "complete" if combined else "failed" if combination else "unavailable",
                                    "reason": None if combined else "failed_combination_or_portfolio" if combination else "no_selection"},
        "quality_audit": {"status": "complete" if quality["complete"] else "unavailable",
                          "raw_occurrences": quality["raw_occurrences"], "coverage": quality["coverage"],
                          "reason": quality["unavailable_reason"]},
        "data": {"status": "synthetic" if gateway.mock else "unqualified",
                 "digest": protocol.data_digest or None,
                 "reason": "synthetic_oracle" if gateway.mock else "market_data_qualification_pending"},
    }
    for population in ("final_pool", "test_selection"):
        result = report["diversity"][population]
        reasons = {name: result[name]["reason"] for name in ("structure", "signal") if result[name]["reason"]}
        report["completeness"][population + "_diversity"] = {
            "status": "complete" if not reasons else "partial", "reason": ", ".join(
                f"{name}: {reason}" for name, reason in reasons.items()) or None,
            "structure_pairs": result["structure"]["pairs"], "signal_pairs": result["signal"]["pair_count"]}
    report["completeness"]["report"] = {"status": "complete", "artifact": "result.json / report.md / trajectory.csv"}
    # Mandatory analysis capability is checked before claiming a complete report.
    report["complete_t3"] = False
    report["pending_capabilities"] = ["data_qualification", "native_algorithm_recovery", "assay_market_qualification",
                                      "persistent_harness", "compiled_policy", "assay_portfolio_endpoint",
                                      "full_metrics_and_quality_coverage", "crash_and_guest_isolation", "full_matrix_qualification"]
    atomic_json_write(runtime.run_dir / "result.json", report)
    (runtime.run_dir / "report.md").write_text(render_report(report), encoding="utf-8")
    with (runtime.run_dir / "trajectory.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("search_attempt", "candidate_id", "status", "objective",
            "successful_search_observations", "physical_search_jobs", "cumulative_search_jobs",
            "oracle_elapsed_seconds", "cumulative_oracle_elapsed_seconds",
            "model_requests_run_total", "tool_calls_run_total")); writer.writeheader()
        successful, jobs, elapsed = 0, 0, 0.0
        for index, item in enumerate(row for row in observations if row.candidate.source != "initialization"):
            raw = search_raw[index]
            successful += item.evaluation.succeeded
            physical_jobs = len(raw["jobs"])
            jobs += physical_jobs
            duration = raw.get("elapsed_seconds")
            elapsed = elapsed + duration if elapsed is not None and type(duration) in (int, float) and math.isfinite(duration) else None
            writer.writerow({"search_attempt": index+1, "candidate_id": item.candidate_id,
                             "status": item.evaluation.status,
                             "objective": item.evaluation.metrics.get(("mock_" if gateway.mock else "")+protocol.objective),
                             "successful_search_observations": successful,
                             "physical_search_jobs": physical_jobs, "cumulative_search_jobs": jobs,
                             "oracle_elapsed_seconds": duration, "cumulative_oracle_elapsed_seconds": elapsed,
                             "model_requests_run_total": costs["model_requests"],
                             "tool_calls_run_total": costs["tool_calls"]})
    atomic_json_write(runtime.run_dir / "report_manifest.json", {
        "protocol_digest": protocol.identity,
        "files": {name: hashlib.sha256((runtime.run_dir / name).read_bytes()).hexdigest()
                  for name in ("result.json", "report.md", "trajectory.csv")}})
    return report
