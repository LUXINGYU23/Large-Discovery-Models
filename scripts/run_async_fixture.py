#!/usr/bin/env python3
"""Run a deterministic async numeric fixture from the pinned runtime root.

The working directory must be the runtime root containing experiment_service,
ldm_tts, tasks/numeric_search and scripts/simulate_numeric.py. The run directory
must already exist; its runtime pin is left in place. No model or device is used.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--metric", required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args(argv)
    if not args.run_dir.is_dir():
        parser.error("--run-dir must be an existing directory")
    if not args.task_id.strip() or not args.metric.strip():
        parser.error("--task-id and --metric must not be blank")
    if not 1 <= args.budget <= 101 or args.concurrency < 1:
        parser.error("require 1 <= budget <= 101 and concurrency >= 1")

    root = Path.cwd()
    sys.path.insert(0, str(root))
    from experiment_service import VirtualClock
    from experiment_service.manifest import ExecutorContext
    from experiment_service.testing import running_service
    from ldm_tts.async_runtime import CampaignConfig, Controller, ESClient, FixedProposer, RuntimeOptions, check_invariants
    from ldm_tts.contracts import ObjectiveSet, ObjectiveSpec
    from scripts.simulate_numeric import _drive
    from tasks.numeric_search.core import executors, experiments
    from tasks.numeric_search.core.domain import NumericDomain
    from tasks.numeric_search.core.proposers import fixed_candidates, numeric_sampler

    class FixtureCodec(experiments.NumericCodec):
        def __init__(self, metric):
            super().__init__(experiments.SIM_KIND)
            self.metric = metric
            self.name = f"numeric-fixture[{metric}]"

        def metrics(self, candidate, result):
            # The numeric task maximizes a nonpositive objective; expose its
            # negation as the requested bounded metric to minimize.
            return {self.metric: -float(result["objective"])}

    run_dir = args.run_dir.resolve()
    names = ("controller.sqlite3", "experiment-service.sqlite3", "summary.json", "budget.json", "events.jsonl")
    if any((run_dir / name).exists() for name in names):
        raise FileExistsError("run directory already contains fixture outputs")
    clock = VirtualClock()
    sim = executors.make_simulated_executor(ExecutorContext(
        experiments.SIM_KIND, clock, {"seed": args.seed, "duration": ["fixed", 1]}))
    candidates = fixed_candidates(NumericDomain(), numeric_sampler, args.budget, args.seed)
    config = CampaignConfig("fixture", budget=args.budget, concurrency=args.concurrency, seed=args.seed)
    options = RuntimeOptions(anti_entropy_interval=300, long_poll_seconds=0)
    objectives = ObjectiveSet.from_specs([ObjectiveSpec(args.metric, "minimize", "Synthetic fixture metric")])
    with running_service([sim], pools={"default": args.concurrency}, clock=clock,
                         database=run_dir / "experiment-service.sqlite3") as es:
        with Controller(run_dir / "controller.sqlite3", config, codec=FixtureCodec(args.metric),
                        client=ESClient(es.base_url), clock=clock, proposer=FixedProposer(candidates),
                        objectives=objectives, options=options) as ctrl:
            _drive(ctrl, es, clock)
            items = ESClient(es.base_url).changes(config.campaign_id, limit=1000)["items"]
            observations = ctrl.observations()
            successful = [o for o in observations if o.evaluation.succeeded]
            incumbent = min(successful, key=lambda o: o.evaluation.metrics[args.metric], default=None)
            expected = {c.payload: -executors.evaluate_spec({"x": c.payload})["objective"] for c in candidates}
            actual = {o.candidate.payload: o.evaluation.metrics.get(args.metric) for o in successful}
            problems = check_invariants(ctrl.store, config, items, synced=True)
            if len(observations) != args.budget or len(successful) != args.budget or actual != expected:
                problems.append("observations do not exactly match the deterministic fixture budget and values")
            if len(items) != args.budget or any(item["state"] != "SUCCEEDED" for item in items):
                problems.append("experiment-service resources do not all succeed")
            if len(sim.launches) != args.budget or len(set(sim.launches)) != args.budget:
                problems.append("physical attempts are missing or duplicated")
            if ctrl.budget().consumed != args.budget or ctrl.ledger.campaign_row()["stop_reason"] != "budget_exhausted":
                problems.append("campaign did not exhaust its budget")
            if ctrl.anomalies():
                problems.append("controller has open anomalies")
            summary = {
                "task": args.task_id,
                "metric": args.metric,
                "observation_count": len(observations),
                "successful_evaluation_count": len(successful),
                "failed_evaluation_count": len(observations) - len(successful),
                "incumbent": incumbent.to_dict() if incumbent else None,
                "stop_reason": ctrl.ledger.campaign_row()["stop_reason"],
                "invariant_problems": problems,
            }
            budget, events = ctrl.budget().to_dict(), ctrl.events()

    def write_json(path, value):
        path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")

    write_json(run_dir / "budget.json", budget)
    (run_dir / "events.jsonl").write_text("".join(json.dumps(event, sort_keys=True, allow_nan=False) + "\n"
                                                   for event in events), encoding="utf-8")
    write_json(run_dir / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True, allow_nan=False))
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
