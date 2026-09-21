#!/usr/bin/env python3
"""Verify a declared full reconstruction grid and emit a horizontal table.

Requires a predeclared JSON suite: methods (id/label/search_method/proposal_mode/
model/trials_per_target), targets (1000 ordered SMILES per dataset), assets
(frozen digest dictionaries per dataset), and cases (method/dataset/seed/path).
Paths are relative to the suite manifest unless absolute. No inference is run.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from tasks.reasyn.core.chemistry import canonicalize
from tasks.reasyn.core.metrics import reconstruction_metrics

DATASETS = ("enamine", "chembl", "zinc250k")
SEEDS = (0, 1, 2)
CYCLES = {"enamine": 12, "chembl": 24, "zinc250k": 16}
METRICS = ("reconstruction_rate", "mean_similarity", "product_diversity", "building_block_diversity")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def target_directories(folder):
    return {p.name for p in folder.glob("target-*") if p.is_dir()}


def close_metrics(left, right, context):
    for key in METRICS:
        require(math.isfinite(left[key]) and math.isfinite(right[key])
                and math.isclose(left[key], right[key], rel_tol=1e-9, abs_tol=1e-10),
                f"{context}: recomputed {key} differs from saved result")


def validate_suite(suite):
    require(suite.get("benchmark") == "reconstruction", "not a reconstruction suite")
    seeds = suite.get("seeds")
    protocol = suite.get("replication_protocol", "three_seed")
    require(protocol in ("three_seed", "single_seed"), "unknown replication protocol")
    require(isinstance(seeds, list) and seeds and all(type(s) is int and s >= 0 for s in seeds), "seeds must be explicit nonnegative integers")
    require((protocol == "three_seed" and seeds == list(SEEDS)) or
            (protocol == "single_seed" and len(seeds) == 1),
            "three_seed requires seeds 0, 1, 2; single_seed requires exactly one declared seed")
    require(set(suite["targets"]) == set(DATASETS), "all three released datasets are required")
    for dataset in DATASETS:
        targets = [canonicalize(t, stereo=False) for t in suite["targets"][dataset]]
        require(len(targets) == len(set(targets)) == 1000, f"{dataset}: require 1000 unique valid targets")
        require(bool(suite["assets"][dataset]), f"{dataset}: missing frozen asset identities")
    methods = {row["id"]: row for row in suite["methods"]}
    require(methods and len(methods) == len(suite["methods"]), "methods must be nonempty and unique")
    for method in methods.values():
        require(type(method["trials_per_target"]) is int and method["trials_per_target"] in (1, 4), "supported protocols are original single projection or four matched trials")
        require(method["trials_per_target"] != 1 or method["search_method"] == "baseline",
                "single-projection protocol is reserved for original ReaSyn")
        require(method["proposal_mode"] in ("baseline", "openai", "harness"), "invalid proposal backend")
        if method["proposal_mode"] != "baseline":
            keys = {"acquisition_alpha", "acquisition_eta", "acquisition_beta", "acquisition_z_clip",
                    "reservoir_size", "proposal_batch_size", "bo_pool_size", "llm_max_tokens"}
            if method["proposal_mode"] == "harness":
                keys |= {"harness_sessions", "harness_thinking", "harness_sidecar_image", "policy_runner_image"}
            require(all(keys <= set(method.get("settings_by_dataset", {}).get(d, {})) for d in DATASETS),
                    "search hyperparameters must be locked per dataset in the suite manifest")
    cases = [(c["method"], c["dataset"], c["seed"]) for c in suite["cases"]]
    expected = {(m, d, s) for m in methods for d in DATASETS for s in seeds}
    require(len(cases) == len(set(cases)) and set(cases) == expected, "missing, duplicate, or unexpected method/dataset/seed cases")
    paths = [str(c["path"]) for c in suite["cases"]]
    require(len(set(paths)) == len(paths), "a run directory cannot represent multiple seeds or methods")
    return methods


def verify_target(folder, *, target, dataset, seed, method, assets):
    folder = Path(folder)
    config = read(folder / "config.json")
    identity = read(folder / "scientific_identity.json")
    result = read(folder / "result.json")
    ledger = read(folder / "budget.json")
    budget = ledger["counters"]
    require(read(folder / "status.json")["status"] == "completed", f"{folder}: unfinished target")
    require(result.get("mock") is False and config.get("mock") is False, f"{folder}: mock or missing real-mode receipt")
    require(result["benchmark"] == "reconstruction" and config["dataset"] == dataset and config["seed"] == seed,
            f"{folder}: wrong dataset or seed")
    require(identity["original_target"] == target and identity["asset_digests"] == assets,
            f"{folder}: mismatched target or frozen assets")
    for key in ("search_method", "proposal_mode"):
        require(config[key] == method[key], f"{folder}: wrong {key}")
    for key, value in method.get("settings_by_dataset", {}).get(dataset, {}).items():
        require(config.get(key) == value, f"{folder}: changed locked hyperparameter {key}")
    if method.get("model"):
        require(config["llm_model"] == method["model"], f"{folder}: wrong proposal model")
    trials = method["trials_per_target"]
    settings = {"search_width": 8, "exhaustiveness": 4, "num_editflow_samples": 100,
                "num_cycles": CYCLES[dataset] // trials, "max_results": 100}
    require(config["iterations"] == trials and config["evaluations_per_round"] == 1, f"{folder}: wrong round/evaluation schedule")
    for key, value in settings.items():
        require(config[key] == identity[key] == value, f"{folder}: non-paper {key}")
    require(result["successful_evaluations"] == budget["successful_evaluations"] == trials,
            f"{folder}: incomplete scientific evaluation budget")
    require(trials <= budget["projection_targets"] <= ledger["limits"]["projection_targets"],
            f"{folder}: invalid projection/recovery accounting")
    requests = sorted((folder / "projections").glob("*/request.json"))
    require(len(requests) == trials, f"{folder}: missing or unexpected projection trials")
    projected_rows = []
    for path in requests:
        request = read(path)
        projection = read(path.with_name("result.json"))
        require(request["asset_digests"] == assets, f"{path}: changed frozen assets")
        require(len(request["targets"]) == 1 and request["settings"].get("exact_break") is True,
                f"{path}: wrong projection target count or early-stop semantics")
        require(all(request["settings"][k] == v for k, v in settings.items()), f"{path}: non-paper projection settings")
        require(projection.get("complete") is True and projection.get("parameter_count") == 341397459,
                f"{path}: incomplete or incorrect frozen model pair")
        require(projection["request_sha256"] == hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest(),
                f"{path}: projection receipt belongs to another request")
        require(all(r.get("pathway_verified") is True and r["target"] == request["targets"][0]
                    for r in projection["rows"]), f"{path}: unverified pathway or wrong query")
        projected_rows.extend({**r, "target": target, "projection_query": r["target"]} for r in projection["rows"])
    signatures = lambda rows: Counter(json.dumps(r, sort_keys=True) for r in rows)
    require(signatures(projected_rows) == signatures(result["reconstruction_rows"]),
            f"{folder}: reported pathways differ from selected projection receipts")
    # Empty, completed projections legitimately contribute zero. Never omit them.
    metrics = reconstruction_metrics([target], projected_rows)
    close_metrics(metrics, result["metrics"], str(folder))
    decisions = Counter()
    if method["proposal_mode"] == "harness":
        manifest = read(folder / "harness/manifest.json")
        require(manifest["model"] == method["model"] and manifest["wireApi"] == "responses", f"{folder}: incorrect Harness provider")
        require(len(manifest["profiles"]) == config["harness_sessions"], f"{folder}: reduced research session count")
        require(budget.get("harness_provider_calls", 0) > 0 and budget.get("harness_tool_calls", 0) > 0,
                f"{folder}: missing real Harness calls/tools")
        if method["search_method"] == "ldm_harness_compiled":
            policy_manifest = read(folder / "policy_harness/manifest.json")
            require(policy_manifest["model"] == method["model"] and policy_manifest["wireApi"] == "responses",
                    f"{folder}: incorrect independent policy provider")
            policy = list((folder / "policy_harness/rounds").glob("round_*/result.json"))
            require(len(policy) == trials - 1 and budget.get("policy_provider_requests", 0) > 0,
                    f"{folder}: missing policy decisions or actual policy model calls")
            decisions.update(read(p)["action"] for p in policy)
    elif method["proposal_mode"] == "openai":
        batches = list((folder / "proposal_batches").glob("round-*/batch-*.json"))
        require(batches and budget.get("llm_requests", 0) >= len(batches), f"{folder}: missing provider receipts")
        require(all(read(p)["response"]["metadata"].get("model") == method["model"] for p in batches),
                f"{folder}: response model does not match requested model")
    return {"metrics": {k: metrics[k] for k in METRICS}, "budget": budget,
            "policy_actions": dict(decisions), "result_sha256": digest(folder / "result.json")}


def aggregate(manifest_path):
    manifest_path = Path(manifest_path).resolve()
    suite = read(manifest_path)
    methods = validate_suite(suite)
    seeds = suite["seeds"]
    found, receipts = {}, []
    resolved_paths = set()
    for case in suite["cases"]:
        key = (case["method"], case["dataset"], case["seed"])
        folder = (manifest_path.parent / case["path"]).resolve()
        require(folder not in resolved_paths, "multiple cases resolve to the same run directory")
        resolved_paths.add(folder)
        targets = suite["targets"][case["dataset"]]
        saved = read(folder / "benchmark_result.json")
        require(saved.get("mock") is False and saved.get("benchmark") == "reconstruction", f"{folder}: not a real reconstruction result")
        require(saved["target_count"] == saved["requested_campaigns"] == saved["completed_campaigns"] == 1000,
                f"{folder}: incomplete full target denominator")
        actual = read(folder / "targets_manifest.json")
        require(actual.get("mock") is False and actual["targets"] == targets, f"{folder}: targets differ from predeclared suite")
        expected_names = {f"target-{index:05d}" for index in range(1000)}
        require(target_directories(folder) == expected_names,
                f"{folder}: missing or extra target directories")
        summaries = []
        budget, decisions = Counter(), Counter()
        for index, target in enumerate(targets):
            item = verify_target(folder / f"target-{index:05d}", target=target,
                                 dataset=case["dataset"], seed=case["seed"], method=methods[case["method"]],
                                 assets=suite["assets"][case["dataset"]])
            summaries.append(item)
            budget.update(item["budget"])
            decisions.update(item["policy_actions"])
        measured = {metric: statistics.mean(item["metrics"][metric] for item in summaries) for metric in METRICS}
        close_metrics(measured, saved, str(folder))
        found[key] = measured
        receipts.append({**case, "metrics": measured, "budget": dict(budget), "policy_actions": dict(decisions),
                         "benchmark_result_sha256": digest(folder / "benchmark_result.json"),
                         "target_result_sha256": [item["result_sha256"] for item in summaries]})
    rows = []
    for method in suite["methods"]:
        row = {"method": method["id"], "label": method["label"], "datasets": {}}
        for dataset in DATASETS:
            row["datasets"][dataset] = {metric: ({
                "mean": statistics.mean(found[method["id"], dataset, seed][metric] for seed in seeds),
                "std": statistics.pstdev(found[method["id"], dataset, seed][metric] for seed in seeds),
            } if len(seeds) > 1 else {"value": found[method["id"], dataset, seeds[0]][metric]}) for metric in METRICS}
        rows.append(row)
    return {"benchmark": "reconstruction", "scope": "paper_scale", "targets_per_dataset": 1000,
            "seeds": seeds, "replication_protocol": suite.get("replication_protocol", "three_seed"),
            **({"std_ddof": 0} if len(seeds) > 1 else {}), "suite_sha256": digest(manifest_path),
            "rows": rows, "case_receipts": receipts,
            "qualification_note": "Full-grid result audit, not an assertion of clean-git provenance, identical upstream sampling trajectories, or global SOTA."}


def markdown(result):
    headers = ["Method"] + [f"{d}/{m}" for d in DATASETS for m in ("Recon.%", "Sim.", "Div.P", "Div.BB")]
    seeds = result.get("seeds", list(SEEDS))
    description = (f"1,000 targets/dataset; seed={seeds[0]} only. Full-target point estimates; no CI or cross-seed standard deviation."
                   if len(seeds) == 1 else "1,000 targets/dataset; seeds 0, 1, 2; mean ± population SD (ddof=0).")
    lines = ["# ReaSyn — full reconstruction results", "",
             description, "",
             "| " + " | ".join(headers) + " |", "| --- | " + " | ".join(["---:"] * 12) + " |"]
    for row in result["rows"]:
        cells = [row["label"].replace("|", "\\|").replace("\n", " ")]
        for dataset in DATASETS:
            for metric in METRICS:
                item = row["datasets"][dataset][metric]
                scale, decimals = (100, 1) if metric == "reconstruction_rate" else (1, 3)
                cells.append(f"{item['value']*scale:.{decimals}f}" if len(seeds) == 1 else
                             f"{item['mean']*scale:.{decimals}f} ± {item['std']*scale:.{decimals}f}")
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "Original ReaSyn uses one continuous 12/24/16-cycle projection. Search methods use four restarted 3/6/4-cycle projections; summed cycle caps match but trajectories and actual compute differ.",
              "All rows use width 8, exhaustiveness 4, 100 EB samples/cycle and the same frozen assets. All cells are newly measured, not copied from the paper. Read case_receipts for actual costs and disable/keep/replace policy decisions.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = aggregate(args.suite)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "reconstruction_full.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.output / "reconstruction_full.md").write_text(markdown(result))
    print(json.dumps({"status": "verified", "cases": len(result["case_receipts"]), "output": str(args.output)}))


if __name__ == "__main__":
    main()
