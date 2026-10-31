"""Offline T3 metric projections with explicit domains and undefined values."""

import ast
import itertools
import json
import math
import numpy as np

from .grammar import parse_expression
from .protocol import digest


def daily_metrics(rows, *, direction=1, ddof=1):
    result = {"direction": direction, "ddof": ddof, "annualized": False}
    for name in ("ic", "rank_ic"):
        values = np.array([row[name] for row in rows if type(row.get(name)) in (int, float) and math.isfinite(row[name])])
        result[name+"_n"] = len(values)
        if not len(values):
            result[name+"_derived"] = {"mean": None, "ir": None, "winrate": None, "skewness": None, "reason": "no_valid_days"}
            continue
        mean = float(np.mean(values)); centered = values-mean
        variance = float(np.mean(centered**2))
        std = float(np.std(values, ddof=ddof)) if len(values) > ddof else 0
        result[name+"_derived"] = {"mean": mean, "ir": mean/std if std else None,
            "winrate": float(np.mean(values*direction > 0)),
            "skewness": float(np.mean(centered**3) / variance**1.5) if variance else None,
            "skewness_estimator": "central_moment_m3_over_m2_pow_1.5",
            "reason": None if std and variance else "insufficient_or_constant_series"}
    return result


def search_metrics(records):
    ics = [record.get("metrics", {}).get("ic") if record["success"] else None for record in records]
    valid = [value for value in ics if type(value) in (int, float) and math.isfinite(value)]
    success = next((index+1 for index, value in enumerate(ics)
                    if type(value) in (int, float) and math.isfinite(value) and value > .03), None)
    return {"threshold": .03, "strict_comparison": ">", "attempts": len(records),
            "successful_measurements": len(valid), "threshold_discovery_evaluations": success,
            "gain": max(0., min(max(valid, default=0.)/.03, 1.)),
            "run_success": success is not None}


def generation_costs(records):
    harness = [record for record in records if record.get("attempt_kind") == "harness_provider_calls"]
    if harness and len(harness) != len(records):
        direct = [record for record in records if record.get("attempt_kind") != "harness_provider_calls"]
        return {"unit": "mixed", "step_count": len(records), "total_attempts": None,
                "unavailable_reason": "different_attempt_units",
                "components": {"direct": generation_costs(direct), "harness": generation_costs(harness)}}
    attempts = [record["attempt_count"] for record in records]
    if harness and any(value is not None and (type(value) is not int or value < 0) for value in attempts):
        raise ValueError("Harness provider call counts must be nonnegative integers or unavailable")
    known = all(type(value) is int and value >= 0 for value in attempts)
    if not known and not harness:
        raise ValueError("generation attempt counts must be nonnegative integers")
    result = {"unit": "sidecar provider calls per committed Harness turn" if harness else
                       "model attempts per logical generation step", "step_count": len(records),
            "attempts_per_step": attempts, "total_attempts": sum(attempts) if known else None,
            "mean": sum(attempts) / len(attempts) if attempts and known else None,
            "failed_steps": sum(not record["complete"] and not record.get("interrupted", False) for record in records),
            "interrupted_steps": sum(record.get("interrupted", False) for record in records)}
    if harness:
        result["unavailable_reason"] = None if known else "sidecar_provider_usage_missing"
        result["submission_attempts_per_step"] = [record["submission_attempts"] for record in records]
        result["sidecar_usage_per_step"] = [record["sidecar_usage"] for record in records]
    return result


def ea_update_metrics(states, *, planned_rounds):
    snapshots = [item for item in states if item["kind"] == "ea.pool"]
    if not snapshots:
        return {"applicable": True, "rounds": 0, "planned_rounds": planned_rounds,
                "sum_U_t": 0, "update_rate": None, "best_update_events": 0,
                "reason": "no_ea_pool_snapshots",
                "definition": "new generation expressions entering the retained top pool per completed round",
                "events": []}
    rounds = [item["state"]["round"] for item in snapshots]
    if rounds != list(range(len(rounds))) or len(rounds) > planned_rounds + 1:
        raise ValueError("EA pool snapshots are not contiguous from round zero")
    events = []
    for before, after in itertools.pairwise(snapshots):
        previous = {item["expression"] for item in before["state"]["pool"]}
        current = [item["expression"] for item in after["state"]["pool"]]
        generated = {item["expression"] for item in after["state"]["candidates"]}
        entered = [expression for expression in current if expression not in previous]
        if len(set(current)) != len(current) or not set(entered) <= generated:
            raise ValueError("EA pool update differs from the recorded generation")
        prior_best = before["state"]["pool"][0]["expression"] if before["state"]["pool"] else None
        current_best = current[0] if current else None
        events.append({"round": after["state"]["round"], "boundary": after["boundary"],
                       "U_t": len(entered), "entered_expressions": entered,
                       "best_updated": current_best != prior_best,
                       "previous_best": prior_best, "current_best": current_best})
    count = len(events)
    return {"applicable": True, "rounds": count, "planned_rounds": planned_rounds,
            "sum_U_t": sum(item["U_t"] for item in events),
            "update_rate": sum(item["U_t"] for item in events) / count if count else None,
            "best_update_events": sum(item["best_updated"] for item in events),
            "reason": None if count == planned_rounds else "incomplete_rounds",
            "definition": "new generation expressions entering the retained top pool per completed round",
            "events": events}


def aggregate_fraction_success(roster, reports, *, invalid_evidence=None, invalid_rule="count_as_failure"):
    if invalid_rule not in {"count_as_failure", "exclude_with_evidence"}:
        raise ValueError("unknown pre-registered infrastructure-invalid rule")
    invalid_evidence = invalid_evidence or {}
    ids = [item["run_id"] for item in roster]
    if len(ids) != len(set(ids)) or set(reports) - set(ids) or set(invalid_evidence) - set(ids):
        raise ValueError("aggregate inputs differ from the pre-registered run roster")
    rows = []
    for item in roster:
        run_id, report = item["run_id"], reports.get(item["run_id"])
        if not item.get("protocol_digest") or type(item.get("seed")) is not int:
            raise ValueError("roster requires a frozen protocol digest and seed per run")
        if report is not None:
            if (report["protocol_digest"] != item["protocol_digest"] or
                    report["protocol"]["random_seed"] != item["seed"] or
                    report["method"] not in {"alphabench_cot", "alphabench_tot"} or
                    type(report["search"]["run_success"]) is not bool):
                raise ValueError("run report differs from its pre-registered CoT/ToT identity")
        evidence = invalid_evidence.get(run_id)
        if evidence is not None and (not isinstance(evidence, str) or not evidence.strip() or
                                     report is not None and report["search"]["run_success"]):
            raise ValueError("infrastructure invalidation needs evidence and cannot remove a successful run")
        excluded = evidence is not None and invalid_rule == "exclude_with_evidence"
        rows.append({"run_id": run_id, "seed": item["seed"], "protocol_digest": item["protocol_digest"],
                     "included": not excluded, "search_success": report["search"]["run_success"] if report else False,
                     "result_available": report is not None, "infrastructure_invalid_evidence": evidence,
                     "reason": "infrastructure_invalid" if evidence else "missing_result" if report is None else None})
    included = [row for row in rows if row["included"]]
    wins = sum(row["search_success"] for row in included)
    return {"metric": "CoE/ToT FracSuccess", "invalid_rule": invalid_rule,
            "pre_registered_runs": len(roster), "included_runs": len(included),
            "infrastructure_invalid_runs": sum(row["infrastructure_invalid_evidence"] is not None for row in rows),
            "successes": wins, "frac_success": wins / len(included) if included else None,
            "unavailable_reason": None if included else "no_included_runs", "runs": rows}


def render_report(report):
    protocol = report["protocol"]
    lines = ["# AlphaBench T3 Run", "",
             f"Method: `{report['method']}` | Profile: `{protocol['profile']}` | "
             f"Market: `{protocol['backend']}:{protocol['market']}`",
             f"Protocol SHA-256: `{report['protocol_digest']}`",
             f"Data SHA-256: `{protocol['data_digest'] or 'unavailable'}`", "",
             "## Stage Completeness", "",
             "| Stage | Status | Reason |", "| --- | --- | --- |"]
    stages = ("initialization", "search", "validation", "selection_frozen", "test",
              "independent_combination", "quality_audit", "data", "final_pool_diversity",
              "test_selection_diversity", "report")
    for stage in stages:
        value = report["completeness"][stage]
        lines.append(f"| {stage} | {value['status']} | {value.get('reason') or ''} |")
    search, combined, quality = report["search"], report["including_initialization"], report["quality_audit"]
    filter_key = protocol["backend"] + "_" + quality["check_kind"] + "_success_rate"
    lines.extend(["", "## Search", "",
        "| Population | Attempts | Successful measurements | First strict IC > 0.03 | Gain | Run success |",
        "| --- | ---: | ---: | ---: | ---: | --- |"])
    for name, value in (("new search", search), ("including initialization", combined)):
        lines.append(f"| {name} | {value['attempts']} | {value['successful_measurements']} | "
                     f"{value['threshold_discovery_evaluations']} | {value['gain']} | {value['run_success']} |")
    for name, value in (("Search", search), ("Including initialization", combined)):
        cost = value["generation_cost"]
        lines.append(f"{name} generation cost ({cost['unit']}): "
                     f"{cost.get('attempts_per_step', cost.get('components'))}")
    update = search["ea_update"]
    if update["applicable"]:
        lines.extend([f"EA U_t / completed T: {update['sum_U_t']} / {update['rounds']}; "
                      f"UpdateRate: {update['update_rate']}; best-update events: {update['best_update_events']}; "
                      f"reason: {update['reason']}",
                      f"EA entered expressions per round: {[(event['round'], event['entered_expressions']) for event in update['events']]}"])
    lines.extend([
        f"Model requests (run total): {report['costs']['model_requests']}",
        f"Tool calls (run total): {report['costs']['tool_calls']}",
        "", "## Quality", "",
        f"Raw occurrences: {quality['raw_occurrences']}",
        f"Format failures (outside occurrence denominator): {quality['format_failures']}",
        f"Audit coverage: {quality['coverage']}; reason: {quality['unavailable_reason']}",
        f"Static success rate: {quality['static_success_rate']}",
        f"{filter_key}: {quality[filter_key]}",
        f"Paper dynamic success rate: {quality['paper_dynamic_success_rate']}; "
        f"coverage: {quality['paper_coverage']}; reason: {quality['paper_unavailable_reason']}",
        f"Unique admission: {quality['unique_admitted']} / {quality['unique_generated']} "
        f"({quality['unique_admission_rate']})",
        f"Native quality: {json.dumps(quality['native_quality'], sort_keys=True)}",
        "", "## Diversity", "",
        "| Population | AST pairs | AST mean / max | Signal pairs | Signal diversity | Reason |",
        "| --- | ---: | --- | ---: | ---: | --- |"])
    for name, value in report["diversity"].items():
        structure, signal = value["structure"], value["signal"]
        lines.append(f"| {name} | {structure['pairs']} | {structure['mean']} / {structure['max']} | "
                     f"{signal['pair_count']} / {signal['total_pairs']} | {signal['diversity']} | "
                     f"{report['completeness'][name + '_diversity']['reason'] or ''} |")
    lines.extend(["", "## Frozen Test", "",
        "| Factor | IC | RankIC | ICIR | RankICIR | IC WinRate | IC skewness | Daily rows | Portfolio |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"])
    for row in report["test"]:
        raw, derived = row["raw"], row["derived"]
        metrics = raw.get("metrics", {})
        lines.append(f"| `{row['candidate_id']}` | {metrics.get('ic')} | {metrics.get('rank_ic')} | "
                     f"{metrics.get('icir')} | {metrics.get('rank_icir')} | "
                     f"{derived['ic_derived']['winrate']} | {derived['ic_derived']['skewness']} | "
                     f"{len(raw.get('daily', []))} | {isinstance(raw.get('portfolio'), dict)} |")
    combination = report["independent_combination"]
    lines.extend(["", "## Independent Combination", "",
                  f"Success: {combination.get('success') if combination else None}; "
                  f"portfolio: {isinstance(combination.get('portfolio'), dict) if combination else False}",
        "", "## Budget", "", "| Counter | Used | Limit |", "| --- | ---: | ---: |"])
    for name, used in sorted(report["budget"]["counters"].items()):
        lines.append(f"| {name} | {used} | {report['budget']['limits'].get(name, '')} |")
    lines.extend(["", f"Qualification: `{report['qualification']}`; complete T3: `{str(report['complete_t3']).lower()}`.", ""])
    return "\n".join(lines)


def structure_diversity(expressions):
    import zss

    def convert(node):
        if isinstance(node, ast.Constant):
            return None
        label = (node.func.id if isinstance(node, ast.Call) else node.id if isinstance(node, ast.Name)
                 else type(node.op).__name__ if isinstance(node, (ast.BinOp, ast.UnaryOp))
                 else type(node.ops[0]).__name__ if isinstance(node, ast.Compare) else type(node).__name__)
        children = node.args if isinstance(node, ast.Call) else list(ast.iter_child_nodes(node))
        return zss.Node(label, [child for item in children if (child := convert(item)) is not None
                              and not isinstance(item, (ast.Load, ast.operator, ast.unaryop, ast.cmpop))])
    trees = [convert(parse_expression(expression, backend="assay").tree.body) for expression in expressions]
    trees = [tree for tree in trees if tree is not None]
    pairs = [float(zss.simple_distance(a, b, label_dist=lambda a, b: int(a != b)))
             for a, b in itertools.combinations(trees, 2)]
    mean, maximum = (float(np.mean(pairs)), max(pairs)) if pairs else (None, None)
    return {"pairs": len(pairs), "mean": mean, "max": maximum,
            "normalized_mean": mean / maximum if maximum else (0. if pairs else None),
            "normalized_max": 1. if maximum else (0. if pairs else None),
            "normalization": "distance divided by maximum pairwise distance; zero maximum maps to zero",
            "edit_cost": "unit insert/delete/relabel; constants removed",
            "reason": None if pairs else "fewer_than_two_factor_trees"}


def signal_diversity(factors):
    ids = [item["candidate_id"] for item in factors]
    indexed, missing = [], []
    for factor in factors:
        values = {}
        for item in factor["scores"]:
            key = (item["date"], item["instrument"])
            if key in values:
                raise ValueError("duplicate factor score sample")
            values[key] = item["score"]
        indexed.append(values)
        if not values:
            missing.append(factor["candidate_id"])
    pairs = []
    for left, right in itertools.combinations(range(len(ids)), 2):
        a, b = indexed[left], indexed[right]
        shared = sorted(set(a) & set(b))
        keys = [key for key in shared if all(type(value) in (int, float) and math.isfinite(value) for value in (a[key], b[key]))]
        rho, reason = None, None
        if len(keys) < 2:
            reason = "insufficient_finite_samples"
        else:
            values = np.array([(a[key], b[key]) for key in keys])
            if np.any(values.std(axis=0) == 0):
                reason = "constant_series"
            else:
                rho = float(np.clip(np.corrcoef(values.T)[0, 1], -1., 1.))
        pairs.append({"left": ids[left], "right": ids[right], "left_index": left, "right_index": right,
                      "shared_samples": len(shared), "finite_samples": len(keys),
                      "sample_index_digest": digest(keys), "correlation": rho, "reason": reason})
    correlations = [row["correlation"] for row in pairs if row["correlation"] is not None]
    available = bool(correlations) and not missing
    return {"candidate_ids": ids, "candidate_set_digest": digest(sorted(set(ids))), "missing_candidates": missing,
            "pair_count": len(correlations), "total_pairs": len(pairs), "pairs": pairs,
            "diversity": 1-float(np.mean(np.abs(correlations))) if available else None,
            "signed_mean_correlation": float(np.mean(correlations)) if available else None,
            "reason": "missing_factor_scores" if missing else "no_defined_pairs" if not correlations else None,
            "alignment": "intersection of date/instrument; jointly finite; undefined pairs excluded without imputation"}
