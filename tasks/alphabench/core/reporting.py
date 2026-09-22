"""Offline T3 metric projections with explicit domains and undefined values."""

import ast
import itertools
import math
import numpy as np

from .grammar import parse_expression


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
    attempts = [record["attempt_count"] for record in records]
    return {"unit": "model attempts per logical generation step", "step_count": len(records),
            "attempts_per_step": attempts, "total_attempts": sum(attempts),
            "mean": sum(attempts) / len(attempts) if attempts else None,
            "failed_steps": sum(not record["complete"] for record in records)}


def structure_diversity(expressions):
    import zss

    def convert(node):
        if isinstance(node, ast.Constant):
            return None
        label = node.func.id if isinstance(node, ast.Call) else node.id if isinstance(node, ast.Name) else type(node).__name__
        children = node.args if isinstance(node, ast.Call) else list(ast.iter_child_nodes(node))
        return zss.Node(label, [child for item in children if (child := convert(item)) is not None
                              and not isinstance(item, (ast.Load, ast.operator, ast.unaryop, ast.cmpop))])
    trees = [convert(parse_expression(expression, backend="assay").tree.body) for expression in expressions]
    trees = [tree for tree in trees if tree is not None]
    pairs = [(float(zss.simple_distance(a, b)), max(1, len(list(a.iter())) + len(list(b.iter()))))
             for a, b in itertools.combinations(trees, 2)]
    return {"pairs": len(pairs), "mean": float(np.mean([d for d, _ in pairs])) if pairs else None,
            "max": max((d for d, _ in pairs), default=None),
            "normalized_mean": float(np.mean([d/n for d, n in pairs])) if pairs else None,
            "normalized_max": max((d/n for d, n in pairs), default=None)}


def signal_diversity(score_sets):
    correlations = []
    for left, right in itertools.combinations(score_sets, 2):
        a = {(item["date"], item["instrument"]): item["score"] for item in left}
        b = {(item["date"], item["instrument"]): item["score"] for item in right}
        keys = sorted(set(a)&set(b))
        pairs = [(a[key], b[key]) for key in keys if a[key] is not None and b[key] is not None
                 and math.isfinite(a[key]) and math.isfinite(b[key])]
        if len(pairs) < 2: continue
        values = np.array(pairs)
        if np.any(values.std(axis=0) == 0): continue
        correlations.append(float(np.corrcoef(values.T)[0, 1]))
    return {"pair_count": len(correlations), "total_pairs": len(score_sets)*(len(score_sets)-1)//2,
            "diversity": 1-float(np.mean(np.abs(correlations))) if correlations else None,
            "signed_mean_correlation": float(np.mean(correlations)) if correlations else None,
            "alignment": "intersection of date/instrument; finite pairs; constant series excluded"}
