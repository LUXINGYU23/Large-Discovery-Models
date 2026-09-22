"""Offline T3 metric projections with explicit domains and undefined values."""

import ast
import itertools
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
    attempts = [record["attempt_count"] for record in records]
    return {"unit": "model attempts per logical generation step", "step_count": len(records),
            "attempts_per_step": attempts, "total_attempts": sum(attempts),
            "mean": sum(attempts) / len(attempts) if attempts else None,
            "failed_steps": sum(not record["complete"] and not record.get("interrupted", False) for record in records),
            "interrupted_steps": sum(record.get("interrupted", False) for record in records)}


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
    if len(ids) != len(set(ids)):
        raise ValueError("signal diversity requires distinct candidate identities")
    indexed, missing = {}, []
    for factor in factors:
        values = {}
        for item in factor["scores"]:
            key = (item["date"], item["instrument"])
            if key in values:
                raise ValueError("duplicate factor score sample")
            values[key] = item["score"]
        indexed[factor["candidate_id"]] = values
        if not values:
            missing.append(factor["candidate_id"])
    pairs = []
    for left, right in itertools.combinations(ids, 2):
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
        pairs.append({"left": left, "right": right, "shared_samples": len(shared), "finite_samples": len(keys),
                      "sample_index_digest": digest(keys), "correlation": rho, "reason": reason})
    correlations = [row["correlation"] for row in pairs if row["correlation"] is not None]
    available = bool(correlations) and not missing
    return {"candidate_ids": ids, "candidate_set_digest": digest(sorted(ids)), "missing_candidates": missing,
            "pair_count": len(correlations), "total_pairs": len(pairs), "pairs": pairs,
            "diversity": 1-float(np.mean(np.abs(correlations))) if available else None,
            "signed_mean_correlation": float(np.mean(correlations)) if available else None,
            "reason": "missing_factor_scores" if missing else "no_defined_pairs" if not correlations else None,
            "alignment": "intersection of date/instrument; jointly finite; undefined pairs excluded without imputation"}
