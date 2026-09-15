"""Released ReaSyn evaluation formulas, with explicit full requested denominators."""

from __future__ import annotations
import math
import re
from .chemistry import canonicalize, similarity

ORACLES = (
    "amlodipine_mpo",
    "celecoxib_rediscovery",
    "drd2",
    "fexofenadine_mpo",
    "gsk3b",
    "jnk3",
    "median1",
    "median2",
    "osimertinib_mpo",
    "perindopril_mpo",
    "ranolazine_mpo",
    "sitagliptin_mpo",
    "zaleplon_mpo",
)
WIDTH_ONE = frozenset(
    ("drd2", "gsk3b", "perindopril_mpo", "sitagliptin_mpo", "zaleplon_mpo")
)


def top_mean(scores, n=10):
    return (
        sum(sorted(scores, reverse=True)[:n]) / min(n, len(scores)) if scores else 0.0
    )


def top_auc(scores, *, top_n=10, max_calls=10000, frequency=100, finish=False):
    """Port of scripts/optimize_tdc.py:53; scores MUST be unique-call order.

    Coarse trapezoids start at zero; partial runs are never padded unless the
    caller explicitly declares finish. Reject over-budget inputs, fixing the
    upstream Oracle.score_smi > / >= boundary without changing legal-run AUC.
    """
    if max_calls < 1 or frequency < 1 or top_n < 1 or len(scores) > max_calls:
        raise ValueError("invalid AUC limits or over-budget trajectory")
    if any(not math.isfinite(s) for s in scores):
        raise ValueError("nonfinite oracle score")
    if not scores:
        return 0.0
    area = prev = 0.0
    called = 0
    for idx in range(frequency, min(len(scores), max_calls), frequency):
        value = top_mean(scores[:idx], top_n)
        area += frequency * (value + prev) / 2
        prev, called = value, idx
    value = top_mean(scores, top_n)
    area += (len(scores) - called) * (value + prev) / 2
    if finish and len(scores) < max_calls:
        area += (max_calls - len(scores)) * value
    return area / max_calls


def reconstruction_metrics(targets, rows, *, mock=False, diversity=None):
    """Every requested target contributes; missing outputs contribute zero.

    Published exact reconstruction ignores stereochemistry. Similarity is
    recomputed to original target, never the possibly altered projection query.
    Diversity follows released eval_recon.py, including its product multiplicity.
    """
    targets = [canonicalize(t, mock=mock, stereo=False) for t in targets]
    if not targets or len(set(targets)) != len(targets):
        raise ValueError("targets must be nonempty and unique after canonicalization")
    groups = {t: [] for t in targets}
    for row in rows:
        target = canonicalize(row["target"], mock=mock, stereo=False)
        if target not in groups:
            raise ValueError("output target outside the requested manifest")
        smi = canonicalize(row["smiles"], mock=mock, stereo=False)
        groups[target].append(
            {**row, "smiles": smi, "score": similarity(target, smi, mock=mock)}
        )
    if diversity is None and not mock:
        from tdc import Evaluator

        diversity = Evaluator("diversity")
    if diversity is None:
        diversity = lambda smiles: 0.0  # Fixture values are explicitly non-scientific.
    product_div = bb_div = total_sim = exact = success = 0.0
    per_target = []
    for target, outputs in groups.items():
        best = max((r["score"] for r in outputs), default=0.0)
        hit = any(r["smiles"] == target for r in outputs)
        total_sim += best
        exact += hit
        success += bool(outputs)
        products, bbs = [], set()
        for row in outputs:
            if row["score"] >= 0.8:
                products.append(row["smiles"])
                for token in row.get("synthesis", "").split(";"):
                    if token and not re.match(r"R\d+", token):
                        bbs.add(canonicalize(token, mock=mock, stereo=False))

        def measured(items):
            if len(items) <= 1:
                return 0.0
            value = float(diversity(list(items)))
            return value if math.isfinite(value) else 0.0

        product_div += measured(products)
        bb_div += measured(bbs)
        per_target.append(
            {
                "target": target,
                "output_count": len(outputs),
                "similarity": best,
                "reconstructed": hit,
            }
        )
    n = len(targets)
    return {
        "target_count": n,
        "success_rate": success / n,
        "reconstruction_rate": exact / n,
        "mean_similarity": total_sim / n,
        "product_diversity": product_div / n,
        "building_block_diversity": bb_div / n,
        "per_target": per_target,
    }
