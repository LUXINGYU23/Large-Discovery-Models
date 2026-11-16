"""Shared GP acquisition with one deterministic reservation per branch."""

import math
from ldm_tts.contracts import Candidate
from ldm_tts.optimization.records import BOObservation
from ..core.surrogate import LossSelector


def reserve_and_rank(predictions, count, reserved_per_branch=1):
    """Reserve each branch's best proposals, then select highest remaining UCBs."""
    if not predictions or type(count) is not int or not 1 <= count <= len(predictions):
        raise ValueError("invalid proposal pool or batch size")
    if type(reserved_per_branch) is not int or reserved_per_branch < 0:
        raise ValueError("invalid reservation per branch")
    ids = [p["program_id"] for p in predictions]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate candidate in acquisition pool")
    if any(type(p["branch"]) is not int or p["branch"] < 0
           or type(p["acquisition"]) not in (int, float) or not math.isfinite(p["acquisition"])
           for p in predictions):
        raise ValueError("invalid acquisition prediction")
    ranked = sorted(predictions, key=lambda p: (-p["acquisition"], p["program_id"]))
    reserved = [p for b in sorted({p["branch"] for p in ranked})
                for p in [q for q in ranked if q["branch"] == b][:reserved_per_branch]]
    if count < len(reserved):
        raise ValueError("batch cannot hold the per-branch reservations")
    reserved_ids = {p["program_id"] for p in reserved}
    chosen = [(p, "reserved") for p in reserved]
    chosen += [(p, "acquisition") for p in ranked if p["program_id"] not in reserved_ids][:count-len(chosen)]
    return [{**p, "selection": reason} for p, reason in chosen]


def select(pool, history, encoder, count, reserved_per_branch=1):
    """Fit the existing loss GP on all charged sources and score this pool."""
    if not history:
        raise ValueError("the initial solver must be measured before GP selection")
    def candidate(sha):
        return Candidate(sha, {"sha256": sha, "source_path": f"candidates/{sha}/solver.py"},
                         sha, "rsi")
    observations = []
    for row in history:
        vector = encoder.encode(candidate(row["program_id"]))
        observations.append(BOObservation.scalar(row["program_id"], row["risk"], vector.values,
                                                  feature_version=vector.version))
    candidates = [candidate(p["program_id"]) for p in pool]
    vectors = {c.candidate_id: encoder.encode(c) for c in candidates}
    selector = LossSelector("fusion")
    selector.fit(observations)
    result = selector.select(candidates, vectors, count=len(candidates))
    owners = {p["program_id"]: p["branch"] for p in pool}
    predictions = [{"program_id": p.candidate_id, "branch": owners[p.candidate_id],
                    "mean": p.scalar_mean, "std": p.scalar_std,
                    "acquisition": p.acquisition_score} for p in result.predictions]
    return {"predictions": predictions, "selected": reserve_and_rank(predictions, count, reserved_per_branch),
            "surrogate": result.metadata}
