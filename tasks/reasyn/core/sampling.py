"""Task-local empirical proposal measure and finite-pool LDM sampling.

Reconstruction uses a canonical query plus its independent projection seed as
the probability identity. TDC uses the canonical projected product. Repeated
unmeasured products contribute occurrences before the shared reservoir dedups.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
import hashlib
import json

import numpy as np

from ldm_tts.contracts import CandidateRejection
from .candidate import ReaSynDomain

Q0_METADATA_KEY = "reasyn_empirical_q0"
EPSILON = 1e-12


def attach_empirical_base_measure(proposals, *, benchmark, mock, observations=()):
    domain = ReaSynDomain(benchmark, mock=mock)
    evaluated = {item.canonical_key for item in observations}
    admitted = []
    for proposal in proposals:
        candidate = domain.admit(proposal)
        if isinstance(candidate, CandidateRejection):
            raise ValueError(f"invalid trusted proposal: {candidate.message}")
        if candidate.canonical_key not in evaluated:
            admitted.append((proposal, candidate.canonical_key))
    counts = Counter(key for _, key in admitted)
    annotations = {}
    for proposal, key in admitted:
        lineage = proposal.metadata.get("harness_lineage")
        if isinstance(lineage, dict):
            annotations.setdefault(key, []).append(dict(lineage))
    total = sum(counts.values())
    space = "canonical_query_and_sampling_seed" if benchmark == "reconstruction" else "canonical_product"
    return tuple(
        replace(proposal, metadata={
            **proposal.metadata,
            **({"research_annotations": annotations[key]} if key in annotations else {}),
            Q0_METADATA_KEY: {
                "identity_space": space,
                "occurrence_count": counts[key],
                "valid_occurrence_count": total,
                "probability": counts[key] / total,
            },
        })
        for proposal, key in admitted
    )


def empirical_base_masses(candidates):
    counts = []
    totals = set()
    for candidate in candidates:
        record = candidate.metadata.get(Q0_METADATA_KEY, {})
        count = record.get("occurrence_count")
        total = record.get("valid_occurrence_count")
        if (isinstance(count, bool) or not isinstance(count, int) or count < 1
                or isinstance(total, bool) or not isinstance(total, int) or total < count):
            raise ValueError("ReaSyn candidates require valid empirical q0 occurrence records")
        probability = record.get("probability")
        if probability is None or not np.isclose(float(probability), count / total):
            raise ValueError("ReaSyn empirical q0 does not match occurrence counts")
        counts.append(count)
        totals.add(total)
    if len(totals) > 1:
        raise ValueError("ReaSyn empirical q0 totals are inconsistent")
    if counts and sum(counts) > next(iter(totals)):
        raise ValueError("ReaSyn q0 contains duplicated candidate identities")
    values = np.asarray(counts, dtype=float)
    return values / values.sum() if len(values) else values


def robust_z(values, *, clip=5.0):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("acquisition values must be a finite vector")
    if not np.isfinite(clip) or clip <= 0:
        raise ValueError("robust-z clip must be finite and positive")
    if not len(values):
        return values
    center = float(np.median(values))
    scale = 1.4826 * float(np.median(np.abs(values - center)))
    if scale <= EPSILON:
        scale = float(np.std(values))
    if scale <= EPSILON:
        return np.zeros_like(values)
    return np.clip((values - center) / (scale + EPSILON), -clip, clip)


def tilted_distribution(q0, acquisition, *, alpha, eta, z_clip=5.0):
    if any(not np.isfinite(weight) or weight < 0 for weight in (alpha, eta)):
        raise ValueError("LDM alpha and eta must be finite and nonnegative")
    q0 = np.asarray(q0, dtype=float)
    if q0.ndim != 1 or not np.isfinite(q0).all() or np.any(q0 <= 0):
        raise ValueError("empirical q0 must contain finite positive masses")
    q0 = q0 / q0.sum()
    normalized = robust_z(acquisition, clip=z_clip)
    if q0.shape != normalized.shape:
        raise ValueError("q0 and acquisition vectors must align")
    logits = alpha * np.log(q0 + EPSILON) + eta * normalized
    probabilities = np.exp(logits - np.max(logits))
    return probabilities / probabilities.sum(), logits, normalized


def candidate_set_seed(seed, round_idx, candidates, *, phase):
    payload = json.dumps({
        "seed": seed, "round_idx": round_idx, "phase": phase,
        "candidate_ids": sorted(candidate.candidate_id for candidate in candidates),
    }, sort_keys=True).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def gumbel_top_k(probabilities, count, *, seed):
    probabilities = np.asarray(probabilities, dtype=float)
    if (probabilities.ndim != 1 or not np.isfinite(probabilities).all()
            or np.any(probabilities <= 0)):
        raise ValueError("sampling requires finite positive probabilities")
    rng = np.random.default_rng(seed)
    values = np.log(probabilities) + rng.gumbel(size=len(probabilities))
    return tuple(int(i) for i in np.argsort(values)[::-1][:min(count, len(values))])
