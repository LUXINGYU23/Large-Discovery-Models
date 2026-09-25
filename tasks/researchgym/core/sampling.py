"""Empirical proposal measure q0 and the finite-pool LDM sampling distribution.

Occurrences are counted before the shared reservoir deduplicates canonical
programs, so agreement between independent requests or sessions raises q0.
Only authoritative measured candidates are excluded.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import replace

import numpy as np

from ldm_tts.contracts import CandidateRejection

Q0_KEY = "researchgym_empirical_q0"
EPSILON = 1e-12


def attach_empirical_base_measure(proposals, domain, *, measured_keys):
    admitted = []
    for proposal in proposals:
        candidate = domain.admit(proposal)
        if isinstance(candidate, CandidateRejection):
            raise ValueError(f"validated proposal failed admission: {candidate.message}")
        if candidate.canonical_key in measured_keys:
            raise ValueError("validated proposal repeats a measured candidate")
        admitted.append((proposal, candidate.canonical_key))
    counts = Counter(key for _, key in admitted)
    notes = {}
    for proposal, key in admitted:
        note = proposal.metadata.get("research_note")
        if note:
            notes.setdefault(key, []).append(note)
    total = len(admitted)
    return tuple(
        replace(proposal, metadata={
            **{k: v for k, v in proposal.metadata.items() if k != "research_note"},
            "research_annotations": notes.get(key, []),
            Q0_KEY: {"occurrence_count": counts[key], "valid_occurrence_count": total,
                     "probability": counts[key] / total},
        })
        for proposal, key in admitted
    )


def empirical_base_masses(candidates):
    counts, totals = [], set()
    if len({c.canonical_key for c in candidates}) != len(candidates):
        raise ValueError("q0 requires canonical candidates")
    for candidate in candidates:
        record = candidate.metadata.get(Q0_KEY) or {}
        count, total = record.get("occurrence_count"), record.get("valid_occurrence_count")
        if any(isinstance(v, bool) or not isinstance(v, int) for v in (count, total)) or not 1 <= count <= total:
            raise ValueError("candidate lacks a valid empirical q0 occurrence record")
        counts.append(count)
        totals.add(total)
    if len(totals) > 1 or (counts and sum(counts) > next(iter(totals))):
        raise ValueError("inconsistent empirical q0 occurrence totals")
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
    """p_i proportional to q0_i^alpha * exp(eta * robust_z(acquisition)_i)."""
    if any(not np.isfinite(weight) or weight < 0 for weight in (alpha, eta)):
        raise ValueError("LDM alpha and eta must be finite and nonnegative")
    q0 = np.asarray(q0, dtype=float)
    if q0.ndim != 1 or not np.isfinite(q0).all() or np.any(q0 <= 0):
        raise ValueError("empirical q0 must contain finite positive masses")
    q0 = q0 / q0.sum()
    normalized = robust_z(acquisition, clip=z_clip)
    if normalized.shape != q0.shape:
        raise ValueError("q0 and acquisition vectors must align")
    logits = alpha * np.log(q0 + EPSILON) + eta * normalized
    probabilities = np.exp(logits - logits.max())
    return probabilities / probabilities.sum(), logits, normalized


def candidate_set_seed(seed, round_idx, candidates, *, phase):
    payload = json.dumps({"seed": seed, "round_idx": round_idx, "phase": phase,
                          "candidate_ids": sorted(c.candidate_id for c in candidates)}, sort_keys=True)
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big")


def gumbel_top_k(probabilities, count, *, seed):
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 1 or not np.isfinite(probabilities).all() or np.any(probabilities <= 0):
        raise ValueError("sampling requires finite positive probabilities")
    keys = np.log(probabilities) + np.random.default_rng(seed).gumbel(size=len(probabilities))
    return tuple(int(i) for i in np.argsort(keys)[::-1][:min(count, len(keys))])
