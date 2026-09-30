"""Direct sequence proposal reservoir for the antibody paper variants."""
from __future__ import annotations

import json
import random
from typing import Any, Iterable

import numpy as np

from tasks.antibody.core.ldm_light.ldm_acq import (
    AA,
    AROMATIC,
    N_GLYCO,
    longest_hydrophobic_run,
    net_charge,
    passes_developability,
    random_candidates,
    seqs_to_indices,
    valid_seq,
)


def _history_payload(rows: list[dict[str, Any]], top_k: int) -> dict[str, Any]:
    best = sorted(rows, key=lambda row: float(row["LastValue"]))[:top_k]
    return {
        "num_observed": len(rows),
        "best": [
            {"sequence": row["LastProtein"], "score": float(row["LastValue"])}
            for row in best
        ],
        "recent": [
            {
                "iteration": int(row["Index"]),
                "sequence": row["LastProtein"],
                "score": float(row["LastValue"]),
                "best_so_far": float(row["BestValue"]),
            }
            for row in rows[-top_k:]
        ],
    }


def build_direct_prompt(
    *,
    antigen: str,
    seq_len: int,
    num_sequences: int,
    observed: Iterable[str],
    rows: list[dict[str, Any]],
    antigen_context: dict[str, Any] | None,
    history_top_k: int,
) -> str:
    """Build the direct token-level sequence generation contract."""
    payload = {
        "task": "direct_cdrh3_generation",
        "objective": "Minimize Absolut binding energy; lower is better.",
        "antigen": antigen,
        "constraints": {
            "length": int(seq_len),
            "alphabet": AA,
            "num_sequences": int(num_sequences),
            "do_not_repeat": sorted(str(seq) for seq in observed)[-200:],
            "developability": {
                "max_cysteine": 1,
                "max_hydrophobic_run": 4,
                "max_aromatic_FWY": 2,
                "net_charge_range": [-1.0, 2.0],
                "forbid_n_glycosylation_NXS_or_NXT": True,
            },
        },
        "history": _history_payload(rows, int(history_top_k)),
        "antigen_context": antigen_context or {},
        "required_output": ["G" * int(seq_len)],
        "output_rules": [
            "Return one top-level JSON list of amino-acid strings only.",
            "Generate sequences directly; do not output DSL, code, scores, or explanations.",
        ],
    }
    return json.dumps(payload, indent=2)


def _load_json_list(raw: str) -> list[Any]:
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("empty LLM response")
    decoder = json.JSONDecoder()
    for index, char in enumerate(raw):
        if char != "[":
            continue
        try:
            value, _ = decoder.raw_decode(raw[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, list):
            return value
    raise ValueError("LLM response must contain a top-level JSON list")


def parse_direct_sequences(
    raw: str,
    *,
    seq_len: int,
    observed: Iterable[str],
    max_sequences: int | None = None,
) -> list[str]:
    """Parse, validate, deduplicate, and cap direct CDRH3 proposals."""
    sequences, _ = _parse_direct_sequences_with_rejections(
        raw,
        seq_len=seq_len,
        observed=observed,
        max_sequences=max_sequences,
    )
    return sequences


def _parse_direct_sequences_with_rejections(
    raw: str,
    *,
    seq_len: int,
    observed: Iterable[str],
    max_sequences: int | None = None,
) -> tuple[list[str], list[dict[str, Any]]]:
    observed_set = {str(seq).strip().upper() for seq in observed}
    used: set[str] = set()
    sequences: list[str] = []
    rejections: list[dict[str, Any]] = []
    for item_index, item in enumerate(_load_json_list(raw)):
        if not isinstance(item, str):
            rejections.append({
                "item_index": item_index,
                "value": item,
                "reasons": ["not_string"],
            })
            continue
        sequence = item.strip().upper()
        reasons: list[str] = []
        if len(sequence) != int(seq_len):
            reasons.append("length")
        if any(aa not in AA for aa in sequence):
            reasons.append("alphabet")
        if valid_seq(sequence, int(seq_len)):
            if sequence.count("C") > 1:
                reasons.append("max_cysteine")
            if longest_hydrophobic_run(sequence) > 4:
                reasons.append("max_hydrophobic_run")
            if sum(1 for aa in sequence if aa in AROMATIC) > 2:
                reasons.append("max_aromatic_FWY")
            if not -1.0 <= net_charge(sequence) <= 2.0:
                reasons.append("net_charge_range")
            if N_GLYCO.search(sequence) is not None:
                reasons.append("n_glycosylation_NXS_or_NXT")
        if sequence in observed_set:
            reasons.append("already_observed")
        if sequence in used:
            reasons.append("duplicate_response")
        if not reasons and passes_developability(sequence):
            sequences.append(sequence)
            used.add(sequence)
            if max_sequences is not None and len(sequences) >= int(max_sequences):
                break
        else:
            rejections.append({
                "item_index": item_index,
                "sequence": sequence,
                "reasons": reasons or ["developability"],
            })
    return sequences, rejections


def _fallback_sequences(
    rng: random.Random,
    *,
    n: int,
    seq_len: int,
    observed: set[str],
) -> list[str]:
    return [
        item["sequence"]
        for item in random_candidates(rng, int(n), int(seq_len), observed)
    ]


def propose_direct_batch(
    *,
    llm: Any,
    rng: random.Random,
    antigen: str,
    seq_len: int,
    n: int,
    observed: set[str],
    rows: list[dict[str, Any]],
    antigen_context: dict[str, Any] | None,
    args: Any,
    independent: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Generate n direct candidates, optionally as independent completions."""
    errors: list[dict[str, Any]] = []
    accumulated: list[str] = []
    prompt = ""
    raw_outputs: list[str] = []

    for attempt in range(1, max(1, int(args.max_retries)) + 1):
        needed = int(n) - len(accumulated)
        prompt = build_direct_prompt(
            antigen=antigen,
            seq_len=seq_len,
            num_sequences=1 if independent else needed,
            observed=observed.union(accumulated),
            rows=rows,
            antigen_context=antigen_context,
            history_top_k=int(args.history_top_k),
        )
        if independent:
            if str(getattr(args, "planner_mode", "choices")) == "choices":
                outputs = llm.call_many(
                    prompt,
                    temperature=float(args.temperature),
                    timeout_s=int(args.timeout_s),
                    n=needed,
                )
            else:
                outputs = [
                    llm.call(
                        prompt,
                        temperature=float(args.temperature),
                        timeout_s=int(args.timeout_s),
                    )
                    for _ in range(needed)
                ]
        else:
            outputs = [
                llm.call(
                    prompt,
                    temperature=float(args.temperature),
                    timeout_s=int(args.timeout_s),
                )
            ]
        raw_outputs.extend(outputs)
        for output_index, raw in enumerate(outputs):
            try:
                parsed, rejections = _parse_direct_sequences_with_rejections(
                    raw,
                    seq_len=seq_len,
                    observed=observed.union(accumulated),
                    max_sequences=1 if independent else needed,
                )
                accumulated.extend(parsed)
                if not parsed:
                    errors.append({
                        "attempt": attempt,
                        "output_index": output_index,
                        "error": "no candidates passed admission",
                        "rejections": rejections,
                        "raw_response": raw,
                    })
            except Exception as exc:
                errors.append({
                    "attempt": attempt,
                    "output_index": output_index,
                    "error": str(exc),
                    "raw_response": raw,
                })
        if len(accumulated) >= int(n):
            break

    source = "llm_direct"
    if len(accumulated) < int(n):
        if not bool(args.fallback_random):
            raise RuntimeError(json.dumps(errors, indent=2))
        accumulated.extend(_fallback_sequences(
            rng,
            n=int(n) - len(accumulated),
            seq_len=seq_len,
            observed=observed.union(accumulated),
        ))
        source = "llm_direct_fallback_random"

    candidates = [{"sequence": sequence, "score": None} for sequence in accumulated[: int(n)]]
    return candidates, {
        "source": source,
        "generation_mode": (
            (
                "independent_choices"
                if str(getattr(args, "planner_mode", "choices")) == "choices"
                else "independent_calls"
            )
            if independent
            else "single_batch"
        ),
        "n_requested": int(n),
        "n_valid": len(candidates),
        "prompt": prompt,
        "raw_outputs": raw_outputs,
        "errors": errors,
    }


def score_direct_candidates(
    candidates: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    *,
    args: Any,
) -> list[dict[str, Any]]:
    """Fit the current GP and attach posterior/acquisition values."""
    import torch

    from tasks.antibody.core.ldm_light.ldm_acq import fit_gp_and_make_acquisition

    gp, acquisition = fit_gp_and_make_acquisition(
        rows,
        acq_name=str(args.acq),
        beta=float(args.acq_beta),
        xi=float(args.acq_xi),
        device=getattr(args, "device", "cpu") or "cpu",
    )
    device = torch.device(getattr(args, "device", "cpu") or "cpu")
    encoded = torch.tensor(
        seqs_to_indices([candidate["sequence"] for candidate in candidates]),
        dtype=torch.float32,
        device=device,
    )
    with torch.no_grad():
        posterior = gp.likelihood(gp(encoded))
        scores = acquisition(encoded).detach().cpu().numpy().reshape(-1)
        means = posterior.mean.detach().cpu().numpy().reshape(-1)
        stddevs = posterior.stddev.detach().cpu().numpy().reshape(-1)

    scored: list[dict[str, Any]] = []
    for candidate, score, mean, stddev in zip(candidates, scores, means, stddevs):
        item = dict(candidate)
        item.update({
            "acquisition_score": float(score),
            "acquisition_raw": float(score),
            "mu": float(mean),
            "sigma": float(stddev),
            "source": "direct_generation",
        })
        scored.append(item)
    return scored
