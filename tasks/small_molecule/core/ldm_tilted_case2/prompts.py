"""Prompt builders and compact history summaries for tilted case2."""

from __future__ import annotations

from difflib import SequenceMatcher
import json
from typing import Sequence

from ldm_tts.optimization.acquisition import pareto_front
from tasks.small_molecule.core.ldm_tilted_case2.config import TiltedLDMCase2Config
from tasks.small_molecule.core.ldm_tilted_case2.m1_context import build_m1_molecule_context


SYSTEM_TEXT = (
    "You propose candidate SMILES for an iterative molecular optimization loop. "
    "Return JSON only. Do not estimate docking score, activity, EHVI, uncertainty, "
    "rank, constraint probability, or any other candidate value."
)


def summarize_history(
    history: Sequence[tuple[str, Sequence[float | None]]],
    *,
    minimize: Sequence[bool],
    max_pareto: int = 8,
    max_failures: int = 8,
    max_extreme: int = 5,
    max_avoid: int = 96,
) -> dict:
    finite = [(smiles, tuple(float(x) for x in scores)) for smiles, scores in history if _finite_pair(scores)]
    front_scores = set(pareto_front([scores for _smiles, scores in finite], minimize))
    failures = [{"smiles": smiles, "scores": list(scores)} for smiles, scores in history if not _finite_pair(scores)]
    top_vina = sorted(finite, key=lambda item: item[1][0])[:max_extreme]
    top_activity = sorted(finite, key=lambda item: item[1][1], reverse=True)[:max_extreme]
    balanced = sorted(finite, key=_balanced_rank_key)[:max_extreme]
    summary = {
        "objective": "minimize Vina, maximize neural activity",
        "pareto_front": [
            {"smiles": smiles, "scores": list(scores)}
            for smiles, scores in finite
            if scores in front_scores
        ][:max_pareto],
        "top_low_vina": [{"smiles": smiles, "scores": list(scores)} for smiles, scores in top_vina],
        "top_high_activity": [{"smiles": smiles, "scores": list(scores)} for smiles, scores in top_activity],
        "balanced_elites": [{"smiles": smiles, "scores": list(scores)} for smiles, scores in balanced],
        "recent_selected": [
            {"smiles": smiles, "scores": list(scores) if scores else None}
            for smiles, scores in list(history)[-max_pareto:]
        ],
        "failures": failures[:max_failures],
        "avoid_exact_smiles": [smiles for smiles, _scores in list(history)[-max_avoid:]],
        "n_evaluated": len(history),
    }
    alert = _recent_diversity_alert(finite)
    if alert:
        summary["recent_diversity_alert"] = alert
    return summary


def build_m1_prompt(
    summary: dict,
    cfg: TiltedLDMCase2Config,
    *,
    sample_count: int | None = None,
    strategy: str | None = None,
    feedback: dict | None = None,
) -> tuple[str, str]:
    schema = {"direct_smiles": [{"smiles": "CCO", "rationale": "brief intent"}]}
    requested = cfg.m1_k_direct_llm if sample_count is None else sample_count
    focus = strategy or (
        "Balance local edits of observed useful molecules, recombinations of observed motifs, "
        "and broader alternatives when the supplied history supports uncertainty or diversity."
    )
    if summary.get("recent_diversity_alert"):
        focus += (
            " The history summary contains recent_diversity_alert; treat it as a warning to "
            "avoid simple extensions or near-copies of recent_selected, and choose alternative "
            "parents and edit types from the supplied history."
        )
    return SYSTEM_TEXT, _m1_prompt(requested, focus, summary, schema, cfg, feedback=feedback)


def build_m1_analog_seed_prompt(
    summary: dict,
    cfg: TiltedLDMCase2Config,
    *,
    feedback: dict | None = None,
) -> tuple[str, str]:
    per_seed_budget = max(1, cfg.m1_analog_k_total // max(1, cfg.m1_analog_n_llm_seeds))
    schema = {
        "seeds": [
            {
                "smiles": "CCO",
                "budget": per_seed_budget,
                "intent": "analogue expansion seed",
            }
        ]
    }
    task = (
        f"Choose {cfg.m1_analog_n_llm_seeds} diverse seed molecules for ReaSyn analogue "
        f"expansion. Each seed should use budget={per_seed_budget}. Return the full seed "
        "set unless no valid unused seed can be inferred from the supplied history."
    )
    return SYSTEM_TEXT, _prompt(task, summary, schema, cfg, feedback=feedback)


def _prompt(
    task: str,
    summary: dict,
    schema: dict,
    cfg: TiltedLDMCase2Config,
    *,
    feedback: dict | None = None,
) -> str:
    constraints = (
        "Use JSON only. LLM output contributes only to the proposal/base measure. "
        "Never include scores, ranks, uncertainty, or predicted properties."
    )
    chemistry_rules = (
        "SMILES hygiene for generated molecules: output canonical-looking single-component organic SMILES; "
        "no salts or dot-disconnected mixtures; no reaction arrows; no atom maps; no prose inside smiles; "
        "use valid organic valence patterns; avoid metals, isotopes, radicals, and charged salts; "
        "common organic halogen substituents such as fluorine, chlorine, bromine, and iodine are allowed "
        "when they are syntactically valid and useful for a history-derived proposal; "
        "keep ring/branch syntax balanced; respect smiles_max_len. For M1 direct proposals, avoid letting the "
        "batch collapse to exact duplicates, tiny seed variants, or a monotonic homologous series unless that "
        "choice is directly supported by the observed history. Maintain diversity across local edits, crossover, "
        "and scaffold-hop proposals without naming or hard-coding a preferred structural class. "
        "Fill as much of the requested sample count as possible with valid unique molecules."
    )
    parts = [
        task,
        constraints,
        chemistry_rules,
        f"max_candidates_per_round={cfg.max_candidates_per_round}",
        f"smiles_max_len={cfg.smiles_max_len}",
        "Do not output candidate or seed SMILES longer than smiles_max_len.",
        "History summary:",
        json.dumps(summary, ensure_ascii=False),
        "JSON schema example:",
        json.dumps(schema, ensure_ascii=False),
    ]
    if feedback:
        parts.extend(["Reservoir feedback:", json.dumps(feedback, ensure_ascii=False)])
    return "\n".join(parts)


def _m1_prompt(
    requested: int,
    generation_focus: str,
    summary: dict,
    schema: dict,
    cfg: TiltedLDMCase2Config,
    *,
    feedback: dict | None = None,
) -> str:
    target_context = (
        "The optimization task is for KRAS G12D small-molecule candidates. The activity "
        "objective is based on a target-specific model trained from public KRAS G12D IC50 "
        "records, and the docking objective evaluates binding with AutoDock Vina.\n\n"
        "Public medicinal-chemistry literature has shown that small molecules can engage "
        "the switch-II pocket region. Use this information only as broad target context, "
        "not as a command to copy known drugs, named scaffolds, or exact chemotypes."
    )
    background = (
        "This is an iterative optimization loop for two objectives: lower Vina docking score "
        "and higher neural activity. Your output is used only to build a proposal reservoir/base "
        "measure. External scorers and a Bayesian optimization selector decide which candidates "
        "are evaluated. Therefore, infer useful proposal patterns from the observed history and "
        "molecule context, but do not predict scores or label candidates as good or bad."
    )
    molecule_context = build_m1_molecule_context(summary)
    principles = (
        "- Use the supplied Pareto, low-Vina, high-activity, balanced, and recent history as evidence "
        "for proposing new molecules.\n"
        "- Balance exploitation and exploration: include local edits of observed useful molecules, "
        "recombinations of observed motifs, and broader alternatives when the history supports "
        "uncertainty or diversity.\n"
        "- Explore changes in size class, ring pattern, aromaticity, heteroatom balance, polarity, "
        "flexibility, and target-relevant motif compatibility.\n"
        "- Prefer chemically plausible small organic molecules suitable for docking and QSAR scoring.\n"
        "- Use the provided elite molecules as parents for mutation, crossover, and scaffold-hop proposals.\n"
        "- Do not copy any molecule in avoid_exact_smiles.\n"
        "- Do not let the batch collapse to exact duplicates, trivial near-copies, tiny seed variants, "
        "or a simple monotonic size series unless the observed history directly supports that behavior.\n"
        "- Do not name or hard-code any preferred structural class.\n"
        "- Do not force any single mechanism, scaffold, or structural class.\n"
        "- Initial seed molecules may be very small; do not assume seed size is the target molecular size.\n"
        "- If uncertain about validity, emit fewer valid high-quality candidates rather than many noisy ones."
    )
    smiles_hygiene = (
        f"Output single-component organic SMILES only. Avoid salts, dot-disconnected mixtures, "
        f"reaction arrows, atom maps, prose inside SMILES, metals, isotopes, radicals, and charged salts. "
        f"Use ordinary organic valence patterns, balanced ring/branch syntax, and keep every SMILES "
        f"within smiles_max_len={cfg.smiles_max_len}. Do not over-restrict valid history-derived organic "
        f"substituents solely because they are uncommon."
    )
    parts = [
        "Task:",
        f"Generate up to {requested} valid, unique candidate SMILES. Use compact minified JSON and keep each rationale under 8 words.",
        "Target context:",
        target_context,
        "Background:",
        background,
        "Molecule context table:",
        json.dumps(molecule_context, ensure_ascii=False),
        "How to use the molecule context:",
        (
            "The property labels are qualitative descriptors, not scores. Use them as evidence "
            "for proposing new molecules. Look for combinations of history role and molecular "
            "properties that suggest useful local edits, recombinations, or scaffold-neighbor alternatives."
        ),
        "Generation principles:",
        principles,
        "SMILES hygiene:",
        smiles_hygiene,
        "Generation focus:",
        generation_focus,
    ]
    if feedback:
        parts.extend([
            "Round feedback:",
            "Use this only to avoid invalid, duplicate, overlength, and already-evaluated outputs.",
            json.dumps(feedback, ensure_ascii=False),
        ])
    parts.extend([
        "History summary:",
        json.dumps(summary, ensure_ascii=False),
        "JSON output format:",
        json.dumps(schema, ensure_ascii=False),
    ])
    return "\n".join(parts)


def _finite_pair(scores: Sequence[float | None]) -> bool:
    if len(scores) != 2:
        return False
    return scores[0] is not None and scores[1] is not None


def _balanced_rank_key(item: tuple[str, tuple[float, float]]) -> tuple[float, float]:
    _smiles, scores = item
    vina, activity = scores
    return (float(vina) - float(activity), float(vina))


def _recent_diversity_alert(
    finite_history: Sequence[tuple[str, tuple[float, float]]],
    *,
    max_recent: int = 4,
    min_recent: int = 3,
    similarity_threshold: float = 0.72,
) -> dict | None:
    recent = [smiles for smiles, _scores in list(finite_history)[-max_recent:]]
    if len(recent) < min_recent:
        return None
    pair_scores = [
        _string_similarity(left, right)
        for idx, left in enumerate(recent)
        for right in recent[idx + 1:]
    ]
    has_nested_extension = any(
        len(left) >= 4 and len(right) >= 4 and (left in right or right in left)
        for idx, left in enumerate(recent)
        for right in recent[idx + 1:]
    )
    max_similarity = max(pair_scores, default=0.0)
    if not has_nested_extension and max_similarity < similarity_threshold:
        return None
    return {
        "status": "recent_selected_are_too_similar",
        "max_pairwise_string_similarity": max_similarity,
        "instruction": (
            "Avoid simple extensions or near-copies of recent_selected. "
            "Use other supplied Pareto/extreme/balanced history entries as parents and vary the edit type."
        ),
    }


def _string_similarity(left: str, right: str) -> float:
    return float(SequenceMatcher(None, left, right).ratio())
