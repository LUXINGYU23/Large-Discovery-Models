"""core/ldm/orchestrator/prompts.py — system + user prompt templates."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from tasks.antibody.core.ldm.config import DSLConfig
from tasks.antibody.core.ldm.orchestrator.status import OrchestratorStatus

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


def _load(name: str) -> str:
    """Load a prompt template from disk."""
    path = PROMPTS_DIR / name
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def render_history_csv(
    history: Iterable[tuple],
    max_rows: int | None = None,
) -> str:
    """Render history as a CSV string: ``seq,score,iter``."""
    rows = ["seq,score,iter"]
    items = list(history)
    if max_rows is not None and len(items) > max_rows:
        # Keep first (max_rows // 2) and last (max_rows // 2)
        half = max_rows // 2
        items = items[:half] + items[-(max_rows - half):]
    for seq, score, it in items:
        if isinstance(seq, (list, tuple)):
            seq = "".join("ACDEFGHIKLMNPQRSTVWY"[int(i)] for i in seq)
        rows.append(f"{seq},{float(score):.4f},{int(it)}")
    return "\n".join(rows)


def build_system_prompt(config: DSLConfig) -> str:
    """Load and return the system prompt template, injecting config values."""
    from tasks.antibody.core.ldm.prompts.strategies import STRATEGIES

    template = _load("system.txt")
    if not template:
        return _DEFAULT_SYSTEM_PROMPT
    # Compute steps for "Or of 2 LocalSearch(restart=2, steps=???)" example
    # that fits within budget: restart=2, steps=S → budget = 2*(S+1) <= budget_total
    # so S <= budget_total/2 - 1
    half_budget_steps = max(1, config.acq_search_budget // 2 - 1)
    strategy_text = STRATEGIES.get(config.strategy, STRATEGIES["ldm-default"])
    return (
        template
        .replace("{strategy_section}", strategy_text)
        .replace("{acq_n_candidates}", str(config.acq_n_candidates))
        .replace("{init_pool_size}", str(config.init_pool_size))
        .replace("{acq_search_budget}", str(config.acq_search_budget))
        .replace("{num_llm_review}", str(config.num_llm_review))
        .replace("{acq_max_rounds}", str(config.acq_max_rounds))
        .replace("{acq_div2_2}", str(half_budget_steps))
        .replace("{batch_size}", str(config.batch_size))
    )


def build_user_prompt(
    status: OrchestratorStatus,
    config: DSLConfig,
    last_rejection_reason: str | None = None,
) -> str:
    """Build the user prompt from a status snapshot."""
    history_csv = render_history_csv(
        status.full_history, max_rows=config.history_max_in_prompt
    )
    tr_source = _safe_source(status.current_search_dsl) or "(AntBO default — no DSL)"
    bias_source = _safe_source(status.current_bias_dsl) or "(zero bias)"
    size_est = status.current_search_size_estimate
    tr_size_str = f"{size_est:.3e}" if size_est is not None else "(unknown)"
    feedback = last_rejection_reason or "(none — first attempt this iteration)"
    ag_ctx = status.antigen_context or {}
    ag_str = str(ag_ctx)[:800]

    return (
        f"# Status @ iter {status.iteration}\n"
        f"\n"
        f"Antigen: {status.antigen_id}\n"
        f"Seed: {status.antigen_seed}\n"
        f"Iteration: {status.iteration}\n"
        f"\n"
        f"## Antigen context\n"
        f"{ag_str}\n"
        f"\n"
        f"## Full observation history (CSV)\n"
        f"seq,score,iter\n"
        f"{history_csv}\n"
        f"\n"
        f"## BO state\n"
        f"- best_value: {status.best_value:.4f}\n"
        f"- best_sequence: {status.best_sequence}\n"
        f"- n_evals: {status.n_evals}\n"
        f"- n_iters_without_improvement: {status.n_iters_without_improvement}\n"
        f"\n"
        f"## Current trust region\n"
        f"- source: {tr_source}\n"
        f"- evals_in_current_tr: {status.n_evals_in_current_tr}\n"
        f"- estimated |TR|: {tr_size_str}\n"
        f"\n"
        f"## Current bias\n"
        f"- source: {bias_source}\n"
        f"\n"
        f"## Previous attempt feedback (if any)\n"
        f"{feedback}\n"
        f"\n"
        f"Decide what to update. Output ONLY valid JSON.\n"
    )


def _safe_source(atom) -> str | None:
    """Render an atom to its source representation via __repr__."""
    if atom is None:
        return None
    return repr(atom)


# Fallback in case system.txt is missing
_DEFAULT_SYSTEM_PROMPT = """\
You control the trust region and acquisition bias for BO of antibody CDRH3.

Amino acid alphabet (one-letter code):
  A=Ala C=Cys D=Asp E=Glu F=Phe G=Gly H=His I=Ile K=Lys
  L=Leu M=Met N=Asn P=Pro Q=Gln R=Arg S=Ser T=Thr V=Val W=Trp Y=Tyr

[2] OUTPUT FORMAT
  Initialization: {"update_trust_region": "...", "update_bias": "..."}
  BO search plan: {"rationale": "...",
                   "update_trust_region": "... (OPTIONAL)",
                   "update_bias": "... (OPTIONAL)"}
  BO review: {"action": "take", "ids": [0]} or {"action": "search", "update_trust_region": "..."}

  Both update_trust_region and update_bias are OPTIONAL in the BO loop.
  Omit a key to keep the current value. Include it only to change it.

[3] ATOMS
  LocalSearch(center, radius=3, restart=2, steps=100)
  NeighborSampling(center, mut_pr=0.5, budget=1000)
  LatinHyperCubeSampling(num=1000)
  Or(Atom, Atom)

[4] BIAS ATOMS (combine with +)
  - MaxCysteine(max): penalise if C count > max (-1.0 per excess). Prevents disulfide scrambling.
  - MaxHydrophobicRun(max): penalise longest run of {A,I,L,M,F,W,V,Y} > max (-0.5 per excess). Prevents polyreactivity.
  - MaxAromatic(max): penalise F+W+Y > max (-0.25 per excess); bonus +0.15 each for 1-2 aromatics.
  - NetChargeRange(min, max): charge=(R+K+0.1*H)-(D+E). Penalise outside range (-0.5 per unit).
  - NoNGlycosylation(): penalise N-X-S/T motif (-1.0 flat). Prevents unwanted glycosylation.
  Default: MaxCysteine(1) + MaxHydrophobicRun(4) + MaxAromatic(2) + NetChargeRange(-1, 2) + NoNGlycosylation()

[5] Note on stochasticity:
  BO evaluates a batch of candidates per iteration. Occasional non-improving
  samples are normal. If the current trust region is producing reasonable
  candidates, omit update_trust_region to keep it. Drastic shifts should be
  considered sparingly. n_evals_in_current_tr shows how many evaluations have
  been done under the current trust region.
"""