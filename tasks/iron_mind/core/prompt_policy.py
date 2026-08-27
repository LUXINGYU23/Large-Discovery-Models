"""Shared policy data for Iron Mind reaction proposal prompts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tasks.iron_mind.core.schema import ReactionValue


BASELINE_PROMPT_POLICY = "baseline_v1"
DIRECT_PROMPT_POLICY = "direct_v1"
PORTFOLIO_PROMPT_POLICY = "portfolio_v1"
DEFAULT_PROMPT_POLICY = PORTFOLIO_PROMPT_POLICY
PROMPT_POLICIES = frozenset({BASELINE_PROMPT_POLICY, DIRECT_PROMPT_POLICY, PORTFOLIO_PROMPT_POLICY})

SYSTEM_PROMPT = (
    "You are the proposal component of a closed-loop reaction-condition optimizer. "
    "Use chemical knowledge and supplied experimental evidence to form a private hypothesis. "
    "A separate GP-UCB selector ranks the candidate reservoir and a frozen oracle evaluates it. "
    "Do not predict scores, rank candidates, or explain your reasoning. Return JSON only."
)

DIRECT_SYSTEM_PROMPT = (
    "You are the direct proposal component of a closed-loop reaction-condition search. "
    "Your one source-valid condition will be evaluated immediately without a GP selector. "
    "Choose an unevaluated condition and return JSON only."
)

INITIAL_ROLE_INSTRUCTIONS = (
    (
        "chemical_prior",
        "Use chemical plausibility to choose the unfixed conditions; do not assume an outcome.",
    ),
    (
        "coverage_prior",
        "Use the unfixed conditions to cover a complementary operational regime.",
    ),
    (
        "operational_contrast",
        "Use the unfixed conditions to create a chemically plausible contrast to common defaults.",
    ),
    (
        "interaction_prior",
        "Use the unfixed conditions to test a plausible interaction between condition families.",
    ),
)

EVIDENCE_ROLE_INSTRUCTIONS = (
    (
        "evidence_exploitation",
        "Use high-scoring observations to choose the unfixed conditions while preserving novelty.",
    ),
    (
        "counterfactual_probe",
        "Use the unfixed conditions to distinguish competing explanations suggested by history.",
    ),
    (
        "underexplored_coverage",
        "Prefer underexplored unfixed options when they remain chemically plausible.",
    ),
    (
        "mechanistic_divergence",
        "Use the unfixed conditions to explore a plausible alternative to the incumbent regime.",
    ),
)


@dataclass(frozen=True)
class ProposalSlotPlan:
    """One deterministic portfolio allocation for an independent request."""

    policy: str
    role: str
    role_instruction: str
    focus: tuple[tuple[str, ReactionValue], ...] = ()
    focus_capacity: int = 1
    focus_position: int = 0

    def focus_payload(self) -> dict[str, ReactionValue]:
        return dict(self.focus)

    def metadata(self) -> dict[str, Any]:
        return {
            "prompt_policy": self.policy,
            "prompt_version": self.policy,
            "proposal_role": self.role,
            "slot_focus": self.focus_payload(),
            "slot_focus_capacity": self.focus_capacity,
            "slot_focus_position": self.focus_position,
        }
