"""Deterministic LDMEngine smoke adapter for ``iron_mind``."""

from __future__ import annotations

from pathlib import Path

from ldm_tts.engine.run_store import CampaignRuntime
from ldm_tts.contracts import (
    CallableCandidateEvaluator,
    Candidate,
    CandidateRejection,
    LDMTaskSpec,
    RawProposal,
)
from ldm_tts.engine import LDMEngine, LDMEngineConfig, LDMEngineResult
from ldm_tts.engine.expansion import (
    CallableReservoirExpander,
    ExpansionRequest,
    ExpansionResult,
)


class DraftCandidateDomain:
    """Replace with task-owned canonicalization and scientific validation."""

    def admit(self, proposal: RawProposal) -> Candidate | CandidateRejection:
        try:
            value = int(proposal.payload)
        except (TypeError, ValueError):
            return CandidateRejection("invalid", "mock integer required", proposal.source)
        return Candidate(
            candidate_id=f"draft-{value}",
            payload=value,
            canonical_key=str(value),
            source=proposal.source,
        )


def _expand(request: ExpansionRequest) -> ExpansionResult:
    return ExpansionResult(
        proposals=(RawProposal(request.round_idx, "deterministic_mock"),),
    )


def run_mock_campaign(
    task_spec: LDMTaskSpec,
    *,
    iterations: int,
    run_dir: Path,
) -> LDMEngineResult:
    runtime = CampaignRuntime.open(
        run_dir,
        task="iron_mind",
        config={"mode": "mock", "iterations": iterations},
        task_spec=task_spec,
        budget_limits={"external_evaluations": iterations},
    )
    engine = LDMEngine(
        task_spec=task_spec,
        expander=CallableReservoirExpander(_expand),
        candidate_domain=DraftCandidateDomain(),
        evaluator=CallableCandidateEvaluator(
            lambda candidate: {"objective": float(candidate.payload)}
        ),
        runtime=runtime,
    )
    return engine.run(
        LDMEngineConfig(
            iterations=iterations,
            reservoir_size=1,
            evaluations_per_round=1,
        )
    )
