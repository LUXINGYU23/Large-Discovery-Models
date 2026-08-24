from __future__ import annotations

from pathlib import Path

import pytest

from ldm_tts.contracts import (
    AcquisitionSpec,
    Candidate,
    CandidateDomainSpec,
    CandidateRejection,
    CallableCandidateEvaluator,
    LDMTaskSpec,
    ObjectiveSpec,
    RawProposal,
    ReservoirExpansionSpec,
    ReservoirSpec,
    ResponseSpaceSpec,
    SurrogateSpaceSpec,
)
from ldm_tts.engine import LDMEngine, LDMEngineConfig
from ldm_tts.engine.expansion import CallableReservoirExpander, ExpansionResult
from ldm_tts.engine.run_store import CampaignRuntime
from ldm_tts.transport import ProposalResponse


class UnusedCandidateDomain:
    def admit(self, proposal: RawProposal) -> Candidate | CandidateRejection:
        raise AssertionError("attempt-only expansion must not admit candidates")


def attempt_only_task_spec() -> LDMTaskSpec:
    return LDMTaskSpec(
        task="attempt_only",
        candidate_domain=CandidateDomainSpec("integer", "integer", 1),
        objectives=(ObjectiveSpec("score", "maximize"),),
        response_spaces=(ResponseSpaceSpec("integers", "json"),),
        acquisition=AcquisitionSpec("first", ("score",), "maximize", "first"),
        reservoir=ReservoirSpec(
            "integers",
            (
                ReservoirExpansionSpec(
                    "emit_integers",
                    "emit_candidate",
                    "integers",
                    True,
                ),
            ),
            "integer range",
            "integer string",
            max_size=1,
        ),
        surrogate=SurrogateSpaceSpec("none", "not used", "none"),
    )


def test_expansion_result_accepts_attempts_without_candidates() -> None:
    attempt = ProposalResponse(text="{}")

    assert ExpansionResult(attempts=(attempt,)).attempts == (attempt,)
    with pytest.raises(ValueError, match="expansion must emit"):
        ExpansionResult()
    with pytest.raises(ValueError, match="selection mode"):
        ExpansionResult(attempts=(attempt,), selection_mode="unknown")


def test_engine_counts_attempt_only_expansion_without_selection_or_evaluation(
    tmp_path: Path,
) -> None:
    runtime = CampaignRuntime.open(tmp_path / "attempt-only", task="attempt_only")
    engine = LDMEngine(
        task_spec=attempt_only_task_spec(),
        expander=CallableReservoirExpander(
            lambda request: ExpansionResult(
                attempts=(ProposalResponse(text="{}"),)
            )
        ),
        candidate_domain=UnusedCandidateDomain(),
        evaluator=CallableCandidateEvaluator(lambda candidate: {"score": 1.0}),
        runtime=runtime,
    )

    result = engine.run(
        LDMEngineConfig(
            iterations=1,
            reservoir_size=1,
            max_empty_reservoir_rounds=1,
        )
    )

    assert result.stop_reason == "empty_reservoir_limit"
    assert runtime.budget.counters["outer_iterations"] == 1
    assert runtime.budget.counters["proposal_attempts"] == 1
    for counter in (
        "valid_search_candidates",
        "selected_candidates",
        "external_evaluations",
        "expensive_evaluation_attempts",
        "successful_evaluations",
        "benchmark_jobs",
    ):
        assert runtime.budget.counters.get(counter, 0) == 0
