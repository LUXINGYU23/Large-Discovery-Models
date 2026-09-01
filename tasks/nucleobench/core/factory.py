"""Task-local assembly for the deterministic NucleoBench mock campaign."""

from __future__ import annotations

from ldm_tts.contracts import LDMTaskSpec
from ldm_tts.data import DataCollectionSink
from ldm_tts.engine import LDMEngine
from ldm_tts.engine.run_store import CampaignRuntime
from tasks.nucleobench.core.candidate import NucleoBenchCandidateDomain
from tasks.nucleobench.core.evaluator import NucleoBenchEvaluator
from tasks.nucleobench.core.mock import (
    MOCK_CONTEXT,
    build_mock_expander,
    mock_energies,
)


def build_mock_engine(
    runtime: CampaignRuntime,
    sink: DataCollectionSink,
    task_spec: LDMTaskSpec,
) -> LDMEngine:
    domain = NucleoBenchCandidateDomain(MOCK_CONTEXT, sink=sink)
    return LDMEngine(
        task_spec=task_spec,
        expander=build_mock_expander(),
        candidate_domain=domain,
        evaluator=NucleoBenchEvaluator(MOCK_CONTEXT, mock_energies),
        runtime=runtime,
    )


__all__ = ["build_mock_engine"]
