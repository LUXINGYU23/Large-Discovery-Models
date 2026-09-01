"""Task-local component assembly for NucleoBench campaign methods."""

from __future__ import annotations

from collections.abc import Callable

from ldm_tts.contracts import LDMTaskSpec
from ldm_tts.data import DataCollectionSink
from ldm_tts.engine import LDMEngine
from ldm_tts.engine.expansion import ReservoirExpander
from ldm_tts.engine.run_store import CampaignRuntime
from ldm_tts.optimization.records import AcquisitionSelector
from ldm_tts.transport import ProposalClient
from tasks.nucleobench.core.candidate import (
    MutationContext,
    NucleoBenchCandidateDomain,
)
from tasks.nucleobench.core.constants import DIRECT_SEARCH_METHODS, SEARCH_METHODS
from tasks.nucleobench.core.evaluator import NucleoBenchEvaluator
from tasks.nucleobench.core.hamming_gp import (
    HammingGPUCBConfig,
    HammingGPUCBSelector,
    NucleotideHammingEncoder,
)
from tasks.nucleobench.core.mock import (
    MOCK_CONTEXT,
    build_mock_expander,
    mock_energies,
)
from tasks.nucleobench.core.proposals import (
    DEFAULT_PROPOSAL_MAX_WORKERS,
    DEFAULT_PROPOSAL_REQUEST_WAVES,
    DirectMutationProposalExpander,
    ScoreBlindMutationPoolExpander,
)
from tasks.nucleobench.core.selection import AcquisitionTiltedSelector


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


def build_surrogate_components(
    search_method: str,
    context: MutationContext,
    *,
    evaluations_per_round: int,
    seed: int = 0,
    gp_config: HammingGPUCBConfig | None = None,
    alpha: float = 1.0,
    eta: float = 1.0,
    z_clip: float = 5.0,
) -> tuple[NucleotideHammingEncoder | None, AcquisitionSelector | None]:
    """Build the sole task-local surrogate path for one campaign method."""

    if search_method not in SEARCH_METHODS:
        raise ValueError(f"unknown NucleoBench search method: {search_method!r}")
    if evaluations_per_round < 1:
        raise ValueError("evaluations_per_round must be positive")
    if search_method in DIRECT_SEARCH_METHODS:
        return None, None

    encoder = NucleotideHammingEncoder(context)
    base = HammingGPUCBSelector(
        objective_name="utility",
        feature_dimension=encoder.dimension,
        feature_version=encoder.version,
        config=gp_config,
    )
    if search_method == "bo":
        return encoder, base
    return encoder, build_ldm_selector(
        base,
        evaluations_per_round=evaluations_per_round,
        seed=seed,
        alpha=alpha,
        eta=eta,
        z_clip=z_clip,
    )


def build_proposal_expander(
    search_method: str,
    context: MutationContext,
    *,
    seed: int,
    evaluations_per_round: int,
    client: ProposalClient | None = None,
    max_workers: int = DEFAULT_PROPOSAL_MAX_WORKERS,
    max_request_waves: int = DEFAULT_PROPOSAL_REQUEST_WAVES,
    before_requests: Callable[[int], None] | None = None,
) -> ReservoirExpander:
    """Build the task-local BO or direct-model mutation generator."""

    if search_method == "bo":
        return ScoreBlindMutationPoolExpander(context, seed=seed)
    if search_method not in {"ldm", "llm"}:
        raise ValueError(
            f"proposal expander for search method {search_method!r} is not implemented"
        )
    if client is None:
        raise ValueError(f"search method {search_method!r} requires a proposal client")
    return DirectMutationProposalExpander(
        client,
        NucleoBenchCandidateDomain(context),
        search_method=search_method,
        evaluations_per_round=evaluations_per_round,
        seed=seed,
        max_workers=max_workers,
        max_request_waves=max_request_waves,
        before_requests=before_requests,
    )


def build_ldm_selector(
    base_selector: AcquisitionSelector,
    *,
    evaluations_per_round: int,
    seed: int,
    alpha: float,
    eta: float,
    z_clip: float,
) -> AcquisitionTiltedSelector:
    """Apply the shared 4B proposal and 3B maintained-pool policy."""

    return AcquisitionTiltedSelector(
        base_selector,
        alpha=alpha,
        eta=eta,
        z_clip=z_clip,
        seed=seed,
        pool_size=3 * evaluations_per_round,
        proposal_sample_count=4 * evaluations_per_round,
    )


__all__ = [
    "build_ldm_selector",
    "build_mock_engine",
    "build_proposal_expander",
    "build_surrogate_components",
]
