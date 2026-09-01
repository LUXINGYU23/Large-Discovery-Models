"""Task-local component assembly for NucleoBench campaign methods."""

from __future__ import annotations

from ldm_tts.contracts import LDMTaskSpec
from ldm_tts.data import DataCollectionSink
from ldm_tts.engine import LDMEngine
from ldm_tts.engine.run_store import CampaignRuntime
from ldm_tts.optimization.records import AcquisitionSelector
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
    "build_surrogate_components",
]
