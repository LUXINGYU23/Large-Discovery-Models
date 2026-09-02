"""Task-local component assembly for NucleoBench campaign methods."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from ldm_tts.engine.expansion import ReservoirExpander
from ldm_tts.harness import HarnessClient, HarnessProfile
from ldm_tts.optimization.records import AcquisitionSelector
from ldm_tts.transport import ProposalClient
from tasks.nucleobench.core.candidate import (
    MutationContext,
    NucleoBenchCandidateDomain,
)
from tasks.nucleobench.core.constants import DIRECT_SEARCH_METHODS, SEARCH_METHODS
from tasks.nucleobench.core.hamming_gp import (
    HammingGPUCBConfig,
    HammingGPUCBSelector,
    NucleotideHammingEncoder,
)
from tasks.nucleobench.core.harness import (
    DIRECT_HARNESS_PROFILE_ID,
    HARNESS_PROFILE_IDS,
    NucleoBenchHarnessExpander,
)
from tasks.nucleobench.core.proposals import (
    DEFAULT_PROPOSAL_MAX_WORKERS,
    DirectMutationProposalExpander,
    ScoreBlindMutationPoolExpander,
)
from tasks.nucleobench.core.selection import AcquisitionTiltedSelector


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
    return encoder, AcquisitionTiltedSelector(
        base,
        alpha=alpha,
        eta=eta,
        z_clip=z_clip,
        seed=seed,
        pool_size=3 * evaluations_per_round,
        proposal_sample_count=4 * evaluations_per_round,
    )


def build_proposal_expander(
    search_method: str,
    context: MutationContext,
    *,
    seed: int,
    evaluations_per_round: int,
    client: ProposalClient | None = None,
    harness_client: HarnessClient | None = None,
    harness_session_profiles: Sequence[HarnessProfile] = (),
    campaign_id: str = "",
    first_active_round: int = 1,
    max_workers: int = DEFAULT_PROPOSAL_MAX_WORKERS,
    before_requests: Callable[[int], None] | None = None,
    account: Callable[[dict[str, int]], None] | None = None,
) -> ReservoirExpander:
    """Build the sole task-local proposal path for one campaign method."""

    if search_method == "bo":
        return ScoreBlindMutationPoolExpander(context, seed=seed)
    if search_method in {"ldm_harness", "harness"}:
        if harness_client is None or not harness_session_profiles:
            raise ValueError(
                f"search method {search_method!r} requires a harness client and profiles"
            )
        expected_profiles = (
            HARNESS_PROFILE_IDS
            if search_method == "ldm_harness"
            else (DIRECT_HARNESS_PROFILE_ID,)
        )
        if (
            tuple(profile.profile_id for profile in harness_session_profiles)
            != expected_profiles
        ):
            raise ValueError(
                f"search method {search_method!r} requires profiles {expected_profiles}"
            )
        if any(
            profile.candidates_per_turn != evaluations_per_round
            for profile in harness_session_profiles
        ):
            raise ValueError(
                "every NucleoBench harness profile must submit evaluations_per_round candidates"
            )
        return NucleoBenchHarnessExpander(
            harness_client,
            NucleoBenchCandidateDomain(context),
            profiles=harness_session_profiles,
            campaign_id=campaign_id,
            first_active_round=first_active_round,
            attach_empirical_q0=search_method == "ldm_harness",
            account=account,
        )
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
        before_requests=before_requests,
    )


__all__ = [
    "build_proposal_expander",
    "build_surrogate_components",
]
