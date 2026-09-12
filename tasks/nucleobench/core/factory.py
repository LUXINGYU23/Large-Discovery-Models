"""Task-local component assembly for NucleoBench campaign methods."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from ldm_tts.engine.expansion import ReservoirExpander
from ldm_tts.harness import HarnessClient, HarnessProfile, PolicyResearchController
from ldm_tts.optimization.records import AcquisitionSelector
from ldm_tts.transport import ProposalClient
from tasks.nucleobench.core.candidate import (
    MutationContext,
    NucleoBenchCandidateDomain,
)
from tasks.nucleobench.core.constants import (
    COMPILED_POLICY_METHOD,
    DEFAULT_LDM_ALPHA,
    DEFAULT_LDM_ETA,
    DIRECT_SEARCH_METHODS,
    PARALLEL_HARNESS_METHODS,
    PERSISTENT_HARNESS_METHODS,
    SEARCH_METHODS,
)
from tasks.nucleobench.core.hamming_gp import (
    HammingGPUCBConfig,
    HammingGPUCBSelector,
    NucleotideHammingEncoder,
)
from tasks.nucleobench.core.harness import (
    DIRECT_HARNESS_PROFILE_ID,
    NucleoBenchHarnessExpander,
)
from tasks.nucleobench.core.proposals import (
    DEFAULT_PROPOSAL_MAX_WORKERS,
    DirectMutationProposalExpander,
    ScoreBlindMutationPoolExpander,
)
from tasks.nucleobench.core.optimization_policy import NucleoOptimizationPolicyAdapter
from tasks.nucleobench.core.benchmark_clock import BenchmarkClock
from tasks.nucleobench.core.selection import AcquisitionTiltedSelector


def build_surrogate_components(
    search_method: str,
    context: MutationContext,
    *,
    evaluations_per_round: int,
    proposal_samples: int | None = None,
    bo_pool_size: int | None = None,
    seed: int = 0,
    gp_config: HammingGPUCBConfig | None = None,
    alpha: float = DEFAULT_LDM_ALPHA,
    eta: float = DEFAULT_LDM_ETA,
    z_clip: float = 5.0,
    policy_controller: PolicyResearchController | None = None,
    policy_adapter: NucleoOptimizationPolicyAdapter | None = None,
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
        pool_size=3 * evaluations_per_round if bo_pool_size is None else bo_pool_size,
        proposal_sample_count=(
            4 * evaluations_per_round if proposal_samples is None else proposal_samples
        ),
        policy_controller=policy_controller,
        policy_adapter=policy_adapter,
        policy_mode=search_method == COMPILED_POLICY_METHOD,
    )


def build_proposal_expander(
    search_method: str,
    context: MutationContext,
    *,
    seed: int,
    evaluations_per_round: int,
    client: ProposalClient | None = None,
    harness_client: HarnessClient | None = None,
    harness_artifact_root: Path | None = None,
    harness_session_profiles: Sequence[HarnessProfile] = (),
    harness_candidates_per_session: int | None = None,
    harness_unique_candidates: bool = False,
    campaign_id: str = "",
    first_active_round: int = 1,
    max_workers: int = DEFAULT_PROPOSAL_MAX_WORKERS,
    before_requests: Callable[[int], None] | None = None,
    account: Callable[..., Any] | None = None,
    benchmark_clock: BenchmarkClock | None = None,
    surrogate_query_config: HammingGPUCBConfig | None = None,
) -> ReservoirExpander:
    """Build the sole task-local proposal path for one campaign method."""

    if search_method == "bo":
        return ScoreBlindMutationPoolExpander(context, seed=seed)
    if search_method in PERSISTENT_HARNESS_METHODS:
        if harness_client is None or not harness_session_profiles or harness_artifact_root is None:
            raise ValueError(
                f"search method {search_method!r} requires a harness client, profiles, and artifact root"
            )
        if (
            search_method == "harness"
            and tuple(profile.profile_id for profile in harness_session_profiles)
            != (DIRECT_HARNESS_PROFILE_ID,)
        ):
            raise ValueError("Direct Harness requires the direct_research profile")
        return NucleoBenchHarnessExpander(
            harness_client,
            NucleoBenchCandidateDomain(context),
            profiles=harness_session_profiles,
            artifact_root=harness_artifact_root,
            candidates_per_profile=(
                evaluations_per_round
                if harness_candidates_per_session is None
                else harness_candidates_per_session
            ),
            campaign_id=campaign_id,
            first_active_round=first_active_round,
            attach_empirical_q0=search_method in PARALLEL_HARNESS_METHODS,
            allow_repeated_occurrences=(
                search_method in PARALLEL_HARNESS_METHODS
                and not harness_unique_candidates
            ),
            account=account,
            benchmark_clock=benchmark_clock,
            surrogate_query_config=surrogate_query_config,
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
