"""Task-owned SynthonBench assembly for LDM, BO, and direct-LLM comparisons."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

from ldm_tts.contracts import LDMTaskSpec
from ldm_tts.data import DataCollectionSink
from ldm_tts.engine import InitialRoundReservoirExpander, LDMEngine
from ldm_tts.engine.expansion import ReservoirExpander
from ldm_tts.engine.run_store import CampaignRuntime
from ldm_tts.harness import HarnessClient, HarnessProfile
from ldm_tts.optimization.records import AcquisitionSelector
from ldm_tts.transport import ProposalClient

from tasks.synthonbench.core import task_spec as task_contracts
from tasks.synthonbench.core.candidate import SynthonCandidateDomain
from tasks.synthonbench.core.catalog import SynthonProposalCatalog
from tasks.synthonbench.core.constants import (
    DEFAULT_ACQUISITION_ETA,
    DEFAULT_GP_REACTION_WEIGHT,
    OBJECTIVE_NAME,
)
from tasks.synthonbench.core.evaluator import OfficialSynthonEvaluator
from tasks.synthonbench.core.harness import SynthonHarnessExpander
from tasks.synthonbench.core.ldm_selector import AcquisitionTiltedSelector
from tasks.synthonbench.core.nystrom_encoder import SynthonNystromEncoder
from tasks.synthonbench.core.prompting import DEFAULT_PROMPT_POLICY, validate_prompt_policy
from tasks.synthonbench.core.proposals import SynthonBenchProposalExpander
from tasks.synthonbench.core.search import (
    INITIALIZATION_MODES,
    RandomSynthonPoolExpander,
    SEARCH_METHODS,
    SynthonInitializationExpander,
)
from tasks.synthonbench.core.space_order import ordered_reactions
from tasks.synthonbench.core.tanimoto_gp import TanimotoGPUCBConfig, SynthonTanimotoGPUCBSelector


@dataclass(frozen=True)
class CampaignComponentOptions:
    """Dependencies and immutable choices for one SynthonBench campaign."""

    client: ProposalClient | None
    official_task: object
    runtime: CampaignRuntime
    sink: DataCollectionSink
    target: str
    proposal_samples: int
    bo_pool_size: int
    bo_search_samples: int = 64
    evaluations_per_round: int = 1
    search_method: str = "ldm"
    initialization_mode: str = "none"
    proposal_candidates_per_request: int = 16
    proposal_max_workers: int = 4
    slate_size: int = 24
    reaction_allocation: str = "product_weighted"
    selection_seed: int = 0
    fingerprint_bits: int = 2048
    gp_landmarks: int = 256
    gp_kernel_jitter: float = 1.0e-8
    gp_signal_std: float = 1.0
    gp_mean_std: float = 1.0
    gp_observation_noise_std: float = 1.0
    gp_reaction_weight: float = DEFAULT_GP_REACTION_WEIGHT
    acquisition_beta: float = 1.0
    alpha: float = 1.0
    eta: float = DEFAULT_ACQUISITION_ETA
    z_clip: float = 5.0
    prompt_policy: str = DEFAULT_PROMPT_POLICY
    before_requests: Callable[[int], None] | None = None
    proposal_backend: str = "direct"
    harness_client: HarnessClient | None = None
    harness_profiles: tuple[HarnessProfile, ...] = ()
    account_harness_usage: Callable[[dict[str, int]], None] | None = None

    def __post_init__(self) -> None:
        if self.search_method not in SEARCH_METHODS:
            raise ValueError(f"Unknown search method: {self.search_method!r}")
        if self.initialization_mode not in INITIALIZATION_MODES:
            raise ValueError(f"Unknown initialization mode: {self.initialization_mode!r}")
        if min(
            self.proposal_samples,
            self.bo_pool_size,
            self.bo_search_samples,
            self.evaluations_per_round,
            self.proposal_candidates_per_request,
        ) < 1:
            raise ValueError("Proposal, pool, and evaluation counts must be positive")
        if self.search_method == "ldm" and self.proposal_samples <= self.bo_pool_size:
            raise ValueError("LDM proposal samples must exceed the BO pool size")
        if self.evaluations_per_round > self.bo_pool_size and self.search_method != "llm":
            raise ValueError("BO-based methods require evaluations_per_round <= bo_pool_size")
        if self.proposal_backend not in {"direct", "harness"}:
            raise ValueError("proposal_backend must be direct or harness")
        if self.proposal_backend == "harness":
            if self.search_method != "ldm":
                raise ValueError("The harness backend is available only for LDM search")
            if self.harness_client is None or not self.harness_profiles:
                raise ValueError("Harness search requires a client and profile set")
            expected = sum(profile.candidates_per_turn for profile in self.harness_profiles)
            if self.proposal_samples != expected:
                raise ValueError("proposal_samples must equal the harness minibatch total")
        elif self.search_method in {"ldm", "llm"} and self.client is None:
            raise ValueError("Direct model search methods require a proposal client")
        if self.proposal_backend == "direct" and self.search_method in {"ldm", "llm"}:
            breadth = (
                self.proposal_samples
                if self.search_method == "ldm"
                else self.evaluations_per_round
            )
            if breadth % self.proposal_candidates_per_request:
                raise ValueError(
                    "direct proposal breadth must divide evenly by candidates per request"
                )
        _validate_options(self)
        if self.proposal_backend == "direct":
            validate_prompt_policy(self.prompt_policy)


@dataclass(frozen=True)
class CampaignComponents:
    """All task adapters connected to one shared :class:`LDMEngine`."""

    task_spec: LDMTaskSpec
    domain: SynthonCandidateDomain
    expander: ReservoirExpander
    encoder: SynthonNystromEncoder | None
    selector: AcquisitionSelector | None
    evaluator: OfficialSynthonEvaluator
    engine: LDMEngine


def build_campaign_components(options: CampaignComponentOptions) -> CampaignComponents:
    """Assemble a task-local method without reading credentials or global state."""

    official_task = options.official_task
    reactions = ordered_reactions(official_task.allowed_reactions)
    encoder, selector = _search_components(options, reactions)
    declared_task = task_contracts.build_synthon_task_spec(
        encoder=encoder,
        acquisition=(
            selector.describe()
            if selector is not None
            else task_contracts.build_direct_acquisition()
        ),
        proposal_samples=options.proposal_samples,
        evaluations_per_round=options.evaluations_per_round,
        bo_pool_size=options.bo_pool_size,
        bo_search_samples=options.bo_search_samples,
        proposal_candidates_per_request=options.proposal_candidates_per_request,
        proposal_max_workers=options.proposal_max_workers,
        slate_size=options.slate_size,
        reaction_allocation=options.reaction_allocation,
        prompt_policy=options.prompt_policy,
        search_method=options.search_method,
        initialization_mode=options.initialization_mode,
        proposal_backend=options.proposal_backend,
        harness_profile_count=len(options.harness_profiles),
    )
    domain = SynthonCandidateDomain(official_task.space, reactions, options.target, options.sink)
    expander = _expander(options, domain, reactions)
    evaluator = OfficialSynthonEvaluator(official_task)
    engine = LDMEngine(
        task_spec=declared_task,
        expander=expander,
        candidate_domain=domain,
        evaluator=evaluator,
        runtime=options.runtime,
        selector=selector,
        surrogate_encoder=encoder,
    )
    return CampaignComponents(declared_task, domain, expander, encoder, selector, evaluator, engine)


def build_synthon_selector(
    *, encoder: SynthonNystromEncoder, selection_seed: int, gp_signal_std: float,
    gp_mean_std: float, gp_observation_noise_std: float, acquisition_beta: float,
    alpha: float, eta: float, z_clip: float, bo_pool_size: int, proposal_samples: int,
) -> AcquisitionTiltedSelector:
    """Build the existing empirical-q0 LDM selector."""

    return AcquisitionTiltedSelector(
        build_base_synthon_selector(
            encoder=encoder, gp_signal_std=gp_signal_std, gp_mean_std=gp_mean_std,
            gp_observation_noise_std=gp_observation_noise_std, acquisition_beta=acquisition_beta,
        ),
        alpha=alpha, eta=eta, z_clip=z_clip, seed=selection_seed,
        pool_size=bo_pool_size, proposal_sample_count=proposal_samples,
    )


def build_base_synthon_selector(
    *, encoder: SynthonNystromEncoder, gp_signal_std: float, gp_mean_std: float,
    gp_observation_noise_std: float, acquisition_beta: float,
) -> SynthonTanimotoGPUCBSelector:
    """Build the plain local Tanimoto GP-UCB comparator."""

    return SynthonTanimotoGPUCBSelector(
        objective_name=OBJECTIVE_NAME,
        feature_dimension=encoder.dimension,
        feature_version=encoder.version,
        config=TanimotoGPUCBConfig(
            beta=acquisition_beta, signal_std=gp_signal_std, mean_std=gp_mean_std,
            observation_noise_std=gp_observation_noise_std,
        ),
    )


def _search_components(options: CampaignComponentOptions, reactions):
    if options.search_method == "llm":
        return None, None
    encoder = SynthonNystromEncoder(
        options.official_task.space, reactions, landmark_count=options.gp_landmarks,
        seed=options.selection_seed, fingerprint_bits=options.fingerprint_bits,
        kernel_jitter=options.gp_kernel_jitter, reaction_weight=options.gp_reaction_weight,
    )
    if options.search_method == "bo":
        return encoder, build_base_synthon_selector(
            encoder=encoder, gp_signal_std=options.gp_signal_std, gp_mean_std=options.gp_mean_std,
            gp_observation_noise_std=options.gp_observation_noise_std,
            acquisition_beta=options.acquisition_beta,
        )
    return encoder, build_synthon_selector(
        encoder=encoder, selection_seed=options.selection_seed, gp_signal_std=options.gp_signal_std,
        gp_mean_std=options.gp_mean_std, gp_observation_noise_std=options.gp_observation_noise_std,
        acquisition_beta=options.acquisition_beta, alpha=options.alpha, eta=options.eta,
        z_clip=options.z_clip, bo_pool_size=options.bo_pool_size, proposal_samples=options.proposal_samples,
    )


def _expander(options: CampaignComponentOptions, domain: SynthonCandidateDomain, reactions) -> ReservoirExpander:
    if options.search_method == "bo":
        search: ReservoirExpander = RandomSynthonPoolExpander(
            options.official_task.space, reactions, seed=options.selection_seed
        )
    elif options.proposal_backend == "harness":
        assert options.harness_client is not None
        search = SynthonHarnessExpander(
            options.harness_client,
            domain,
            target=options.target,
            profiles=options.harness_profiles,
            campaign_id=options.runtime.run_id,
            first_active_round=1 if options.initialization_mode == "shared_random" else 0,
            account=options.account_harness_usage,
        )
    else:
        catalog = SynthonProposalCatalog(
            options.official_task.space, allowed_reactions=reactions, slate_size=options.slate_size,
            seed=options.selection_seed, reaction_allocation=options.reaction_allocation,
            unique_anchors=True,
            proposals_per_round=(
                options.evaluations_per_round
                if options.search_method == "llm"
                else options.proposal_samples
            ),
            first_round=1 if options.initialization_mode == "shared_random" else 0,
            restrict_to_complete_tuples=options.search_method == "llm",
        )
        assert options.client is not None
        search = SynthonBenchProposalExpander(
            options.client, domain, catalog, target=options.target, before_requests=options.before_requests,
            candidates_per_request=options.proposal_candidates_per_request,
            max_workers=options.proposal_max_workers, prompt_policy=options.prompt_policy,
        )
    if options.initialization_mode == "none":
        return search
    return InitialRoundReservoirExpander(
        initializer=SynthonInitializationExpander(
            options.official_task.space, reactions, seed=options.selection_seed,
            attach_q0=options.search_method == "ldm",
        ),
        search_expander=search,
        initial_reservoir_size=options.evaluations_per_round,
    )


def _validate_options(options: CampaignComponentOptions) -> None:
    values = (options.acquisition_beta, options.alpha, options.eta, options.gp_mean_std, options.gp_reaction_weight)
    if any(not math.isfinite(item) or item < 0 for item in values):
        raise ValueError("Acquisition and non-negative GP parameters must be finite")
    positive = (options.gp_kernel_jitter, options.gp_signal_std, options.gp_observation_noise_std)
    if any(not math.isfinite(item) or item <= 0 for item in positive):
        raise ValueError("Positive GP parameters must be finite")
    if not math.isfinite(options.z_clip) or options.z_clip <= 0:
        raise ValueError("z_clip must be finite and positive")


__all__ = [
    "CampaignComponentOptions", "CampaignComponents", "build_base_synthon_selector",
    "build_campaign_components", "build_synthon_selector",
]
