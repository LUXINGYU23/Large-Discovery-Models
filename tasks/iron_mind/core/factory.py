"""Task-owned assembly for Iron Mind campaign methods."""

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

from tasks.iron_mind.core import task_spec as task_contracts
from tasks.iron_mind.core.candidate import IronMindCandidateDomain
from tasks.iron_mind.core.constants import OBJECTIVE_NAME
from tasks.iron_mind.core.data import FrozenReactionTable
from tasks.iron_mind.core.evaluator import FrozenReactionEvaluator
from tasks.iron_mind.core.harness import IronMindHarnessExpander
from tasks.iron_mind.core.ldm_selector import AcquisitionTiltedSelector
from tasks.iron_mind.core.ldm_policy import DEFAULT_ETA
from tasks.iron_mind.core.prompting import DEFAULT_PROMPT_POLICY, validate_prompt_policy
from tasks.iron_mind.core.proposals import DEFAULT_PROPOSAL_MAX_WORKERS, IronMindProposalExpander
from tasks.iron_mind.core.reaction_gp import ReactionCategoricalGPUCBSelector
from tasks.iron_mind.core.schema import ReactionDatasetSchema
from tasks.iron_mind.core.search import (
    FullReactionDomainExpander,
    INITIALIZATION_MODES,
    IronMindInitializationExpander,
    SEARCH_METHODS,
    finite_domain_size,
)
from tasks.iron_mind.core.surrogate import ReactionOneHotEncoder


@dataclass(frozen=True)
class CampaignComponentOptions:
    """Dependencies injected by the workflow for one task-local search method."""

    client: ProposalClient | None
    schema: ReactionDatasetSchema
    table: FrozenReactionTable
    sink: DataCollectionSink
    runtime: CampaignRuntime
    proposal_samples: int
    bo_pool_size: int
    search_method: str = "ldm"
    initialization_mode: str = "none"
    proposal_max_workers: int = DEFAULT_PROPOSAL_MAX_WORKERS
    before_requests: Callable[[int], None] | None = None
    acquisition_beta: float = 1.0
    acquisition_alpha: float = 1.0
    acquisition_eta: float = DEFAULT_ETA
    acquisition_z_clip: float = 5.0
    selection_seed: int = 0
    prompt_policy: str = DEFAULT_PROMPT_POLICY
    harness_client: HarnessClient | None = None
    harness_profiles: tuple[HarnessProfile, ...] = ()
    account_harness_usage: Callable[[dict[str, int]], None] | None = None

    def __post_init__(self) -> None:
        if self.table.schema != self.schema:
            raise ValueError("Campaign table schema does not match the supplied schema.")
        if self.search_method not in SEARCH_METHODS:
            raise ValueError(f"Unknown search method: {self.search_method!r}.")
        if self.initialization_mode not in INITIALIZATION_MODES:
            raise ValueError(f"Unknown initialization mode: {self.initialization_mode!r}.")
        if self.proposal_samples < 1 or self.bo_pool_size < 1:
            raise ValueError("Proposal and BO pool sizes must be positive.")
        if self.search_method in {"ldm", "ldm_harness"} and self.proposal_samples <= self.bo_pool_size:
            raise ValueError("LDM proposal samples must exceed the BO pool size.")
        if self.search_method in {"ldm_harness", "harness"}:
            if self.harness_client is None or not self.harness_profiles:
                raise ValueError("Harness search requires a client and profile set")
            expected = sum(profile.candidates_per_turn for profile in self.harness_profiles)
            if self.proposal_samples != expected:
                raise ValueError("proposal_samples must equal the harness minibatch total")
        elif self.search_method in {"ldm", "llm"} and self.client is None:
            raise ValueError("Direct model search methods require a proposal client.")
        if self.proposal_max_workers < 1 or self.selection_seed < 0:
            raise ValueError("Worker count must be positive and seed non-negative.")
        if self.search_method != "harness":
            _validate_acquisition_options(self)
        if self.search_method in {"ldm", "llm"}:
            validate_prompt_policy(self.prompt_policy)


@dataclass(frozen=True)
class CampaignComponents:
    """Task adapters connected to one shared :class:`LDMEngine`."""

    task_spec: LDMTaskSpec
    domain: IronMindCandidateDomain
    expander: ReservoirExpander
    encoder: ReactionOneHotEncoder | None
    selector: AcquisitionSelector | None
    evaluator: FrozenReactionEvaluator
    engine: LDMEngine


def build_campaign_components(options: CampaignComponentOptions) -> CampaignComponents:
    """Assemble the selected method without reading credentials or global state."""

    encoder, selector = _search_components(options)
    declared_task = task_contracts.build_reaction_task_spec(
        options.schema,
        (
            selector.describe()
            if selector is not None
            else task_contracts.build_direct_acquisition(options.search_method)
        ),
        proposal_samples=options.proposal_samples,
        bo_pool_size=options.bo_pool_size,
        proposal_max_workers=options.proposal_max_workers,
        prompt_policy=options.prompt_policy,
        search_method=options.search_method,
        initialization_mode=options.initialization_mode,
        surrogate=(
            encoder.describe()
            if encoder is not None
            else task_contracts.disabled_surrogate(options.search_method)
        ),
        domain_size=finite_domain_size(options.table),
        harness_profile_count=len(options.harness_profiles),
    )
    domain = IronMindCandidateDomain(options.schema, options.table, sink=options.sink)
    expander = _expander(options, domain)
    evaluator = FrozenReactionEvaluator(options.table)
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


def build_reaction_selector(**kwargs) -> AcquisitionTiltedSelector:
    """Build the empirical-q0 LDM selector retained from the main task method."""

    base = build_base_reaction_selector(
        schema=kwargs["schema"], beta=kwargs["beta"], feature_version=kwargs["feature_version"]
    )
    return AcquisitionTiltedSelector(
        base,
        alpha=kwargs["alpha"],
        eta=kwargs["eta"],
        z_clip=kwargs["z_clip"],
        seed=kwargs["seed"],
        pool_size=kwargs["pool_size"],
        proposal_sample_count=kwargs["proposal_sample_count"],
    )


def build_base_reaction_selector(
    *, schema: ReactionDatasetSchema, beta: float, feature_version: str
) -> ReactionCategoricalGPUCBSelector:
    """Build plain factor-aware categorical GP-UCB for the BO comparator."""

    return ReactionCategoricalGPUCBSelector(
        schema=schema,
        objective_name=OBJECTIVE_NAME,
        beta=beta,
        feature_version=feature_version,
    )


def _search_components(options: CampaignComponentOptions):
    if options.search_method in {"llm", "harness"}:
        return None, None
    encoder = ReactionOneHotEncoder(options.schema)
    if options.search_method == "bo":
        return encoder, build_base_reaction_selector(
            schema=options.schema,
            beta=options.acquisition_beta,
            feature_version=encoder.version,
        )
    return encoder, build_reaction_selector(
        schema=options.schema,
        beta=options.acquisition_beta,
        alpha=options.acquisition_alpha,
        eta=options.acquisition_eta,
        z_clip=options.acquisition_z_clip,
        seed=options.selection_seed,
        pool_size=options.bo_pool_size,
        proposal_sample_count=options.proposal_samples,
        feature_version=encoder.version,
    )


def _expander(options: CampaignComponentOptions, domain: IronMindCandidateDomain) -> ReservoirExpander:
    if options.search_method == "bo":
        search: ReservoirExpander = FullReactionDomainExpander(options.table)
    elif options.search_method in {"ldm_harness", "harness"}:
        assert options.harness_client is not None
        search = IronMindHarnessExpander(
            options.harness_client,
            domain,
            profiles=options.harness_profiles,
            campaign_id=options.runtime.run_id,
            first_active_round=1 if options.initialization_mode == "shared_random" else 0,
            attach_empirical_q0=options.search_method == "ldm_harness",
            account=options.account_harness_usage,
        )
    else:
        assert options.client is not None
        search = IronMindProposalExpander(
            options.client,
            domain,
            before_requests=options.before_requests,
            max_workers=options.proposal_max_workers,
            prompt_policy=options.prompt_policy,
            slot_seed=options.selection_seed,
        )
    if options.initialization_mode == "none":
        return search
    return InitialRoundReservoirExpander(
        initializer=IronMindInitializationExpander(
            options.table,
            seed=options.selection_seed,
            attach_q0=options.search_method in {"ldm", "ldm_harness"},
        ),
        search_expander=search,
        initial_reservoir_size=1,
    )


def _validate_acquisition_options(options: CampaignComponentOptions) -> None:
    values = (options.acquisition_beta, options.acquisition_alpha, options.acquisition_eta)
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("Acquisition parameters must be finite and non-negative.")
    if not math.isfinite(options.acquisition_z_clip) or options.acquisition_z_clip <= 0:
        raise ValueError("Acquisition z-clip must be finite and positive.")


__all__ = [
    "CampaignComponentOptions", "CampaignComponents", "build_base_reaction_selector",
    "build_campaign_components", "build_reaction_selector",
]
