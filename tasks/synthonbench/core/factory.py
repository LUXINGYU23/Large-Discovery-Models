"""Task-owned SynthonBench assembly for LDM, BO, and direct-LLM comparisons."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

from ldm_tts.contracts import (
    AcquisitionSpec,
    CandidateDomainSpec,
    LDMTaskSpec,
    ObjectiveSpec,
    ProposalSearchSpec,
    ReservoirExpansionSpec,
    ReservoirSpec,
    ResponseSpaceSpec,
    SurrogateSpaceSpec,
)
from ldm_tts.data import DataCollectionSink
from ldm_tts.engine import InitialRoundReservoirExpander, LDMEngine
from ldm_tts.engine.expansion import ReservoirExpander
from ldm_tts.engine.run_store import CampaignRuntime
from ldm_tts.optimization.records import AcquisitionSelector
from ldm_tts.transport import ProposalClient

from tasks.synthonbench.core.candidate import SynthonCandidateDomain
from tasks.synthonbench.core.catalog import SynthonProposalCatalog
from tasks.synthonbench.core.constants import (
    DEFAULT_ACQUISITION_ETA,
    DEFAULT_GP_REACTION_WEIGHT,
    OBJECTIVE_NAME,
    TASK_ID,
)
from tasks.synthonbench.core.evaluator import OfficialSynthonEvaluator
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
    proposal_max_workers: int = 64
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

    def __post_init__(self) -> None:
        if self.search_method not in SEARCH_METHODS:
            raise ValueError(f"Unknown search method: {self.search_method!r}")
        if self.initialization_mode not in INITIALIZATION_MODES:
            raise ValueError(f"Unknown initialization mode: {self.initialization_mode!r}")
        if min(self.proposal_samples, self.bo_pool_size, self.bo_search_samples, self.evaluations_per_round) < 1:
            raise ValueError("Proposal, pool, and evaluation counts must be positive")
        if self.search_method == "ldm" and self.proposal_samples <= self.bo_pool_size:
            raise ValueError("LDM proposal samples must exceed the BO pool size")
        if self.evaluations_per_round > self.bo_pool_size and self.search_method != "llm":
            raise ValueError("BO-based methods require evaluations_per_round <= bo_pool_size")
        if self.search_method in {"ldm", "llm"} and self.client is None:
            raise ValueError("Model search methods require a proposal client")
        _validate_options(self)
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
    task_spec = build_synthon_task_spec(
        encoder=encoder,
        acquisition=selector.describe() if selector is not None else build_direct_acquisition(),
        proposal_samples=options.proposal_samples,
        bo_pool_size=options.bo_pool_size,
        bo_search_samples=options.bo_search_samples,
        proposal_max_workers=options.proposal_max_workers,
        slate_size=options.slate_size,
        reaction_allocation=options.reaction_allocation,
        prompt_policy=options.prompt_policy,
        search_method=options.search_method,
        initialization_mode=options.initialization_mode,
    )
    domain = SynthonCandidateDomain(official_task.space, reactions, options.target, options.sink)
    expander = _expander(options, domain, reactions)
    evaluator = OfficialSynthonEvaluator(official_task)
    engine = LDMEngine(
        task_spec=task_spec,
        expander=expander,
        candidate_domain=domain,
        evaluator=evaluator,
        runtime=options.runtime,
        selector=selector,
        surrogate_encoder=encoder,
    )
    return CampaignComponents(task_spec, domain, expander, encoder, selector, evaluator, engine)


def build_synthon_task_spec(
    *,
    encoder: SynthonNystromEncoder | None,
    acquisition: AcquisitionSpec,
    proposal_samples: int,
    bo_pool_size: int,
    bo_search_samples: int = 64,
    proposal_max_workers: int,
    slate_size: int,
    reaction_allocation: str,
    prompt_policy: str,
    search_method: str = "ldm",
    initialization_mode: str = "none",
) -> LDMTaskSpec:
    """Describe finite tuple, response, and method-specific acquisition semantics."""

    if encoder is None and search_method != "llm":
        raise ValueError("BO-based SynthonBench methods require a surrogate encoder")
    return LDMTaskSpec(
        task=TASK_ID,
        candidate_domain=CandidateDomainSpec(
            name="Source-pinned synthon tuples", kind="reaction_synthon_tuple", dimension=None,
            representation="One reaction ID and one ordered valid synthon ID per official reaction slot.",
        ),
        objectives=(ObjectiveSpec(OBJECTIVE_NAME, "maximize", "Official fixed-oracle SynthonBench utility."),),
        response_spaces=(_response_space(),),
        acquisition=acquisition,
        reservoir=_reservoir_spec(
            bo_search_samples if search_method == "bo" else proposal_samples,
            proposal_max_workers,
            search_method,
            initialization_mode,
        ),
        surrogate=encoder.describe() if encoder is not None else disabled_surrogate(),
        proposal_search=_proposal_search(search_method, proposal_samples, proposal_max_workers),
        metadata={
            "search_method": search_method,
            "initialization_mode": initialization_mode,
            "proposal_samples": proposal_samples,
            "bo_pool_size": bo_pool_size,
            "bo_search_samples": bo_search_samples,
            "slate_size": slate_size,
            "reaction_allocation": reaction_allocation,
            "prompt_policy": prompt_policy,
        },
    )


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


def build_direct_acquisition() -> AcquisitionSpec:
    return AcquisitionSpec(
        name="direct_llm_reservoir_order", objective_names=(OBJECTIVE_NAME,), score_direction="maximize",
        selection_rule="evaluate every admitted direct LLM candidate in reservoir order",
    )


def disabled_surrogate() -> SurrogateSpaceSpec:
    return SurrogateSpaceSpec(kind="none", representation="No surrogate for direct LLM baseline.", dimension_policy="none")


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
    else:
        assert options.client is not None
        catalog = SynthonProposalCatalog(
            options.official_task.space, allowed_reactions=reactions, slate_size=options.slate_size,
            seed=options.selection_seed, reaction_allocation=options.reaction_allocation,
            direct_unique=options.search_method == "llm",
            direct_proposal_count=options.evaluations_per_round if options.search_method == "llm" else None,
            direct_start_round=1 if options.initialization_mode == "shared_random" else 0,
        )
        search = SynthonBenchProposalExpander(
            options.client, domain, catalog, target=options.target, before_requests=options.before_requests,
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


def _reservoir_spec(samples, workers, method, initialization) -> ReservoirSpec:
    expansions = []
    if initialization == "shared_random":
        expansions.append(ReservoirExpansionSpec(
            name="shared_random_initialization", action_kind="emit_candidate",
            response_space="synthon_tuple_json", produces_candidates=True,
            description="Deterministic product-uniform initial sample from the official space.",
        ))
    description = "Score-blind random unseen product pool for base GP-UCB." if method == "bo" else (
        f"Launch {samples} independent one-candidate requests with at most {workers} workers."
    )
    expansions.append(ReservoirExpansionSpec(
        name=f"{method}_search_expansion", action_kind="emit_candidate",
        response_space="synthon_tuple_json", produces_candidates=True, description=description,
    ))
    return ReservoirSpec(
        name="synthon_tuple_reservoir", expansions=tuple(expansions),
        candidate_validator="tasks.synthonbench.core.candidate:SynthonCandidateDomain",
        deduplication_key="official SynthonBench product_id", max_size=samples,
    )


def _proposal_search(method: str, samples: int, workers: int) -> ProposalSearchSpec:
    if method == "bo":
        return ProposalSearchSpec(
            name="score_blind_random_pool_bo", breadth=samples,
            evaluation_policy="nystrom_tanimoto_gp_ucb", parameters={"model_requests": 0},
        )
    if method == "llm":
        return ProposalSearchSpec(
            name="parallel_independent_direct_llm", breadth=samples,
            evaluation_policy="reservoir_order_direct_evaluation",
            parameters={"max_workers": workers, "one_candidate_per_request": True},
        )
    return ProposalSearchSpec(
        name="parallel_independent_requests", breadth=samples,
        evaluation_policy="empirical_q0_maintained_acquisition_tilted",
        parameters={"max_workers": workers, "one_candidate_per_request": True},
    )


def _response_space() -> ResponseSpaceSpec:
    return ResponseSpaceSpec(
        name="synthon_tuple_json", output_kind="json_object",
        parser="tasks.synthonbench.core.proposal_parsing:parse_synthon_response",
        description="One source-valid tuple chosen only from a supplied finite public synthon slate.",
        schema={"type": "object", "additionalProperties": False,
                "required": ["reaction_id", "synthon_ids"],
                "properties": {"reaction_id": {"type": "string"}, "synthon_ids": {"type": "array"}}},
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
    "build_campaign_components", "build_direct_acquisition", "build_synthon_selector",
    "build_synthon_task_spec", "disabled_surrogate",
]
