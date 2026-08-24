"""Task-owned assembly for Iron Mind LDM, BO, and direct-LLM campaigns."""

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

from tasks.iron_mind.core.candidate import IronMindCandidateDomain
from tasks.iron_mind.core.data import FrozenReactionTable
from tasks.iron_mind.core.evaluator import FrozenReactionEvaluator
from tasks.iron_mind.core.ldm_selector import AcquisitionTiltedSelector
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


TASK_ID = "iron_mind"
OBJECTIVE_NAME = "reaction_score"


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
    acquisition_eta: float = 3.0
    acquisition_z_clip: float = 5.0
    selection_seed: int = 0
    prompt_policy: str = DEFAULT_PROMPT_POLICY

    def __post_init__(self) -> None:
        if self.table.schema != self.schema:
            raise ValueError("Campaign table schema does not match the supplied schema.")
        if self.search_method not in SEARCH_METHODS:
            raise ValueError(f"Unknown search method: {self.search_method!r}.")
        if self.initialization_mode not in INITIALIZATION_MODES:
            raise ValueError(f"Unknown initialization mode: {self.initialization_mode!r}.")
        if self.proposal_samples < 1 or self.bo_pool_size < 1:
            raise ValueError("Proposal and BO pool sizes must be positive.")
        if self.search_method == "ldm" and self.proposal_samples <= self.bo_pool_size:
            raise ValueError("LDM proposal samples must exceed the BO pool size.")
        if self.search_method in {"ldm", "llm"} and self.client is None:
            raise ValueError("Model search methods require a proposal client.")
        if self.proposal_max_workers < 1 or self.selection_seed < 0:
            raise ValueError("Worker count must be positive and seed non-negative.")
        _validate_acquisition_options(self)
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
    task_spec = build_reaction_task_spec(
        options.schema,
        selector.describe() if selector is not None else build_direct_acquisition(),
        proposal_samples=options.proposal_samples,
        bo_pool_size=options.bo_pool_size,
        proposal_max_workers=options.proposal_max_workers,
        prompt_policy=options.prompt_policy,
        search_method=options.search_method,
        initialization_mode=options.initialization_mode,
        surrogate=encoder.describe() if encoder is not None else disabled_surrogate(),
        domain_size=finite_domain_size(options.table),
    )
    domain = IronMindCandidateDomain(options.schema, options.table, sink=options.sink)
    expander = _expander(options, domain)
    evaluator = FrozenReactionEvaluator(options.table)
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


def build_reaction_task_spec(
    schema: ReactionDatasetSchema,
    acquisition: AcquisitionSpec,
    *,
    proposal_samples: int,
    bo_pool_size: int,
    proposal_max_workers: int,
    prompt_policy: str = DEFAULT_PROMPT_POLICY,
    search_method: str = "ldm",
    initialization_mode: str = "none",
    surrogate: SurrogateSpaceSpec | None = None,
    domain_size: int | None = None,
) -> LDMTaskSpec:
    """Describe one fixed-schema reaction search with method-specific semantics."""

    if search_method not in SEARCH_METHODS or initialization_mode not in INITIALIZATION_MODES:
        raise ValueError("Unknown Iron Mind search or initialization method.")
    if proposal_samples < 1 or bo_pool_size < 1 or proposal_max_workers < 1:
        raise ValueError("Proposal, pool, and worker counts must be positive.")
    if search_method == "ldm" and bo_pool_size >= proposal_samples:
        raise ValueError("LDM BO pool must be smaller than proposal samples.")
    active_surrogate = surrogate or ReactionOneHotEncoder(schema).describe()
    return LDMTaskSpec(
        task=TASK_ID,
        candidate_domain=_candidate_domain_spec(schema),
        objectives=(ObjectiveSpec(OBJECTIVE_NAME, "maximize", "Frozen-table reaction score; higher is better."),),
        response_spaces=(_reaction_response_space(),),
        acquisition=acquisition,
        reservoir=_reaction_reservoir_spec(
            proposal_samples,
            proposal_max_workers,
            search_method,
            initialization_mode,
            domain_size,
        ),
        surrogate=active_surrogate,
        proposal_search=_proposal_search(search_method, proposal_samples, proposal_max_workers, domain_size),
        metadata={
            "dataset_id": schema.dataset_id,
            "schema_sha256": schema.schema_sha256,
            "search_method": search_method,
            "initialization_mode": initialization_mode,
            "proposal_samples": proposal_samples,
            "bo_pool_size": bo_pool_size,
            "proposal_max_workers": proposal_max_workers,
            "proposal_transport": "openai_chat_completions_single_choice",
            "sampling_mode": "local_concurrent_independent_requests",
            "prompt_policy": validate_prompt_policy(prompt_policy),
            "prompt_version": validate_prompt_policy(prompt_policy),
        },
    )


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
    if options.search_method == "llm":
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
    else:
        assert options.client is not None
        search = IronMindProposalExpander(
            options.client,
            domain,
            before_requests=options.before_requests,
            max_workers=options.proposal_max_workers,
            prompt_policy=options.prompt_policy,
        )
    if options.initialization_mode == "none":
        return search
    return InitialRoundReservoirExpander(
        initializer=IronMindInitializationExpander(
            options.table,
            seed=options.selection_seed,
            attach_q0=options.search_method == "ldm",
        ),
        search_expander=search,
        initial_reservoir_size=1,
    )


def _proposal_search(method: str, samples: int, workers: int, domain_size: int | None) -> ProposalSearchSpec:
    if method == "bo":
        return ProposalSearchSpec(
            name="full_finite_domain_bo",
            breadth=domain_size or samples,
            evaluation_policy="factor_aware_gp_ucb",
            parameters={"model_requests": 0, "domain_size": domain_size},
        )
    if method == "llm":
        return ProposalSearchSpec(
            name="parallel_independent_direct_llm",
            breadth=samples,
            evaluation_policy="reservoir_order_direct_evaluation",
            parameters={"max_workers": workers, "one_candidate_per_request": True},
        )
    return ProposalSearchSpec(
        name="parallel_independent_requests",
        breadth=samples,
        evaluation_policy="q0_maintained_acquisition_tilted",
        parameters={"max_workers": workers},
    )


def _reaction_reservoir_spec(samples, workers, method, initialization, domain_size) -> ReservoirSpec:
    expansions = []
    if initialization == "shared_random":
        expansions.append(ReservoirExpansionSpec(
            name="shared_random_initialization", action_kind="emit_candidate",
            response_space="reaction_candidate_json", produces_candidates=True,
            description="Deterministic seed-controlled sample from the official finite table.",
        ))
    description = "Expose all unseen official conditions to GP-UCB." if method == "bo" else (
        f"Launch {samples} independent one-candidate requests with at most {workers} workers."
    )
    expansions.append(ReservoirExpansionSpec(
        name=f"{method}_search_expansion", action_kind="emit_candidate",
        response_space="reaction_candidate_json", produces_candidates=True, description=description,
    ))
    return ReservoirSpec(
        name="reaction_condition_reservoir", expansions=tuple(expansions),
        candidate_validator="tasks.iron_mind.core.candidate:IronMindCandidateDomain",
        deduplication_key="SHA-256 canonical reaction payload",
        max_size=domain_size if method == "bo" else samples,
    )


def _candidate_domain_spec(schema: ReactionDatasetSchema) -> CandidateDomainSpec:
    return CandidateDomainSpec(
        name="Source-pinned finite reaction conditions", kind="finite_reaction_conditions",
        dimension=len(schema.factors), representation="One exact typed option per tracked reaction factor.",
        constraints={"dataset_id": schema.dataset_id, "factor_order": list(schema.factor_names)},
    )


def _reaction_response_space() -> ResponseSpaceSpec:
    return ResponseSpaceSpec(
        name="reaction_candidate_json", output_kind="json_object", schema=_reaction_response_schema(),
        parser="tasks.iron_mind.core.proposal_parsing:parse_reaction_response",
        description="One source-valid finite reaction candidate.",
    )


def _reaction_response_schema() -> dict[str, object]:
    return {"type": "object", "additionalProperties": False, "required": ["dataset_id", "conditions"],
            "properties": {"dataset_id": {"type": "string"}, "conditions": {"type": "object"}}}


def build_direct_acquisition() -> AcquisitionSpec:
    return AcquisitionSpec(
        name="direct_llm_reservoir_order", objective_names=(OBJECTIVE_NAME,), score_direction="maximize",
        selection_rule="evaluate every admitted direct LLM candidate in reservoir order",
    )


def disabled_surrogate() -> SurrogateSpaceSpec:
    return SurrogateSpaceSpec(kind="none", representation="No surrogate for direct LLM baseline.", dimension_policy="none")


def _validate_acquisition_options(options: CampaignComponentOptions) -> None:
    values = (options.acquisition_beta, options.acquisition_alpha, options.acquisition_eta)
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("Acquisition parameters must be finite and non-negative.")
    if not math.isfinite(options.acquisition_z_clip) or options.acquisition_z_clip <= 0:
        raise ValueError("Acquisition z-clip must be finite and positive.")


__all__ = [
    "CampaignComponentOptions", "CampaignComponents", "OBJECTIVE_NAME", "build_base_reaction_selector",
    "build_campaign_components", "build_direct_acquisition", "build_reaction_selector",
    "build_reaction_task_spec", "disabled_surrogate",
]
