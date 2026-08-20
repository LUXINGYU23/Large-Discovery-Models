"""Task-owned assembly of Iron Mind components around shared LDM runtime types."""

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
)
from ldm_tts.data import DataCollectionSink
from ldm_tts.engine import LDMEngine
from ldm_tts.engine.run_store import CampaignRuntime
from ldm_tts.transport import ProposalClient

from tasks.iron_mind.core.candidate import IronMindCandidateDomain
from tasks.iron_mind.core.data import FrozenReactionTable
from tasks.iron_mind.core.evaluator import FrozenReactionEvaluator
from tasks.iron_mind.core.ldm_selector import AcquisitionTiltedSelector
from tasks.iron_mind.core.prompting import DEFAULT_PROMPT_POLICY, validate_prompt_policy
from tasks.iron_mind.core.proposals import (
    DEFAULT_PROPOSAL_MAX_WORKERS,
    IronMindProposalExpander,
)
from tasks.iron_mind.core.reaction_gp import ReactionCategoricalGPUCBSelector
from tasks.iron_mind.core.schema import ReactionDatasetSchema
from tasks.iron_mind.core.surrogate import ReactionOneHotEncoder


TASK_ID = "iron_mind"
OBJECTIVE_NAME = "reaction_score"


@dataclass(frozen=True)
class CampaignComponentOptions:
    """Dependencies injected by workflow code for one Iron Mind campaign."""

    client: ProposalClient
    schema: ReactionDatasetSchema
    table: FrozenReactionTable
    sink: DataCollectionSink
    runtime: CampaignRuntime
    proposal_samples: int
    bo_pool_size: int
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
        if self.proposal_samples < 1:
            raise ValueError("Proposal sample count must be positive.")
        if self.bo_pool_size < 1:
            raise ValueError("BO pool size must be positive.")
        if self.proposal_samples <= self.bo_pool_size:
            raise ValueError("Proposal sample count must exceed BO pool size.")
        if self.proposal_max_workers < 1:
            raise ValueError("Proposal max workers must be positive.")
        if not math.isfinite(self.acquisition_beta) or self.acquisition_beta < 0:
            raise ValueError("Acquisition beta must be finite and non-negative.")
        for name, value in (
            ("Acquisition alpha", self.acquisition_alpha),
            ("Acquisition eta", self.acquisition_eta),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative.")
        if not math.isfinite(self.acquisition_z_clip) or self.acquisition_z_clip <= 0:
            raise ValueError("Acquisition z-clip must be finite and positive.")
        if self.selection_seed < 0:
            raise ValueError("Selection seed must be non-negative.")
        validate_prompt_policy(self.prompt_policy)


@dataclass(frozen=True)
class CampaignComponents:
    """Complete task adapters connected to one shared :class:`LDMEngine`."""

    task_spec: LDMTaskSpec
    domain: IronMindCandidateDomain
    expander: IronMindProposalExpander
    encoder: ReactionOneHotEncoder
    selector: AcquisitionTiltedSelector
    evaluator: FrozenReactionEvaluator
    engine: LDMEngine


def build_campaign_components(options: CampaignComponentOptions) -> CampaignComponents:
    """Assemble one engine without reading credentials, files, or global task state."""

    encoder = ReactionOneHotEncoder(options.schema)
    selector = build_reaction_selector(
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
    task_spec = build_reaction_task_spec(
        options.schema,
        selector.describe(),
        proposal_samples=options.proposal_samples,
        bo_pool_size=options.bo_pool_size,
        proposal_max_workers=options.proposal_max_workers,
        prompt_policy=options.prompt_policy,
    )
    domain = IronMindCandidateDomain(
        options.schema,
        options.table,
        sink=options.sink,
    )
    expander = IronMindProposalExpander(
        options.client,
        domain,
        before_requests=options.before_requests,
        max_workers=options.proposal_max_workers,
        prompt_policy=options.prompt_policy,
    )
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
) -> LDMTaskSpec:
    """Describe one fixed-schema reaction search with separate proposal and BO pools."""
    if proposal_samples < 1:
        raise ValueError("Proposal sample count must be positive.")
    if bo_pool_size < 1 or bo_pool_size >= proposal_samples:
        raise ValueError("BO pool size must be positive and smaller than proposal samples.")
    if proposal_max_workers < 1:
        raise ValueError("Proposal max workers must be positive.")
    prompt_policy = validate_prompt_policy(prompt_policy)

    return LDMTaskSpec(
        task=TASK_ID,
        candidate_domain=_candidate_domain_spec(schema),
        objectives=(
            ObjectiveSpec(
                OBJECTIVE_NAME,
                "maximize",
                "Frozen-table reaction score; higher is better.",
            ),
        ),
        response_spaces=(_reaction_response_space(),),
        acquisition=acquisition,
        reservoir=_reaction_reservoir_spec(proposal_samples, proposal_max_workers),
        surrogate=ReactionOneHotEncoder(schema).describe(),
        proposal_search=ProposalSearchSpec(
            name="parallel_independent_requests",
            breadth=proposal_samples,
            evaluation_policy="q0_maintained_acquisition_tilted",
            parameters={"max_workers": proposal_max_workers},
        ),
        metadata={
            "dataset_id": schema.dataset_id,
            "schema_sha256": schema.schema_sha256,
            "proposal_samples": proposal_samples,
            "bo_pool_size": bo_pool_size,
            "proposal_max_workers": proposal_max_workers,
            "proposal_transport": "openai_chat_completions_single_choice",
            "sampling_mode": "local_concurrent_independent_requests",
            "prompt_policy": prompt_policy,
            "prompt_version": prompt_policy,
        },
    )


def build_reaction_selector(
    *,
    schema: ReactionDatasetSchema,
    beta: float,
    alpha: float,
    eta: float,
    z_clip: float,
    seed: int,
    pool_size: int,
    proposal_sample_count: int,
    feature_version: str,
) -> AcquisitionTiltedSelector:
    base_selector = ReactionCategoricalGPUCBSelector(
        schema=schema,
        objective_name=OBJECTIVE_NAME,
        beta=beta,
        feature_version=feature_version,
    )
    return AcquisitionTiltedSelector(
        base_selector,
        alpha=alpha,
        eta=eta,
        z_clip=z_clip,
        seed=seed,
        pool_size=pool_size,
        proposal_sample_count=proposal_sample_count,
    )


def _candidate_domain_spec(schema: ReactionDatasetSchema) -> CandidateDomainSpec:
    return CandidateDomainSpec(
        name="Source-pinned finite reaction conditions",
        kind="finite_reaction_conditions",
        dimension=len(schema.factors),
        representation="One exact typed option per tracked reaction factor.",
        constraints={"dataset_id": schema.dataset_id, "factor_order": list(schema.factor_names)},
    )


def _reaction_response_space() -> ResponseSpaceSpec:
    return ResponseSpaceSpec(
        name="reaction_candidate_json",
        output_kind="json_object",
        schema=_reaction_response_schema(),
        parser="tasks.iron_mind.core.proposal_parsing:parse_reaction_response",
        description="One source-valid finite reaction candidate per independent request.",
    )


def _reaction_reservoir_spec(candidate_count: int, max_workers: int) -> ReservoirSpec:
    return ReservoirSpec(
        name="reaction_condition_reservoir",
        expansions=(
            ReservoirExpansionSpec(
                name="reaction_condition_proposal",
                action_kind="emit_candidate",
                response_space="reaction_candidate_json",
                produces_candidates=True,
                description=(
                    f"Launch {candidate_count} independent requests with at most {max_workers} "
                    "workers; every request emits one strict reaction-condition candidate."
                ),
            ),
        ),
        candidate_validator="tasks.iron_mind.core.candidate:IronMindCandidateDomain",
        deduplication_key="SHA-256 canonical reaction payload",
        max_size=candidate_count,
    )


def _reaction_response_schema() -> dict[str, object]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["dataset_id", "conditions"],
        "properties": {
            "dataset_id": {"type": "string"},
            "conditions": {"type": "object"},
        },
    }


__all__ = [
    "CampaignComponentOptions",
    "CampaignComponents",
    "build_campaign_components",
    "build_reaction_selector",
    "build_reaction_task_spec",
]
