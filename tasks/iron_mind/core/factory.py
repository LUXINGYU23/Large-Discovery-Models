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
    reservoir_size: int
    proposal_max_workers: int = DEFAULT_PROPOSAL_MAX_WORKERS
    before_requests: Callable[[int], None] | None = None
    acquisition_beta: float = 1.0

    def __post_init__(self) -> None:
        if self.table.schema != self.schema:
            raise ValueError("Campaign table schema does not match the supplied schema.")
        if self.reservoir_size < 1:
            raise ValueError("Reservoir size must be positive.")
        if self.proposal_max_workers < 1:
            raise ValueError("Proposal max workers must be positive.")
        if not math.isfinite(self.acquisition_beta) or self.acquisition_beta < 0:
            raise ValueError("Acquisition beta must be finite and non-negative.")


@dataclass(frozen=True)
class CampaignComponents:
    """Complete task adapters connected to one shared :class:`LDMEngine`."""

    task_spec: LDMTaskSpec
    domain: IronMindCandidateDomain
    expander: IronMindProposalExpander
    encoder: ReactionOneHotEncoder
    selector: ReactionCategoricalGPUCBSelector
    evaluator: FrozenReactionEvaluator
    engine: LDMEngine


def build_campaign_components(options: CampaignComponentOptions) -> CampaignComponents:
    """Assemble one engine without reading credentials, files, or global task state."""

    encoder = ReactionOneHotEncoder(options.schema)
    selector = ReactionCategoricalGPUCBSelector(
        schema=options.schema,
        objective_name=OBJECTIVE_NAME,
        beta=options.acquisition_beta,
        feature_version=encoder.version,
    )
    task_spec = build_reaction_task_spec(
        options.schema,
        selector.describe(),
        options.reservoir_size,
        options.proposal_max_workers,
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
    reservoir_size: int,
    proposal_max_workers: int,
) -> LDMTaskSpec:
    """Describe one fixed-schema reaction search with a configurable reservoir."""

    if reservoir_size < 1:
        raise ValueError("Reservoir size must be positive.")
    if proposal_max_workers < 1:
        raise ValueError("Proposal max workers must be positive.")

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
        reservoir=_reaction_reservoir_spec(reservoir_size, proposal_max_workers),
        surrogate=ReactionOneHotEncoder(schema).describe(),
        proposal_search=ProposalSearchSpec(
            name="parallel_independent_requests",
            breadth=reservoir_size,
            evaluation_policy="acquisition_selected",
            parameters={"max_workers": proposal_max_workers},
        ),
        metadata={
            "dataset_id": schema.dataset_id,
            "schema_sha256": schema.schema_sha256,
            "reservoir_size": reservoir_size,
            "proposal_max_workers": proposal_max_workers,
            "proposal_transport": "openai_chat_completions_single_choice",
            "sampling_mode": "local_concurrent_independent_requests",
        },
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
        parser="tasks.iron_mind.core.proposals:parse_reaction_response",
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
    "build_reaction_task_spec",
]
