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
from ldm_tts.optimization.gp import RBFGPUCBSelector
from ldm_tts.transport import ProposalClient

from tasks.iron_mind.core.candidate import IronMindCandidateDomain
from tasks.iron_mind.core.data import FrozenReactionTable
from tasks.iron_mind.core.evaluator import FrozenReactionEvaluator
from tasks.iron_mind.core.proposals import (
    REQUIRED_CANDIDATE_COUNT,
    IronMindProposalExpander,
)
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
    before_request: Callable[[], None] | None = None
    acquisition_beta: float = 1.0

    def __post_init__(self) -> None:
        if self.table.schema != self.schema:
            raise ValueError("Campaign table schema does not match the supplied schema.")
        if not math.isfinite(self.acquisition_beta) or self.acquisition_beta < 0:
            raise ValueError("Acquisition beta must be finite and non-negative.")


@dataclass(frozen=True)
class CampaignComponents:
    """Complete task adapters connected to one shared :class:`LDMEngine`."""

    task_spec: LDMTaskSpec
    domain: IronMindCandidateDomain
    expander: IronMindProposalExpander
    encoder: ReactionOneHotEncoder
    selector: RBFGPUCBSelector
    evaluator: FrozenReactionEvaluator
    engine: LDMEngine


def build_campaign_components(options: CampaignComponentOptions) -> CampaignComponents:
    """Assemble one engine without reading credentials, files, or global task state."""

    encoder = ReactionOneHotEncoder(options.schema)
    selector = RBFGPUCBSelector(
        objective_name=OBJECTIVE_NAME,
        beta=options.acquisition_beta,
        feature_version=encoder.version,
    )
    task_spec = build_reaction_task_spec(options.schema, selector.describe())
    domain = IronMindCandidateDomain(
        options.schema,
        options.table,
        sink=options.sink,
    )
    expander = IronMindProposalExpander(
        options.client,
        domain,
        before_request=options.before_request,
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
    schema: ReactionDatasetSchema, acquisition: AcquisitionSpec
) -> LDMTaskSpec:
    """Describe the exact fixed-schema reaction search assembled by the factory."""

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
        reservoir=_reaction_reservoir_spec(),
        surrogate=ReactionOneHotEncoder(schema).describe(),
        proposal_search=ProposalSearchSpec(
            name="single_turn_fixed_reservoir",
            breadth=REQUIRED_CANDIDATE_COUNT,
            evaluation_policy="acquisition_selected",
        ),
        metadata={
            "dataset_id": schema.dataset_id,
            "schema_sha256": schema.schema_sha256,
            "required_candidate_count": REQUIRED_CANDIDATE_COUNT,
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
        name="reaction_candidates_json",
        output_kind="json_object",
        schema=_reaction_response_schema(),
        parser="tasks.iron_mind.core.proposals:parse_reaction_candidates",
        description="Exactly four source-valid finite reaction candidates.",
    )


def _reaction_reservoir_spec() -> ReservoirSpec:
    return ReservoirSpec(
        name="reaction_condition_reservoir",
        expansions=(
            ReservoirExpansionSpec(
                name="reaction_condition_proposal",
                action_kind="emit_candidate",
                response_space="reaction_candidates_json",
                produces_candidates=True,
                description="Emit four strict reaction-condition candidates.",
            ),
        ),
        candidate_validator="tasks.iron_mind.core.candidate:IronMindCandidateDomain",
        deduplication_key="SHA-256 canonical reaction payload",
        max_size=REQUIRED_CANDIDATE_COUNT,
    )


def _reaction_response_schema() -> dict[str, object]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["candidates"],
        "properties": {
            "candidates": {
                "type": "array",
                "minItems": REQUIRED_CANDIDATE_COUNT,
                "maxItems": REQUIRED_CANDIDATE_COUNT,
            }
        },
    }


__all__ = [
    "CampaignComponentOptions",
    "CampaignComponents",
    "build_campaign_components",
    "build_reaction_task_spec",
]
