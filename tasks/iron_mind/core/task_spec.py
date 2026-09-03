"""Declarative Iron Mind task contracts for each comparison method."""

from __future__ import annotations

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

from tasks.iron_mind.core.constants import OBJECTIVE_NAME, TASK_ID
from tasks.iron_mind.core.prompting import DEFAULT_PROMPT_POLICY, validate_prompt_policy
from tasks.iron_mind.core.schema import ReactionDatasetSchema
from tasks.iron_mind.core.search import INITIALIZATION_MODES, SEARCH_METHODS
from tasks.iron_mind.core.surrogate import ReactionOneHotEncoder


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
    harness_profile_count: int = 0,
) -> LDMTaskSpec:
    """Describe one fixed-schema reaction search with method-specific semantics."""

    if search_method not in SEARCH_METHODS or initialization_mode not in INITIALIZATION_MODES:
        raise ValueError("Unknown Iron Mind search or initialization method.")
    if proposal_samples < 1 or bo_pool_size < 1 or proposal_max_workers < 1:
        raise ValueError("Proposal, pool, and worker counts must be positive.")
    if search_method in {"ldm", "ldm_harness"} and bo_pool_size >= proposal_samples:
        raise ValueError("LDM BO pool must be smaller than proposal samples.")
    active_surrogate = surrogate or ReactionOneHotEncoder(schema).describe()
    validated_prompt_policy = validate_prompt_policy(prompt_policy)
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
        reservoir=_reaction_reservoir_spec(
            samples=proposal_samples,
            workers=proposal_max_workers,
            method=search_method,
            initialization=initialization_mode,
            domain_size=domain_size,
        ),
        surrogate=active_surrogate,
        proposal_search=_proposal_search(
            search_method,
            proposal_samples,
            proposal_max_workers,
            domain_size,
            harness_profile_count,
        ),
        metadata={
            "dataset_id": schema.dataset_id,
            "schema_sha256": schema.schema_sha256,
            "search_method": search_method,
            "initialization_mode": initialization_mode,
            "proposal_samples": proposal_samples,
            "bo_pool_size": bo_pool_size,
            "proposal_max_workers": proposal_max_workers,
            "proposal_transport": (
                "openai_responses_persistent_sessions"
                if search_method in {"ldm_harness", "harness"}
                else "openai_chat_completions_single_choice"
            ),
            "sampling_mode": (
                "persistent_parallel_research_sessions"
                if search_method == "ldm_harness"
                else "persistent_direct_research_session"
                if search_method == "harness"
                else "local_concurrent_independent_requests"
            ),
            "prompt_policy": (
                "task_local_agents"
                if search_method in {"ldm_harness", "harness"}
                else validated_prompt_policy
            ),
            "prompt_version": (
                "task_local_agents"
                if search_method in {"ldm_harness", "harness"}
                else validated_prompt_policy
            ),
        },
    )


def build_direct_acquisition(search_method: str) -> AcquisitionSpec:
    """Describe reservoir-order selection for a direct-evaluation method."""

    return AcquisitionSpec(
        name=f"direct_{search_method}_reservoir_order",
        objective_names=(OBJECTIVE_NAME,),
        score_direction="maximize",
        selection_rule=(
            "evaluate every admitted direct LLM candidate in reservoir order"
            if search_method == "llm"
            else "evaluate every admitted Harness candidate in reservoir order"
        ),
    )


def disabled_surrogate(search_method: str) -> SurrogateSpaceSpec:
    """Describe a direct-evaluation method with no surrogate."""

    return SurrogateSpaceSpec(
        kind="none",
        representation=(
            "No surrogate for direct LLM baseline."
            if search_method == "llm"
            else "No surrogate for direct Harness evaluation."
        ),
        dimension_policy="none",
    )


def _proposal_search(
    method: str,
    samples: int,
    workers: int,
    domain_size: int | None,
    harness_profile_count: int,
) -> ProposalSearchSpec:
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
    if method in {"ldm_harness", "harness"}:
        if harness_profile_count < 1 or samples % harness_profile_count:
            raise ValueError("harness breadth must divide evenly across profiles")
        return ProposalSearchSpec(
            name=(
                "persistent_parallel_research_sessions"
                if method == "ldm_harness"
                else "persistent_direct_research_session"
            ),
            breadth=samples,
            evaluation_policy=(
                "q0_maintained_acquisition_tilted"
                if method == "ldm_harness"
                else "reservoir_order_direct_evaluation"
            ),
            parameters={
                "profile_count": harness_profile_count,
                "candidates_per_session": samples // harness_profile_count,
                "persistent_sessions": True,
                "skills_loaded": False,
                "candidate_source": "structured_source_pinned_reaction_space_tools",
            },
        )
    return ProposalSearchSpec(
        name="parallel_independent_requests",
        breadth=samples,
        evaluation_policy="q0_maintained_acquisition_tilted",
        parameters={"max_workers": workers},
    )


def _reaction_reservoir_spec(
    *,
    samples: int,
    workers: int,
    method: str,
    initialization: str,
    domain_size: int | None,
) -> ReservoirSpec:
    expansions = []
    if initialization == "shared_random":
        expansions.append(
            ReservoirExpansionSpec(
                name="shared_random_initialization",
                action_kind="emit_candidate",
                response_space="reaction_candidate_json",
                produces_candidates=True,
                description="Deterministic seed-controlled sample from the official finite table.",
            )
        )
    description = (
        "Expose all unseen official conditions to GP-UCB."
        if method == "bo"
        else "Collect exact legal conditions generated by persistent research sessions."
        if method in {"ldm_harness", "harness"}
        else f"Launch {samples} independent one-candidate requests with at most {workers} workers."
    )
    expansions.append(
        ReservoirExpansionSpec(
            name=f"{method}_search_expansion",
            action_kind="emit_candidate",
            response_space="reaction_candidate_json",
            produces_candidates=True,
            description=description,
        )
    )
    return ReservoirSpec(
        name="reaction_condition_reservoir",
        expansions=tuple(expansions),
        candidate_validator="tasks.iron_mind.core.candidate:IronMindCandidateDomain",
        deduplication_key="SHA-256 canonical reaction payload",
        max_size=domain_size if method == "bo" else samples,
    )


def _candidate_domain_spec(schema: ReactionDatasetSchema) -> CandidateDomainSpec:
    return CandidateDomainSpec(
        name="Source-pinned finite reaction conditions",
        kind="finite_reaction_conditions",
        dimension=len(schema.factors),
        representation="One exact typed option per tracked reaction factor.",
        constraints={
            "dataset_id": schema.dataset_id,
            "factor_order": list(schema.factor_names),
        },
    )


def _reaction_response_space() -> ResponseSpaceSpec:
    return ResponseSpaceSpec(
        name="reaction_candidate_json",
        output_kind="json_object",
        schema={
            "type": "object",
            "additionalProperties": False,
            "required": ["dataset_id", "conditions"],
            "properties": {
                "dataset_id": {"type": "string"},
                "conditions": {"type": "object"},
            },
        },
        parser="tasks.iron_mind.core.proposal_parsing:parse_reaction_response",
        description="One source-valid finite reaction candidate.",
    )


__all__ = ["build_direct_acquisition", "build_reaction_task_spec", "disabled_surrogate"]
