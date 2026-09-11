"""Declarative LDM contract for one NucleoBench case."""

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
from tasks.nucleobench.core.cases import NucleoBenchCase
from tasks.nucleobench.core.harness import HARNESS_SKILL_IDS, harness_submission_contract
from tasks.nucleobench.core.constants import (
    COMPILED_POLICY_METHOD,
    DIRECT_SEARCH_METHODS,
    PARALLEL_HARNESS_METHODS,
    SEARCH_METHODS,
    TASK_ID,
)


def build_task_spec(
    case: NucleoBenchCase,
    *,
    search_method: str = "ldm",
    evaluations_per_round: int = 1,
    proposal_samples: int | None = None,
    proposal_max_workers: int = 4,
    harness_profile_count: int = 4,
    harness_candidates_per_session: int | None = None,
    harness_unique_candidates: bool = False,
    bo_pool_size: int | None = None,
    acquisition: AcquisitionSpec | None = None,
    surrogate: SurrogateSpaceSpec | None = None,
) -> LDMTaskSpec:
    """Describe the shared mutation-patch search used by every execution profile."""

    if search_method not in SEARCH_METHODS:
        raise ValueError(f"unknown NucleoBench search method: {search_method!r}")
    if evaluations_per_round < 1 or proposal_max_workers < 1:
        raise ValueError("evaluation and worker counts must be positive")
    if harness_candidates_per_session is None:
        harness_candidates_per_session = evaluations_per_round
    if harness_profile_count < 1 or harness_candidates_per_session < 1:
        raise ValueError("Harness profile and candidate counts must be positive")
    direct = search_method in DIRECT_SEARCH_METHODS
    if surrogate is None:
        surrogate = (
            SurrogateSpaceSpec(
                kind="none",
                representation="Disabled for direct candidate evaluation.",
                dimension_policy="none",
            )
            if direct
            else SurrogateSpaceSpec(
                kind="kernel",
                representation=(
                    "Categorical equality over official editable positions with a "
                    "normalized Hamming exponential kernel."
                ),
                dimension_policy="implicit",
            )
        )
    if direct != (surrogate.kind == "none"):
        raise ValueError(
            "direct methods disable the surrogate; BO-based methods require it"
        )
    if acquisition is None:
        acquisition = (
            AcquisitionSpec(
                name=(
                    "direct_harness_reservoir_order"
                    if search_method == "harness"
                    else "direct_evaluation"
                ),
                objective_names=("utility",),
                score_direction="reservoir_order",
                selection_rule="evaluate the emitted minibatch without surrogate ranking",
            )
            if direct
            else AcquisitionSpec(
                name=(
                    "hamming_gp_ucb"
                    if search_method == "bo"
                    else "hamming_gp_ucb_ldm_tilt"
                ),
                objective_names=("utility",),
                score_direction="maximize",
                selection_rule=(
                    "Highest task-local Hamming GP-UCB values."
                    if search_method == "bo"
                    else "Empirical q0 tilted by task-local Hamming GP-UCB."
                ),
                parameters=(
                    {} if search_method == "bo" else {
                        "pool_size": 3 * evaluations_per_round if bo_pool_size is None else bo_pool_size,
                    }
                ),
            )
        )
    if search_method == "ldm":
        response_name = "mutation_patch_batch_json"
        response_spaces = (_indexed_batch_response_space(evaluations_per_round),)
        reservoir_size = 4 * evaluations_per_round
        proposal_name = "parallel_independent_minibatch_requests"
        proposal_parameters = {
            "request_count": 4,
            "candidates_per_request": evaluations_per_round,
            "max_workers": min(proposal_max_workers, 4),
        }
        model_request_count: int | None = 4
        session_turn_count: int | None = None
        candidates_per_model_request: int | None = evaluations_per_round
    elif search_method in PARALLEL_HARNESS_METHODS:
        response_name = "harness_candidate_file"
        response_spaces = (_harness_batch_response_space(harness_candidates_per_session),)
        reservoir_size = harness_profile_count * harness_candidates_per_session
        proposal_name = "persistent_parallel_research_sessions"
        proposal_parameters = {
            "profile_count": harness_profile_count,
            "candidates_per_session": harness_candidates_per_session,
            "unique_candidates_per_session": harness_unique_candidates,
            "skills_loaded": True,
            "skill_ids": list(HARNESS_SKILL_IDS),
        }
        if search_method == COMPILED_POLICY_METHOD:
            proposal_parameters.update(
                policy_profile_count=1,
                policy_skills_loaded=True,
                editable_optimization_components=["prior_mean", "alpha", "eta"],
            )
        model_request_count = None
        session_turn_count = harness_profile_count
        candidates_per_model_request = None
    elif search_method == "bo":
        response_name = "mutation_patch_json"
        response_spaces = (_single_response_space(),)
        reservoir_size = 4 * evaluations_per_round if proposal_samples is None else proposal_samples
        if reservoir_size < evaluations_per_round:
            raise ValueError("BO proposal samples must cover the evaluation batch")
        proposal_name = "bo_mutation_search"
        proposal_parameters = {
            "request_count": 0,
            "candidates_per_request": 1,
            "max_workers": 0,
        }
        model_request_count = 0
        session_turn_count = None
        candidates_per_model_request = None
    elif search_method == "llm":
        response_name = "mutation_patch_json"
        response_spaces = (_single_response_space(),)
        reservoir_size = evaluations_per_round
        proposal_name = "parallel_independent_single_candidate_requests"
        proposal_parameters = {
            "request_count": evaluations_per_round,
            "candidates_per_request": 1,
            "max_workers": min(proposal_max_workers, evaluations_per_round),
        }
        model_request_count = evaluations_per_round
        session_turn_count = None
        candidates_per_model_request = 1
    else:
        response_name = "harness_candidate_file"
        response_spaces = (_harness_batch_response_space(evaluations_per_round),)
        reservoir_size = evaluations_per_round
        proposal_name = "persistent_direct_research_session"
        proposal_parameters = {
            "profile_count": 1,
            "candidates_per_session": evaluations_per_round,
            "skills_loaded": True,
            "skill_ids": list(HARNESS_SKILL_IDS),
        }
        model_request_count = None
        session_turn_count = 1
        candidates_per_model_request = None
    return LDMTaskSpec(
        task=TASK_ID,
        candidate_domain=CandidateDomainSpec(
            name="Start-relative nucleotide mutation patches",
            kind="nucleotide_mutation_patch",
            dimension=case.editable_position_count,
            representation="Zero-based editable positions paired with replacement DNA bases.",
            constraints={
                "case_id": case.case_id,
                "sequence_length": case.sequence_length,
                "editable_position_count": case.editable_position_count,
                "alphabet": ["A", "C", "G", "T"],
            },
        ),
        objectives=(
            ObjectiveSpec(
                name="utility",
                direction="maximize",
                description="Negative energy from the unchanged official model wrapper.",
            ),
        ),
        response_spaces=response_spaces,
        acquisition=acquisition,
        reservoir=ReservoirSpec(
            name="mutation_patch_reservoir",
            expansions=(
                ReservoirExpansionSpec(
                    name="mutation_patch_proposals",
                    action_kind="emit_candidate",
                    response_space=response_name,
                    produces_candidates=True,
                ),
            ),
            candidate_validator="task-local editable-position mutation validation",
            deduplication_key=(
                "case, paired-start digest, start index, and rebuilt sequence digest"
            ),
            max_size=reservoir_size,
        ),
        surrogate=surrogate,
        proposal_search=ProposalSearchSpec(
            name=proposal_name,
            breadth=reservoir_size,
            evaluation_policy="official model evaluation through the shared LDM engine",
            parameters=proposal_parameters,
        ),
        metadata={
            "case_id": case.case_id,
            "case_state": case.state,
            "model_family": case.model_family,
            "model_name": case.model_name,
            "target": case.target,
            "max_seconds": case.max_seconds,
            "search_method": search_method,
            "model_requests_per_round": model_request_count,
            "model_session_turns_per_round": session_turn_count,
            "candidates_per_model_request": candidates_per_model_request,
            "search_breadth": reservoir_size,
        },
    )


def _mutation_schema() -> dict[str, object]:
    return {
        "type": "array",
        "minItems": 1,
        "items": {
            "type": "object",
            "additionalProperties": False,
            "required": ["position", "base"],
            "properties": {
                "position": {"type": "integer", "minimum": 0},
                "base": {"type": "string", "enum": ["A", "C", "G", "T"]},
            },
        },
    }


def _single_response_space() -> ResponseSpaceSpec:
    return ResponseSpaceSpec(
        name="mutation_patch_json",
        output_kind="json_object",
        description="One non-empty mutation patch relative to the paired start.",
        schema={
            "type": "object",
            "additionalProperties": False,
            "required": ["mutations"],
            "properties": {"mutations": _mutation_schema()},
        },
    )


def _indexed_batch_response_space(candidate_count: int) -> ResponseSpaceSpec:
    return ResponseSpaceSpec(
        name="mutation_patch_batch_json",
        output_kind="json_object",
        description="One indexed mutation patch for every independent batch slot.",
        schema={
            "type": "object",
            "additionalProperties": False,
            "required": ["candidates"],
            "properties": {
                "candidates": {
                    "type": "array",
                    "minItems": candidate_count,
                    "maxItems": candidate_count,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["proposal_index", "mutations"],
                        "properties": {
                            "proposal_index": {
                                "type": "integer",
                                "minimum": 0,
                                "maximum": candidate_count - 1,
                            },
                            "mutations": _mutation_schema(),
                        },
                    },
                }
            },
        },
    )


def _harness_batch_response_space(candidate_count: int) -> ResponseSpaceSpec:
    return ResponseSpaceSpec(
        name="harness_candidate_file",
        output_kind="json_object",
        description="Workspace file reference; the immutable JSON snapshot contains mutation patches with pre-evaluation change_summary and rationale notes.",
        schema=harness_submission_contract(candidate_count).payload_schema,
    )


__all__ = ["build_task_spec"]
