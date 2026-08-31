"""Declarative SynthonBench task contracts for each comparison method."""

from __future__ import annotations

import math

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

from tasks.synthonbench.core.constants import OBJECTIVE_NAME, TASK_ID
from tasks.synthonbench.core.nystrom_encoder import SynthonNystromEncoder


def build_synthon_task_spec(
    *,
    encoder: SynthonNystromEncoder | None,
    acquisition: AcquisitionSpec,
    proposal_samples: int,
    evaluations_per_round: int,
    bo_pool_size: int,
    bo_search_samples: int = 64,
    proposal_candidates_per_request: int,
    proposal_max_workers: int,
    slate_size: int,
    reaction_allocation: str,
    prompt_policy: str,
    search_method: str = "ldm",
    initialization_mode: str = "none",
    proposal_backend: str = "direct",
    harness_profile_count: int = 0,
) -> LDMTaskSpec:
    """Describe finite tuple, response, and method-specific acquisition semantics."""

    if encoder is None and search_method != "llm":
        raise ValueError("BO-based SynthonBench methods require a surrogate encoder")
    search_breadth = _search_breadth(
        search_method,
        proposal_samples,
        bo_search_samples,
        evaluations_per_round,
    )
    direct_batch_size = (
        proposal_candidates_per_request
        if proposal_backend == "direct" and search_method in {"ldm", "llm"}
        else 1
    )
    response_space = _response_space(direct_batch_size)
    return LDMTaskSpec(
        task=TASK_ID,
        candidate_domain=CandidateDomainSpec(
            name="Source-pinned synthon tuples",
            kind="reaction_synthon_tuple",
            dimension=None,
            representation="One reaction ID and one ordered valid synthon ID per official reaction slot.",
        ),
        objectives=(
            ObjectiveSpec(
                OBJECTIVE_NAME,
                "maximize",
                "Official fixed-oracle SynthonBench utility.",
            ),
        ),
        response_spaces=(response_space,),
        acquisition=acquisition,
        reservoir=_reservoir_spec(
            samples=search_breadth,
            workers=proposal_max_workers,
            method=search_method,
            initialization=initialization_mode,
            backend=proposal_backend,
            candidates_per_request=direct_batch_size,
            response_space=response_space.name,
        ),
        surrogate=encoder.describe() if encoder is not None else disabled_surrogate(),
        proposal_search=_proposal_search(
            search_method,
            search_breadth,
            proposal_max_workers,
            proposal_backend,
            harness_profile_count,
            direct_batch_size,
        ),
        metadata={
            "search_method": search_method,
            "initialization_mode": initialization_mode,
            "proposal_backend": proposal_backend,
            "proposal_samples": proposal_samples,
            "model_requests_per_round": (
                0
                if search_method == "bo"
                else None
                if proposal_backend == "harness"
                else math.ceil(search_breadth / direct_batch_size)
            ),
            "candidates_per_model_request": (
                None if proposal_backend == "harness" or search_method == "bo" else direct_batch_size
            ),
            "model_session_turns_per_round": (
                harness_profile_count if proposal_backend == "harness" else 0
            ),
            "search_breadth": search_breadth,
            "bo_pool_size": bo_pool_size,
            "bo_search_samples": bo_search_samples,
            "slate_size": None if proposal_backend == "harness" else slate_size,
            "reaction_allocation": None if proposal_backend == "harness" else reaction_allocation,
            "prompt_policy": (
                "task_local_agents" if proposal_backend == "harness" else prompt_policy
            ),
        },
    )


def build_direct_acquisition() -> AcquisitionSpec:
    return AcquisitionSpec(
        name="direct_llm_reservoir_order",
        objective_names=(OBJECTIVE_NAME,),
        score_direction="maximize",
        selection_rule="evaluate every admitted direct LLM candidate in reservoir order",
    )


def disabled_surrogate() -> SurrogateSpaceSpec:
    return SurrogateSpaceSpec(
        kind="none",
        representation="No surrogate for direct LLM baseline.",
        dimension_policy="none",
    )


def _reservoir_spec(
    *,
    samples: int,
    workers: int,
    method: str,
    initialization: str,
    backend: str,
    candidates_per_request: int,
    response_space: str,
) -> ReservoirSpec:
    expansions = []
    if initialization == "shared_random":
        expansions.append(ReservoirExpansionSpec(
            name="shared_random_initialization",
            action_kind="emit_candidate",
            response_space="synthon_tuple_json",
            produces_candidates=True,
            description="Deterministic product-uniform initial sample from the official space.",
        ))
    if method == "bo":
        description = "Score-blind random unseen product pool for base GP-UCB."
    elif backend == "harness":
        description = "Collect exact official-space tuples generated by persistent research sessions."
    else:
        requests = math.ceil(samples / candidates_per_request)
        description = (
            f"Launch {requests} independent requests with at most {workers} workers; "
            f"each request emits {candidates_per_request} candidate occurrences."
        )
    expansions.append(ReservoirExpansionSpec(
        name=f"{method}_search_expansion",
        action_kind="emit_candidate",
        response_space=response_space,
        produces_candidates=True,
        description=description,
    ))
    return ReservoirSpec(
        name="synthon_tuple_reservoir",
        expansions=tuple(expansions),
        candidate_validator="tasks.synthonbench.core.candidate:SynthonCandidateDomain",
        deduplication_key="official SynthonBench product_id",
        max_size=samples,
    )


def _proposal_search(
    method: str,
    breadth: int,
    workers: int,
    backend: str,
    harness_profile_count: int,
    candidates_per_request: int,
) -> ProposalSearchSpec:
    if method == "bo":
        return ProposalSearchSpec(
            name="score_blind_random_pool_bo",
            breadth=breadth,
            evaluation_policy="nystrom_tanimoto_gp_ucb",
            parameters={"model_requests": 0},
        )
    if method == "llm":
        request_count = math.ceil(breadth / candidates_per_request)
        return ProposalSearchSpec(
            name="parallel_independent_direct_llm",
            breadth=breadth,
            evaluation_policy="reservoir_order_direct_evaluation",
            parameters={
                "request_count": request_count,
                "candidates_per_request": candidates_per_request,
                "max_workers": min(workers, request_count),
            },
        )
    if backend == "harness":
        if harness_profile_count < 1 or breadth % harness_profile_count:
            raise ValueError("harness breadth must divide evenly across profiles")
        return ProposalSearchSpec(
            name="persistent_parallel_research_sessions",
            breadth=breadth,
            evaluation_policy="empirical_q0_maintained_acquisition_tilted",
            parameters={
                "profile_count": harness_profile_count,
                "candidates_per_session": breadth // harness_profile_count,
                "persistent_sessions": True,
                "skills_loaded": False,
                "agent_selects_reaction": True,
                "candidate_source": "structured_official_synthon_space_tools",
            },
        )
    request_count = math.ceil(breadth / candidates_per_request)
    return ProposalSearchSpec(
        name="parallel_independent_minibatch_requests",
        breadth=breadth,
        evaluation_policy="empirical_q0_maintained_acquisition_tilted",
        parameters={
            "request_count": request_count,
            "candidates_per_request": candidates_per_request,
            "max_workers": min(workers, request_count),
        },
    )


def _search_breadth(
    method: str,
    proposal_samples: int,
    bo_search_samples: int,
    evaluations_per_round: int,
) -> int:
    if method == "bo":
        return bo_search_samples
    return evaluations_per_round if method == "llm" else proposal_samples


def _response_space(candidates_per_request: int) -> ResponseSpaceSpec:
    tuple_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["reaction_id", "synthon_ids"],
        "properties": {
            "reaction_id": {"type": "string"},
            "synthon_ids": {
                "type": "array",
                "minItems": 1,
                "items": {"type": "integer"},
            },
        },
    }
    if candidates_per_request > 1:
        batch_item = {
            **tuple_schema,
            "required": ["proposal_index", "reaction_id", "synthon_ids"],
            "properties": {
                "proposal_index": {"type": "integer", "minimum": 0},
                **tuple_schema["properties"],
            },
        }
        return ResponseSpaceSpec(
            name="synthon_tuple_batch_json",
            output_kind="json_object",
            parser="tasks.synthonbench.core.proposal_parsing:parse_synthon_batch_response",
            description=(
                f"Exactly {candidates_per_request} source-valid reaction-component tuples, "
                "one for each supplied proposal slot."
            ),
            schema={
                "type": "object",
                "additionalProperties": False,
                "required": ["candidates"],
                "properties": {
                    "candidates": {
                        "type": "array",
                        "minItems": candidates_per_request,
                        "maxItems": candidates_per_request,
                        "items": batch_item,
                    }
                },
            },
        )
    return ResponseSpaceSpec(
        name="synthon_tuple_json",
        output_kind="json_object",
        parser="tasks.synthonbench.core.proposal_parsing:parse_synthon_response",
        description="One complete source-valid reaction-component tuple from the official SynthonSpace.",
        schema=tuple_schema,
    )


__all__ = ["build_direct_acquisition", "build_synthon_task_spec", "disabled_surrogate"]
