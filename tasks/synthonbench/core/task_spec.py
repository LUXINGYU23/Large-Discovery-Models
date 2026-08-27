"""Declarative SynthonBench task contracts for each comparison method."""

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
    search_breadth = _search_breadth(
        search_method,
        proposal_samples,
        bo_search_samples,
        evaluations_per_round,
    )
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
        response_spaces=(_response_space(),),
        acquisition=acquisition,
        reservoir=_reservoir_spec(
            samples=search_breadth,
            workers=proposal_max_workers,
            method=search_method,
            initialization=initialization_mode,
        ),
        surrogate=encoder.describe() if encoder is not None else disabled_surrogate(),
        proposal_search=_proposal_search(search_method, search_breadth, proposal_max_workers),
        metadata={
            "search_method": search_method,
            "initialization_mode": initialization_mode,
            "proposal_samples": proposal_samples,
            "model_requests_per_round": (
                0 if search_method == "bo" else search_breadth
            ),
            "search_breadth": search_breadth,
            "bo_pool_size": bo_pool_size,
            "bo_search_samples": bo_search_samples,
            "slate_size": slate_size,
            "reaction_allocation": reaction_allocation,
            "prompt_policy": prompt_policy,
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
    description = (
        "Score-blind random unseen product pool for base GP-UCB."
        if method == "bo"
        else f"Launch {samples} independent one-candidate requests with at most {workers} workers."
    )
    expansions.append(ReservoirExpansionSpec(
        name=f"{method}_search_expansion",
        action_kind="emit_candidate",
        response_space="synthon_tuple_json",
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
) -> ProposalSearchSpec:
    if method == "bo":
        return ProposalSearchSpec(
            name="score_blind_random_pool_bo",
            breadth=breadth,
            evaluation_policy="nystrom_tanimoto_gp_ucb",
            parameters={"model_requests": 0},
        )
    if method == "llm":
        return ProposalSearchSpec(
            name="parallel_independent_direct_llm",
            breadth=breadth,
            evaluation_policy="reservoir_order_direct_evaluation",
            parameters={"max_workers": workers, "one_candidate_per_request": True},
        )
    return ProposalSearchSpec(
        name="parallel_independent_requests",
        breadth=breadth,
        evaluation_policy="empirical_q0_maintained_acquisition_tilted",
        parameters={"max_workers": workers, "one_candidate_per_request": True},
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


def _response_space() -> ResponseSpaceSpec:
    return ResponseSpaceSpec(
        name="synthon_tuple_json",
        output_kind="json_object",
        parser="tasks.synthonbench.core.proposal_parsing:parse_synthon_response",
        description=(
            "One complete source-valid reaction-component tuple chosen from a supplied "
            "finite public synthon slate."
        ),
        schema={
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
        },
    )


__all__ = ["build_direct_acquisition", "build_synthon_task_spec", "disabled_surrogate"]
