from ldm_tts.contracts import (
    AcquisitionSpec,
    CandidateDomainSpec,
    LDMTaskSpec,
    ObjectiveSpec,
    ReservoirExpansionSpec,
    ReservoirSpec,
    ResponseSpaceSpec,
    ProposalSearchSpec,
    SurrogateSpaceSpec,
)


def describe_ldm_task(args):
    return LDMTaskSpec(
        task="copper_inversion",
        candidate_domain=CandidateDomainSpec(
            name="inversion_algorithm",
            kind="python_source",
            dimension=None,
            representation="Immutable solver.py defining solve(problem, budget, seed)",
        ),
        objectives=(
            ObjectiveSpec(
                name="loss_total",
                direction="minimize",
                description="Independent objective_v1 geometry fit",
            ),
        ),
        response_spaces=(
            ResponseSpaceSpec(
                name="solver_artifact",
                output_kind="json",
                description="One sealed Python artifact, parent and research rationale",
            ),
        ),
        acquisition=AcquisitionSpec(
            name="reservoir_order",
            objective_names=("loss_total",),
            score_direction="minimize",
            selection_rule="Evaluate the sole admitted candidate",
        ),
        reservoir=ReservoirSpec(
            name="solver_reservoir",
            expansions=(
                ReservoirExpansionSpec(
                    name="design_solver",
                    action_kind="emit_candidate",
                    response_space="solver_artifact",
                    produces_candidates=True,
                    description="Persistent agent designs full inversion algorithm",
                ),
            ),
            candidate_validator="Code snapshot + read-only interface fixtures",
            deduplication_key="SHA256 of UTF-8 source bytes",
        ),
        surrogate=SurrogateSpaceSpec(
            kind="none",
            representation="Single-agent first stage",
            dimension_policy="none",
        ),
        proposal_search=ProposalSearchSpec(name="single_turn"),
        metadata={
            "case_id": args.case_id,
            "mock": args.mock,
            "research_strategy": "independent",
            "objective_version": "objective_v1",
        },
    )
