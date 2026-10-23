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
    from .surrogate import CodeEncoder, LossSelector

    guided = args.search_method == "ldm_harness"
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
                description="Array of sealed Python algorithms, parents and research rationales",
            ),
        ),
        acquisition=LossSelector(args.encoding).describe() if guided else AcquisitionSpec(
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
        surrogate=CodeEncoder(args.out_dir, args.encoding).describe() if guided else SurrogateSpaceSpec(
            kind="none",
            representation="Single-agent first stage",
            dimension_policy="none",
        ),
        proposal_search=ProposalSearchSpec(name="shared_research_tree" if args._research_strategy.requires_main else "persistent_sessions"),
        metadata={
            "case_id": args.case_id,
            "mock": args.mock,
            "research_strategy": args.harness_research_strategy_identity,
            "search_method": args.search_method,
            "objective_version": "objective_v1",
        },
    )
