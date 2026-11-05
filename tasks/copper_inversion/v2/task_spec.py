"""One fusion representation and one cross-case program for copper v2."""
from ldm_tts.contracts import (
    CandidateDomainSpec, LDMTaskSpec, ObjectiveSpec, ReservoirExpansionSpec,
    ReservoirSpec, ResponseSpaceSpec, ProposalSearchSpec,
)
from ..core.encoding import encoding_spec
from ..core.surrogate import LossSelector


def describe_task(suite_digest):
    if len(suite_digest)!=64 or any(c not in "0123456789abcdef" for c in suite_digest):
        raise ValueError("frozen training suite SHA256 required")
    return LDMTaskSpec(
        task="copper_inversion",
        candidate_domain=CandidateDomainSpec(name="inversion_algorithm",kind="python_source",dimension=None,
            representation="One immutable solve(problem, budget, seed) program for every case"),
        objectives=(ObjectiveSpec(name="loss_total",direction="minimize",
            description="Parent-macro independent measurement and geological-prior loss L/(1+L); failed solves count as one; IoU is diagnostic"),),
        response_spaces=(ResponseSpaceSpec(name="solver_artifact",output_kind="json",
            description="Sealed solver sources with parent and research rationale"),),
        acquisition=LossSelector("fusion").describe(),
        reservoir=ReservoirSpec(name="solver_reservoir",expansions=(ReservoirExpansionSpec(
            name="design_solver",action_kind="emit_candidate",response_space="solver_artifact",
            produces_candidates=True,description="Design one algorithm on aggregate training feedback"),),
            candidate_validator="Read-only v2 interface fixtures",deduplication_key="SHA256 of source bytes"),
        surrogate=encoding_spec("fusion"),
        proposal_search=ProposalSearchSpec(name="shared_research_tree"),
        metadata=dict(problem_schema="problem_schema_v2",geometry="occupancy_v2",objective="objective_v2",
            suite_digest=suite_digest,encoding="fusion",block_weights=[.5,.5],
            outer_objective="parent_macro_confirmation_loss",solver_objective="public_data_and_geological_priors"),
    )
