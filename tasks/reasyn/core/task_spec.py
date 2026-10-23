"""Declarative ReaSyn candidate, objective, proposal and selection contract."""

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
from .metrics import ORACLES
from .surrogate import MoleculeEncoder


def describe_ldm_task(args):
    recon = args.benchmark == "reconstruction"
    objective = "similarity" if recon else "oracle_score"
    direct = args.search_method in ("llm", "harness", "baseline")
    return LDMTaskSpec(
        task="reasyn",
        candidate_domain=CandidateDomainSpec(
            name="ReaSyn projection queries"
            if recon
            else "ReaSyn synthesis-verified molecules",
            kind="molecular_projection_query" if recon else "synthesizable_molecule",
            dimension=None,
            representation="Canonical target SMILES and controlled restart seed"
            if recon
            else "Canonical product SMILES with replay-verified stock/reaction trace",
            constraints={"benchmark": args.benchmark, "mock": args.mock},
        ),
        objectives=(
            ObjectiveSpec(
                objective,
                "maximize",
                "Morgan r2/4096 similarity to original target"
                if recon
                else "Released TDC oracle score",
            ),
        ),
        response_spaces=(
            ResponseSpaceSpec(
                name="projection_targets_json",
                output_kind="json",
                schema={
                    "type": "object",
                    "required": ["candidates"],
                    "properties": {
                        "candidates": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "required": ["target_smiles"],
                                "properties": {"target_smiles": {"type": "string"}},
                                "additionalProperties": False,
                            },
                        }
                    },
                    "additionalProperties": False,
                },
                parser="tasks.reasyn.core.proposals:parse_targets",
            ),
        ),
        acquisition=AcquisitionSpec(
            name="direct_proposal_order" if direct else "empirical_q0_tanimoto_gp_ucb" if args.search_method.startswith("ldm") else "tanimoto_gp_ucb",
            objective_names=(objective,),
            score_direction="maximize",
            selection_rule="Empirical group q0 with alpha*log(group_q0+epsilon)-log(group_trial_count)+eta*robust_z(UCB) sampling" if args.search_method.startswith("ldm") else "GP-UCB" if args.search_method == "bo" else "Direct proposal order",
            parameters={
                "beta": args.acquisition_beta,
                "history_limit": args.gp_history_limit,
                "alpha": args.acquisition_alpha, "eta": args.acquisition_eta,
                "bo_pool_size": args.bo_pool_size,
            },
        ),
        reservoir=ReservoirSpec(
            name="projection_query_reservoir"
            if recon
            else "projected_product_reservoir",
            expansions=(
                ReservoirExpansionSpec(
                    name="feedback_conditioned_targets",
                    action_kind="emit_candidate",
                    response_space="projection_targets_json",
                    produces_candidates=True,
                    description="LDM proposes targets; frozen ReaSyn supplies actual valid synthesis pathways",
                ),
            ),
            candidate_validator="tasks.reasyn.core.candidate:ReaSynDomain",
            deduplication_key="Canonical query plus controlled restart seed"
            if recon
            else "Canonical isomeric product SMILES SHA256",
            max_size=args.reservoir_size,
        ),
        surrogate=SurrogateSpaceSpec(
            kind="none",
            representation="Direct proposal order without a surrogate",
            dimension_policy="none",
        ) if direct else MoleculeEncoder(mock=args.mock).describe(),
        proposal_search=ProposalSearchSpec(
            name="independent_minibatch_empirical_q0" if args.search_method.startswith("ldm") else args.search_method,
            breadth=args.reservoir_size,
            evaluation_policy="gp_ucb_selected_frozen_projection"
            if recon
            else "gp_ucb_selected_tdc_oracle",
        ),
        metadata={
            "benchmark": args.benchmark,
            "policy_capabilities": ["prior_mean@1", "ldm_weights@1"],
            "alpha_configurable": True,
            "mock": args.mock,
            "proposal_mode": args.proposal_mode,
            "search_method": args.search_method,
            "official_suite_oracles": list(ORACLES),
        },
    )
