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
from tasks.nucleobench.core.constants import TASK_ID


def build_task_spec(case: NucleoBenchCase) -> LDMTaskSpec:
    """Describe the shared mutation-patch search used by every execution profile."""

    response_name = "mutation_patch_json"
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
        response_spaces=(
            ResponseSpaceSpec(
                name=response_name,
                output_kind="json_object",
                description="One non-empty mutation patch relative to the paired start.",
                schema={
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["mutations"],
                    "properties": {
                        "mutations": {
                            "type": "array",
                            "minItems": 1,
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "required": ["position", "base"],
                                "properties": {
                                    "position": {"type": "integer", "minimum": 0},
                                    "base": {
                                        "type": "string",
                                        "enum": ["A", "C", "G", "T"],
                                    },
                                },
                            },
                        }
                    },
                },
            ),
        ),
        acquisition=AcquisitionSpec(
            name="hamming_gp_ucb_ldm_tilt",
            objective_names=("utility",),
            score_direction="maximize",
            selection_rule="Empirical q0 tilted by task-local Hamming GP-UCB.",
        ),
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
        ),
        surrogate=SurrogateSpaceSpec(
            kind="kernel",
            representation="Sparse categorical Hamming similarity over edited positions.",
            dimension_policy="implicit",
        ),
        proposal_search=ProposalSearchSpec(
            name="method_specific_mutation_search",
            evaluation_policy="official model evaluation through the shared LDM engine",
        ),
        metadata={
            "case_id": case.case_id,
            "case_state": case.state,
            "model_family": case.model_family,
            "model_name": case.model_name,
            "target": case.target,
            "max_seconds": case.max_seconds,
        },
    )


__all__ = ["build_task_spec"]
