"""Runtime-faithful science and search contract."""

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


def describe_ldm_task(args=None):
    attempts = getattr(args, "attempts_per_sample", 1)
    tool_mode = getattr(args, "proposal_format", "cif") == "operations"
    response_space = "operation_plan" if tool_mode else "cif_answer"
    return LDMTaskSpec(
        task="atomworld",
        candidate_domain=CandidateDomainSpec(
            "atomworld_answer",
            "structured_text",
            None,
            "Public sample ID plus raw model output containing a complete CIF",
            constraints={
                "max_output_chars": 1000000,
                "malformed_cif": "officially scored as incorrect",
            },
        ),
        objectives=(
            ObjectiveSpec(
                "correct",
                "maximize",
                "Official per-answer correctness; hidden from refinement and final-answer choice",
            ),
        ),
        response_spaces=(
            ResponseSpaceSpec(
                response_space,
                "json" if tool_mode else "text",
                parser="json.loads + atomworld_tools.execute_operations"
                if tool_mode
                else "tasks.atomworld.core.proposals.extract_cif",
                description="Bounded geometry operation list produces a scored CIF"
                if tool_mode
                else "Last <cif>...</cif> block, as in the official evaluator",
            ),
        ),
        acquisition=AcquisitionSpec(
            "chronological_submission",
            ("correct",),
            "maximize",
            "Evaluate the sole submitted answer; final answer is the last scheduled attempt, independent of score",
        ),
        reservoir=ReservoirSpec(
            "cif_submissions",
            (
                ReservoirExpansionSpec(
                    "answer", "emit_candidate", response_space, True
                ),
                ReservoirExpansionSpec(
                    "blind_self_review",
                    "edit_candidate",
                    response_space,
                    True,
                    description="Original question, previous draft and public syntax feedback only",
                ),
            ),
            "tasks.atomworld.core.proposals.AtomWorldDomain",
            "sha256(sample_id + canonical answer)",
            max_size=1,
        ),
        surrogate=SurrogateSpaceSpec(
            "none", "No surrogate: target-blind chronological refinement", "none"
        ),
        proposal_search=ProposalSearchSpec(
            "blind_refinement",
            breadth=1,
            depth=attempts,
            beam_width=1,
            evaluation_policy="each_submission",
        ),
        metadata={
            "feedback_policy": "no targets, judge scores, judge errors, oracle parent or cross-question answers",
            "comparison": "one-shot and extended compute reported separately",
        },
    )
