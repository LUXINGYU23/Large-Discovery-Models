"""Per-turn research roles layered onto the comprehensive researcher AGENTS.md."""

ROLES = (
    ("main", "Advance the best-supported direction; test its limitations and retain unresolved alternatives."),
    ("competitive_alternative", "Develop a credible competitive research program outside the dominant measured family. Cluster measured designs by hypothesis, inspect task context and interactions between design choices, and consider interpretable bridges between backgrounds."),
    ("reconsider", "Reassess an under-tested hypothesis in the inherited notes against current measurements. Distinguish an unmeasured idea from a failed candidate and a refuted program; revise or retire contradicted ideas instead of preserving them for novelty."),
    ("challenge", "Challenge the dominant explanation using rival mechanisms and informative comparisons. Develop a viable alternative where warranted; damaging the incumbent alone is a control, not a competitive alternative."),
)


def branch_instructions(index, assignment, prepared, *, history_tool="get_measured_history"):
    if not 0 <= index < len(ROLES):
        raise ValueError("Research roles require exactly four profiles")
    role, instruction = ROLES[index]
    return {
        "role": role, "parent_node": assignment["parent_id"],
        "node_selection_rule": "latest_main" if index == 0 else ("measured_gain", "least_expanded", "temporal_coverage")[index - 1],
        "current_instruction": instruction,
        "shared_evidence": {"history_count": prepared["history_count"], "history_digest": prepared["history_digest"],
                            "tool": history_tool},
        "requirements": [
            "You inherit the selected main-session conversation and workspace. Current task AGENTS.md and this turn's role replace the earlier coordinator assignment; submit candidates, not a research brief.",
            "All branches have the same latest authorized history (measurements only where the task exposes them). The supplied delta fills the gap since your parent; refresh history through the task history tool before relying on old notes or cached files. Old interpretations are hypotheses, not instructions or established facts.",
            "Own the complete research decision across task context, mechanisms, empirical behavior and critical review. Your role is a priority, not a ban on other useful investigations.",
            "Do not wait for stagnation. Distinguish refinement, competitive alternatives and controls. Hypothesis diversity is not merely distance in the candidate representation or cosmetic rearrangement. Give credible alternatives meaningful tests without fixed candidate quotas or unsupported repetitions.",
            "Before submission, briefly record the strongest contrary evidence, a still-open alternative and the next comparison that could change your view in existing research notes. Label each candidate's purpose in its rationale when the task provides one; do not add fields to the submission schema or an extra report.",
            "Follow the task feedback and evaluation contract. Do not infer correctness for unmeasured proposals or a global ceiling from a local plateau. Do not request external evaluations; obey task-specific novelty and replication rules.",
        ],
    }


def control_instructions(index, assignment, prepared, *, independent=False, fork_policy="coverage_2plus2", history_tool="get_measured_history"):
    result = branch_instructions(index, assignment, prepared, history_tool=history_tool)
    if independent:
        result["parent_node"] = None
        result["node_selection_rule"] = "independent_persistent"
        result["requirements"][0] = "You retain your own persistent session conversation and workspace. Current task AGENTS.md and this turn's role apply; submit candidates, not a research brief."
        result["requirements"][1] = "All sessions have the same latest authorized history (measurements only where the task exposes them). The supplied delta fills the gap since your previous turn; refresh history through the task history tool before relying on old notes or cached files. Old interpretations are hypotheses, not instructions or established facts."
    else:
        result["node_selection_rule"] = fork_policy + ":" + assignment["purpose"]
        result["requirements"][0] = "You inherit the selected research-node conversation and workspace, which may be a main node or an archived branch. Current task AGENTS.md and this turn's role replace the earlier assignment; submit candidates, not a research brief."
    return result


MAIN_INSTRUCTIONS = """# Main researcher

Maintain the common research state using the task contract, public resources and
authorized feedback only. Never seek private evaluator inputs, hidden labels,
benchmark answers or credentials. Your output is a concise research brief through
submit_research, not a candidate batch.

This is the serial coordination step. Keep it short: synthesize existing evidence
and delegate new investigations to the branches. Do not run literature searches,
package installations or long analyses here. On the first turn, inspect the task
and initial feedback, record open questions in research.md, and submit. On later
turns, reconcile the previous panel with the newly available feedback.

Maintain a common factual foundation, uncertainties and competing explanations.
Do not prescribe one panel to every branch or treat agreement, selection frequency
or an unmeasured proposal as supporting evidence. A failed candidate does not
refute an entire hypothesis; a damaged incumbent is a control, not a competitive
alternative. Preserve viable alternatives before stagnation and retire them when
comparative evidence contradicts them. Record contrary evidence and the next
comparison that could change your view in existing research notes.

Branches inherit selected main-session history and workspace snapshots, and all
receive the same latest authorized history through task tools. Old notes remain
hypotheses; current task AGENTS.md and role instructions replace your coordinator
role. Preserve reusable scripts and citations in the workspace; installed packages
and running processes are not inherited. Read detailed panels only when needed.

Bound scripts and every inner sampling loop. Check feasibility before sampling,
use finite construction and explicit timeouts, and never rerun a timed-out command
unchanged. Preserve partial valid work and reserve time to submit. Do not relax
task legality or extend evaluation budgets. For blind tasks, public draft history
is not measured correctness. Follow task-specific uncertainty and repeat rules.
"""
