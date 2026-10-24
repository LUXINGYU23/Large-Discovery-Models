# AlphaBench T3 factor research

The current turn supplies the market, backend, objective, required candidate count and new evaluated search observations. Use `mcp__t3__get_contract` for the authoritative operator and field contract. Search history tools expose only evaluated search observations. Static validation is free; `mcp__t3__check_expression` is a paid backend check. A GP query tool is available only in the named LDM query variant and reports a frozen start-of-round posterior, not an Oracle measurement.

Write `/workspace/candidates.json` as a UTF-8 object containing only a `candidates` array. Each element has a short `name`, an `expression`, and optionally a `research_note`. Submit `candidates.json` with `submit_candidates`. The array must contain exactly `candidate_count` distinct canonical expressions that have never been evaluated, including failed earlier evaluations. Repair any indexed rejection and resubmit the complete array.

Research only the supplied search history and public operator contract. Do not seek validation/test labels, Host files, market archives or provider credentials.


## Shared research roles and reflection

When the current turn supplies research_branch, adopt that role and output contract
instead of the inherited coordinator assignment. You remain responsible for the
whole task; the role is a research priority, not a restriction to one specialty.
Use task-specific tools and skills for exact constraints, representations and
feedback. All branches receive the same latest authorized history even when their
conversation and workspace come from an older main-session node. Refresh that
history before relying on inherited notes; old interpretations are hypotheses.

Distinguish refinement, competitive alternatives and controls. Seek alternatives
before a plateau; cosmetic edits or representation distance alone do not establish
hypothesis diversity. Do not infer support from selection frequency or agreement
between branches. An unmeasured proposal is not a failure, and one failed candidate
does not refute its whole hypothesis. Retire ideas when comparative evidence
contradicts them. A damaged incumbent is a control, not a competitive alternative.

Before submission, briefly record the strongest contrary evidence, a still-open
alternative, and the next comparison that could change your view in existing
research notes. Explain each candidate's purpose in its rationale field when the task provides one;
do not add a separate report or change the task's submission schema. Never invent
measurements or labels; public drafts in blind tasks are not correctness feedback.

Check feasibility before running constrained generators. Bound every randomized
search and inner repair loop, including rejected draws. Use explicit short tool
timeouts, preserve partial valid work, and never repeat a timed-out command
unchanged. Optional analysis must not prevent a complete, legal submission.
