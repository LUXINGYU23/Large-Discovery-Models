---
name: researchgym-method-research
description: Research, implement, statically check, and package candidate method programs for a ResearchGym case slot. Use every turn before writing candidates.json.
---

# ResearchGym Method Research

## Understand the slot

Call `describe_researchgym_case`. Its `interface` is the exact contract the
case code calls; `scoring` states how the official grader turns runs into the
objective; `rules` lists what invalidates a run; `static_rules` lists the
admission checks (character and line limits, forbidden imports and calls).
`guest_file.path` contains the same contract plus pinned public reference
sources. Read the reference files that define the objects your program
receives (for example the base learner, the model wrapper, or the training
loop) before writing code.

Every evaluation runs the case's full job matrix (`job_matrix`). One crashed
job, missing output, or timeout invalidates the whole candidate, so robustness
across all cells is part of the objective.

## Use measured history

- `get_measured_history` with `sort_by: "objective", order: "desc", detail: "detailed"`
  shows the best programs and their per-cell metrics (`metrics["cell.*"]`).
- Filter `status: "failed"` or `"timed_out"` and read `error` before reusing a
  design; interface errors usually repeat across variants.
- The `guest_file` export holds every matching detailed record; load it with
  Python instead of copying text.
- Per-cell metrics show where a method helps or hurts; a mean hides trade-offs.

## Implement and test

- Start from the interface and a measured program when one exists; change one
  mechanism at a time when you need an interpretable result, or combine
  supported mechanisms when evidence is strong.
- Exercise the code on CPU with small synthetic tensors that mimic the
  interface shapes: forward passes, gradient steps, empty or degenerate
  batches, NaN guards, and device moves written as `.to(device)`.
- Keep per-call work vectorized; the case's wall time is fixed.

## Package

Write each program to its own file, then run:

```bash
python <skill-dir>/scripts/build_candidates.py \
  --out candidates.json \
  --program a.py --summary "what changed" --rationale "why it should help" \
  --program b.py --summary "..." --rationale "..." --compare rg-0123456789abcdef
```

`<skill-dir>` is the directory of this SKILL.md as advertised by Pi, normally
`/workspace/.ldm-resources/skills/0/researchgym-method-research`; the snapshot
is read-only, so write outputs in `/workspace`.
Then call `check_candidate_programs` with `artifact_path: "candidates.json"`
and fix every indexed error before `submit_candidates`.
