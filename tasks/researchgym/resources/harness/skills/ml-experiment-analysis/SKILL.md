---
name: ml-experiment-analysis
description: Interpret noisy per-cell benchmark measurements, separate real effects from seed noise, and plan informative method variants. Use when deciding what to change next from measured history.
---

# Analyze ML Method Measurements

## Read the measurement correctly

- Each objective is an average over cells (datasets, corruptions, or seeds).
  Load per-cell values from the history export and compare cell by cell; a
  gain on easy cells can hide a loss on hard ones.
- Training runs are noisy. Seed-to-seed spread in reinforcement learning or
  small-epoch fine-tuning can exceed the difference between two methods.
  Treat a single small improvement as weak evidence; look for consistency
  across cells and across related programs.
- A failed or timed-out candidate is information about the interface or the
  compute budget, not about the method's quality.

## Plan the next variants

- State the mechanism a variant tests and what result would refute it.
- Prefer variants that differ from a measured program in one identifiable
  mechanism when the goal is to learn; combine mechanisms when several have
  independently consistent support.
- Include a credible alternative family when all measured programs share one
  idea and progress has stalled; do not abandon a supported family because of
  one noisy result.
- Record the hypothesis and the compared measured candidate IDs in each
  entry's `rationale` and `comparison_candidate_ids`.

## Guard against self-deception

- CPU sanity checks and literature results are not measurements of this case.
- Do not reuse a public method's reported number as evidence for this
  sub-protocol; the case's metric, data, and budget differ from papers.
- Keep conclusions proportional to the number of measured programs.
