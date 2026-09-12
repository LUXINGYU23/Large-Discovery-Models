---
name: experimental-design
description: Build interpretable sequence panels with matched controls, factorial contrasts, and competing regulatory programs. Use when a candidate should test a mechanism or when several changes could confound the interpretation of a score.
license: MIT
metadata:
  runtime: Python 3.11 with pyDOE3 1.6.2, NumPy, and pandas is preinstalled in the research guest.
  source: K-Dense-AI/scientific-agent-skills
  source-commit: 1e5eeffbdad3749125afe7ab48a39694e27f181c
---

# Design Informative Sequence Panels

Choose designs that can improve the campaign objective and resolve an important
uncertainty. A useful panel may refine a supported program, develop a credible
alternative, or discriminate between explanations. Choose effort from evidence
and remaining time; this skill does not impose slot quotas or round schedules.

## Define the Contrast

Before building sequences, state the proposed mechanism, a rival explanation,
the variables that differ, and what outcome would distinguish them. Name the
measured candidate ID when using an existing control.

For a matched sequence contrast:

1. Retrieve the named parent's exact bases with `get_sequence_window`, using
   its measured candidate ID, and verify the returned checksum in analysis code.
2. Copy that exact sequence. Modify the declared feature while holding unrelated
   positions and relevant module context constant.
3. Compute the complete base-level difference between parent and variant.
   Check motif hits on both strands, background composition, and overlaps.
4. Describe all material differences in the rationale. A redesigned background
   is not a matched motif ablation, even if the design labels differ by one factor.

A control that damages the leading program is not a competitive alternative
program. Give an alternative a plausible implementation and a comparison that
can distinguish a weak implementation from a weak hypothesis.

Check that construction requirements are mutually consistent before expanding
a panel. Requiring a motif in an inserted module while forbidding it anywhere
in the final sequence is infeasible. Hold useful native background fixed when
possible, build one interpretable contrast first, and save valid designs
incrementally. A difficult control should not prevent unrelated hypotheses
from being submitted. If a diagnostic contradicts the intended contrast,
revise that design or its interpretation and document the discrepancy.

## Factorial Designs

For a few interacting features, a full factorial can test main effects and
interactions. Generate coded layouts with the installed `pyDOE3`, then map
each row to a valid sequence on controlled backgrounds:

```python
import pandas as pd
from pyDOE3 import ff2n

design = pd.DataFrame(ff2n(3), columns=["module_a", "module_b", "background"])
assert len(design.drop_duplicates()) == 8
```

The -1/+1 columns are design levels, not mutation positions or scores. Define
their sequence-level meaning before implementation. Factorial size grows as
2**k; only use factors for which the allocated panel can support a useful test.
Fractional designs (`fracfact`) trade samples for aliasing: specify which main
effects and interactions are confounded before interpreting them. Inspect rank
and factor balance after sequence construction, not just in the coded matrix.

Use a factorial only when its contrasts are identifiable and its sequences are
individually useful proposals. A matched variant of an already measured parent
often answers the question with less construction and analysis. Loading this
Skill does not require using pyDOE3 or building a factorial every turn.

The downstream selector may measure only part of the proposed panel. Check the
actual measured subset before claiming a complete contrast or fitting all
factor interactions. Unmeasured rows remain open experiments; selection is not
evidence for or against their quality.

## Evidence Units and Adaptation

This campaign uses a deterministic evaluator, not a wet-lab randomized trial.
Repeated occurrences of a sequence are not independent measurements and do not
provide replication, extra precision, or a new biological background. Reuse
its existing measurement when making a contrast; obey the turn's novelty
contract when proposing candidates.

Use multiple relevant backgrounds to probe whether an effect transfers. Vary
one controlled factor or a declared factorial set within each background.
Adapt after observations, retain pre-evaluation rationales, and distinguish
exploratory hypotheses from results of planned comparisons. A single poor
implementation does not refute every implementation of a program, and many
nearby winners do not rule out programs that have barely been tested.

Keep layouts and checks in scratch analysis files. Submit only the task's
candidate schema; no extra design fields or separate approval workflow is needed.
