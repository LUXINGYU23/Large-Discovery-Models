---
name: experimental-design
description: Design interpretable chemical candidate panels with matched controls and identifiable factor contrasts. Use when multiple changes confound an effect or when choosing between refinement and a competitive alternative.
license: MIT
metadata:
  source: K-Dense-AI/scientific-agent-skills
  source-commit: 1e5eeffbdad3749125afe7ab48a39694e27f181c
---

# Informative Chemical Panels

Define the hypothesis, a rival explanation, the intended changes, and what
measured result would distinguish them. Retrieve exact measured candidate IDs
and their original rationales through `get_measured_history`, using detailed
records only for the comparisons you need.

Compare exact reaction and ordered-synthon tuples. Keep other slots fixed for
a matched substitution and inspect their public structures; an identifier
change alone does not quantify the chemical change. Different reactions can
produce closely related chemistry, so assess mechanisms as well as labels.

Small factorial layouts can separate supported main effects and interactions.
The guest provides `pyDOE3.ff2n(k)` and `fracfact`. Map coded levels to exact
legal task values; check rank, balance, and aliasing on the actual candidate
panel, not only its coded design. A matched variant of an already measured
candidate often answers the question more cheaply than a full factorial.
Do not require a factorial or a package call every round.

A damaged leader is a control, not a competitive alternative. Give unresolved
alternatives plausible implementations and comparisons that can distinguish
a weak construct from a weak hypothesis. Allocate refinement, alternatives,
and controls from current evidence and remaining time, not fixed quotas.

The downstream selector may evaluate only part of the panel. Inspect the
measured subset before fitting interactions or claiming a complete contrast.
Unmeasured proposals remain open experiments, not failed experiments.
Identical candidate occurrences are not independent replicates; the task's
existing measurement is the evidence. Retain pre-evaluation rationales,
including contradicted hypotheses, without adding extra submission fields.
