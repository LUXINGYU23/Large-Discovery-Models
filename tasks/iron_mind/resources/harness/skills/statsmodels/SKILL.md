---
name: statsmodels
description: Diagnose identifiable chemical factor or descriptor associations in measured history. Use explanatory models for a concrete contrast and confounding checks, not sparse initialization or mandatory per-round fitting.
license: MIT
metadata:
  source: K-Dense-AI/scientific-agent-skills
  source-commit: 1e5eeffbdad3749125afe7ab48a39694e27f181c
---

# Quantitative Checks on Measured Chemistry

Use one row per measured candidate ID. Join labels by exact identity, not
position, approximate names, annotations, or occurrence count. Omit failed
evaluations with missing scores from numerical fits; never replace them with
zero. Local models guide research, not the framework GP or official evaluator.

Start with descriptive summaries and matched comparisons. Skip fitting when
the sampled levels, reactions, or backgrounds cannot identify the effect.
Use a few schema-defined factors and supported interactions, checking coverage
within substrate and reagent backgrounds before interpreting level effects.
The guest provides statsmodels/patsy, pandas, NumPy/SciPy and scikit-learn.

Inspect constant columns, design rank, feature correlations and influential
points. Formula models include an intercept; use `C(factor)` for categorical
coding, and introduce interactions only when observations support them.
One-hot groups plus intercepts and highly correlated descriptors need reference
coding or regularization. Fit statistics are not causal validation.

Use a small regularized scikit-learn pipeline for predictive ranking. Fit
scaling and feature selection on training rows only. Hold out later rounds for
forward prediction or entire reaction/scaffold/background groups for transfer;
random row splits can leak near-identical candidates. With too few independent
groups, report the diagnostic as unstable instead of fabricating a score.

Compare against a simple baseline on identical held-out rows. Inspect residuals
and errors by family, not only in-sample R-squared. Adaptive selection and
shared chemistry are not repaired by nominal p-values or robust standard errors.
Do not remove a legitimate high scorer merely to improve a regression fit.
Keep exact split IDs, computed features and scripts in the workspace.
