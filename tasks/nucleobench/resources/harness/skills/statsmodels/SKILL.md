---
name: statsmodels
description: Diagnose a specific feature contrast in measured history when enough distinct constructs identify it. Use small explanatory models for associations and confounding, not one-point initialization or routine per-turn fitting; use scikit-learn for predictive validation.
license: MIT
metadata:
  runtime: Python 3.11 with statsmodels 0.14.6, patsy, NumPy, SciPy, pandas, and scikit-learn is preinstalled in the research guest.
  source: K-Dense-AI/scientific-agent-skills
  source-commit: 1e5eeffbdad3749125afe7ab48a39694e27f181c
---

# Quantitative Checks on Measured History

Use a small model when it can test a concrete explanation or change the next
panel. Features come from reconstructed sequences; responses come only from
measured campaign history. Local model outputs are predictions, not oracle
measurements, and do not alter the framework GP or selection policy.

Start with exact sequence contrasts and descriptive summaries. Skip fitting
when history lacks the feature variation or independent backgrounds needed to
answer the question, including the single paired-start initialization. Neither
loading this Skill nor receiving a new batch requires a statistical model.

## Prepare the Evidence

Use one row per measured candidate ID, not one row per annotation or proposal
occurrence. Keep the measured round, biological program, and sequence background
where these groupings can be defined from the constructs. Join labels by exact
candidate ID, never by row order or similar names.

Start with a few interpretable features such as a measured sequence contrast,
verified motif counts/spacing, and GC fraction. Avoid fitting hundreds of
features to one small, highly correlated family. Inspect missing values, rank,
constant columns, and feature correlations before interpreting coefficients.

## Fit and Diagnose

The formula API includes the intercept and encodes interactions explicitly:

```python
import statsmodels.formula.api as smf

# frame contains measured, distinct candidates and computed feature columns.
fit = smf.ols("utility ~ motif_a * motif_b + gc_fraction", data=frame).fit()
print(fit.params)
print(fit.condition_number)
influence = fit.get_influence()
print(influence.cooks_distance[0])
```

Use residuals, rank, and influence to discover where a proposed explanation
fails. Check whether a coefficient changes when an influential family or
background is left out. A large condition number may reflect scale as well as
collinearity; inspect both rather than applying a universal cutoff. Do not
delete a legitimate high-scoring observation merely to improve a fit.

OLS coefficients describe associations in the sampled data. HC3 standard
errors can address heteroskedasticity under their assumptions, but cannot
repair adaptive selection, shared sequence ancestry, missing contrasts, or
confounding. Deterministic scores and repeated related constructs do not
constitute biological replicates. Do not promote nominal p-values or confidence
intervals into proof of a mechanism or global search performance.

## Validate Predictions

For candidate ranking, prefer a small regularized scikit-learn model and a
validation split that tests the intended use:

- Fit transforms and feature selection on training rows only, using a pipeline.
- Hold out complete sequence families/backgrounds when testing transfer;
  `GroupKFold` or `LeaveOneGroupOut` keeps related examples together.
- Hold out later measured rounds when testing forward generalization. A random
  row split can leak near-identical constructs into both sets.
- If too few independent groups or rounds exist, say the estimate is unstable
  and keep the model exploratory. Do not manufacture a validation score.

Compare with a simple baseline on the same held-out observations. Examine
ranking errors and performance by program, not just in-sample R-squared.
Keep scripts, feature definitions, split IDs, and predictions in the workspace.
Use model disagreement to choose an informative candidate; a local model's
extrapolation alone cannot establish that an unmeasured program is inferior.
