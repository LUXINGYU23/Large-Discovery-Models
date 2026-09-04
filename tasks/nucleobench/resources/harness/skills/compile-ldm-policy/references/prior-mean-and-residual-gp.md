# Prior Mean and Residual GP

The generated policy supplies a deterministic mean function. The task owns the
representation, kernel family, hyperparameter procedure, noise model, posterior,
uncertainty, and acquisition function.

## Probability semantics

For raw measured utilities (u_i), the task supplies a location (ell_t) and
positive scale (s_t) in the exact `mean_context` returned by
`inspect_policy_contract`:

```text
z_i = (u_i - location_t) / scale_t
h_t(x) = E[z(x) | public task context and justified evidence]
r_i = z_i - h_t(x_i)
```

The policy returns (h_t(x)), one standardized conditional expectation per
feature row. It must not return a raw utility, predicted maximum, rank,
probability of optimality, UCB, or acquisition score.

At fixed GP hyperparameters, the residual posterior mean has the standard form

```text
mu_z(x) = h_t(x) + k(x, X) [K(X, X) + noise]^-1 (z - h_t(X))
mu_u(x) = location_t + scale_t * mu_z(x).
```

The task may fit its existing kernel hyperparameters to residuals, but generated
code never performs GP inference. Predictive variance, numerical linear algebra,
and acquisition remain task-owned. A zero (h_t) recovers the static task
baseline.

A mean shifts beliefs away from observations; nearby measured residuals let the
GP correct it. Unsupported structure can still bias unobserved regions. Model
misspecification therefore has a real optimization cost, so added structure
must earn its complexity.

## Evidence discipline

Use evidence in this order:

1. direct measurements from this campaign;
2. exact released task schema, structures, descriptors, and constraints;
3. transferable primary literature for the same mechanism or target;
4. explicitly labeled mechanistic speculation.

A generic literature claim must yield to repeated contradictory campaign
measurements. Literature can motivate a feature or sign; it does not provide an
unobserved benchmark label.

The policy estimates its parameters from the same measurements on which the
residual GP conditions. This is a valid model decomposition, but the GP does not
propagate uncertainty in an overfit generated mean. Keep the mean deliberately
simpler and more strongly regularized than a stand-alone predictor.

## Small-data construction

Start with (h_t(x)=0). Add only structure that has support:

- Standardize raw `history_utilities` with the supplied target location and
  scale before fitting any coefficient. Standardized zero, not
  `target_location`, is the natural intercept.
- Prefer a weak fixed scientific contrast or a ridge-shrunk additive model.
  A typical linear form is
  `theta = solve(Phi.T @ Phi + lambda * P, Phi.T @ z)`.
- Treat one-hot and compositional groups as collinear. Use centered group
  effects, a reference level, or ridge regularization instead of interpreting
  arbitrary full-rank coefficients causally.
- Require observations on multiple combinations before adding interactions.
  Confounded factor combinations do not identify separate causal effects.
- Bound extrapolation by design. Do not rely on the runner's final
  `mean_clip`; repeated clipping means the model is too aggressive.
- Return zeros for empty history unless a weak, explicit public prior is
  defensible. One observation rarely supports transferable fitted effects.

Never normalize from `query_features`, use query-batch statistics, branch on
row index, or make one query depend on its neighbors. The same row must receive
the same result alone, permuted, or in a larger batch.

## Evaluating a draft

`evaluate_policy_draft` reports:

- standardized history-utility and prior-residual summaries;
- zero-prior and draft-prior RMSE on the current history;
- their difference and prior/target Pearson correlation;
- history/query prior ranges and the number of clipped values.

These metrics are labeled `in_sample_prior_fit_only`. They detect scale errors,
sign reversals, extreme means, and obvious non-fit. They do not establish
out-of-sample accuracy and should not reward interpolation. When sample size
allows, use sandbox calculations for leave-one-round-out or otherwise honest
held-out comparisons, and prefer the least complex model that remains
scientifically coherent.

## Further reading

- Rasmussen and Williams, [Gaussian Processes for Machine Learning, Chapter 2](https://gaussianprocess.org/gpml/chapters/RW2.pdf), gives the fixed-mean residual posterior.
- Yuan et al., [LLM-Guided Bayesian Optimization](https://arxiv.org/abs/2605.17976), formalizes semantic guidance as a GP mean shift while retaining numerical BO.
- Bogunovic and Krause, [Misspecified Gaussian Process Bandit Optimization](https://arxiv.org/abs/2111.05008), shows why mismatch cannot be treated as cost-free.
