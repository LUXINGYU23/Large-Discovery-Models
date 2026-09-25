# Prior Mean and Residual GP

For raw measured utilities u_i the task supplies `target_location` and positive
`target_scale` in `mean_context`:

```text
z_i = (u_i - target_location) / target_scale
h(x) = E[z(x) | public program features and supported evidence]
mu_z(x) = h(x) + k(x, X) [K(X, X) + noise I]^-1 (z - h(X))
```

The GP is fixed: an exact RBF kernel on the unscaled program features
(feature scale floor 1), the configured length scale and noise, the most
recent `history_limit` measurements, and the task's target-scale floor. A zero
`h` reproduces the fixed baseline exactly. The policy returns `h` only.

## Program features

`feature_semantics` in `mean_context` defines every column. Columns 0-5 are
capped size statistics (length, functions, loops, branches, calls, numeric
literals); columns 6-11 count called or accessed names matching method-family
keywords (confidence, selection, geometry, memory, views, control), divided by
6 and capped at 1. They are collinear and coarse: two different methods that
call the same names share a feature vector, and failures carry no utility.

## Small-data construction

- Standardize `history_utilities` with the supplied location and scale before
  fitting anything; standardized zero is the natural intercept.
- Prefer zero, a single supported contrast, or a strongly ridge-shrunk
  additive model: `theta = solve(Phi.T @ Phi + lam * I, Phi.T @ z)`.
- Return zeros for empty or one-point histories unless an explicit public
  prior is defensible. Never normalize with query statistics.

## Evaluating a draft

`evaluate_policy_draft` holds out whole measured rounds, fits the zero-mean GP
on each earlier prefix, and compares baseline and draft predictions on the
held-out round. You have seen those labels, so repeated selection overfits the
check; use `prediction_feedback` for prospective evidence.
