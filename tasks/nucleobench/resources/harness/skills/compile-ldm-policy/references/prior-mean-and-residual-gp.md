# Prior Mean and Residual GP

The generated policy supplies a mean function; the task owns the representation, covariance kernel, noise model, posterior, and acquisition function.

For measured features `X`, standardized utilities `y`, and policy mean `m(X)`, the fixed task GP fits residuals:

```text
r = y - m(X)
posterior_mean(x) = m(x) + GP_residual_mean(x)
```

## Design rules

- Start with zero or a small regularized linear/additive model. Add interactions only when measured evidence or robust public knowledge supports them.
- Derive normalization only from supplied history and context. Handle zero variance and empty history explicitly.
- Return standardized utility means, not raw scores or acquisition values.
- Apply one row-wise rule to every query. The prediction for a row must not depend on its batch neighbors, position, or query order.
- Do not encode candidate identifiers, current `q0`, acquisition, selection probability, or a lookup table of hidden outcomes.
- Prefer bounded coefficients, shrinkage, and stable linear algebra over high-degree extrapolation.

The runner clips accepted means to the task contract's standardized range and reports the clip count. Frequent clipping is evidence that the mean is too aggressive, not a mechanism to rely on.
