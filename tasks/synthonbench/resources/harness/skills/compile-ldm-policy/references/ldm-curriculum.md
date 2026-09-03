# LDM Curriculum Weights

Harness-Compiled LDM combines the empirical proposal distribution and the fixed task acquisition as:

```text
q(x) proportional to q0(x)^alpha * exp(eta * robust_z(acquisition(x)))
```

`q0` is estimated from independent proposal occurrences. Repeated same-round proposals are therefore meaningful probability mass.

## Interpreting the controls

- `alpha = 0` ignores proposal frequency; larger values trust Agent consensus more strongly.
- `eta = 0` ignores acquisition; larger values concentrate on the task GP's acquisition ranking.
- The stage label is descriptive provenance. It does not alter computation by itself.

Use measured progress and aggregate diagnostics to choose a restrained curriculum. Early rounds often warrant broader proposal support; later rounds may justify more acquisition pressure when the residual GP has enough evidence. Stagnation, low diversity, or unstable diagnostics can justify reducing the dominant term.

ESS, entropy, KL, and overlap are diagnostics, not hard safety gates. Do not tune weights solely to hit a diagnostic target, and do not claim a policy is better without measured evidence.
