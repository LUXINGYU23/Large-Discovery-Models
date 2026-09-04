# LDM Curriculum Weights

Harness-Compiled LDM combines independent proposal occurrences with the fixed
task acquisition on the maintained finite pool.

## Distribution

Let `q0(x) > 0` be empirical proposal mass and let
`a_tilde(x) = robust_z(a(x))`. The implemented distribution is

```text
q(x) = exp(alpha * log(q0(x) + epsilon) + eta * a_tilde(x)) / Z
```

and therefore, for two pool members,

```text
log(q(x_i) / q(x_j))
  = alpha * log((q0(x_i) + epsilon) / (q0(x_j) + epsilon))
    + eta * (a_tilde(x_i) - a_tilde(x_j)).
```

This odds equation is the most useful way to reason about the controls:

- `alpha` scales evidence from same-round proposal frequency.
- `eta` scales evidence from the task GP's acquisition.
- Their ratio changes which source dominates.
- Multiplying both by the same positive constant preserves the logit direction
  but changes softmax concentration, like inverse temperature.

Same-round duplicate occurrences are meaningful mass in (q_0); they are not a
validation defect. Historical evaluated candidates have already been rejected
before this stage.

Useful limiting cases are:

- `alpha=1, eta=0`: sample from empirical (q_0).
- `alpha=0, eta>0`: ignore occurrence frequency and use acquisition tilt.
- `alpha>0, eta=0`: ignore acquisition.
- `alpha=0, eta=0`: uniform over the maintained pool.
- task defaults: reproduce the fixed Harness-backed LDM baseline.

The stage string is provenance only. It has no computational effect.

## Read the diagnostics correctly

The exact `weight_context` is visible through `inspect_policy_contract`.
It includes history size, pool and occurrence counts, task defaults, a (q_0)
summary, and a baseline-acquisition summary.

For a pool of size (N):

```text
ESS(q0) = 1 / sum_x q0(x)^2,       1 <= ESS <= N
H(q0)   = -sum_x q0(x) log q0(x),  0 <= H <= log N.
```

Use `ESS/N` or `H/log(N)` when comparing different pool sizes. Low values mean
proposal mass is concentrated; they do not say whether the consensus is
scientifically correct. If baseline acquisition is effectively constant,
robust normalization becomes zero and changing `eta` cannot create a ranking.

Weights should reflect an explicit hypothesis about:

1. how independent and credible the current proposal consensus is;
2. whether measured history supports the task surrogate and compiled mean;
3. whether acquisition meaningfully separates the current pool;
4. whether recent real evaluations are improving or contradicting prior beliefs.

No single entropy, ESS, history-size, or round threshold is sufficient. The host
does not impose a trust gate, so extreme weights are legal but must be justified
by stronger evidence than ordinary defaults.

## Curriculum design

A curriculum may be non-monotone. Early data scarcity can justify broad support,
but a strong task prior can also justify early focus. Later measurements can
increase acquisition weight when the residual model becomes informative, or
decrease it when new evidence exposes misspecification. Proposal consensus may
be sharpened when independent roles converge for defensible reasons, or
flattened when they collapse onto one unsupported family.

Keep decisions legible:

- begin from the task defaults;
- change one or both weights only when the current snapshot provides a reason;
- record the reason in session analysis, not in the stage string;
- use a stable stage label for comparable policy logic;
- evaluate the active artifact on every new snapshot before `keep`;
- do not modify weights merely to make rounds look adaptive.

## Relation to guided BO

The proposal term is a strictly positive support preference, while the compiled
mean is a belief about expected utility. They are not interchangeable. ColaBO
shows how user beliefs over function properties can guide BO while retaining
support across the domain; LGBO shows a tractable mean-shift route for continuous
semantic guidance. This implementation uses the lighter task-local mean plus
finite-pool LDM tilt and does not claim either paper's full guarantees.

- Hvarfner et al., [A General Framework for User-Guided Bayesian Optimization](https://arxiv.org/abs/2311.14645).
- Yuan et al., [LLM-Guided Bayesian Optimization](https://arxiv.org/abs/2605.17976).
