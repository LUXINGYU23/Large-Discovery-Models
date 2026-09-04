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

Same-round duplicate occurrences, including deliberate repetitions within one
proposal minibatch, are meaningful mass in (q_0); they are not a validation
defect. A proposing Agent may allocate several occurrence slots to one legal,
historically unseen candidate when its evidence warrants stronger mass.
Historical evaluated candidates have already been rejected before this stage.

Useful limiting cases are:

- `alpha=1, eta=0`: sample from empirical (q_0).
- `alpha=0, eta>0`: ignore occurrence frequency and use acquisition tilt.
- `alpha>0, eta=0`: ignore acquisition.
- `alpha=0, eta=0`: uniform over the maintained pool.
- task defaults `alpha=2.0, eta=0.25`: reproduce the fixed LDM baseline.

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

## Evidence-defined curriculum

A curriculum state is a diagnosis of the current evidence, not a range of round
numbers. Do not branch on `round_index`, and do not infer readiness from history
size alone. Use the current proposal and acquisition summaries together with
measured progress and contradictions in the research snapshot.

- **Proposal-led:** the surrogate is still prior-like, acquisition is nearly
  flat or unstable, and proposal consensus has a defensible scientific basis.
  Favor `alpha` relative to `eta`.
- **Balanced:** proposal mass and acquisition provide distinct, credible signals
  without a clear conflict. Stay near the released baseline unless diagnostics
  support a material change.
- **Acquisition-led:** measured evidence supports the residual model,
  acquisition separates the pool, and recent evaluations validate its ranking
  better than proposal frequency. Increase `eta` relative to `alpha`.
- **Recovery:** proposal mass has collapsed onto an unsupported family,
  acquisition conflicts with new measurements, or progress has stalled. Flatten
  the unreliable source, and possibly both, to recover useful support.

Transitions may be non-monotone. New evidence can move a campaign back to a
proposal-led or recovery state after the surrogate appeared informative.

Keep decisions legible:

- begin from the task defaults;
- change one or both weights only when the current snapshot provides a reason;
- make the weight audit explicit even when the prior mean is unchanged;
- record the reason in session analysis and use a stable evidence-state label;
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
