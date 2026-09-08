# LDM Curriculum Weights

Harness-Compiled LDM combines empirical proposal occurrences with the fixed
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

- `alpha` scales preference from same-round proposal frequency.
- `eta` scales the influence of the task GP's acquisition.
- Their ratio changes which source dominates.
- Multiplying both by the same positive constant preserves the logit direction
  but changes softmax concentration, like inverse temperature.

Same-round duplicate occurrences, including deliberate repetitions within one
proposal minibatch, are meaningful mass in (q_0); they are not a validation
defect. A proposing Agent may allocate several occurrence slots to one legal,
historically unseen candidate when its evidence warrants stronger mass.
Historical evaluated candidates have already been rejected before this stage.
Multiplicity is an allocation of belief, not a count of independent experiments.
Repeated slots from one session can dominate mass without stronger evidence.
Being selected is not a successful objective measurement; being left unmeasured
is not evidence for extra confidence. Do not reward vote escalation itself.

Useful limiting cases are:

- `alpha=1, eta=0`: sample from empirical (q_0).
- `alpha=0, eta>0`: ignore occurrence frequency and use acquisition tilt.
- `alpha>0, eta=0`: ignore acquisition.
- `alpha=0, eta=0`: uniform over the maintained pool; both final-selection
  signals are disabled. Earlier q0-weighted pool admission still applies.
- task defaults `alpha=2.0, eta=0.25`: reproduce the fixed LDM baseline.

The stage string is provenance only. It has no computational effect.

## Read the diagnostics correctly

The exact `weight_context` is in the exported `input.json["execution_context"]`.
It includes history size, pool and occurrence counts, task defaults, a (q_0)
summary, and a baseline-acquisition summary. `candidate_predictions` adds each
pool member's q0, raw GP mean/std/UCB, normalized acquisition, and default
first-draw probability. `normalization` records the exact robust-z clip.
`prediction_feedback` compares frozen zero-mean/active GP predictions with the
subsequent measurements; `optimization_progress` records measured improvement.
These errors are selected-point evidence, not accuracy on the whole space.

Each measured prediction carries its original `pool_size`, competition ranks
(1 is best, ties share rank), `q0_relative_to_max`, `first_draw_probability`,
and alpha/eta. These are frozen in the measurement's own round. Do not compare
an old q0 against the current pool's maximum; use its saved relative mass or
rank within its original pool. First-draw probability is not the inclusion
probability for a multi-candidate evaluation batch.

Read `requested_evaluation_batch` and `effective_evaluation_batch` from the
weight context, and compare the effective batch with the current pool size.
A large batch can include most candidates even when the first-draw distribution
is concentrated. If the whole pool is evaluated, no weight choice changes the
evaluated set. Do not claim selection benefit from entropy or first-draw odds
alone. With `benchmark_time`, judge the value of further research and information
gathering against the remaining opportunity to evaluate candidates. Time pressure
does not itself establish surrogate reliability or proposal quality.

Separate three questions:

- **Prediction:** an error or standardized residual concerns the utility
  prediction at that measured point. A positive residual means underprediction,
  not that GP ranking or acquisition has been validated.
- **Ranking:** compare predictions frozen under the same model and history
  against multiple subsequent measurements from that round. A round with one
  measured point provides no within-round ranking test. A nonconstant UCB
  alone is not evidence of useful ordering.
- **Decision:** assess observed progress and contradictions under the actual
  selection distribution. GP holdout RMSE does not depend on alpha/eta and
  cannot validate a weight change. Selected-point feedback does not reveal
  the rewards of alternative unmeasured candidates.

Both q0 and acquisition can be wrong, especially on sparsely measured or
confounded regions. Distinguish missing calibration from measured contradictions.
The configured baseline is a working decision rule under uncertainty, not a new
claim that must be proved from scratch each round. With little evidence, retain
it or the last justified nondegenerate policy. A zero mean still leaves a GP
posterior and UCB exploration; sparse data do not imply that eta should be zero.

Use entropy, ESS and weighted logit ranges to check actual influence, not as
optimization objectives. In particular, maximum ESS through alpha=eta=0 abandons
both final-selection signals. Tiny positive weights are not a meaningful repair.
Changing or suppressing a signal should answer a concrete observed failure or
testable opportunity, rather than express a generic preference for neutrality.

Before choosing weights, calculate pairwise log-odds for a high-frequency
candidate and a contrasting high-acquisition candidate. Check whether the
proposed weights can materially change the distribution. For example, a 20:1
q0 ratio contributes about 6 logit units at alpha=2, whereas eta=0.25 with z
clipped to [-2,2] contributes at most 1. No GP ranking can overcome that gap.
Use the actual task clip and evidence rather than adopting these example values.

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

- **Baseline:** calibration is unavailable or inconclusive and no concrete
  failure justifies changing the configured weights. Retain the working LDM
  balance while collecting measurements; no custom artifact is necessary.
- **Proposal-led:** the surrogate is still prior-like, acquisition is nearly
  flat or unstable, and proposal consensus has a defensible scientific basis.
  Favor `alpha` relative to `eta`.
- **Balanced:** proposal mass and acquisition provide distinct, credible signals
  without a clear conflict. Choose weights whose actual logit contributions
  express that balance; the numerical defaults need not be balanced.
- **Acquisition-led:** measured evidence supports the residual model,
  acquisition separates the pool, and recent evaluations validate its ranking
  better than proposal frequency. Increase `eta` relative to `alpha`.
- **Recovery:** comparative measurements contradict the dominant proposal
  family or acquisition ordering. Reduce the implicated source and compare the
  resulting odds with the baseline and active policy. A stall motivates
  diagnosis; by itself it does not establish that both sources are harmful.
  Test the rule on plausible stalled and contradictory contexts, not only the
  current successful point.

Transitions may be non-monotone. Moving back to proposal-led needs independent
support for proposal quality, not just an acquisition failure. Expected
improvement and the information value of a discriminating experiment are
different reasons for retaining meaningful probability on alternatives.

Keep decisions legible:

- start from the task defaults; retain them when no change is justified;
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
