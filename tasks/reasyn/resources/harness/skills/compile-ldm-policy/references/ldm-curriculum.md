# ReaSyn LDM Curriculum Weights

The host combines measured-history GP/UCB with empirical proposal occurrences.
The policy changes only alpha and eta, never candidate identities, kernel,
representation, evaluation budget, or the sampling algorithm.

## Query Groups and Independent Trials

Legal repeated occurrences are retained, including repeats within one session.
For reconstruction, canonical query SMILES defines the proposal group; each
occurrence receives an independent sampling seed and evaluation identity.
A previously seen query is not a previously evaluated trial. For TDC, the group
is the canonical projected product; repeated products contribute proposal mass
before deduplication, but only previously unmeasured products are eligible.

On the maintained BO pool, let Q_g be the normalized empirical mass of group g
and n_g its number of retained trials. Pool maintenance preserves the original
occurrence counts for surviving groups, renormalizes across these groups and
allocates each group's mass equally to its retained trials. The candidate-aligned
`q0` is Q_g/n_g, not Q_g. Read `q0_group_mass` and `group_trial_count` rather than
inferring group mass from row counts after pool maintenance.

For trial i in group g(i), the actual first-draw distribution is:

```text
logit_i = alpha * log(Q_g(i) + epsilon) - log(n_g(i)) + eta * z_i
p_i = exp(logit_i) / sum_j exp(logit_j)
z_i = robust_z(UCB_i)

log(p_i / p_j)
  = alpha * log((Q_g(i) + epsilon) / (Q_g(j) + epsilon))
    - log(n_g(i) / n_g(j)) + eta * (z_i - z_j)
```

TDC has n_g=1. Alpha scales group-frequency preference; eta scales acquisition
influence. Their ratio controls this balance. Common scaling changes the group
preference/acquisition terms but does NOT scale the fixed -log(n_g) allocation.
It is an inverse temperature for the entire trial distribution only when all
retained group sizes are equal.

Useful limits:

- alpha=1, eta=0 reproduces the allocated empirical trial mass, up to epsilon.
- alpha=0, eta>0 removes group-frequency preference but keeps trial allocation.
- alpha>0, eta=0 uses only group-frequency preference and trial allocation.
- alpha=eta=0 is uniform across query groups, then across each group's trials.
  For three A trials and one B trial: [1/6, 1/6, 1/6, 1/2], not four quarters.
  Earlier q0-weighted pool maintenance still affects which trials are available.

Use `default_alpha` and `default_eta` from the authoritative weight context.
Current CLI defaults are 1.0 and 1.0; they are adapter defaults, not a released
ReaSyn LDM baseline. The stage string records evidence provenance only.
Multiplicity is proposal preference, not additional measured scientific evidence.

## Exact Diagnostics

Read `input.json["execution_context"]["weight_context"]`. It contains:

- `history_size`, `unique_candidate_count`, `valid_proposal_occurrences`,
  `requested_evaluation_batch`, `effective_evaluation_batch`, and defaults.
- `q0_summary`: group scope, group_count, entropy and maximum group mass.
  `trial_base_summary`: allocated-trial entropy and maximum trial mass.
- `baseline_acquisition_summary`: mean and std; `acquisition`: UCB name,
  score direction and beta; `normalization`: robust-z epsilon, mad_scale, z_clip.
- `candidate_predictions`: candidate_id, q0, group_trial_count, q0_group_mass,
  baseline_mean, baseline_std, baseline_acquisition, normalized_acquisition,
  default_selection_probability.
- `prediction_feedback` (added by the shared controller): summary, measurements,
  omitted_older_measurements. Summary reports count and, when matched
  measurements exist, baseline/active RMSE and MAE.

Feedback measurements join frozen predictions to subsequently observed utility
by round and candidate identity. They include baseline/active mean, std and
acquisition; q0, group mass/count, normalized acquisition, logit,
selection_probability, first_draw_probability, q0_relative_to_max, pool_size,
alpha, eta, round_index and measured_utility. The relative q0 is the TRIAL mass
relative to the maximum in that original pool, not a query-group confidence.
No competition ranks, `optimization_progress`, or `benchmark_time` object is
promised by this adapter. Derive a clearly labeled progress summary from
available measured history if needed; do not assume fields from another task.

First-draw probability is not the inclusion probability for a multi-trial batch.
A batch covering the entire pool leaves no weight-dependent choice of evaluated
set. Entropy, ESS, KL and selection changes are descriptive, not proof of reward.
For comparing normalized entropy or ESS, use the number of groups for group
mass and the number of retained trials for trial mass; never mix the two.

## Evidence-Driven Changes

Start from the configured baseline or last justified policy. Sparse history and
a zero compiled mean still leave a GP posterior and UCB exploration. Missing
calibration is not evidence that acquisition or proposal preference is harmful.

Separate predictive error, within-round ranking and decision quality. A positive
residual means underprediction, not a validated acquisition ordering. One
measured trial provides no within-round ranking test. Held-out RMSE evaluates
the mean, not alpha/eta, and selected-point feedback cannot reveal rewards of
unmeasured alternatives.

Before changing weights, calculate the full pairwise log-odds above, including
group-size allocation, using the actual z_clip. Audit the resulting distribution
with `evaluate_policy_draft`; do not maximize ESS as an optimization objective.
Use a baseline, proposal-led, balanced, acquisition-led or recovery stage only
when the measured evidence supports that diagnosis. Never select a stage from
round number or history size alone. A stall motivates investigation; it does
not establish that both signals should be removed.

Record the hypothesis and evidence in session analysis, validate a complete
artifact, and reevaluate the active artifact on each new snapshot before keep.
The mean models standardized utility; proposal mass models finite-pool support
preference. These roles are not interchangeable.
