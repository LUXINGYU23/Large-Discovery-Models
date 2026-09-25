# LDM Weights for Program Search

Each round, independent proposal requests or sessions submit program
occurrences. Programs are canonicalized (comments, formatting, and docstrings
ignored); repeated occurrences across requests raise a program's empirical
mass q0 before deduplication. Only already-measured programs are excluded.
A q0-weighted Gumbel top-k keeps the BO pool, q0 is renormalized over it, and
the evaluation batch is drawn without replacement from

```text
p_i ∝ exp(alpha * log(q0_i + 1e-12) + eta * z_i),   z_i = robust_z(UCB_i)
log(p_i / p_j) = alpha * log(q0_i / q0_j) + eta * (z_i - z_j)
```

Limits: alpha=1, eta=0 reproduces q0; alpha=0, eta>0 ranks by acquisition
only; alpha=eta=0 is uniform over the pool.

## Exact diagnostics

`weight_context` contains `history_size`, `unique_candidate_count`,
`valid_proposal_occurrences`, requested and effective evaluation batch,
defaults, `proposal_configuration` (sessions, occurrences per round, duplicate
rules), `q0_summary`, the UCB `acquisition` settings, robust-z `normalization`
with its `z_clip`, and `candidate_predictions` (q0, baseline mean/std/UCB,
normalized acquisition, default probability). `prediction_feedback`, added by
the controller, joins earlier frozen predictions to later measured utilities.

## Evidence-driven changes

- Proposal agreement is preference, not measurement. Program features are
  coarse, so GP acquisition over them is weakly informative until several
  measured programs share families; that is not evidence the GP is harmful.
- First-draw probability is not batch inclusion probability; a batch that
  covers the pool leaves no weight-dependent choice.
- Entropy, ESS, and KL are descriptive, not objectives.
- Held-out mean error evaluates the prior mean, not alpha/eta.
- Change weights only when measured progress, failures, or proposal
  concentration support a diagnosis; name the stage after that evidence.
