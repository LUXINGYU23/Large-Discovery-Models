---
name: compile-ldm-policy
description: Design, test, and submit a deterministic Python prior mean and LDM weight policy from an authoritative campaign snapshot without selecting candidates or changing the task GP.
---

# Compile an LDM Optimization Policy

Produce one complete `optimization_policy.py` for the current campaign round.
You are a belief compiler: translate task evidence into a small auditable mean
function and two LDM weights. The task remains responsible for numerical GP
inference, acquisition, sampling, candidate validation, and evaluation.
This package is a digest-bound, read-only snapshot inside the session guest.
Resolve every referenced file relative to the Skill location advertised by Pi.

## Required workflow

1. Call `inspect_policy_contract`. Its contract, execution contexts, research
   snapshot, and active-policy pointer are authoritative. The snapshot paths in
   the turn message are provenance references, not files in your guest workspace.
2. Read [prior-mean-and-residual-gp.md](references/prior-mean-and-residual-gp.md)
   before designing `compute_prior_mean` and
   [ldm-curriculum.md](references/ldm-curriculum.md) before choosing `alpha` or
   `eta`.
3. State a small set of testable hypotheses. Give priority to direct campaign
   measurements, then exact released task facts, then transferable literature,
   and finally clearly labeled mechanistic speculation. Direct contradictory
   measurements override a generic literature prior.
4. Use the sandbox, public tools, and literature to test useful hypotheses.
   Prefer a zero mean or a shrunken additive model until the observations support
   more structure. Do not confuse in-sample fit with predictive evidence.
5. Write a complete deterministic NumPy-only `optimization_policy.py` in the
   session workspace. Research scripts may use the task guest, but submitted
   code must obey the restricted runtime contract.
6. Call `validate_policy_draft`, then `evaluate_policy_draft`. Interpret its
   draft diagnostics as descriptive in-sample checks only. Repair exact errors
   and rerun both tools when the file changes materially.
7. Submit `replace` only for a justified, validated file. Use `keep` after
   evaluating the active file on the new snapshot when its assumptions still
   hold. Use `disable` when the zero-mean/default-weight task baseline is better
   justified. A policy need not change every round.

## Mathematical contract

`history_utilities` contains raw task utilities. `compute_prior_mean` must use
`context["target_location"]` and `context["target_scale"]` and return the
conditional mean on standardized utility scale, not raw utility, an optimum,
a rank, or an acquisition score. The fixed task GP models the remaining
standardized residual.

The finite-pool LDM distribution is

```text
log q(x) = alpha * log(q0(x) + epsilon)
           + eta * robust_z(acquisition(x)) - log Z.
```

`alpha` controls proposal-frequency evidence; `eta` controls task-GP acquisition
evidence. Their ratio changes the balance and their common scale changes the
concentration. Both must be finite and non-negative.

## Artifact interface

```python
POLICY_API_VERSION = 1
CAPABILITIES = {"prior_mean": 1, "ldm_weights": 1}

def compute_prior_mean(history_features, history_utilities, query_features, context):
    ...

def choose_ldm_weights(context):
    return {"stage": "...", "alpha": 1.0, "eta": 1.0}
```

The mean must be finite, deterministic, query-order equivariant, batch
independent, and valid for empty and one-point histories. It must not use
candidate identity, row position, `q0`, acquisition values, selection
probabilities, files, network access, or hidden labels.

Call the terminal tool with exactly one payload:

- `{"action":"replace","artifact_path":"optimization_policy.py"}`
- `{"action":"keep"}`
- `{"action":"disable"}`

Never include `artifact_path` with `keep` or `disable`.
