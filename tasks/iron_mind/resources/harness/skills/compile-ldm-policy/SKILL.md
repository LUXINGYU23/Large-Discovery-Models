---
name: compile-ldm-policy
description: Design, validate, and submit a deterministic Python optimization policy for Harness-Compiled LDM. Use when a policy architect must define a task-informed prior mean and round-specific LDM alpha/eta weights from an authoritative policy snapshot without selecting candidates or changing the task GP.
---

# Compile an LDM Optimization Policy

Produce one complete `optimization_policy.py` that satisfies the active policy contract.

## Workflow

1. Call `inspect_policy_contract` before designing the policy. Treat its contract and research snapshot as authoritative.
2. Read [prior-mean-and-residual-gp.md](references/prior-mean-and-residual-gp.md) before changing `compute_prior_mean`.
3. Read [ldm-curriculum.md](references/ldm-curriculum.md) before changing `alpha`, `eta`, or the stage label.
4. Research the measured evidence and public task knowledge. Use sandbox calculations when they can test a hypothesis, but never call the true oracle or select the evaluation minibatch.
5. Write a complete, deterministic, NumPy-only `optimization_policy.py` in the session workspace.
6. Call `validate_policy_draft`, then `evaluate_policy_draft`. Read [validation-and-repair.md](references/validation-and-repair.md) when either tool reports an error or an unstable diagnostic.
7. Repair exact failures and repeat validation. Submit `replace` only after the complete file passes. Use `keep` when the active policy remains justified, or `disable` when the static task baseline is preferable.

## Required artifact interface

```python
POLICY_API_VERSION = 1
CAPABILITIES = {"prior_mean": 1, "ldm_weights": 1}

def compute_prior_mean(history_features, history_utilities, query_features, context):
    ...

def choose_ldm_weights(context):
    return {"stage": "...", "alpha": 1.0, "eta": 1.0}
```

Keep the model simple enough to explain from measured evidence. The prior mean must be query-order equivariant, batch independent, finite, and valid for empty or tiny history. It must not use candidate identity, `q0`, acquisition values, or hidden labels. `alpha` and `eta` must be finite and non-negative.
