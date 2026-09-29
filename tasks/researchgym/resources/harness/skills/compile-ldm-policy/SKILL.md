---
name: compile-ldm-policy
description: Design, test, and submit the enabled prior-mean and/or LDM-weight capabilities for the ResearchGym program campaign without selecting candidates or changing the task GP.
---

# Compile an LDM Optimization Policy

Choose whether to retain the task baseline, keep the active policy, or submit
one complete `optimization_policy.py` justified by current campaign evidence.
The task owns GP inference, UCB, q0 estimation, BO-pool maintenance, sampling,
candidate validation, and evaluation. This skill is a read-only snapshot;
resolve referenced files relative to its advertised location.

## Required workflow

1. Call `inspect_policy_contract`. In its read-only `guest_snapshot.directory`,
   load `input.json["execution_context"]`, `research_snapshot.json`, and
   `arrays.npz` with JSON/NumPy. Print compact summaries, not whole files.
2. Read `enabled_capabilities`. For `prior_mean@1` read
   [prior-mean-and-residual-gp.md](references/prior-mean-and-residual-gp.md);
   for `ldm_weights@1` read [ldm-curriculum.md](references/ldm-curriculum.md).
   Do not implement a disabled capability.
3. Query measured programs with `get_measured_history` when the numeric rows
   are not enough (for example to understand why a family failed).
4. State a few testable hypotheses. Direct campaign measurements outrank
   public task facts, which outrank literature, which outranks speculation.
5. If a change is justified, write a deterministic NumPy-only
   `optimization_policy.py`, call `validate_policy_draft`, then
   `evaluate_policy_draft`, and repair the reported errors. Follow
   [validation-and-repair.md](references/validation-and-repair.md).
6. Submit `replace` only for a validated, justified file; `keep` after
   re-evaluating the active file on the new snapshot; `disable` when the
   zero-mean/default-weight baseline is better supported.

## Mathematical contract

`history_utilities` are raw objective values of measured programs.
`compute_prior_mean` returns one standardized conditional mean per row using
`context["target_location"]` and `context["target_scale"]`; the fixed RBF GP
models the remaining standardized residual.

Each canonical program is one trial. On the maintained BO pool,

```text
log p_i = alpha * log(q0_i + epsilon) + eta * robust_z(UCB_i) - log Z
```

`q0_i` is the program's share of the round's valid proposal occurrences,
renormalized over the pool. `alpha` scales proposal-frequency preference and
`eta` scales GP-acquisition influence; their ratio sets the balance and a
common factor acts as an inverse temperature. Both must be finite and
nonnegative. Start from `default_alpha` and `default_eta`.

## Artifact interface

`CAPABILITIES` must match `enabled_capabilities` exactly (`name@1` becomes
`"name": 1`). Baseline artifacts:

```python
import numpy as np

POLICY_API_VERSION = 1
CAPABILITIES = {"prior_mean": 1, "ldm_weights": 1}

def compute_prior_mean(history_features, history_utilities, query_features, context):
    return np.zeros(len(query_features), dtype=float)

def choose_ldm_weights(context):
    return {"stage": "evidence_baseline",
            "alpha": float(context["default_alpha"]), "eta": float(context["default_eta"])}
```

The mean must be finite, deterministic, query-order equivariant, batch
independent, and valid for empty and one-point histories. It must not use
candidate identity, row position, q0, acquisition values, selection
probabilities, files, network access, or hidden labels. Only
`choose_ldm_weights` reads the weight context. Never branch on `round_index`.

Terminal payloads: `{"action":"replace","artifact_path":"optimization_policy.py"}`,
`{"action":"keep"}`, or `{"action":"disable"}`.
