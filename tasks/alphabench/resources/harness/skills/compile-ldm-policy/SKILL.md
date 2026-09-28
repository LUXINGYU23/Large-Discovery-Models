---
name: compile-ldm-policy
description: Compile only the enabled AlphaBench T3 prior mean and LDM weight policy from public, measured search evidence.
---

# Compile an AlphaBench T3 optimization policy

Inspect the authoritative contract and snapshot with `inspect_policy_contract`. Read `input.json`, `research_snapshot.json` and `arrays.npz` through the guest snapshot path. The numeric rows are authoritative; do not rebuild them from prose or market files. The policy artifact runs without network access, with read-only inputs and finite time, memory and process limits.

`history_utilities` are raw signed search IC or RankIC. `compute_prior_mean` returns a vector of conditional means on the standardized scale `(utility - target_location) / target_scale`. The Host subtracts those history means before fitting the fixed exact RBF GP, then adds the query means to its posterior. A zero vector leaves the original GP unchanged. Use only `history_features`, `history_utilities`, `query_features` and `mean_context` in this function; no q0, candidate IDs, session IDs or validation/test outcomes.

The frozen selection logits are `alpha*log(q0) + eta*clip(robust_z(UCB),-5,5)` before Gumbel top-k without replacement. `choose_ldm_weights` receives only `weight_context` and returns exactly `{"stage": str, "alpha": finite_nonnegative, "eta": finite_nonnegative}`. Default weights are `2.0` and `0.25`. Use measured prediction feedback and the current q0/UCB spread to justify a change; history count or round number alone is not evidence. Disabling a custom policy restores the default weights and zero mean, not a disabled GP.

Write a deterministic NumPy-only `optimization_policy.py`. Set `POLICY_API_VERSION = 1` and a literal `CAPABILITIES` dictionary exactly matching the enabled contract. For `prior_mean@1`, implement `compute_prior_mean(history_features, history_utilities, query_features, context)`, returning one finite value per query row. For `ldm_weights@1`, implement `choose_ldm_weights(context)`. Omit a disabled export and its capability key. A conservative two-capability artifact is:

```python
import numpy as np

POLICY_API_VERSION = 1
CAPABILITIES = {"prior_mean": 1, "ldm_weights": 1}

def compute_prior_mean(history_features, history_utilities, query_features, context):
    return np.zeros(len(query_features), dtype=float)

def choose_ldm_weights(context):
    return {"stage": "baseline", "alpha": context["default_alpha"], "eta": context["default_eta"]}
```

For a replacement, run `validate_policy_draft` on the complete artifact. Check finite outputs, row alignment, chronological prediction errors and whether the change has evidence beyond in-sample fit. Submit `replace` with `artifact_path="optimization_policy.py"`; the Host executes the artifact in a restricted Docker runner and returns indexed errors for repair. The other terminal actions are `keep` and `disable`.
