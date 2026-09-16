---
name: public_audit_policy
description: Compile a deterministic untrained public structural prior without accessing benchmark labels.
---

Inspect the current contract with inspect_policy_contract and read the authoritative guest snapshot. The current drafts, public question and feature meanings are in research_snapshot.json. There are no objective measurements: empty history arrays must remain empty. Use public geometry reasoning to justify a small, finite prior over the supplied features, or retain the zero-prior baseline. This prior is an unverified hypothesis, never a measured accuracy estimate. Selection only ranks the current revision's drafts; no prior revision may replace the final scheduled answer.

Write optimization_policy.py using this interface:

```python
import numpy as np

POLICY_API_VERSION = 1
CAPABILITIES = {"prior_mean": 1}

def compute_prior_mean(history_features, history_utilities, query_features, context):
    return np.zeros(len(query_features), dtype=float)
```

The baseline above is complete. If modifying it, return one finite standardized plausibility prior per query row, including empty queries; the runner clips to the contract bound. Do not use output-row order, global state, random generation, file reads, networking, or benchmark answer lookups. Do not implement the disabled ldm_weights capability. There is no fitted GP or retrospective objective-error feedback in this protocol.

Call validate_policy_draft and evaluate_policy_draft on the file; repair all errors. These tools validate execution, not correctness. Submit with submit_optimization_policy using exactly {"action":"replace","artifact_path":"optimization_policy.py"}. Use {"action":"keep"} only when an active policy exists, or {"action":"disable"} for zero prior and deterministic first-profile selection. Keep scientific rationale in persistent research notes, not extra terminal-tool fields.
