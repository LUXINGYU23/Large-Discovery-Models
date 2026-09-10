---
name: compile-ldm-policy
description: Design, test, and submit the enabled prior-mean and/or LDM-weight capabilities from an authoritative campaign snapshot without selecting candidates or changing the task GP.
---

# Compile an LDM Optimization Policy

Choose whether to retain the task baseline, keep an active policy, or submit
one complete `optimization_policy.py` justified by current campaign evidence.
Translate evidence into only the capabilities enabled by the current contract.
The task remains responsible for numerical GP
inference, acquisition, sampling, candidate validation, and evaluation.
This package is a digest-bound, read-only snapshot inside the session guest.
Resolve every referenced file relative to the Skill location advertised by Pi.

## Required workflow

1. Call `inspect_policy_contract`. Its contract and active-policy pointer are
   authoritative. In the read-only `guest_snapshot.directory`, load
   `input.json["execution_context"]`, `research_snapshot.json`, and `arrays.npz`
   with JSON/NumPy. Query selected evidence and print compact summaries, not
   entire files. Do not reconstruct numeric feature rows from SMILES or prose.
   Read `weight_context.proposal_sampling` for actual session counts and
   repeat rules; shared instructions do not imply statistically independent votes.
2. Read `enabled_capabilities`. Only for `prior_mean@1`, read
   [prior-mean-and-residual-gp.md](references/prior-mean-and-residual-gp.md).
   Only for `ldm_weights@1`, read
   [ldm-curriculum.md](references/ldm-curriculum.md).
   Do not research, declare, or implement a disabled capability. The task
   supplies zero mean or default weights for the disabled component.
3. State a small set of testable hypotheses. Give priority to direct campaign
   measurements, then exact released task facts, then transferable literature,
   and finally clearly labeled mechanistic speculation. Direct contradictory
   measurements override a generic literature prior.
4. When `ldm_weights@1` is enabled, audit weights independently of whether the
   prior mean changes. Infer an evidence-defined curriculum state from proposal
   concentration, surrogate readiness and acquisition separation, and measured
   progress or contradiction. Never switch stages from round number or history
   size alone. Start from the configured weights or a justified active policy.
   Missing evidence is not evidence that either signal is harmful; UCB already
   provides uncertainty-driven exploration before its ranking is calibrated.
5. Use the sandbox, public tools, and literature to test useful hypotheses.
   For an enabled prior mean, prefer zero or a shrunken additive model until
   observations support more structure. Do not confuse in-sample fit with
   predictive evidence.
6. If a change is justified, write a complete deterministic NumPy-only
   `optimization_policy.py` in the session workspace. Research scripts may use
   the task guest, but submitted code must obey the restricted runtime contract.
7. For a new file, call `validate_policy_draft`, then `evaluate_policy_draft`.
   For an enabled
   mean, compare chronological fixed-GP holdouts; for enabled weights, compare
   current-pool first-draw probabilities. In either case,
   inspect frozen pre-measurement errors in `weight_context.prediction_feedback`.
   Predictive error checks evaluate the mean, not the utility of alpha/eta.
   A positive residual, successful validation, or a changed distribution is
   not evidence that a weight policy improves optimization.
   Repair exact errors and rerun both tools after a material edit.
8. Submit `replace` only for a justified, validated file. Use `keep` after
   evaluating the active file on the new snapshot when its assumptions still
   hold. Use `disable` when the zero-mean/default-weight task baseline is better
   justified. `disable` removes the custom policy, not the GP or LDM weighting.
   No artifact is needed to retain this baseline.

## Mathematical contract

Apply the mean guidance only for `prior_mean@1` and the weight-design guidance
only for `ldm_weights@1`. Read-only diagnostics do not authorize changing a
disabled component.

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

`alpha` controls proposal-frequency preference; `eta` controls task-GP acquisition
influence. A zero prior mean does not remove the GP posterior or its uncertainty.
The weights' ratio changes the balance and their common scale changes the
concentration. Both must be finite and non-negative. The released baseline is
`alpha=2.0, eta=0.25`; use the actual context defaults as the starting policy.
Adapt from evidence, without fixed round schedules or host-imposed weight floors.
Both weights at zero remove both signals from final selection. Do not choose
that state, or numerically negligible weights, merely to maximize ESS or avoid
uncalibrated predictions. An unsupported new mean calls for zero mean, not zero
acquisition weight.

## Artifact interface

`CAPABILITIES` must be a literal dictionary matching `enabled_capabilities`
exactly, converting each `name@1` to `"name": 1`. Include only its required
functions. These are complete baseline artifacts for the two single-capability
contracts; adapt the enabled function when evidence justifies a change.

Prior mean only:

```python
import numpy as np

POLICY_API_VERSION = 1
CAPABILITIES = {"prior_mean": 1}

def compute_prior_mean(history_features, history_utilities, query_features, context):
    return np.zeros(len(query_features), dtype=float)
```

LDM weights only:

```python
POLICY_API_VERSION = 1
CAPABILITIES = {"ldm_weights": 1}

def choose_ldm_weights(context):
    return {
        "stage": "evidence_baseline",
        "alpha": float(context["default_alpha"]),
        "eta": float(context["default_eta"]),
    }
```

When both capabilities are enabled, combine these two functions and the NumPy
import in one file, keep one `POLICY_API_VERSION = 1`, and declare
`CAPABILITIES = {"prior_mean": 1, "ldm_weights": 1}`.

When weight adaptation is enabled and supported, replace its baseline return with a
deterministic rule over the supplied diagnostics. Stage names describe the
current evidence state. Do not branch on `round_index` or use a fixed
early/middle/late round schedule.

Only the mean is barred from reading proposal and acquisition diagnostics;
`choose_ldm_weights` should use the supplied `weight_context`. Audit actual
log-odds with its exact `normalization.z_clip`, not raw UCB spread. Cross-session
agreement is proposal preference, not independent objective measurements.

The mean must be finite, deterministic, query-order equivariant, batch
independent, and valid for empty and one-point histories. It must not use
candidate identity, row position, `q0`, acquisition values, selection
probabilities, files, network access, or hidden labels.

Call the terminal tool with exactly one payload:

- `{"action":"replace","artifact_path":"optimization_policy.py"}`
- `{"action":"keep"}`
- `{"action":"disable"}`

Never include `artifact_path` with `keep` or `disable`.
