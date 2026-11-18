---
name: compile-ldm-policy
description: Research and compile the CDRH3 utility prior and LDM selection weights from measured energy observations.
---

# Compile a CDRH3 optimization policy

Call `inspect_policy_contract` and read the returned guest snapshot. The
authoritative numeric input uses 11 position-major groups of 20 one-hot amino
acids in the order `ACDEFGHIKLMNPQRSTVWY`. `history_utilities` are exactly
negative measured energies. Larger is better. Candidate features, historical
utilities and recorded feedback are not permission to evaluate a sequence.

For enabled `prior_mean@1`, write `compute_prior_mean(history_features,
history_utilities, query_features, context)`. Return one deterministic finite
standardized utility mean for each query row: `(utility - target_location) /
target_scale`. The host subtracts history priors before fitting its exact RBF
GP and adds query priors to its predictions. The sparse-history GP does not use
the custom mean until at least two observations are available. Prior values are
clipped according to the contract. Use only the supplied numeric inputs and
mean context, and design the function to handle empty history.

For enabled `ldm_weights@1`, write `choose_ldm_weights(context)` returning
exactly `{"stage": string, "alpha": nonnegative finite number, "eta":
nonnegative finite number}`. At selection the host samples with logits
`alpha * log(q0) + eta * robust_z(utility_mean + beta * utility_std)` and
conditions on pending experiments. The application context contains the
generation, target definition and defaults; it does not contain the eventual
reservoir's q0 or future measurements. Encode any justified research decision
in the artifact. Avoid overinterpreting very small histories or selected-point
feedback. Default weights are alpha 1 and eta 1.

Use NumPy and the standard library only. Set `POLICY_API_VERSION = 1` and
`CAPABILITIES = {"prior_mean": 1, "ldm_weights": 1}` for the two enabled
capabilities. Write the artifact at the relative workspace path
`optimization_policy.py`. Validate and evaluate it with the policy tools using
that relative path. Then submit `{"action": "replace", "artifact_path":
"optimization_policy.py"}`. You may instead submit `{"action": "keep"}`
when an active artifact remains appropriate or `{"action": "disable"}` for
the prior-free default. Report uncertainty honestly; never fabricate results.
