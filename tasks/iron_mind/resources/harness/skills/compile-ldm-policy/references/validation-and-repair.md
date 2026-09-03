# Validation and Repair

Use the policy tools in this order:

1. `inspect_policy_contract`
2. write or revise `optimization_policy.py`
3. `validate_policy_draft`
4. `evaluate_policy_draft`
5. repair exact failures
6. submit the terminal action

Validation errors include a JSON Pointer, code, message, and repair hint. Fix the reported contract violation rather than hiding it with broad exception handling or constant fallback output.

Common failures:

- `forbidden_import` or `forbidden_operation`: remove file, process, network, dynamic-code, or non-whitelisted dependency access.
- `invalid_prior_shape` or `non_finite_output`: return exactly one finite scalar per query and stabilize normalization.
- `query_order_dependent` or `batch_dependent_output`: make prediction row-wise and independent of query batch composition.
- `non_deterministic_output`: remove random, time-dependent, or mutable module state.
- `invalid_ldm_weights`: return exactly `stage`, `alpha`, and `eta`, with finite non-negative numeric weights.
- task validation errors: use the supplied task diagnostics to repair behavior without changing fixed GP components or selecting candidates directly.

After a rejection, preserve useful research in the same session and replace only the invalid policy logic. `keep` is invalid when no active policy exists. `disable` intentionally returns to the task's static baseline.
