# Validation and Repair

Use one bounded repair loop:

1. Call `inspect_policy_contract`.
2. Inspect the exact `mean_context`, `weight_context`, research snapshot, and
   active-policy pointer returned by the tool.
3. Write or revise `optimization_policy.py` in the session workspace.
4. Call `validate_policy_draft` for source and interface checks.
5. Call `evaluate_policy_draft` for current-round execution and descriptive
   diagnostics.
6. Repair the reported path, code, message, and hint; repeat both checks after a
   material edit.
7. Submit exactly one terminal action.

Use the tool's `guest_snapshot.directory` to read `arrays.npz`,
`input.json`, and the feature contract with NumPy/JSON. Host-side paths in the
round message remain lineage references.

Common failures:

- `forbidden_import` or `forbidden_operation`: remove file, process, network,
  dynamic-code, or non-whitelisted dependency access from the submitted file.
- `invalid_prior_shape` or `non_finite_output`: return exactly one finite
  scalar per query and stabilize all divisions and solves.
- `query_order_dependent` or `batch_dependent_output`: remove row-position
  and query-batch dependence.
- `non_deterministic_output`: remove random, time-dependent, or mutable module
  state.
- `invalid_ldm_weights`: return exactly `stage`, `alpha`, and `eta`, with
  finite non-negative numeric weights.
- `invalid_submission_shape`: use one of the exact payloads below and no other
  fields.
- task validation errors: repair task semantics without changing fixed GP
  components or selecting candidates.

Do not hide validation failures behind broad exceptions, silent constant
fallbacks, or clipping. Draft diagnostics use the last available chronological
whole-round holdouts. Each earlier training prefix uses the fixed Tanimoto
kernel and jitter, and derives normalization from its bounded most-recent
measured window. The mean does not change these choices online or in diagnostics.
These are development checks on already measured history, not an untouched test
set or a counterfactual optimization run. Check subsequent frozen-prediction
feedback before claiming predictive improvement.

Passing these tools proves executability and reports diagnostics, not scientific
validity. Holdout GP RMSE assesses the mean, not alpha/eta; a lower concentration
or changed first-draw distribution does not prove higher reward. Audit the
weight function separately, including what it returns when improvement stops,
feedback is ambiguous, or both signals lack support. A residual sign is not a
ranking test. Read the exact ReaSyn fields listed in ldm-curriculum.md. Historical
q0_relative_to_max refers to allocated trial mass in that measurement's original
pool, not current-pool mass or a group-level rank. No competition-rank or
optimization-progress object is supplied.

Terminal payloads are exact:

```json
{"action":"replace","artifact_path":"optimization_policy.py"}
```

```json
{"action":"keep"}
```

```json
{"action":"disable"}
```

`keep` re-executes the active artifact on the new inputs and is invalid before
a first accepted epoch. `disable` intentionally returns to the task's static
zero-mean/default-weight policy. After a rejection, keep useful research in the
same session and change only the invalid logic or payload.
