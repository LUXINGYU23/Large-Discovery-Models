# Harness-Compiled LDM Registration

Read [`docs/research-harness.md`](../../../docs/research-harness.md) first. Use
this reference only when one persistent Harness session designs a task-local
optimization policy in addition to the existing proposal Harness.

## Preserve the LDM boundary

- Candidate generation, admission, occurrence counts, empirical `q0`, pool
  maintenance, acquisition, evaluation, and optimization history remain owned
  by the existing task and shared engine.
- The policy Agent must not select candidates, call the oracle, alter the
  evaluated minibatch, or start another optimization loop.
- Keep one independent policy `HarnessClient`, profile, manifest, artifact root,
  session history, tool budget, and budget counters. Do not mix policy turns
  into proposal-session state.

## Implement the task-local policy seam

1. Add a deterministic public feature encoder under `tasks/<task_id>/core/`.
   Features may use released task data and measured observations, but must not
   expose hidden labels, candidate identity, row position, `q0`, acquisition
   values, selection probabilities, or future outcomes.
2. Add an `OptimizationPolicyAdapter` that publishes a versioned
   `PolicyCapabilityContract`, materializes authoritative round inputs, and
   validates execution outputs against task semantics.
3. Wire the prior mean into the existing task surrogate as a residual GP:
   subtract the standardized prior from measured targets, fit the unchanged GP,
   and add the prior back to query predictions. Kernel, variance, noise,
   acquisition, and numerical safeguards remain fixed.
4. Prove zero-prior parity: a zero prior with default `alpha` and `eta` must
   reproduce the existing `ldm_harness` predictions and selection distribution.
5. Add one task-owned `policy_architect/AGENTS.md` and mount the repository
   `compile-ldm-policy` skill read-only. Keep task knowledge in these resources,
   not in shared Harness code.

The current release supports only `prior_mean@1` and `ldm_weights@1`. Add a new
versioned capability and task adapter path before exposing another editable
component; do not broaden the meaning of an existing capability.

## Artifact and fallback contract

Use the generic `HarnessSubmissionContract` with one terminal action:

- `replace`: submit a complete `optimization_policy.py` artifact;
- `keep`: revalidate and reuse the active policy epoch;
- `disable`: use the static zero-prior/default-weight policy.

Snapshot the file before task validation. The Agent may use the built-in policy
MCP to inspect, validate, and evaluate drafts, but the host must execute the
accepted snapshot again through `PolicyExecutor` with read-only inputs, no
network, and bounded CPU, memory, processes, and time. Never execute Agent code
with the host interpreter.

Return exact JSON-Pointer errors to the same session for repair. Bound formal
profiles to a small submission-attempt count. If the turn fails, revalidate the
previous epoch on current inputs; if it is unavailable or invalid, use static
defaults and record `degraded=true`.

## Registration and verification

- Add secret-free six- and twelve-round profiles only when those campaign
  contracts are required. Lock proposal counts, BO pool, evaluation budget,
  policy capabilities, submission attempts, and separate proposal/policy tool
  budgets.
- Record proposal and policy manifests under `<run_dir>/harness/` and
  `<run_dir>/policy_harness/`. Include policy turns, provider/tool/artifact
  usage, validation failures, epoch/source/action, degraded state, and selected
  weights in reports.
- Test feature determinism and leakage boundaries, strict artifact paths and
  digests, repair, isolated execution, zero-prior parity, residual-GP behavior,
  epoch resume, previous/default fallback, and unchanged behavior for every
  pre-existing method.
- Before a full real experiment, present the resolved tasks, methods, cases,
  seeds, rounds, evaluation counts, endpoint, model, wire API, thinking level,
  concurrency, tool budgets, output root, and resume policy for explicit user
  confirmation.
