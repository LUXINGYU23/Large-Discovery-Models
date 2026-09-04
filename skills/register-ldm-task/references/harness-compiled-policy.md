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
   validates execution outputs against task semantics. Its exact
   `mean_context` and `weight_context` must be safe to expose through
   `inspect_policy_contract`; include target location/scale, feature names and
   groups, task defaults, and only the aggregate diagnostics the generated
   functions are allowed to consume.
3. Wire the prior mean into the existing task surrogate as a residual GP:
   subtract the standardized prior from measured targets, fit the unchanged GP,
   and add the prior back to query predictions. Kernel, variance, noise,
   acquisition, and numerical safeguards remain fixed.
4. Prove zero-prior parity: a zero prior with default `alpha` and `eta` must
   reproduce the existing `ldm_harness` predictions and selection distribution.
5. Add one task-owned `policy_architect/AGENTS.md` and one task-owned
   `resources/harness/skills/compile-ldm-policy/` skill. Load both through the
   policy profile from the task's read-only `/resources` mount. The sidecar
   digest-verifies and snapshots the selected files into its session workspace;
   Pi must advertise the read-only guest path under
   `/workspace/.ldm-resources`, not the sidecar-only source path. Do not put task
   runtime skills in the repository-level developer `skills/` directory.
   Document what every feature means, which groups are collinear or masked,
   what scientific structures cannot be represented, and that raw utility must
   be standardized before fitting the prior mean.

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

Draft evaluation should expose enough current-history diagnostics to catch a
raw-versus-standardized scale error, reversed sign, non-finite output, or
aggressive clipping. Label fit statistics as in-sample; do not present them as
prequential accuracy, GP calibration, or evidence that a more flexible mean
will optimize better.

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
  exact execution-context inspection, scale/sign diagnostics, epoch resume,
  previous/default fallback, and unchanged behavior for every pre-existing
  method.
- Before a full real experiment, present the resolved tasks, methods, cases,
  seeds, rounds, evaluation counts, endpoint, model, wire API, thinking level,
  concurrency, tool budgets, output root, and resume policy for explicit user
  confirmation.
