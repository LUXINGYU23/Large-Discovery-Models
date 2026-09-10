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

## Implement the task-local policy adapter

1. Add a deterministic public feature encoder under `tasks/<task_id>/core/`.
   Features may use released task data and measured observations, but must not
   expose hidden labels, candidate identity, row position, `q0`, acquisition
   values, selection probabilities, or future outcomes.
2. Add an `OptimizationPolicyAdapter` that publishes a versioned
   `PolicyCapabilityContract`, materializes authoritative round inputs, and
   validates execution outputs against task semantics. Its exact
   `mean_context` and `weight_context` must be safe to expose through
   `inspect_policy_contract`; include task-defined target transforms, feature
   names and groups, defaults, and the diagnostics each generated function is
   allowed to consume. Keep pool identities, `q0`, and acquisition signals in
   `weight_context`, not in `mean_context`.
   Populate `PolicyRoundInput.history_candidate_ids`, `history_rounds`, and
   `measured_observations` explicitly, aligned with numeric history; do not
   duplicate those fields in `research_snapshot`. Preserve all objective
   columns: `history_utilities` and prior outputs can be vectors or matrices.
   Define objective order, direction, and scaling in the task context.
   Set `PolicyRoundInput.round_index` from the selector's engine-supplied
   `round_idx`, never from the last successful GP observation. Test a round
   following failed measurements so policy epochs cannot collide.
   Keep measured message rows compact and provide task-owned detailed history
   lookup, including original research notes, when full designs are large.
   Expose requested and effective evaluation batch sizes in `weight_context`
   when interpreting without-replacement selection. Those signals must not
   leak into the deployable mean features.
   Derive session counts, minibatch sizes and occurrence rules from the actual
   proposal configuration, not a second set of defaults. Keep these facts in
   the task's weight context. Evaluate selection headroom under the task's
   actual batch contract before expensive policy research.
3. Wire the prior mean into the existing task surrogate as a residual GP:
   subtract the prior on the task-declared target scale, fit the unchanged GP,
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
   what scientific structures cannot be represented, and the exact objective
   order, direction, units, and target transformations. Standardization is the
   reference tasks' choice, not a shared requirement. Document only enabled
   capabilities; a weights-only contract does not require prior-mean research.
   Give the Agent an explicit task baseline and require evidence for changing
   either weight signal or the mean. Use the current measured stage, not fixed
   round switches. Distribution diagnostics such as ESS are not optimization
   objectives; residual errors on selected points do not establish a better
   sampling policy. Numerical validity alone does not demonstrate improvement.
   Make artifact declarations and required exports match `enabled_capabilities`
   exactly. Qualify the documented single-capability examples against the real
   runner, not just the default dual-capability contract.

The current release supports only `prior_mean@1` and `ldm_weights@1`. Add a new
versioned capability and task adapter path before exposing another editable
component; do not broaden the meaning of an existing capability.

Keep GP holdouts, acquisition normalization, prediction errors, and optimization
progress under the task, not in the shared controller or Pi runner. Implement
`with_feedback(round_input, records)` on the adapter. Build task-defined rows
before calling `controller.record_predictions(round_index, predictions)`.
For draft evaluation, register a read-only task resource implementing
`evaluate_policy_draft(prior, inputs, outputs)` through
`ldm_tts.harness.pi.policy_mcp_server(diagnostics_path=..., diagnostics_sha256=...)`; see the
shared Harness documentation for its inputs. Only the task can determine
whether UCB, RMSE, a Pareto metric, or another diagnostic is appropriate.
Declare scientific report paths in the task matrix's `policy_fields` and scalar
aggregates in `policy_mean_fields`; do not add task metric names or scalarization
to the shared reporter.
Proposal profile count belongs to the task, not the shared policy lifecycle.

Use established numerical routines for ties and undefined statistics. Pair
frozen predictions with measurements by candidate identity and evaluation event;
do not pool raw proposal frequencies across unrelated rounds. Distinguish
historical development checks, prospective selected-point accuracy, and actual
optimization benefit. Repeated draft selection can overfit the same holdouts.

## Artifact and fallback contract

Use the generic `HarnessSubmissionContract` with one terminal action:

- `replace`: submit a complete `optimization_policy.py` artifact;
- `keep`: revalidate and reuse the active policy epoch;
- `disable`: use the static zero-prior/default-weight policy.

Snapshot the file before task validation. Pi's built-in policy MCP uses a
sidecar Python subprocess to inspect, validate, and evaluate drafts. The
campaign must execute the
accepted snapshot again through `PolicyExecutor` with read-only inputs, no
network, and bounded CPU, memory, processes, and time. Never execute Agent code
with the host interpreter.

Draft evaluation should expose enough current-history diagnostics to catch a
raw-versus-standardized scale error, reversed sign, non-finite output, or
aggressive clipping. Label fit statistics as in-sample; do not present them as
prequential accuracy, GP calibration, or evidence that a more flexible mean
will optimize better.

Return exact JSON-Pointer errors to the same session for repair. Keep research,
file-editing, and validation tools available after rejection. Bound formal
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
  Attach compiled policy metadata to the shared engine's `candidates_selected`
  event in `events.jsonl`; no additional task-specific selection export is
  required by Pilot Evaluation. Test report collection with the task's actual
  initialization count, which may differ from its optimization batch size.
  Count every policy attempt, including runtime failures that use fallback;
  distinguish committed turns, failed attempts, and incomplete usage. Never
  substitute zero for unavailable provider or tool counts. Cached decisions
  must not charge the same attempt again.
- Test feature determinism and leakage boundaries, strict artifact paths and
  digests, repair, isolated execution, zero-prior parity, residual-GP behavior,
  exact execution-context inspection, scale/sign diagnostics, epoch resume,
  previous/default fallback, and unchanged behavior for every pre-existing
  method.
- Before a full real experiment, present the resolved tasks, methods, cases,
  seeds, rounds, evaluation counts, endpoint, model, wire API, thinking level,
  concurrency, tool budgets, output root, and resume policy for explicit user
  confirmation.
