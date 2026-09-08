# LDM Task Qualification Stages

Use this reference after registration scaffolding and before any production
claim or long-running campaign.

Record the current stage in
`tasks/<task_id>/resources/qualification_evidence.json`. Each completed gate
must cite existing repository-relative evidence paths. Validate a claim with
`scripts/validate_tasks.py --task <task_id> --require-stage <stage>`.

Evidence paths must exist in a clean Git checkout. Runtime directories such as
`runs/`, `ldm_runs/`, and `data/generated/` remain ignored and must not be cited
directly. For a completed campaign, add a compact record under
`tasks/<task_id>/resources/` containing the contract digest, execution result,
complete budget counters, scientific metrics, provenance, and SHA-256 digests
of the retained raw artifacts.

Do not equate a file's local existence with clean-checkout availability. Before
recording a passed gate, run `git status --short --untracked-files=all --
tasks/<task_id> config/<task_id>` and require
`git ls-files --error-unmatch -- <evidence-path>` to succeed for every path
listed by that gate. In particular, ensure `config/<task_id>/mock.yaml` is
tracked before citing it as `mock_verified` evidence; otherwise local validation
can pass while CI checks a checkout in which the entire config directory is
absent.

## 1. Registered

Required evidence:

- `task.json` is discoverable and the adapter exports `main(argv)`.
- `experiment.json` parses with schema version 1 and matches the task ID.
- The contract is `draft` when benchmark provenance or budgets remain unknown.
- No domain implementation lives in `ldm_task/`.
- No secrets, generated runs, environments, or downloaded assets are tracked.

## 2. Mock Verified

Required evidence:

- Mock generation and evaluation need no service, GPU, dataset, or secret.
- The shared config runner resolves the conventional module and working tree.
- At least one accepted action crosses the same parser used by real inference.
- Collection tests validate canonical IR and prevent provenance/outcome leakage.
- Search and evaluation counters match the mock topology exactly.
- A Harness-backed mock uses a protocol-faithful fake sidecar, exercises
  provisional submission validation and correction, and needs no container,
  KVM, endpoint, or secret.

## 3. Contract Verified

Required evidence:

- Candidate validation rejects unsafe, malformed, duplicate, and over-budget
  candidates before expensive evaluation.
- Harness profiles, optional skills, candidate schemas, and task tools have
  stable recorded digests; the task validator returns indexed rejection codes
  and actionable reasons before a turn commits.
- Fixed benchmark code remains unchanged outside the editable region.
- Tensor shapes, dtypes, finite outputs, parameter count, and requested devices
  pass the cheap contract evaluator.
- Parallel benchmark jobs map to the intended devices and per-job timeouts.
- Reported, optimized, and diagnostic metrics are distinct and documented. A
  reported metric may also be optimized when it provides a continuous signal.

## 4. Seed Evaluated

Required evidence:

- Benchmark URL, immutable commit, and task path come from a primary source.
- One seed candidate runs with official datasets, hyperparameters, random seed,
  checkpoint selection, epoch cap, training-hour cap, and parameter limit.
- The summary records every reported and diagnostic metric, source candidate,
  evaluator logs, and parameter count.
- The seed observation is explicitly outside or inside future campaign budget.

Only now change `qualification` from `draft` to `qualified`.

## 5. Tiny Campaign Verified

Required evidence:

- `experiment.json` declares the proposal-provider kind and capabilities.
- When `requires_endpoint_preflight` is true, a short authenticated preflight
  validates connectivity, provider/model identity, response shape, and latency
  before search begins. Endpoint checks are not gates for deterministic,
  dataset-backed, or simulator providers that declare the capability false.
- For a hybrid task, the selected method's provider mapping controls preflight;
  offline methods remain service-free while direct and Harness methods verify
  their actual wire API.
- A Harness backend additionally runs the real sidecar capability smoke using
  its configured wire API, container isolation, tools, and one persistent
  session. A Chat Completions-only probe does not qualify a Responses-based
  Harness.
- One configured test-time-search reservoir is generated and cheaply validated.
- Acquisition scores every valid candidate and selects exactly the configured
  number for expensive evaluation.
- The selected LDM candidate, not an unmodified benchmark baseline or standalone
  benchmark agent, enters the evaluator.
- `experiment_contract.json`, `budget.json`, `status.json`, search manifest,
  selection record, evaluation manifest, and summary are durable.
- Required-service failures open a circuit and produce a resumable paused state
  without consuming expensive evaluation budget.

## 6. Campaign Qualified

Required evidence:

- The real config selects a named `contract_profile`; dry-run validation rejects
  changes to official settings or campaign budget.
- `scripts/validate_tasks.py --task <task_id> --require-qualified` succeeds.
- `budget.json` separately limits and reports outer iterations, LLM requests,
  valid search candidates, expensive attempts, benchmark jobs, and completions;
  every declared counter is present even when its value is zero.
- Harness campaigns also report session turns and measured provider, web,
  Context7, and artifact usage without using those measurements as hidden stop
  conditions unless the experiment contract explicitly says so.
- Resume reconstructs state from terminal manifests and never repeats a
  completed expensive evaluation.
- Harness resume reuses committed turn identity and artifacts, never commits a
  partial submission, and never advances a session twice for the same history
  range and digest.
- Detached launch returns a durable execution handle, unbuffered log, heartbeat
  status, and unique run directory without copying credentials. The handle may
  be a local PID or a remote execution ID.
- Monitoring reports search phase, selected candidate, evaluator phase, device
  assignment, completed/remaining budget, best optimized metric, and official
  reported metric.
- Baseline and LDM comparisons declare the same primary expensive-evaluation
  budget. Extended-budget results are labeled separately.
- Artifact references are run-relative, and completed campaigns provide a
  portable `result.json` plus `trajectory.csv` when the task reports a scalar
  trajectory.

For Delta or another remote execution backend, archive pull, task-aware upload,
blocking cancellation (`kill --wait` or equivalent), and repeated terminal
status/cancel calls must be idempotent. These are backend requirements rather
than repository-local lifecycle implementations.

## Incident Rules

- Endpoint outage: pause and resume after a successful preflight; do not loop
  through the full search space with identical timeouts.
- Candidate contract failure: record a cheap rejection and preserve expensive
  budget.
- Evaluator failure after launch: count an expensive attempt, preserve logs, and
  resume at the next unevaluated selection unless the benchmark says otherwise.
- Contract/profile mismatch: stop before importing the task procedure.
- Missing provenance or official budget: keep qualification at `draft` and do
  not present results as benchmark-comparable.
