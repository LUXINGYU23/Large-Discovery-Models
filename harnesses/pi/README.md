# Pi Research Harness Sidecar

This directory contains the pinned Node sidecar used by persistent LDM research
sessions. It owns Pi session lifecycle, Gondolin-isolated file and shell tools,
web extensions, task-defined terminal submission, and raw model-provider
transport capture. Task validation, optimization history, `q0`, GP inference,
acquisition, and evaluation remain in Python.

The Responses model declares `supportsStrictMode` so Pi explicitly transmits
each tool's strict policy. Ordinary tools use `strict:false`, preserving omitted
optional arguments instead of provider-side schema normalization. Task validators
still enforce complete submissions and scientific legality before commit.

For the task-neutral Python interface, task ownership boundary, resource
layout, and qualification rules, see
[`docs/research-harness.md`](../../docs/research-harness.md).

Build the release image from the repository root:

```bash
docker build -t ldm-pi-harness:latest harnesses/pi
```

Each image build resolves the current SoL-Pi `main` revision and installs its
exact tested Pi package versions. The resolved pair is recorded in
`/app/sol-pi-version.json`; session manifests also record package versions,
the SoL-Pi source URL, and the effective lockfile digest. Pin the resulting
image digest for an entire experiment and its resumes.

## SoL-Pi

SoL-Pi is disabled unless `PiHarnessConfig.sol_pi` supplies its configuration.
NucleoBench exposes this through `--harness-sol-pi-config`:

```json
{
  "version": 1,
  "actionFusion": true,
  "observationPack": true,
  "evidencePreservingReducer": true,
  "onlineContextCompact": true,
  "cacheWriteReadRatio": 50
}
```

Use the provider's cache-miss/cache-hit price ratio for
`cacheWriteReadRatio`; the value above is an example, not a universal default.
The effective `sol-pi.json` is saved in each session's `pi-agent` directory.
The reducer uses that session's model and traced provider. Its calls and Pi
compaction calls are included in the provider trace, alongside ordinary research
requests. Provider request-body settings apply to all these calls. A provider that
does not support forced tool use can set `tool_choice: "auto"` in those settings;
the Harness still requires a validated terminal submission.

The upstream mechanisms remain unchanged. Action Fusion's file operations and
follow-up commands use the task guest; its hash checks use the same shared
workspace files. Observation and evidence archives remain under the native
session directory and are mounted read-only at `/workspace/.sol-pi` for research.
`obs_recall` retrieves archived observations; `update_plan` supplies boundaries
for native compaction and continuation. Enabling the extension does not alter
task submissions, measured-history validation, GP queries, or evaluation budgets.

See the [SoL-Pi documentation](https://github.com/NVlabs/SoL-Pi) for mechanism
details and the supported configuration schema.

## Task Guest

Before the first Harness run for a task, build its task-owned guest into a
user-selected cache and run its offline smoke script:

```bash
npm --prefix harnesses/pi ci
npm --prefix harnesses/pi run build:task-guest -- \
  --task synthonbench --cache-dir /path/to/harness-cache
npm --prefix harnesses/pi run smoke:task-guest -- \
  --task synthonbench --cache-dir /path/to/harness-cache
```

The build command requires Docker, `e2fsprogs`, `cpio`, and `lz4` on a Linux
host. Guest smoke additionally requires Linux KVM and the host-architecture
QEMU system emulator. The build turns the task's pinned Docker recipe into a
digest-addressed Gondolin image. The runtime resolves only that local image; it
neither builds nor downloads a guest during a campaign.

The Python task runner starts the image over stdin/stdout JSONL. API keys are
sent once through the bootstrap frame and are not placed in container arguments,
environment variables, manifests, or session files. Do not invoke the sidecar
manually for normal experiments.

The sidecar declares its package SemVer at startup; the client binds every
subsequent JSONL request and response to that release and one campaign. Turn
inputs include a monotonic history range and digest;
the sidecar advances each persistent session only after an atomic turn commit.
Committed turns are idempotent and partial turns recover from their saved
submission, artifact descriptors, and measured usage.
Initialization pins configuration, profile/tool digests, resolved guest, and
sidecar implementation. Resume rejects a different identity without rewriting
the original manifest. Use a new artifact root for a changed configuration.

During `run_turn`, the terminal tool named by `HarnessSubmissionContract` is
provisional until the Python caller answers a
`submission_validation_requested` frame. `retry` returns task-provided
JSON-Pointer errors to the same session, with research and editing tools still
available for repair; `accept` commits the turn, while
`reject_turn` records a terminal failure for caller fallback. The protocol is
task-neutral: payload meaning and domain validation remain in the task-owned
Python callback.

Execution errors carry `error.turnUsage` entries for available per-profile,
per-turn provider calls, tool calls, and captured artifact bytes, including
work completed before a provider failure or wall-time limit. Failed turns do
not commit or advance history. If the transport exits before delivering usage,
the client preserves unknown counters rather than reporting zero.

Recoverable execution failures use `error.code=recoverable_turn_error`. A caller
may retry within its task-owned recovery window. The same batch and turn IDs
replay committed profiles and continue only unfinished sessions; quota usage and
validation attempts are retained. Continuation messages do not duplicate the
history delta. Authentication, protocol, and integrity errors remain fatal.
On transport exit or response timeout, Python recreates the sidecar with the
same persistent artifact root and replays the turn batch. Credentials remain
private in-memory until close so the replacement process can be bootstrapped;
they are never added to subprocess arguments or inherited environment.
Timeout cancellation aborts Pi compaction as well as ordinary inference.
See [partial-turn recovery](../../docs/research-harness.md) for the Python API
and time-budget semantics.

Contracts may declare file fields. The sidecar rejects unsafe paths, symlink
escapes, unsupported suffixes, missing files, and oversized files, then stores
an immutable per-attempt snapshot before task validation. Wire records contain
artifact descriptors and digests, never duplicate source text.

## Compiled Optimization Policies

Harness-Compiled LDM uses the same sidecar twice: one task-defined pool owns
candidate-proposal sessions, and a separate pool owns one persistent policy
session. The policy terminal contract accepts `replace`, `keep`, or `disable`.
`replace` must reference a complete `optimization_policy.py`; source code is
never embedded in the terminal JSON payload.

The policy pool loads its task-owned `compile-ldm-policy` skill from
`resources/harness/skills/`. The sidecar verifies its digest, snapshots the
selected package into the session workspace, and advertises its read-only guest
path under `/workspace/.ldm-resources`; this lets Pi load the full Skill and its
relative references without mounting the repository. The pool also includes
the built-in `ldm_policy` MCP server. Its `inspect_policy_contract`,
`validate_policy_draft`, and `evaluate_policy_draft` tools let the Agent inspect
the authoritative feature contract and repair a draft before submission. These
tools are advisory. Inspection returns the contract and file paths under
`guest_snapshot.directory`. Its read-only `input.json` contains the exact
task-supplied `execution_context`, including `mean_context` and `weight_context`;
`research_snapshot.json` contains task research context and measured-record
indexes, while `arrays.npz` contains numeric inputs. Exact designs and original
notes are available through task-owned history tools. Agents query the relevant
files or records rather than loading all history into a tool response. The runner preserves
single-objective vectors or multiobjective matrices; objective meaning and GP
diagnostics are task-owned. A trusted read-only diagnostic module is configured
through `LDM_POLICY_DIAGNOSTICS` and `LDM_POLICY_DIAGNOSTICS_SHA256`.
Its digest is verified before loading. See the
[task hook contract](../../docs/research-harness.md#task-responsibilities).
Without a task hook, evaluation reports artifact outputs only.
Draft validation and evaluation use the existing sidecar Python subprocess.
They do not start another container or microVM.

The three reference tasks' draft-evaluation hooks compare
chronological measured-history holdouts with training-prefix GP hyperparameters
frozen, and report current-pool first-draw distribution changes. These are
development diagnostics, not an untouched test or a closed-loop counterfactual.
The weight context also contains exact normalization, candidate predictions,
measured progress, and errors for predictions frozen before real measurements.
Measured feedback retains the original pool-relative q0, competition ranks,
first-draw probability and alpha/eta. Do not compare historical q0 with the
current pool maximum. A positive residual is underprediction, not validation
of acquisition ranking; GP holdout error does not evaluate sampling weights.

After immutable snapshotting, task-owned Python executes the accepted artifact
again in a separate read-only, network-disabled container with bounded CPU,
memory, processes, and time. The host interpreter never imports or executes
Agent-authored code. Terminal payloads are exact: `replace` includes only
`action` and `artifact_path`; `keep` and `disable` include only `action`.

Candidate and policy clients have independent manifests, sessions, turn state,
tool budgets, and usage counters. Reference tasks store them under
`<run_dir>/harness/` and `<run_dir>/policy_harness/` respectively.

Pi sessions use a 262,144-token model context window with built-in automatic
compaction enabled. The release configuration reserves 16,384 tokens for the
next response and retains the most recent 20,000 tokens verbatim; Pi writes
compaction entries into its native session history.

The sidecar writes the configured ordered provider route to Pi Web Access.
`provider: auto` uses automatic fallback, while an Agent may explicitly retry
with any provider in that route when returned sources are unsuitable. Provider
names outside the route are rejected with a model-visible structured reason;
they are never silently substituted. The default route is `parallel-mcp`,
`exa`, then `duckduckgo`; all are usable without a task-owned search API key.

The sidecar can also load explicitly allowlisted MCP tools over stdio or
Streamable HTTP. Python resolves environment- or file-backed secrets before
bootstrap; only secret references and redacted configuration digests are
persisted. MCP tools are exposed as `mcp__<server_id>__<tool_name>`. Server
lifecycle, protocol failures, and calls are recorded in the native Pi session.

Each run stores only:

- native Pi session JSONL with messages, tool calls, and tool results;
- redacted raw model-provider request and response bodies plus `provider_index.jsonl`;
- `input.json`, `submission.json`, and `turn_committed.json` for recovery and lineage;
- one run `manifest.json`, including the resolved guest and its environment snapshot.

Each turn has a wall-time limit and may define hard limits for individual tools.
The Agent sees its initial tool budget and the remaining count after every
call. Unlisted tools are unlimited and a zero limit disables a tool.
The configured terminal tool cannot be limited. A started tool execution consumes one
call even when it fails; policy and budget rejections do not. Reservations are
persisted before execution, so an interrupted turn resumes with the same used
counts. If a provider stream ends before a committed batch, the sidecar
continues the existing work with tools available for repair within the same
wall-time window and retains every raw attempt. Partial-turn recovery continues
attempt numbering rather than replacing earlier artifact snapshots.
Transient provider failures, including structured `server_error` responses,
use the task's recovery window and backoff. Already committed sessions are
replayed without regeneration; authentication and contract errors remain fatal.

The container requires Linux KVM for Gondolin. The task runner mounts run
artifacts, read-only task resources, and the selected guest cache explicitly.
The cache contains the immutable image store, build records, transient build
state, Gondolin session state, and per-session COW overlays; it is outside the
repository and may be removed when no campaign needs the images. Tasks may also register
digest-verified, read-only tool extensions; their tool names and source digests
are recorded in the run manifest.

The Gondolin guest grants its Agent root-level shell and file access inside the
isolated microVM and unrestricted outbound HTTP(S) when a task leaves the host
allow/deny lists empty. The guest sees only its session workspace, not the host
repository, benchmark data, oracle, credentials, or other sessions. Task query
rules that prevent benchmark leakage remain independent from sandbox network
permissions.

### Guest Runtime Contract

Every session receives the same runtime note: `/workspace` is a research
workspace rather than a repository checkout, files are edited through Pi's
registered `read` and `write` tools, and task or MCP tools are invoked directly.
`apply_patch` is not a guest command, and task-tool implementation runtimes such
as Node.js are not part of the guest contract. Guest `git` is reserved for
cloning useful public research material when network policy permits it.
