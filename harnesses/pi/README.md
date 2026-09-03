# Pi Research Harness Sidecar

This directory contains the pinned Node sidecar used by persistent LDM research
sessions. It owns Pi session lifecycle, Gondolin-isolated file and shell tools,
web extensions, task-defined terminal submission, and raw model-provider
transport capture. Task validation, optimization history, `q0`, GP inference,
acquisition, and evaluation remain in Python.

For the task-neutral Python interface, task ownership boundary, resource
layout, and qualification rules, see
[`docs/research-harness.md`](../../docs/research-harness.md).

Build the release image from the repository root:

```bash
docker build -t ldm-pi-harness:latest harnesses/pi
```

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

During `run_turn`, the terminal tool named by `HarnessSubmissionContract` is
provisional until the Python caller answers a
`submission_validation_requested` frame. `retry` returns task-provided
JSON-Pointer errors to the same session; `accept` commits the turn, while
`reject_turn` records a terminal failure for caller fallback. The protocol is
task-neutral: payload meaning and domain validation remain in the task-owned
Python callback.

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

The policy pool includes the repository `compile-ldm-policy` skill and the
built-in `ldm_policy` MCP server. Its `inspect_policy_contract`,
`validate_policy_draft`, and `evaluate_policy_draft` tools let the Agent inspect
the authoritative feature contract and repair a draft before submission. These
tools are advisory: after immutable snapshotting, task-owned Python executes the
accepted artifact again in a separate read-only, network-disabled container
with bounded CPU, memory, processes, and time. The host interpreter never
imports or executes Agent-authored code.

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
continues submission-only recovery within the same wall-time window and retains
every raw attempt.

The container requires Linux KVM for Gondolin. The task runner mounts run
artifacts, read-only task profiles, and the selected guest cache explicitly.
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
