# Pi Research Harness Sidecar

This directory contains the pinned Node sidecar used by persistent LDM research
sessions. It owns Pi session lifecycle, Gondolin-isolated file and shell tools,
web extensions, terminal candidate submission, and raw model-provider transport
capture. Task validation, optimization history, `q0`, GP inference, acquisition,
and evaluation remain in Python.

For the task-neutral Python interface, task ownership boundary, resource
layout, and qualification rules, see
[`docs/research-harness.md`](../../docs/research-harness.md).

Build the release image from the repository root:

```bash
docker build -t ldm-pi-harness:latest harnesses/pi
```

The Python task runner starts the image over stdin/stdout JSONL. API keys are
sent once through the bootstrap frame and are not placed in container arguments,
environment variables, manifests, or session files. Do not invoke the sidecar
manually for normal experiments.

The strict JSONL protocol binds every request and response to one protocol
version and campaign. Turn inputs include a monotonic history range and digest;
the sidecar advances each persistent session only after an atomic turn commit.
Committed turns are idempotent and partial submissions recover from their saved
candidate batch and measured usage.

During `run_turn`, `submit_candidates` is provisional until the Python caller
answers a `submission_validation_requested` frame. Acceptance persists the
batch and allows commit. Rejection is recorded as a model-visible Pi tool error
with indexed task-provided reasons; the same session must correct and resubmit
within the original wall-time window. The protocol is task-neutral: candidate
identity and domain validation remain in the task-owned Python callback.

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

Each run stores only:

- native Pi session JSONL with messages, tool calls, and tool results;
- redacted raw model-provider request and response bodies plus `provider_index.jsonl`;
- `input.json`, `submission.json`, and `turn_committed.json` for recovery and lineage;
- one run `manifest.json`.

Agents may make as many provider and tool calls and write as much trace data as
needed within the configured wall-time limit. Provider, web, Context7, and
artifact usage is measured but never used to stop a turn. If a provider stream
ends before a committed batch, the sidecar continues submission-only recovery
within the same wall-time window and retains every raw attempt.

The container requires Linux KVM for Gondolin. The task runner mounts run
artifacts, read-only task profiles, and the Gondolin image cache explicitly.
Tasks may also register digest-verified, read-only tool extensions; their tool
names and source digests are recorded in the run manifest.

The Gondolin guest grants its Agent root-level shell and file access inside the
isolated microVM and unrestricted outbound HTTP(S) when a task leaves the host
allow/deny lists empty. The guest sees only its session workspace, not the host
repository, benchmark data, oracle, credentials, or other sessions. Task query
rules that prevent benchmark leakage remain independent from sandbox network
permissions.
