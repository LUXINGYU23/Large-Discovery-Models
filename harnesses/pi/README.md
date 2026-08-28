# Pi Research Harness Sidecar

This directory contains the pinned Node sidecar used by persistent LDM research
sessions. It owns Pi session lifecycle, Gondolin-isolated file and shell tools,
web extensions, terminal candidate submission, and raw model-provider transport
capture. Task validation, optimization history, `q0`, GP inference, acquisition,
and evaluation remain in Python.

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

Each run stores only:

- native Pi session JSONL with messages, tool calls, and tool results;
- redacted raw model-provider request and response bodies plus `provider_index.jsonl`;
- `input.json`, `submission.json`, and `turn_committed.json` for recovery and lineage;
- one run `manifest.json`.

Pi provider retries are disabled. The sidecar performs only one bounded,
submission-only recovery when a Responses stream is interrupted before terminal
submission, retaining every raw attempt under the same turn budget.

The container requires Linux KVM for Gondolin. The task runner mounts run
artifacts, read-only task profiles, and the Gondolin image cache explicitly.
