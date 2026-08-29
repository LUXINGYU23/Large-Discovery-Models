# Research Harness Integration

The research Harness is an optional reservoir-expansion backend for tasks that
need persistent Agents to inspect evidence, use tools, run sandbox analysis, and
submit candidate minibatches. It replaces a direct model request inside the
task expander. It does not replace the shared LDM campaign, BO model,
acquisition rule, candidate domain, or evaluator.

SynthonBench and Iron Mind are the reference integrations. Each runs four
persistent Pi sessions with different `AGENTS.md` profiles and routes accepted
candidate occurrences through the same task-owned empirical `q0`, GP-UCB,
acquisition tilt, and official evaluation path used by its direct backend.

## Architecture

```text
shared Campaign / LDMEngine
  -> task-owned ReservoirExpander
     -> ldm_tts.harness.HarnessClient
        -> versioned JSONL sidecar protocol
           -> persistent Pi sessions and isolated tools
        <- provisional candidate submissions
     -> task-owned submission validation
        -> reject with indexed reasons and continue the same turn
        -> or accept and commit the turn
  -> task-owned CandidateDomainAdapter
  -> surrogate and acquisition
  -> authoritative evaluator and Observation history
```

The Campaign owns optimization history. Each Harness session owns its private
research transcript. On every active round, the task projects the authoritative
history into a monotonic delta plus any compact exclusion snapshot needed to
prevent historical repeats.

## Shared Interface

The public `ldm_tts.harness` package provides:

| Type | Purpose |
| --- | --- |
| `HarnessPoolConfig` | Campaign, task, case, provider, profile, strict candidate JSON Schema, tool, network, and limit configuration. |
| `HarnessProfile` | One persistent Agent identity, `AGENTS.md`, optional skill directories, candidate count, and content digests. |
| `HarnessToolExtension` | A digest-verified task tool module and its exact exported tool names. |
| `HarnessLimits` / `HarnessNetworkPolicy` | Per-turn wall-time and network/query policy. |
| `HarnessTurn` | One profile's round, history range and digest, task message, and forbidden query terms. |
| `HarnessClient` | Long-lived sidecar process, secret bootstrap, strict request/response validation, and turn execution. |
| `HarnessSubmissionRequest` | One provisional minibatch submitted by a profile during a turn. |
| `HarnessSubmissionValidation` / `HarnessSubmissionRejection` | Task-owned acceptance or indexed rejection reasons for a provisional submission. |
| `HarnessTurnResult` | Committed candidates, session/turn lineage, measured usage, and artifact references. |

The current Pi sidecar lives in `harnesses/pi` and uses the OpenAI Responses
wire format. It owns session lifecycle, automatic context compaction, isolated
file and shell tools, web and Context7 extensions, terminal candidate
submission, and redacted raw provider capture. The shared Python protocol is
task-neutral and does not understand domain candidate identities.

## Task Responsibilities

A Harness-enabled task keeps its integration in `tasks/<task_id>/core/` and
implements it behind the ordinary `ReservoirExpander` interface. The task must:

1. Start one `HarnessClient` for the campaign and close it in a `finally` block.
2. Define stable profiles and candidate counts. Skills are optional and may be
   omitted while preserving the interface.
3. Build deterministic turns from campaign/profile/round/history identity.
4. Send newly measured observations for reasoning and a compact authoritative
   evaluated-candidate snapshot when historical repeats are forbidden.
5. Validate provisional submissions with the same parser and canonical identity
   used by candidate admission.
6. Return stable rejection codes and actionable messages for every rejected
   index, allowing correction within the same session and wall-time window.
7. Accept only a complete valid minibatch before computing proposal frequencies
   or entering surrogate/acquisition selection.
8. Count Harness turns and measured provider/tool usage through the campaign
   budget ledger.

Duplicate policy is task-owned. A task using empirical proposal frequency must
state whether agreement across independent sessions represents additional
probability mass. It must not globally deduplicate meaningful occurrences before
estimating `q0`.

## Resources And Tools

Versioned task inputs belong under:

```text
tasks/<task_id>/resources/harness/
|-- profiles/<profile_id>/AGENTS.md
|-- profiles/<profile_id>/skills/   # optional
`-- tools/                           # optional structured task tools
```

Record SHA-256 identities for profile instructions, skill directories,
candidate schemas, and tool sources. Mount these inputs read-only. A task tool
may expose a safe structured view of official benchmark data, but scientific
validation remains in Python and cannot be delegated to the Agent or tool.

Pass the actual strict candidate JSON Schema, not only a digest or prose
example. The sidecar combines it with each profile's exact minibatch count for
`submit_candidates`; task Python still performs authoritative domain and
history validation before accepting a provisional submission.

## Secrets And Artifacts

Pass the provider key once through the Harness bootstrap frame. Do not place it
in container arguments, container environment variables, configs, prompts,
manifests, or trace files.

Write Harness outputs below `<run_dir>/harness/`. The Pi sidecar retains native
session JSONL, redacted provider requests and responses, input/submission/commit
records, and a run manifest. These files are raw research traces. They are not
canonical `ldm-2.0` accepted-action records and require a separate conversion
and leakage contract before training use.

## Qualification

A mock test should use a protocol-faithful fake sidecar and cover turn identity,
strict minibatch cardinality, rejection and correction, budget accounting, and
lineage without Docker, KVM, a network endpoint, or a secret.

Before claiming a tiny real Harness campaign, run the sidecar's unit tests and a
real capability smoke with the configured wire API, container isolation,
profiles, and task tools. Then verify one Harness-generated reservoir enters the
normal acquisition and official evaluator path. See
the [Pi sidecar contract](../harnesses/pi/README.md) for the current runtime
requirements and the task README for the concrete workflow.
