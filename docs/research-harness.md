# Research Harness Integration

The research Harness lets a persistent Agent inspect optimization history, use
tools, work in an isolated sandbox, and make a task-defined terminal
submission. It is available through three search methods:

| Method | Candidate generation | Selection before evaluation |
| --- | --- | --- |
| `ldm_harness` | Multiple task-defined persistent Agents submit an over-sampled reservoir. | The task estimates empirical `q0`, fits its surrogate, and applies the fixed LDM acquisition policy. |
| `ldm_harness_compiled` | The same proposal sessions build the reservoir, while one independent persistent policy Agent submits a complete Python policy artifact. | The task validates and executes the artifact, uses its prior mean in the residual GP, and applies its round-specific `alpha` and `eta`. |
| `harness` | One task-defined persistent Agent submits exactly the evaluation minibatch. | None. Every accepted candidate is evaluated in stable submission order. |

Iron Mind, SynthonBench, and NucleoBench are the reference integrations. The shared Harness
does not know their candidate identity, legal search space, duplicate policy,
surrogate, or evaluator.

## Architecture

```text
shared Campaign / LDMEngine
  -> task-owned ReservoirExpander
     -> proposal HarnessClient and persistent Pi sessions
        -> sidecar-release JSONL protocol
           -> isolated shell and file tools
           -> web, Context7, task-local, and configured MCP tools
        <- provisional structured submission
     -> task-owned validation
        -> retry with JSON-Pointer errors and continue the same turn
        -> or accept and commit the turn
  -> ldm_harness_compiled only:
     -> independent policy HarnessClient and persistent policy session
        <- complete optimization_policy.py plus replace/keep/disable action
     -> immutable artifact snapshot
     -> isolated policy runner and task-owned validation
     -> accepted policy epoch or deterministic fallback
  -> ldm_harness: fixed task surrogate and acquisition selector
  -> ldm_harness_compiled: task residual GP and compiled LDM weights
  -> harness: stable reservoir order
  -> authoritative task evaluator and Observation history
```

The Campaign owns measured optimization history. Each Pi session owns its
private research transcript. A task sends an initial history snapshot and then
monotonic deltas, together with the compact exclusion set needed to reject
previously evaluated candidates.

## Shared Interface

The public `ldm_tts.harness` package provides:

| Type | Purpose |
| --- | --- |
| `HarnessPoolConfig` | Backend-neutral campaign identity, profiles, submission contract, and limits. |
| `HarnessProfile` | One persistent Agent identity, `AGENTS.md`, optional skill directories, and content digests. |
| `HarnessSubmissionContract` | Dynamic terminal tool name, strict payload schema, artifact rules, and optional validation-attempt limit. |
| `HarnessArtifactRule` | One JSON-Pointer file reference with allowed suffixes and a size limit. |
| `HarnessToolExtension` | A digest-verified task tool module and its exported tool names. |
| `HarnessMcpServer` | One allowlisted stdio or Streamable HTTP MCP server. |
| `HarnessLimits` | Per-turn wall time and per-tool call budgets. |
| `HarnessTurn` | One profile round with deterministic history lineage and task input. |
| `HarnessClient` | Sidecar lifecycle, secret bootstrap, protocol validation, and turn execution. |
| `HarnessSubmissionValidation` | Task-owned `accept`, `retry`, or `reject_turn` decision with actionable errors. |
| `HarnessTurnResult` | Committed generic submission, artifact descriptors, session lineage, and measured usage. |
| `PolicyCapabilityContract` | Versioned task feature schema and enabled compiled-policy capabilities. |
| `PolicyRoundInput` | Aligned numeric history, explicit history identities and rounds, measured records, and task-owned contexts. |
| `OptimizationPolicyAdapter` | Task-owned prediction feedback and scientific execution validation. |
| `PolicyResearchController` | Policy rounds, artifact validation, epoch state, resume, and deterministic fallback. |
| `PolicyExecutor` | Isolated execution boundary for accepted policy artifacts. |

Pi-specific configuration is imported from `ldm_tts.harness.pi`:
`PiHarnessConfig` extends the pool configuration with its provider, tools,
MCP servers, network policy, and `PiGuestRuntime`. Other backends can extend
`HarnessPoolConfig` without importing Pi or declaring a Gondolin guest.

The Pi sidecar in `harnesses/pi` uses the OpenAI Responses wire format. It owns
session lifecycle, automatic context compaction, isolated file and shell tools,
web and Context7 extensions, MCP clients, terminal submission, and redacted
provider capture.

## MCP Tools

Pass `--harness-mcp-config` a YAML file with an explicit server and tool
allowlist. Supported transports are stdio and Streamable HTTP:

```yaml
servers:
  local_analysis:
    transport: stdio
    command: node
    args: [/absolute/path/to/server.js]
    env:
      SERVICE_TOKEN:
        secret_file: /absolute/path/to/protected-token
    tools: [analyze_candidate]
  literature:
    transport: streamable_http
    url: https://mcp.example.org/mcp
    headers:
      Authorization:
        secret_env: LITERATURE_API_KEY
        prefix: "Bearer "
    tools: [search, fetch]
```

The Agent sees `mcp__local_analysis__analyze_candidate`,
`mcp__literature__search`, and `mcp__literature__fetch`. An empty allowlist is
invalid. HTTP endpoints require HTTPS except for loopback tests, and URLs cannot
contain credentials. Literal values use `{value: ...}`; secrets use
`secret_env` or `secret_file` and are resolved by Python before sidecar
bootstrap. Secret values are excluded from commands, manifests, sessions, and
provider captures.

## Tool Budgets

`--harness-tool-budget NAME=COUNT` sets a hard per-Agent, per-optimization-turn
limit for proposal sessions. Tasks with an independent compiled-policy session
use `--policy-tool-budget NAME=COUNT` for that session. In a runner YAML, use a
list when configuring more than one tool:

```yaml
args:
  harness-tool-budget:
    - web_search=4
    - fetch_content=8
    - mcp__literature__search=2
  policy-tool-budget:
    - web_search=8
    - fetch_content=16
```

The default network budgets are eight `web_search` calls, sixteen
`fetch_content`, sixteen `get_search_content`, four `resolve-library-id`, and eight
`query-docs` calls per turn. Context7 budgets are omitted when Context7 is
disabled. A tool absent from the mapping is unlimited; zero disables it.
The terminal tool named by the submission contract cannot be budgeted.

The Agent receives the budget snapshot at turn start and the remaining count
after each call. A tool execution attempt consumes one call even when the tool
returns an error. A policy rejection or budget rejection does not consume a
call. Reservations are persisted before execution, so resuming an interrupted
turn cannot reset or double-spend its budget.

## Task Responsibilities

A Harness-enabled task keeps its adapter in `tasks/<task_id>/core/` and must:

1. Start one `HarnessClient` per independent pool and close each in a `finally`
   block.
2. Define one profile set and strict submission contract per pool.
3. Build deterministic turns from campaign, profile, round, and history identity.
4. Send newly measured observations and, when historical repeats are forbidden,
   the authoritative evaluated-candidate exclusion snapshot.
5. Validate provisional submissions with the same parser, canonical identity,
   and official-space checks used by candidate admission.
6. Return stable JSON-Pointer paths, rejection codes, messages, and repair hints
   so the Agent can repair the same submission in-session.
7. Refill until the complete valid minibatch is accepted.
8. Preserve meaningful same-round occurrences before estimating `q0` for both
   Harness-backed LDM methods. Define real-evaluation replicate semantics in
   the task contract.
9. Record each pool's turns and measured provider/tool usage separately in the
   campaign budget.

For the reference tasks, only candidates in the authoritative evaluated set are historical repeats.
Candidates proposed in an earlier turn but never evaluated remain eligible.

For `ldm_harness_compiled`, the task must additionally:

1. Keep the policy session, submission contract, feature encoder, and adapter
   separate from candidate generation.
2. Expose task-local numeric features that do not encode candidate identity,
   row order, hidden labels, `q0`, acquisition values, or selection outcomes.
3. Preserve the task's kernel, variance, noise, acquisition, pool maintenance,
   evaluator, and candidate budget. The current release permits only
   `prior_mean@1` and `ldm_weights@1`.
4. Fit the unchanged task GP to residual targets after subtracting the compiled
   prior mean. A zero prior and default weights must reproduce `ldm_harness`.
5. Accept only a complete `optimization_policy.py` through immutable artifact
   snapshotting. Execute it outside the host interpreter with read-only inputs,
   no network, and bounded CPU, memory, processes, and time.
6. Support `replace`, `keep`, and `disable`. If a turn fails after repair, reuse
   the previous valid epoch when it remains valid; otherwise use static task
   defaults and mark the round degraded.

Each policy attempt consumes one `policy_harness_turns`, whether the sidecar
commits a submission or fails. Persisted round decisions and committed-turn
replays do not consume another attempt. Runtime failure does not advance the
session's committed history cursor. Failed responses carry available per-turn
provider/tool/artifact usage; transport loss may leave that usage unknown.
The controller records only measured counters and marks missing usage in the
round report instead of assuming zero. This lifecycle accounting is independent
of the task's objectives and scientific fallback validation.

The shared policy boundary accepts `history_utilities` with shape `(H,)` or
`(H, M)`. Prior outputs preserve that objective shape: `(H,)` and `(Q,)`,
or `(H, M)` and `(Q, M)`. Objective order, direction, units, transforms,
scalarization, and diagnostic metrics belong to the task. The controller and
runner do not select an objective or impose a GP or acquisition family.

`PolicyRoundInput` requires aligned `history_candidate_ids`, `history_rounds`,
and `measured_observations`, alongside numeric history. History is chronological;
replicate semantics and candidate identity checks belong to the task.
Supply these fields directly, not inside `research_snapshot`. The
controller builds the exported history snapshot and sends measurement deltas.
Tasks should send compact measurement indexes and expose complete records through
filtered, paginated tools. Candidate identity, scientific summaries and retrieval
fields remain task-local. Do not serialize full scientific payloads into every
turn when stable IDs and targeted detail retrieval suffice.
Its `record_predictions(round_index, predictions)` persists task-defined JSON
rows unchanged. The adapter's `with_feedback(round_input, records)` constructs
task-specific feedback from earlier prediction records and measured outcomes.

Draft diagnostics are also task-owned. Pi runs drafts in a sidecar Python
subprocess; no additional draft sandbox is created. A task may mount a trusted standalone
module read-only and register it with
`ldm_tts.harness.pi.policy_mcp_server(diagnostics_path=..., diagnostics_sha256=...)`.
Its `evaluate_policy_draft(prior, inputs, outputs)` returns a JSON object.
`inputs` contains `arrays`, `execution_context`, and `contract`;
`outputs` contains history/query prior means and `weights`. The `prior`
callback executes the draft with shape and finiteness checks, or is `None`
when that capability is disabled. The Pi runner verifies the module digest
before loading it. Without a hook it reports artifact outputs only. The accepted
artifact is still re-executed independently by `PolicyExecutor`; advisory
diagnostics do not replace authoritative task validation.

### Reference Task Diagnostics

Iron Mind, SynthonBench, and NucleoBench currently optimize scalar utilities
with task-local UCB GPs. Their measured `history_utilities` remain raw task
values. Each task supplies
`target_location` and positive `target_scale`, and the policy returns a
conditional mean on the standardized scale:

```text
z(x) = (utility(x) - target_location) / target_scale
prior_mean(x) = E[z(x) | public task features and supported evidence].
```

The unchanged task GP is fitted to `z - prior_mean` and adds the mean back to
its posterior. The policy mean is not a maximum, rank, probability of
optimality, or acquisition score. `inspect_policy_contract` returns the contract
and exports authoritative snapshot files read-only into the research guest.
The exact execution contexts remain in `input.json`; full history remains in
`research_snapshot.json`. Query these files locally rather than printing them
in full. `evaluate_policy_draft` compares chronological measured-history
holdouts using GP hyperparameters fitted on each training prefix and frozen.
Current-pool diagnostics describe first-draw probabilities, not batch inclusion.
Historical validation is a development check: the Agent has seen those labels.
Each task constructs zero-mean and active prediction records before real
evaluation; the controller freezes them, and the task adapter computes paired
errors on subsequently measured candidates.
Each record also freezes that round's pool size, q0 relative to its maximum,
competition ranks (1 is best; ties share rank), acquisition and first-draw
probability, and actual alpha/eta. Interpret historical confidence in that
original pool, not against a later pool's maximum. Selected-point residuals do
not establish whole-pool ranking; mean holdout error does not validate weights.
Proposal slots express preference, not independent measurements or evidence
from being selected. Both proposal and acquisition influence require support.
The online task retains its existing GP refitting procedure. No current-query
oracle labels enter research snapshots. Task-local instructions must state the exact
feature semantics, identifiability limits, and evidence hierarchy.

## Resources And Artifacts

Versioned task inputs belong under:

```text
tasks/<task_id>/resources/harness/
|-- profiles/<profile_id>/AGENTS.md
|-- skills/<skill_id>/SKILL.md      # optional task-runtime skills
|-- policy_diagnostics.py          # optional trusted draft-evaluation hook
|-- image/guest-image.json
|-- image/Dockerfile
|-- image/lock/
|-- image/smoke.sh
`-- tools/                           # optional task-local structured tools
```

Record content digests for profile instructions, skills, submission contracts,
and tool sources. Pi pins initialization configuration, resource digests,
resolved guest identity, and sidecar implementation in the run manifest.
Resume requires the same identity; changed configuration or runtime needs a
new artifact root. Before a session starts, the sidecar verifies the selected
`AGENTS.md` and Skill digests, snapshots only those resources under the session
workspace, and exposes the snapshot read-only at `/workspace/.ldm-resources` in
the guest. Pi advertises guest-visible Skill paths so progressive `read` calls
and relative references stay inside the sandbox. The task tree remains the
versioned source of truth. The contract carries the strict payload JSON Schema
and exact minibatch count; Python remains the authoritative scientific
validator. Artifact rules snapshot referenced files before task validation and
expose only relative path, size, and digest metadata on the wire.

## Task Guest Runtime

Each Harness task owns a `resources/harness/image/` recipe. Its descriptor
names the pinned Dockerfile inputs, COW rootfs size, and smoke script. The
descriptor digest becomes the only image reference accepted by the protocol.

### Guest Runtime Contract

Every session receives the same runtime contract. `/workspace` is an isolated
research workspace, not a checkout of the project repository. Agents edit files
through Pi's registered `read` and `write` tools; `apply_patch` is not available
inside the guest. Task and MCP tools are registered structured tools and must be
called directly rather than by running their implementation files. A task guest
must not assume that an implementation runtime such as Node.js is installed.
`git` may be used only to clone useful public research material when the task's
network policy permits it, not to inspect the workspace as a repository.

Task-specific runtime dependencies belong in that task's guest-image lockfile
and smoke test. The Pi sidecar injects this contract into every Agent session;
its implementation-level wording is documented in
[`harnesses/pi/README.md`](../harnesses/pi/README.md#guest-runtime-contract).

Build and smoke the selected task guest before a campaign:

```bash
npm --prefix harnesses/pi ci
npm --prefix harnesses/pi run build:task-guest -- \
  --task <task_id> --cache-dir /path/to/harness-cache
npm --prefix harnesses/pi run smoke:task-guest -- \
  --task <task_id> --cache-dir /path/to/harness-cache
```

Guest building requires Docker, `e2fsprogs`, `cpio`, and `lz4` on a Linux host.
Guest smoke additionally requires Linux KVM and the host-architecture QEMU
system emulator. The selected cache holds `images/`, task build records, temporary build state,
Gondolin session state, and session COW overlays. A campaign resolves only a
complete local image with matching recipe metadata and asset checksums; absent
or mismatched images fail before the sidecar starts.

On POSIX hosts the task runner passes its current UID:GID to the sidecar so
bind-mounted artifacts and cache remain writable. When local KVM is present,
the runner also grants that mapped user the device's supplemental group. Use
`--harness-container-user` only to override that mapping.

Harness artifacts are written below `<run_dir>/harness/`:

- native Pi session JSONL with messages, tool calls, and tool results;
- redacted raw provider requests and responses plus their compact index;
- turn input, provisional submission, validation, and commit records;
- a manifest with provider identity, profiles, MCP configuration digests,
  effective tool budgets, usage, session lineage, resolved guest, and an
  environment snapshot captured before guest shutdown.

The native Pi session is the only full conversation record. Python does not
duplicate model or MCP transcripts. These files are raw research traces, not
canonical `ldm-2.0` accepted-action records.

Tasks can enable partial-turn recovery with
`HarnessClient.run_turn(..., recovery_timeout_seconds=remaining_seconds)`.
The sidecar distinguishes recoverable session timeouts and transient provider
failures from configuration, authentication, validation-contract, and protocol
failures. During the recovery window, the client retries with capped backoff and
resends the same turn identities:
committed profiles replay without another model call; unfinished profiles continue
their existing session and workspace without appending the full history again.
Tool quotas and validation attempt indices persist, and provider usage is cumulative
per turn, not summed twice across retries. There is no partial-batch acceptance.
No new recovery attempt starts after the supplied window; an attempt already in
progress retains its configured session time limit. The task owns the campaign
clock and decides whether recovery is allowed. A zero window disables recovery.

`ldm_harness_compiled` writes the independent policy session below
`<run_dir>/policy_harness/`. It stores round inputs and digests, validation
attempts, immutable policy snapshots, accepted epochs, compiled arrays, the
active-policy pointer, and each round result. Pilot Evaluation also exports
`compiled_policy_rounds.csv` with actions, fallback/degraded status, policy
weights, validation and usage counts, and selection diagnostics.
Scientific report fields are mapped explicitly by the task's evaluation
configuration; vectors and structured objective diagnostics are preserved.
Proposal and policy pools may use independent providers, models, and thinking
levels. Their shared campaign, task, and seed identities must still match.

## Qualification

Use a protocol-faithful fake sidecar to test session identity, strict
cardinality, rejection and correction, budget accounting, MCP allowlists, and
lineage without Docker or credentials. Before a real claim, run the sidecar
tests, the task guest smoke, and one capability smoke with the selected
Responses endpoint, container isolation, profiles, and tools. For
`ldm_harness`, verify that accepted occurrences enter `q0`, surrogate, and
acquisition selection. For `ldm_harness_compiled`, additionally verify separate
proposal and policy manifests, artifact digest and path safety, same-session
validation repair, zero-prior parity, residual-GP behavior, policy epoch resume,
fallback accounting, and isolated execution. For `harness`, verify that the
accepted minibatch goes directly to the official evaluator and that no
surrogate or selector is instantiated.

See the [Pi sidecar contract](../harnesses/pi/README.md) and the task README for
runtime-specific commands.
