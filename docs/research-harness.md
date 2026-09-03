# Research Harness Integration

The research Harness lets a persistent Agent inspect optimization history, use
tools, work in an isolated sandbox, and submit task candidates. It is available
through two distinct search methods:

| Method | Candidate generation | Selection before evaluation |
| --- | --- | --- |
| `ldm_harness` | Multiple task-defined persistent Agents submit an over-sampled reservoir. | The task estimates empirical `q0`, fits its surrogate, and applies the LDM acquisition policy. |
| `harness` | One task-defined persistent Agent submits exactly the evaluation minibatch. | None. Every accepted candidate is evaluated in stable submission order. |

Iron Mind and SynthonBench are the reference integrations. The shared Harness
does not know their candidate identity, legal search space, duplicate policy,
surrogate, or evaluator.

## Architecture

```text
shared Campaign / LDMEngine
  -> task-owned ReservoirExpander
     -> ldm_tts.harness.HarnessClient
        -> sidecar-release JSONL protocol
           -> persistent Pi session
           -> isolated shell and file tools
           -> web, Context7, task-local, and configured MCP tools
        <- provisional candidate submission
     -> task-owned validation
        -> reject with indexed reasons and continue the same turn
        -> or accept and commit the turn
  -> ldm_harness: task surrogate and acquisition selector
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
| `HarnessPoolConfig` | Campaign, provider, profile, candidate schema, tools, MCP servers, network policy, and limits. |
| `HarnessGuestRuntime` | Task-owned digest-addressed Gondolin image, COW rootfs size, and install policy. |
| `HarnessProfile` | One persistent Agent identity, `AGENTS.md`, optional skill directories, candidate count, and content digests. |
| `HarnessToolExtension` | A digest-verified task tool module and its exported tool names. |
| `HarnessMcpServer` | One allowlisted stdio or Streamable HTTP MCP server. |
| `HarnessLimits` | Per-turn wall time and per-tool call budgets. |
| `HarnessTurn` | One profile round with deterministic history lineage and task input. |
| `HarnessClient` | Sidecar lifecycle, secret bootstrap, protocol validation, and turn execution. |
| `HarnessSubmissionValidation` | Task-owned acceptance or indexed rejection of a provisional submission. |
| `HarnessTurnResult` | Committed candidates, session lineage, measured usage, and artifact references. |

The Pi sidecar in `harnesses/pi` uses the OpenAI Responses wire format. It owns
session lifecycle, automatic context compaction, isolated file and shell tools,
web and Context7 extensions, MCP clients, candidate submission, and redacted
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
limit. In a runner YAML, use a list when configuring more than one tool:

```yaml
args:
  harness-tool-budget:
    - web_search=4
    - fetch_content=8
    - mcp__literature__search=2
```

The default network budgets are four `web_search` calls, eight
`fetch_content`, eight `get_search_content`, two `resolve-library-id`, and four
`query-docs` calls per turn. Context7 budgets are omitted when Context7 is
disabled. A tool absent from the mapping is unlimited; zero disables it.
`submit_candidates` cannot be budgeted.

The Agent receives the budget snapshot at turn start and the remaining count
after each call. A tool execution attempt consumes one call even when the tool
returns an error. A policy rejection or budget rejection does not consume a
call. Reservations are persisted before execution, so resuming an interrupted
turn cannot reset or double-spend its budget.

## Task Responsibilities

A Harness-enabled task keeps its adapter in `tasks/<task_id>/core/` and must:

1. Start one `HarnessClient` for the campaign and close it in a `finally` block.
2. Define the profile set and exact candidate count for each search method.
3. Build deterministic turns from campaign, profile, round, and history identity.
4. Send newly measured observations and the authoritative evaluated-candidate
   exclusion snapshot.
5. Validate provisional submissions with the same parser, canonical identity,
   and official-space checks used by candidate admission.
6. Return stable rejection codes, rejected indices, candidate identities, and
   actionable reasons so the Agent can repair the same submission in-session.
7. Refill until the complete valid minibatch is accepted.
8. Preserve meaningful same-round occurrences before estimating `q0` for
   `ldm_harness`; require distinct real evaluations for direct `harness`.
9. Record Harness turns and measured provider/tool usage in the campaign budget.

Only candidates in the authoritative evaluated set are historical repeats.
Candidates proposed in an earlier turn but never evaluated remain eligible.

## Resources And Artifacts

Versioned task inputs belong under:

```text
tasks/<task_id>/resources/harness/
|-- profiles/<profile_id>/AGENTS.md
|-- profiles/<profile_id>/skills/   # optional
|-- image/guest-image.json
|-- image/Dockerfile
|-- image/lock/
|-- image/smoke.sh
`-- tools/                           # optional task-local structured tools
```

Record content digests for profile instructions, skills, candidate schemas, and
tool sources. Mount task inputs read-only. The sidecar receives the strict
candidate JSON Schema and exact minibatch count; Python remains the
authoritative scientific validator.

## Task Guest Runtime

Each Harness task owns a `resources/harness/image/` recipe. Its descriptor
names the pinned Dockerfile inputs, COW rootfs size, and smoke script. The
descriptor digest becomes the only image reference accepted by the protocol.

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
bind-mounted artifacts and cache remain writable. Use
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

## Qualification

Use a protocol-faithful fake sidecar to test session identity, strict
cardinality, rejection and correction, budget accounting, MCP allowlists, and
lineage without Docker or credentials. Before a real claim, run the sidecar
tests, the task guest smoke, and one capability smoke with the selected
Responses endpoint, container isolation, profiles, and tools. For `ldm_harness`, verify that accepted
occurrences enter `q0`, surrogate, and acquisition selection. For `harness`,
verify that the accepted minibatch goes directly to the official evaluator and
that no surrogate or selector is instantiated.

See the [Pi sidecar contract](../harnesses/pi/README.md) and the task README for
runtime-specific commands.
