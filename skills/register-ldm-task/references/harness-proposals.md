# Research Harness Registration Checklist

Read [`docs/research-harness.md`](../../../docs/research-harness.md) before
implementing a persistent proposal backend. That document is authoritative for
the shared interface, ownership boundary, resources, traces, and qualification.
Use this reference only as the task-registration checklist.
For an independent policy-artifact session, use
[harness-compiled-policy.md](harness-compiled-policy.md) instead of extending
the candidate submission schema with policy fields.

## Choose The Backend Deliberately

Use direct `ProposalClient` requests when independent prompt/response sampling
is sufficient. Use `HarnessClient` only when candidate generation benefits from
persistent research context, iterative tools, sandbox analysis, or distinct
Agent roles. Supporting both normally makes `proposal_provider.kind` `hybrid`;
record the method-to-provider mapping and preflight only the selected online
method.

Treat reasoning controls as transport-specific provider fields. OpenAI-style
Chat Completions commonly uses top-level `reasoning_effort`, while a Responses
SDK may map a thinking level to `reasoning.effort`. Do not copy one wire shape
to the other. Qualify the exact configured endpoint with an actual request and
verify the transmitted body or nonzero reasoning-token accounting; a value in
YAML alone does not prove that the provider honored it.

Concurrent direct sampling must tolerate transient provider throttling without
turning a requested batch into a smaller one. Use bounded transport retries
with backoff for the same logical proposal, and set any shared circuit-breaker
threshold above one expected concurrency burst. Force a transient 429 in tests
and verify that candidate cardinality and logical proposal accounting remain
unchanged after recovery.

The Harness remains inside the task's `ReservoirExpander`. Do not add another
Campaign, BO loop, optimization history, evaluator path, or central task branch.
Use `PiHarnessConfig` from `ldm_tts.harness.pi` for Pi-specific provider, tools,
and guest settings. Keep `HarnessPoolConfig` and the shared client backend-neutral.

## Implement The Task Boundary

- Keep one client alive for the campaign and close it in a `finally` block.
- Put turn construction, history projection, canonical identity, validation,
  rejection reasons, occurrence semantics, and budget accounting in
  `tasks/<task_id>/core/`.
- Put versioned `AGENTS.md` profiles, optional skill directories, and optional
  structured task tools under `tasks/<task_id>/resources/harness/`.
- Record SHA-256 identities for profiles, skills, submission contracts, and tool
  sources. Let the sidecar snapshot selected profile and Skill files into the
  session workspace and expose that snapshot read-only at the guest-visible
  `/workspace/.ldm-resources` path.
- Provide the sidecar with one strict `HarnessSubmissionContract`. Its payload
  schema owns the terminal tool's required fields and value constraints, with
  `additionalProperties: false`. For a file submission, validate the exact
  minibatch size and candidate contents from the immutable snapshot in task code.
- Build deterministic `HarnessTurn` identities from campaign, profile, round,
  and history range/digest.
- Prefer compact new-measurement indexes and task-owned, paginated history
  lookup over copying every full candidate into each message. Query by ID or
  round and rank before requesting detail. Project records from engine
  observations; do not create a second optimization history. Proposal and
  policy sessions may share that projection read-only.
- Enforce exact historical membership in task validation, including failed
  evaluations when the task forbids reevaluation. Expose the same membership
  check to the Agent. Do not rely on its context or compaction memory.
- Validate every provisional submission through the same parser and canonical
  identity used by Campaign admission.
- For large or annotated minibatches, submit a workspace-relative file through
  `HarnessArtifactRule`. Validate its immutable snapshot, digest, exact count,
  task fields, and short per-candidate change/rationale notes. Keep notes in
  candidate metadata rather than the oracle payload. Preserve all contributing
  notes across canonical admission and checkpointing; return them with measured
  history, not unmeasured cross-session proposals.
- Return one stable, actionable `HarnessSubmissionError` with a JSON Pointer,
  code, message, and repair hint for each invalid entry. Return `retry` until a
  complete valid minibatch is available; use `reject_turn` only when the
  contract's validation-attempt limit closes the turn.
  Keep research, editing, and validation tools usable after rejection; do not
  force the next model request to call only the terminal tool.
- Turn instructions must translate the hard wall-time into explicit research,
  validation, and first-submission milestones. "Reserve enough time" is not a
  reliable delivery contract for an autonomous Agent; qualify the slowest role
  on its first real turn and require submission before the outer timeout.
- Define same-round occurrence semantics before empirical `q0` aggregation;
  do not globally deduplicate meaningful agreement across independent sessions.
- A single comprehensive template can back independent persistent identities.
  Define within-session uniqueness separately from cross-session agreement.
  Encourage alternatives and controls based on evidence, not hardcoded rounds
  or a fixed quota. When historical repeats are forbidden, only evaluated
  candidates belong to the exclusion set; prior unmeasured proposals remain
  eligible. Sessions must not invent a private exclusion set.
- Select task-relevant Skills on demand; do not inject their full text on every
  turn. Pin adapted sources and licenses, remove irrelevant workflows, and
  preinstall only the dependencies promised by the profile. Run a guest smoke
  that actually exercises those packages. Keep runtime Skills task-local and
  proposal and policy selections separate.
- Size the maintained BO pool from the expected number of unique accepted
  occurrences, with headroom for cross-session agreement. Do not set the pool
  equal to a raw occurrence count that duplicates cannot fill.

Pass the API key only through Harness bootstrap. Keep native sessions, redacted
provider transport, and input/submission/commit lineage below
`<run_dir>/harness/`. Do not write credentials there and do not treat raw
multi-turn traces as `ldm-2.0` accepted-action rows.

Sandbox isolation and network policy are separate concerns. The Pi sidecar may
give the Agent root shell, file, package-installation, and unrestricted HTTP(S)
access inside its isolated microVM without mounting host task data. An empty
host allow list means unrestricted network access. Benchmark-name and
hidden-answer query prohibitions are evaluation-integrity rules, not a
substitute for sandbox isolation.

## Verify Before Qualification

Add focused tests for profile/tool digests, turn identity and history ranges,
strict minibatch cardinality, indexed rejection and correction, committed-turn
idempotence, budget counters, lineage, and credential redaction. A mock uses a
protocol-faithful fake sidecar and no external systems.
Enable the shared client's bounded partial-turn recovery where the task budget
allows it. Replay accepted peers and resume failed sessions without appending
the entire history or silently shrinking the reservoir. Keep policy recovery
separate and preserve its existing validated-epoch/default behavior on failure.
Test partial-turn recovery without overwriting attempts and reject resume when
configuration, resource digests, or runtime identity changes.

The shared recovery window includes the first attempt. Leave room for a full
timeout before recovery, or supply the remaining benchmark allowance. Count
measured failure usage from `HarnessError.turn_usage`, not only successful
returns. Use stable per-turn `usage_key` values with
`CampaignRuntime.consume_many` for cumulative counters, and test a failed call
followed by ledger reload and committed-turn replay without double charging.

Before `tiny_campaign_verified`, run the actual sidecar unit tests and one real
capability smoke with the configured wire API, isolation, profiles, and tools.
Verify that one accepted Harness reservoir enters the normal acquisition and
official evaluator path. For Pi-specific runtime requirements, follow
`harnesses/pi/README.md`.

After a multi-round run, audit native traces rather than relying only on the
summary. Check malformed submissions and wrong cardinalities, whether sessions
incorrectly exclude unmeasured prior proposals, cross-profile overlap before
`q0`, unique-count headroom for the maintained pool, acquisition effective
sample size, web/tool failures, and whether roles actually use the sandbox
capabilities claimed by their `AGENTS.md`.
