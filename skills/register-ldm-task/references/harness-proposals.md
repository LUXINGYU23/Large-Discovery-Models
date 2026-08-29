# Research Harness Registration Checklist

Read [`docs/research-harness.md`](../../../docs/research-harness.md) before
implementing a persistent proposal backend. That document is authoritative for
the shared interface, ownership boundary, resources, traces, and qualification.
Use this reference only as the task-registration checklist.

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

## Implement The Task Boundary

- Keep one client alive for the campaign and close it in a `finally` block.
- Put turn construction, history projection, canonical identity, validation,
  rejection reasons, occurrence semantics, and budget accounting in
  `tasks/<task_id>/core/`.
- Put versioned `AGENTS.md` profiles, optional skill directories, and optional
  structured task tools under `tasks/<task_id>/resources/harness/`.
- Record SHA-256 identities for profiles, skills, candidate schemas, and tool
  sources; mount these resources read-only.
- Provide the sidecar with the task's strict candidate JSON Schema, including
  required fields, value constraints, and `additionalProperties: false`. The
  sidecar must expose that schema through `submit_candidates` and require the
  exact profile minibatch size; task validation remains the authoritative
  admission boundary.
- Build deterministic `HarnessTurn` identities from campaign, profile, round,
  and history range/digest.
- Send newly measured observations for reasoning and a compact authoritative
  evaluated-candidate snapshot when historical repeats are forbidden.
- State explicitly that only the authoritative evaluated snapshot is excluded.
  Candidates proposed in an earlier turn but not evaluated remain eligible;
  persistent sessions must not invent a private exclusion set.
- Validate every provisional submission through the same parser and canonical
  identity used by Campaign admission.
- Return one stable, actionable `HarnessSubmissionRejection` for each rejected
  index and accept only a complete valid minibatch.
- Define same-round occurrence semantics before empirical `q0` aggregation;
  do not globally deduplicate meaningful agreement across independent sessions.
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
