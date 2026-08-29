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
- Build deterministic `HarnessTurn` identities from campaign, profile, round,
  and history range/digest.
- Send newly measured observations for reasoning and a compact authoritative
  evaluated-candidate snapshot when historical repeats are forbidden.
- Validate every provisional submission through the same parser and canonical
  identity used by Campaign admission.
- Return one stable, actionable `HarnessSubmissionRejection` for each rejected
  index and accept only a complete valid minibatch.
- Define same-round occurrence semantics before empirical `q0` aggregation;
  do not globally deduplicate meaningful agreement across independent sessions.

Pass the API key only through Harness bootstrap. Keep native sessions, redacted
provider transport, and input/submission/commit lineage below
`<run_dir>/harness/`. Do not write credentials there and do not treat raw
multi-turn traces as `ldm-2.0` accepted-action rows.

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
