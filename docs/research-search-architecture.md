# Shared Pi research trees

This is the implemented research runtime used by NucleoBench, Iron Mind,
SynthonBench, ReaSyn, AtomWorld and AlphaBench. New LDM Harness campaigns default
to **CoverageForks 2+2**. It selects research history; task candidate generation,
empirical q0, surrogate fitting, selection and evaluation remain task-owned.
There is no second campaign engine and no LLM call inside a fork policy.

## Default and explicit strategies

With four research profiles, each active step runs one persistent main researcher
followed by four parallel proposal branches:

| Config in `config/research_strategies/` | Proposal branches |
| --- | --- |
| `coverage_2plus2.json` (default) | 2 latest-main forks + 1 older-main restart + 1 archived-lineage continuation |
| `main_forks.json` | 4 forks of the latest main node |
| `independent.json` | 4 independent persistent sessions; no main researcher |

Use the same CLI argument in all six tasks:

```sh
--harness-research-strategy-config config/research_strategies/coverage_2plus2.json
```

Shared-runner YAML uses `args.harness-research-strategy-config`. AlphaBench also
accepts the JSON object as `research_strategy_config` in its frozen protocol;
a CLI override must agree with that protocol. The resolved configuration and
versioned strategy identity are recorded in each campaign's configuration.

The single-session `harness` baselines in NucleoBench, Iron Mind, SynthonBench,
ReaSyn and AlphaBench stay independent. AtomWorld's multi-profile `harness` and
`harness_public_audit` also support the tree, with public-only feedback. Its four
default identities use two geometry and two audit role templates. Its explicit
independent strategy can still use two profiles. NucleoBench's in-process mock
path does not execute Pi; SoL-Pi is supported only with the independent strategy.

AtomWorld and AlphaBench now default to four proposal identities instead of two.
At the same per-profile quota this doubles their proposed occurrences; their
evaluation limits do not change. Matched comparisons must explicitly align the
profile count and total proposal budget. Other tasks already use four identities.

The 2+2 rule is deterministic. It rotates profile assignments, restarts from an
older main anchor with low recent use, and continues a retained side lineage with
low expansion count/recent use. If no archive exists yet, the continuation slot
uses a main restart; early steps can share the only available parent. A small
archive retains distinct proposal sets, including an unmeasured alternative.
This promotes exploration but does not guarantee escape from local optima.
See [policy details and prior experiments](main-research-forks.md).

## Responsibilities and interfaces

| Component | Responsibility |
| --- | --- |
| `ldm_tts.harness.research_strategy` | Load trusted configuration; resolve fork/archive rules and their versioned identities |
| `research_tree` | Pure parent selection and archive retention |
| `research_store.ResearchTree` | Freeze plans, verify parent hashes, materialize snapshots, record branch outcomes and archive leaves |
| `research_runtime.ResearchCoordinator` | Run the main turn, deliver the preceding branch panel, close it before snapshotting, run tree prepare/commit |
| Task adapter | Export authorized history, define credit keys and finite maximization scores, validate candidate submissions, preserve q0 and budgets |
| Pi Harness | Native history, compaction, sandbox lifecycle, strict turn commits, provider traces and recovery guards |
| Campaign / LDMEngine | Runtime assembly, admission, selection and round progression |
| `engine/state.py` | Lifecycle configuration, observations and checkpoint representation |
| `engine/evaluation.py` | Evaluation reservation, receipt validation, dispatch, result publication and replay |
| `engine/run_store.py` | Durable budget, event, status and checkpoint storage |

`ResearchCoordinator.prepare(step, history, measurements, message=...)` returns a
frozen plan and one `PiForkSource` per profile. The task runs the branch pool and
calls `commit(prepared, sessions, proposal_keys, proposals=...)` only after every
planned branch has accepted its exact task submission. `MeasuredResearchExpander`
is the common adapter for Iron Mind and SynthonBench; the other tasks retain their
own submission/evidence adapters around this coordinator.

A node contains native session bytes, ordinary workspace files and a history
cursor. Shared tool exports are included in the workspace snapshot. Child role
and skill resources are installed by Pi; writable workspaces are never shared.
Forks of older nodes receive all missing evidence from their inherited cursor,
with compact indexes and task tools for full details. They do not receive only
the previous physical round's feedback.

The main session sees the preceding branch panel and task-authorized evidence.
Side conversations are not appended wholesale to its transcript. An archived
continuation forks the frozen side snapshot and retains its lineage identity.
It is not a second permanently running process.

Custom policies implement the existing `ForkPolicy.plan(ForkContext)` or
`ArchivePolicy.retain(nodes)` interface, with deterministic `identity()` values.
Trusted JSON can name `module:factory` plus options. Policies cannot modify
observations, schedule extra evaluator calls, or change task privacy boundaries.
No additional scheduling plugin interface is needed.

## Task boundaries

| Task | Tree step and measured evidence | Candidate/evaluation contract |
| --- | --- | --- |
| NucleoBench | One active search round; finite utility | Mutation validation, global empirical q0 and selected oracle batches |
| Iron Mind | One active search round; measured reaction score | Exact legal reaction conditions and task-owned historical exclusion |
| SynthonBench | One active search round; measured synthon utility | Official tuple/product identity and exact-count candidate files |
| ReaSyn | One proposal minibatch, including bounded replenishment; actual measured projection outcomes credited to originating proposal occurrences | Targets still pass through projection; products/query+seed remain evaluator identities |
| AtomWorld | One revision within one sample; a separate tree per sample | Blind modes expose only public drafts; LDM modes additionally expose past scalar correctness, never private CIFs or judge internals |
| AlphaBench | One active search round; signed search-period objective | Host-authorized checks/requests, occurrence-based q0; validation/test feedback stays private |

ReaSyn freezes a monotonic tree step in its allocation receipt because one engine
round can contain several proposal minibatches. A short minibatch still runs the
four planned research roles, but quotas sum to the requested count: a zero-quota
role submits an empty candidate array and may preserve research notes. This adds
no target, projection or oracle call. Repeated target SMILES are not evidence that
two projected products have the same score. Credit follows the stored occurrence
lineage and its actual observed result.

Research feedback is refreshed from authoritative observations on the next
`prepare`. Final-step observations remain in the campaign checkpoint; an offline
analysis must join that checkpoint instead of interpreting pending final leaves
as failures. There is no automatic finalizer or extra post-budget research turn.

## Budgets, provenance and recovery

For R active steps, four proposal profiles use at most 5R research turns under
2+2 (R main + 4R branch); independent uses 4R. Main turns do not create candidate
occurrences or consume proposal-attempt quotas. Compiled-policy turns are a
separate task budget. ReaSyn uses its existing proposal request cap for this upper
bound; AtomWorld sums scheduled revisions across samples. AlphaBench retains its
explicit finite protocol budgets and charges main turns to `harness_turns`.
For a fixed profile count and quota, changing the strategy adds no candidate,
evaluation, retry or per-request output allowance. Account for all provider, summary, failed and recovered calls, including
cache hits, when comparing methods. Token/cost is not a policy objective.
AlphaBench's `costs` includes the main researcher's Host-authorized requests and
tool usage; its `generation_cost` remains scoped to candidate-generating turns.

A tree root contains:

```text
main/sessions/main_research/{session.jsonl,workspace/}
branches/round_0001/
  fork-source/{session.jsonl,workspace/,snapshot.json}
  fork-plan.json
  fork-sources/<profile>/{session.jsonl,workspace/,snapshot.json}
  sessions/<profile>/...
  tree.json
  panel.json
```

Task measurement exports and per-pool manifests live beside these records.
`harness_provenance.json` lists actual main, branch and policy pool manifests;
AtomWorld includes each sample's pools. Preserve provider traces at these active
paths; do not count backup snapshots as fresh usage.

Replay requires the same history prefix, strategy identity, parent snapshots and
immutable turn inputs. Changed plans or corrupt snapshots fail closed. A partial
barrier reuses committed native turns; no new plan replaces the unfinished one.
Research `max_output_tokens` stops, failed compactions and unknown provider
outcomes retain the existing recovery guards. A strategy never clears them or
adds retries. Research request overrides and native-summary parameters remain
separately scoped in the fixed provider proxy.

Changing the default affects **new campaigns**. Existing completed results,
archived experiments and live run identities are not rewritten. Reproducing an
independent/4fork comparison requires its explicit config. A different strategy
or implementation does not justify bypassing a failed identity check to resume.

## Validation and limits

Task adapter tests execute native-snapshot orchestration with synthetic Harness
responses and task validators, including two-step 2+2 plans, inherited cursors,
q0, main/branch accounting, ReaSyn partial batches and projection credit,
AtomWorld feedback isolation, and AlphaBench Host checks. Shared tests cover
snapshot tampering, deterministic plans, interrupted barriers and failed-call
accounting. Run these on the remote development server under `/mnt/data1/`.

These are engineering checks, not real-provider optimization experiments.
The optional NucleoBench measurement coverage policy stays task-specific and
disabled by default; the other five
adapters reject it instead of ignoring it.
