# Experimental main research session and round forks

## Configurable research strategies

Select the lifecycle and rules with `--harness-research-strategy-config PATH`.
The four default research profiles and candidate/evaluation budgets are configured
as before. Four ready-to-use JSON configurations live in
`config/nucleobench/research_strategies/`:

| File | Behavior with four profiles |
| --- | --- |
| `independent.json` | Four persistent independent sessions, no main researcher |
| `main_forks.json` | One persistent main researcher, four fresh forks of its latest node |
| `coverage_2plus2.json` | Two latest-main forks, one historical restart, one archived-lineage continuation |
| `joint_gain_1plus3.json` | One latest-main fork, three jointly selected historical restarts or archived-lineage continuations |

For example, add this to the existing real campaign command:

```sh
--harness-research-strategy-config config/nucleobench/research_strategies/coverage_2plus2.json
```

No strategy config means independent sessions. Candidate admission, global empirical q0, GP/acquisition,
compiled policy and true-evaluation selection are unchanged. No protected
exploration evaluation slots are introduced.

`ResearchStrategy` (`ldm_tts.harness.research_strategy`) defines persistent versus
forked execution. `ForkPolicy.plan(ForkContext)` selects one `ForkAssignment` per
profile; `ArchivePolicy.retain(nodes)` selects reusable leaves. Both rules expose
`identity()` with semantic version and parameters, which are pinned in durable
round plans. Rules are pure: they neither invoke LLMs nor modify files. Execution,
snapshot validation, history delivery, billing and replay remain framework-owned.

Built-in configuration:

```json
{
  "name": "main_forks",
  "fork_policy": {"name": "coverage"},
  "archive_policy": {"name": "diverse", "capacity": 3, "overlap_threshold": 0.5}
}
```

Either rule can instead be a trusted, importable Python factory:

```json
{
  "name": "main_forks",
  "fork_policy": {"factory": "my_research.rules:MyForkPolicy", "options": {"seed": 42}},
  "archive_policy": {"factory": "my_research.rules:MyArchivePolicy", "options": {"capacity": 3}}
}
```

A top-level `{"factory": "my_research.rules:make_strategy", "options": {}}`
can construct the entire `ResearchStrategy`. Factories must be deterministic and
side-effect free. Put user modules on `PYTHONPATH`, and change the rule identity
when changing its behavior. A fork assignment carries profile ID, available parent
node ID, purpose (`main`, `restart`, `continue`) and lineage ID. Continuation keeps
the parent's lineage ID; restarts create a new one. Custom rules choose sources
without modifying the Pi transport or candidate selector.

### Coverage and archive semantics

- Roles rotate once per round. With N profiles there are N-2 latest-main forks and
  two exploration forks (N must be at least three).
- Historical restart anchors are the first, one-third and two-thirds nodes in the
  older main history, rounded down and deduplicated. Choose the least recently
  expanded anchor, breaking ties by visit count and age. Round one necessarily
  shares the only available main node; distinct history develops subsequently.
- Continuation chooses the archived lineage with the fewest cumulative expansions,
  then oldest visit, then node ID. Counts survive replacement of a lineage's leaf.
  With no archive, use another historical anchor when one exists.
- Before running any child, write `fork-plan.json` with policy identities, measured
  history digest, parent nodes and assignments. Resumption reuses this exact plan.
- Each profile gets its own immutable `fork-sources/PROFILE` and history cursor.
  It receives every measurement since that cursor, plus access to the full current
  evidence store, without injecting the latest main narrative into old parents.
- After the proposal barrier closes, snapshot exploratory children into
  `nodes/PROFILE`. Commit the updated tree atomically to `tree.json`; proposals
  carry `research_fork` lineage. Closed child runtimes do not remain active.
- New leaves stay pending until next round's real measurements arrive. Then retain
  at most three selectable leaves by default. Preserve one unmeasured direction,
  rank measured directions by their best observed utility, and merge proposal sets
  whose Jaccard overlap is at least 0.5. Evidence is cumulative within a lineage.
  Unselected proposals are unknown, never failures. Pending records can temporarily
  exceed the selectable archive limit between evaluation and the next round.
- Proposal-set overlap is a simple exact-candidate descriptor, not a guarantee of
  different biological mechanisms. Replace `ArchivePolicy` for richer descriptors.
  There is no extra model judge, tree-level UCB or oracle evaluation.

Only the selected archive is bounded; historical trace/snapshot artifacts remain
on disk for audit. Final-round leaves have no subsequent scheduling pass to
reconcile their measurements; the authoritative campaign checkpoint retains them.

### Experimental joint-gain selection

Use `--harness-research-strategy-config config/nucleobench/research_strategies/joint_gain_1plus3.json`.
It loads `ldm_tts.harness.joint_gain:JointGainForks` through the existing factory
interface. Independent sessions, latest-main forks and coverage selection remain
separate choices. No token price, token count or cost estimate enters this policy.
Existing usage accounting is unchanged.

One rotating profile forks the latest main node. Each of the other three profiles
can restart from **any older main node** or continue **any retained archive leaf**.
There is no quota for restarts or continuations: jointly enumerate all feasible
three-action combinations. An archived lineage can have only one successor per
round. Different profiles may restart from the same historical node, each creating
its own lineage. With no archive all three restart; round one shares the only root.
The existing `DiverseArchive` rule still retains at most three selectable leaves.

A small shared Gaussian process models a research action `(parent, profile,
purpose)`, using only previous rounds' measured feedback. Its signed target is
the best utility measured from that branch's current proposal batch, minus the
incumbent before that batch. This differs from cumulative lineage quality. An
unmeasured batch has no target; it is never given zero fitness. A later round's
measurement does not retroactively label an earlier batch as successful. Archive
quality, separately, still incorporates all available measurements of its lineage.

The action features are log round, relative parent age, log expansion count,
whether parent quality is known, the scaled parent-to-incumbent gap, profile and
purpose indicators, and a task-supplied sequence descriptor. For NucleoBench,
archived leaves use per-position nucleotide frequencies of their latest batch;
main nodes use the incumbent sequence when their snapshot was created. Frequencies
are divided by the square root of the number of editable positions. These describe
sequence location, not the semantic contents of session notes or biological mechanisms.

The RBF kernel has unit length scale. Targets are scaled by the median absolute
nonzero measured gap (unit scale before any such gap), with a zero-gap prior.
Observation noise defaults to 0.25 in these scaled units. Branches sharing the
same best measured canonical key share observation noise, rather than gaining
independent confidence from the same oracle result. NumPy handles the small
matrix solves; no new package, language-model call or oracle call is needed.

For each feasible triple, combine its actions with the single fixed main action and
estimate `mean(max(0, sampled_gap_1, ..., sampled_gap_4))` from **joint** posterior
samples. Taking the maximum, rather than summing individual expected gains, makes
complementarity part of the objective. Every triple uses the same 512 antithetic
draws; the configured seed and round index determine them. The JSON factory
options expose `samples`, `seed` and `noise_variance`; all are pinned in the policy
identity. The joint acquisition follows the
[batch expected-improvement pattern](https://botorch.org/docs/v0.17.1/batching).

This is a one-step acquisition **proxy**, conditional on branches receiving
measurements. It models uncertainty in their expected best measured gap; it does
not simulate future candidate generation, competition in the global selector, or
all stochastic variation of LLM outputs. Base-frequency similarity and the kernel
are modeling assumptions, not proof of independent discoveries. With sparse data,
the prior can dominate. Better utility or avoidance of local optima requires new
controlled campaigns; replaying historical contexts cannot establish either.

`tree.json` records every branch's batch keys and its immutable decision-time
parent, including main branches and pruned lineages. Before the next plan, the
task joins those keys to successful finite measurements from the same round.
`fork-plan.json` freezes this feedback, the incumbent, descriptors, assignments,
and the preceding tree's SHA-256. Replay reuses that plan and rejects changed
evidence or policies; it neither resamples nor increments visits twice. These
feedback-bearing plans have a new evidence identity. Existing campaigns retain
their frozen runtime and provenance; this implementation does not migrate them.

Validation on 2026-10-11: 82 strategy, Harness and task-boundary tests passed on
the remote server. They cover all restart/continuation allocations, three concurrent
continuation snapshots, lineage feedback, interruption/replay and unchanged budgets.
Read-only checks of 60 recorded contexts from seeds 42, 43 and 10086 produced
deterministic valid plans, averaging 0.158 seconds per preparation (maximum 0.381).
All 183 source artifacts retained their hashes. No provider or oracle calls were
made. These checks establish execution behavior, not improved search performance.
Receipts and isolated fixtures are in
`/mnt/data1/Large-Discovery-Models/research/joint-gain-1plus3-20261011/`.

## State, recovery and costs

Fork strategies require a real, round-budgeted `ldm_harness` or
`ldm_harness_compiled` campaign. SoL-Pi is excluded because its external reduction
archives are outside the native session/workspace snapshot.

The main researcher reconciles measurements with all previous branch proposals
and updates its notes. New investigations belong in the parallel branches.
The compiled-policy researcher remains a separate persistent session.

- `harness/main`: persistent main identity and turn commits.
- `harness/branches/round_N/fork-source`: immutable session/workspace snapshot.
- `harness/branches/round_N/fork-sources/PROFILE`: selected parent snapshot for each child.
- `harness/branches/round_N/sessions`: isolated native child sessions and traces.
- `harness/branches/round_N/panel.json`: candidate hypotheses, lineage and main brief.
- `harness/measured_history`: authoritative evaluated data, including final results.
- `policy_harness`: unchanged compiled-policy session.

Fork initialization stages the native session and workspace together, then renames
the completed directory. Replay validates hashes and does not reset an existing
child. Existing turn input digests, submission snapshots and cumulative usage keys
provide recovery. The main researcher has an extra counted turn each round; its
successful and failed calls contribute to the same LLM accounting as the children.
Full token comparison reads raw provider responses across all three roles. Cached
input and reasoning are subsets, never additional tokens added to the total.

Only ordinary workspace files are inherited, not guest package installations,
processes or a VM snapshot. Child notes/scripts remain archived; the main receives
candidate annotations rather than complete child tool transcripts. At the final
round, measured results are saved without an additional post-campaign LLM turn.
Changing role system prompts and tools can reduce shared provider cache hits;
native forking itself is **not** a guarantee of free input tokens or KV reuse.

## Initial experiment

Remote workspace: `/mnt/data1/Large-Discovery-Models/research/main-research-forks-20261009`.
Run `experiments/main-research-forks/run.py WORKDIR` using the remote official
NucleoBench environment after supplying the source snapshot and built sidecar.
The runner refuses existing experiment paths, performs dependency/dry-run checks,
qualifies two real rounds, verifies native ancestry/cursors/panel counts, then runs
four independent campaigns and writes `comparison/summary.json` and `report.md`.
Failed qualification prevents the full experiment from starting.
Before hashing the deployed source, the runner copies the October 7 deployment's
guest Dockerfile, recording its path and SHA-256. That recipe changes only OCI and
Debian mirrors and reuses the existing qualified guest with its pinned base digest.
The sidecar is rebuilt with the current locked Pi 0.85.1 dependencies.

| Setting | Value |
|---|---|
| Task / shared start | Enformer muscle-not-liver / official 1258, index 62 |
| Seeds | 42, 43, 3407, 10086 |
| Model | DeepSeek-Flash, responses, max reasoning |
| Budget | 20 active rounds × 16 evaluations + shared start = 321 |
| Proposal / BO | 4 roles × 32 candidates; BO pool 64 |
| Roles | target biology, regulatory grammar, sequence landscape, evidence critic |
| Oracle | CPU, 16 threads, microbatch 8 |
| GP queries | Disabled |
| Policy | Existing compiled-policy researcher |

Primary reference is the completed `without_sol_pi` arm in
`research/sol-pi-validation-20260921`: same seeds, start, model, evaluation and
proposal budgets. It used four identical comprehensive research identities.
Historical mean best utility is 6391.50; per-seed scores are 2233.20, 6348.70,
8860.25 and 8123.83. Reuse its recorded raw results rather than spending a new
baseline budget. The later Pure Harness/Vanilla Pi ablation provides secondary
context but has a smaller proposal pool and is not the primary control.

Report paired per-seed utility, trajectories, wall time, calls, total input/output,
cached/uncached input, reasoning and costs split by main/children/policy. Missing
usage must be reported. Four-seed historical results cannot isolate the fork
mechanism: role prompts, source revisions, rebuilt runtime and endpoint time also
change. A smaller token total alone is not evidence of better performance or
lower billed cost; compare uncached/output use and per-seed outcomes too.

## Qualification result (2026-10-09)

The two-round seed-42 qualification completed 33 true evaluations, two turns in
the same main identity, eight distinct native forks, and two policy turns. The
second parent retained the main session ID and contained no newly created trace
nodes from the previous children. Real calls confirmed copied parent histories;
the unit tests also verify workspace isolation and immutable snapshot replay.
241 Python checks and 38 Pi checks passed on the remote server.

| First two active rounds, seed 42 | Main research forks | Historical LDM |
|---|---:|---:|
| Best measured utility | 67.29297 | -1612.30664 |
| Provider calls | 229 | 224 |
| Input + output tokens | 20,050,748 | 18,612,784 |
| Uncached input tokens | 544,237 | 289,222 |
| Output tokens | 386,383 | 412,394 |

All 229 responses had recorded usage. The main researcher alone consumed
1,580,333 tokens. Total tokens were 7.7% higher than the historical two-round
prefix; this qualification does **not** establish token savings or stable utility
improvement. The earlier aborted open-ended-main trial consumed another 405,602
tokens, retained separately as development/qualification overhead.

The qualification exposed two reporting assumptions: a single proposal-pool
manifest and equal proposal-attempt/harness-turn counts. The implementation now
declares main, per-round proposal and policy pools through `harness_provenance.json`
and validates the extra main turns separately. Existing completed measurements
were republished with the repaired reporter without further provider/oracle calls.

Four formal 20-round campaigns were launched at revision `b1a49c68`. Their source
snapshot is frozen in `formal-launch.json` and `source-manifest.json`; the earlier
qualification snapshot is retained separately. They are running, so no final
four-seed claim is available yet. The remote driver will generate the comparison
JSON, Markdown, trajectories and plots after all four campaigns finish.

## History and recovery audit (2026-10-10)

The original fresh seed 3407 reached round 14 (225 observations), then repeated
the round-14 context failure in round 15. It was interrupted for this repair;
its frozen source, checkpoints, sessions, and incurred usage remain intact.
The completed independent repeat remains user-invalidated, not a replacement.

Read-only audit of provider traces found the following. Failed summaries count
terminal non-completed responses with usage; interrupted calls without usage
are reported separately in the audit JSON, not assumed successful or free.

| Campaign | Summary calls | Failed summaries | Research calls capped at 16 output tokens |
| --- | ---: | ---: | ---: |
| 42 | 59 | 15 | 0 |
| 43 | 48 | 18 | 4 |
| 10086 | 94 | 60 | 6 |
| 3407 original, interrupted in round 15 | 256 | 229 | 139 |
| 3407 independent repeat, invalidated | 51 | 18 | 12 |

The parent copied into round 15 carried 235,768 context tokens. The previous
262,144-token context setting reserved only 16,384 tokens, despite research
responses permitting 32,768 output tokens. All four children inherited the
large parent. Summaries also inherited research thinking level `max`, while
Pi allowed only 13,107 output tokens for the history summary. In original seed
3407, 217 summary calls spent their entire output on reasoning without summary
text. The harness ignored the resulting failed compaction/length stop and kept
appending submission reminders; a 1,800-second timeout was automatically
retryable, producing another attempt on the same unusable history.

Not all timeouts have this cause: seed 42 round 7 ran an unbounded generated
rejection sampler with infeasible sequence constraints. That incident and its
manual script repair remain separately recorded.

The repair uses the installed Pi 0.85.1 public compaction extension hook and
native summary/tree implementation, following its
[compaction interface](https://github.com/earendil-works/pi/blob/main/packages/coding-agent/docs/compaction.md).
It does not replace the transcript or synthesize a new research identity:

- Reserve 65,536 context tokens, so normal compaction starts around 196,608;
  keep 20,000 recent tokens. Parent maintenance is shared before later forks.
- Summarize at `low` thinking with a separate 40,960-token reserve (native
  history summary cap 32,768), a 180-second deadline, and at most one retry
  for transient provider failures. Research remains at its configured level.
- Persist failed-history state and allow at most two submission continuation
  prompts per turn, including across restarts. An unresolved length stop or
  failed compaction blocks upstream dispatch rather than adding more prompts.
  A successful explicit native compaction can repair a blocked history.
- Treat wall-time expiry as terminal after draining cancellation; do not
  automatically give the unchanged turn another full 1,800 seconds.
- Record compaction starts, generated summaries' sizes/usage, native commits,
  failures, retries, and continuations in `history-events.jsonl`. Store recovery
  state in `history-recovery.json`. Existing provider traces include summary
  calls in the same turn's accounting.

Remote validation: 45 Pi checks pass, including real SDK compaction against a
fixture Responses server, automatic failure dispatch blocking, transient
recovery, restart persistence, and raw-history preservation. Another 28 Python
client/fork checks pass on the remote server. A separate real
DeepSeek qualification on a read-only copy of the round-15 parent completed in
57.539 seconds with one provider call: 235,768 tokens before, approximately
27,932 projected message tokens after (estimate, excludes request overhead).
All 459 original entries were preserved. The call used 193,408 total tokens,
including 168,832 cache-read tokens. Qualification artifacts are separate from
benchmark results. This verifies recovery functionality, not 20-round utility
or cost improvement; a fresh controlled run is still required for those claims.

Remote audit and qualification: `research/main-research-forks-20261009/history-recovery-audit-20261010/`.
The changed history policy is pinned in initialization identity. The old
experiment is not silently resumed under the new source/runtime identity.
