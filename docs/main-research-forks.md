# Experimental main research session and round forks

Enable `--harness-main-research` with a real, round-budgeted `ldm_harness` or
`ldm_harness_compiled` campaign. The ordinary workflow remains the default.
SoL-Pi is excluded because its external reduction archives are outside the native
session/workspace snapshot. This experiment changes proposal organization only;
candidate admission, empirical q0, GP, BO pool, compiled policy and oracle remain
in the existing pipeline.

Each round:

1. The persistent main researcher receives new authoritative measurements and an
   archive of **all** previous branch proposals and their original hypotheses.
   It revises workspace notes and submits a bounded common research brief.
2. Close its runtime, then atomically freeze its native Pi session and ordinary
   workspace files. SHA-256 binds both components and the measured-history cursor.
3. Use Pi's `SessionManager.forkFrom` to create one child per configured research
   role. Each child inherits the main conversation and an independent file copy,
   receives the current role's AGENTS prompt and the candidate submission contract,
   and submits its full panel through the existing task validator.
4. Close the children and archive their panels. Existing LDM acquisition prunes
   the candidate pool and requests true evaluations. The next main turn reconciles
   those results. Previous child conversations never become the next parent.

The tree has a persistent trunk and one-round leaves. This is not multi-depth
MCTS: no invented reward, UCB over researcher identities, or extra oracle calls.
The policy researcher remains a separate persistent session. Ending child runtimes
does not erase their traces, files or cost records.

The main role performs evidence synthesis, not new remote investigations. An
initial qualification attempt exposed serialized BLAST polling before any branch
started; that attempt is retained as `qualification-open-ended-main`, including
its consumed tokens. The main prompt was tightened before the formal experiment.
New searches and long analyses belong in the parallel branches.

## State, recovery and costs

- `harness/main`: persistent main identity and turn commits.
- `harness/branches/round_N/fork-source`: immutable session/workspace snapshot.
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
