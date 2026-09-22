# AlphaBench T3 implementation review

Implementation base: `56dd03f2a62ca888a5505e49014b86629ad47554`.
Branch: `codex/alphabench-t3-implementation`. All execution takes place on the
designated remote host under `/mnt/data1/Large-Discovery-Models/`.

## Change 1: durable shared selection lifecycle

Cross-check against the complete T3 plan: W02, X05–X06 and Y05. The engine stores
the entire selected batch, candidate payloads, selection metadata and schema
before preparation or dispatch. Recovery processes this batch before budget
exhaustion checks or another proposal/policy call. Stored results precede
idempotent evaluation events and checkpoints. An ambiguous dispatch without a
matching task receipt pauses; it cannot become a scored failed candidate.
New selections always clamp to all remaining attempt limits. The Host-owned
effective batch context cannot be overridden by the caller.

Complexity review: removed the old paid-prefix-only recovery path and its
separate pending-candidate machinery. Fresh and recovered selections share one
execution function. Removed tests that required repeating proposals or implicitly
resending an evaluation after interruption; retained the budget boundary tests
and added end-to-end crash boundary coverage. No second runtime or checkpoint
migration path was added. Manifest discovery permits newly registered tasks.

Validation: 87 targeted engine/campaign/reporting tests and the shared suite
(486 passed, 1 skipped) passed on the remote host using its locked environment.
This review does not qualify any market, backend or T3 method.

## Change 2: draft task lifecycle and offline data preparation

Cross-check: W00–W01, W03–W05 and the current W07–W09/W11/W14 subset. The task
registers through the existing runner and uses the shared Campaign, GP,
transport, budget ledger and collection sink. Direct LLM and LDM execute
initialization, search, private validation, frozen test selection and analysis.
The full pinned operator vocabulary is admitted; generated expressions retain
static/dynamic rejection evidence. Physical requests have durable identities;
an unknown result pauses instead of receiving a fabricated score or a free retry.

The user's network constraint is enforced by the data CLI boundary: acquisition
runs on the local computer; source/data installation and coverage audits run
offline on the server. Archive and member hashes, snapshot-qualified SQL,
response completeness, historical membership and trading-session coverage are
checked. The official AlphaBench archive was inspected and contains four
precomputed IC/RankIC tables, not the market data needed by arbitrary new T3
expressions. Public data stays unqualified while its recorded gaps remain.

Cross-check fixes made before committing: moved CLI parsing and task description
into the actual manifest entry point; fixed its run directory so the runner's
task working directory cannot create nested task paths; enforced the mock
contract profile; included initialization costs separately in the final report;
used UTF-8 and stable LF JSON bytes across Windows/Linux transfers; rejected
truncated HTTP bodies and stopped batch acquisition on repeated connection
failures; checked all required Qlib fields on observed trading sessions.

Complexity review: deleted the obsolete scaffold mock engine and the import of
a nonexistent Assay worker. Unsupported native profiles, methods and backends
fail explicitly. Removed unused imports and unused coverage counters, kept one
shared campaign path, and used the same acquisition/cache/transfer primitives
for the data sources. No alternate evaluator or automatic data-provider fallback
was added. Source archives remain external; raw data, environments and traces
are excluded from Git. Environment locks use the domestic PyPI mirror.

Validation evidence: remote registration, dependency checks, runner dry run and
the `mock_ldm` profile pass. The measured mock includes 30 initialization seeds,
two search evaluations, full test outputs and paired IR/SFT publication;
completed replay leaves all 193 persisted files unchanged. Compact records live
under `tasks/alphabench/resources/evidence/`. The shared suite passes 486 tests
with one skip; 112 task tests pass. Registration, dependency checks, the mock
profile, paired collection and completed replay were checked on the server.

Open work is still substantial: data qualification; full Assay evaluation and
portfolio adaptation; original CoT/ToT/EA concurrency and recovery; source
profile parameters; persistent Harness and compiled policy; remaining full-pool
metrics/quality coverage; process crash and guest isolation checks; all 72
method/backend/market rows and clean real reproduction. This commit only advances
registration to `mock_verified`. The experiment stays `draft`, real data stays
`blocked`, and reports keep `complete_t3=false`.
