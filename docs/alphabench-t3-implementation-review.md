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

## Change 3: recover an interrupted event append

Cross-check against W02/W15: an interrupted last JSONL append must not prevent
reconciliation of already completed evaluation receipts. Resume parses the
committed prefix, saves an incomplete final fragment with its byte offset, and
truncates only that fragment. A complete final event without a newline is kept
and terminated before appending. Corruption in a terminated record still fails
without modifying the log. Existing event keys and budget records survive.

Complexity review: the existing JSONL reader owns this narrowly requested resume
behavior; normal reads do not repair or suppress errors. No new log format,
recovery subsystem, compatibility path or generic workflow layer was added.
Repair assumes the existing single-writer campaign ownership contract.

Validation on the remote server: 91 focused campaign/engine/reporting tests,
490 shared tests (one skipped), and all 112 AlphaBench task tests pass. Tests
cover interrupted UTF-8, a complete unterminated event, committed corruption,
unchanged budgets, event-key deduplication and continued sequence numbering.

## Change 4: generation cost and complete quality denominators

Cross-check against W14 and Spec section 13 found a mislabeled metric: the first
threshold-crossing evaluation is a discovery diagnostic, whereas Search Cost
counts model attempts per logical generation step. Reports now name these
separately and retain failed steps, per-step counts, search-only statistics and
statistics including initialization. Failed evaluations cannot establish a
successful threshold crossing even if a malformed result includes an IC.

Generation occurrences now reference their actual dynamic-check receipt. The
frozen quality audit verifies request/response hashes, expression, protocol,
interval and operation before reusing a result. An occurrence with its own
check preserves that outcome. Previously unchecked duplicates reuse an existing
check; unchecked new expressions use the independent quality budget. Exhausting
that budget leaves the dynamic rates unavailable with explicit coverage, rather
than treating unmeasured items as failures. Static invalid items remain known
failures. Quality outcomes do not enter proposal history or training inputs.

Complexity review: removed the obsolete metric names and the unconditional
second evaluation of every checked expression. Kept receipt references in the
existing occurrence journal and counters in the shared budget ledger; no cache
service, compatibility projection or new reporting layer was introduced. Removed
a redundant CLI parser assertion while retaining the complete procedure test.

Validation: 114 remote task tests pass, including complete replay, failed-step
cost, threshold boundaries, receipt contract mismatch and incomplete audit
coverage. The fresh mock reaches full audit coverage with zero additional
quality calls; completed replay leaves 185 persisted files unchanged. Other
metrics, native methods, backend and matrix qualifications remain open.

## Change 5: local CN status capture and complete offline source audits

Cross-check against W04 and the user's local-download constraint: the official
AlphaBench download contains precomputed IC/RankIC tables, not the OHLCV assets
needed to evaluate arbitrary new T3 expressions. The local US download completed
all 3,100 pinned queries with zero failures. A verified 10,069-file bundle was
transferred and installed without any server-side network acquisition.

Added an explicit BaoStock acquisition step for the unresolved CN sessions.
Queries are bound to a security and one calendar year, below the client's
pagination boundary; errors never become valid cache entries. The live source
has no snapshot API, so offline reproduction binds the saved timestamped
responses and hashes. Only an exact tradestatus=0 corroboration removes a gap.
Absent rows, reported trading, and membership extending past reported delisting
stay unresolved. Audit provenance binds both source response inventories.

Complexity review: this is one optional acquisition dependency group and one
source-specific module using the existing bundle/import/audit path. No provider
fallback, synthesized prices, automatic qualification, or generalized data
service was added. Factored the existing Dolt response reader to share the same
offline validation instead of trusting mutable earlier audit summaries.

Remote validation: all 116 task tests pass. The actual offline US audit reads
1,609,180 price rows and records remaining coverage and asset failures. CN gains
774 corroborated suspension dates; remaining gaps are 107/1,028/259 security-days
for CSI300/500/1000. Compact hashes and counts are committed, raw assets stay under
/mnt/data1/. Data and full T3 qualification remain blocked.

## Change 6: complete source parsing and initialization recovery boundaries

Cross-check against W08 found that the official JSONL final pool could not be
imported, empty initialization was rejected, and an interrupted initialization
could not resume before the main campaign existed. All source formats now reduce
to names/expressions, with original content retained privately. External outcome
fields never enter candidates or substitute for fresh measurements. Source
digests and indexed admission/rejection/duplicate records survive recovery.

The previous initialization run ID was the same directory basename for every
run, allowing cross-run oracle request collisions. Fresh initialization and
search campaigns now use independent UUID identities, retained on resume.
Provider pauses before engine startup are durable; nonnumerical failure metrics
are excluded. Empty initialization uses zero initialization/model budget while
leaving the complete new search budget available.

Complexity review: removed the previous duplicate file/pool loaders and reused
the existing source receipt, Campaign, grammar and oracle implementations. There
is no import of old scores, second evaluation loop or compatibility parser.
Generator construction is limited to cold initialization that actually needs it.

Remote validation: 126 task tests cover source formats, unknown/duplicate/invalid
entries, empty initialization, partial evaluation failure, a pause before search
exists, provider pause before engine startup, unchanged completed replay and
distinct cross-run request identities. The fixed upstream loader yields the
13/29/38/42 entry groups; all 125 records in its seed file pass static admission.
These are source-contract checks, not data-backed seed qualification. Shared
evaluated seed-bundle import and native source profiles remain open W08 work.

## Change 7: shared evaluated initialization for matched methods

Cross-check against W08: matched methods can now import a completed initialization
through --initialization-bundle. The verifier binds the source/model/scientific
contract and requires complete initialization and validation receipts. It checks
candidate identity, original admission order, authoritative observations, raw
response hashes, phase/interval/expression, and creation accounting. Synthetic
records cannot be imported into a real campaign. Method/search/portfolio knobs
do not change the initialization contract; data, label, objective, filter, model,
initialization source settings and runtime limits do.

Each target uses the same public observations and keeps validation on the Host.
Its own initialization counters are zero, while source creation cost and the
public-information digest are preserved separately. Reports include source
generation attempts in statistics including initialization. Import can recover
before its finish record without generating or evaluating seeds again.

Complexity review: the source is the existing completed initialization directory;
no second bundle format, score migration, copying of model sessions, or generic
cache service was introduced. Import uses one existing CampaignRuntime and an
inventory of the exact verified artifacts. Fresh generation and shared reuse
retain distinct costs. The fresh cold-start budget check moved to dispatch
preparation so a verified import can legitimately have zero initialization budget.

Remote validation: 132 task tests pass, including cross-method observation
identity, zero import cost and full new E, changed scientific/mock identity,
corrupt or unfinished receipts, substituted measurements, interrupted import,
and byte-identical completed replay. Real data qualification, native execution
and the remaining full T3 work packages are still open.

## Change 8: offline Assay factor and independent portfolio worker

Cross-check against W04/W06 found three incorrect native defaults for this task:
period-end constituents, static group vectors and an equal-weight benchmark
proxy. The adapter now loads the historical union, preserves pre-entry history,
masks every cross-sectional operation by effective membership, and supplies
per-session group labels and the actual index series. Native parser, numerical
kernels, adjustment, IC and independent portfolio accounting remain in use.
The guide's absent Tanh/Mask kernels are registered with their exact definitions.

Every offline asset is content-verified on each worker invocation. Duplicate
identities, unknown knowledge dates, insufficient interval/warmup coverage,
missing tradability evidence and mismatched benchmark/market fail explicitly.
Single-horizon labels carry their execution convention and purge n or n+1
sessions. The two-week check returns no performance information. Full results
retain raw IC and portfolio reports, scores, sample counts, NAV, costs, actions
and holdings. The independent Assay configuration is fully frozen; Qlib drop
parameters are not silently translated into another strategy. Combination
identity includes its finite-factor averaging rule and frozen direction.

Complexity review: one adapter subclasses the existing engine only at the
historical cross-section boundary, and the existing backtester only at its
prepared-input boundary. It does not reimplement numerical or accounting
engines, introduce a second service or add a backend fallback. Knowledge-date
checks share one helper; the old unconditional Assay rejection is removed.
Health/dispatch now bind market for both backends. Portfolio-only settings are
excluded from the shared initialization contract, where they have no effect.

Remote validation: nine Assay tests pass, including all 84 guide operators,
independent numerical boundary cases, CN/US portfolio fixtures, actual child
process execution, completed replay with unchanged artifacts and market
mismatch rejection. The task suite passes 132 tests, with the Assay module
skipped in the lightweight task environment and tested separately in its lock.
Registration validation passes and continues to report draft/mock_verified.
There are still no qualified Assay data snapshots; acquisition, complete
operator numerical audits and the real matrix remain required for W06 closure.

## Change 9: independent validation ranking and auditable diversity sets

Cross-check against W09/W14 found that validation ranking reused the search
objective, structural normalization used node counts instead of the fixed
source's maximum pair distance, and arithmetic AST labels lost the actual
operator. The frozen validation metric now independently accepts IC, RankIC,
ICIR or RankICIR, with deterministic identity tie-breaking. Addition/subtraction
and unary signs remain distinguishable after constants are removed. Distances
use unit tree-edit costs and the documented maximum-distance normalization.

Both required sets now have their own reports. The entire successful measured
pool uses existing private validation scores, and the frozen test selection uses
its test scores. No new free evaluation or pre-selection signal computation was
introduced. Each correlation records candidate identities, shared and finite
sample counts, sample-index digest and an explicit undefined reason. Missing
member scores make the aggregate unavailable; missing pairs are not filled with
zero. Scope, interval and selection identity are retained in the report.

Complexity review: removed the superseded ambiguous diversity fields and the
incorrect normalization; there is one projection for both sets. Reuses the
existing validation responses and grammar rather than copying oracle artifacts
or adding a metrics service. Portfolio/validation-selection controls are excluded
from the common initialization contract because initialization evaluates and
retains the same complete metric set independently of those later choices.

Remote validation: 137 task tests pass, plus the separately pinned Assay tests
from Change 8. Four complete mock campaigns demonstrate different validation
winners while the search objective stays fixed. Numerical tests cover known
tree distances, sign/operator distinctions, positive/negative correlation,
pair sample alignment, missing members and duplicate sample rejection. Full
W14 closure still requires native update events, preregistered run aggregation,
and the remaining full-task quality/reporting/collection gates.

## Change 10: bounded Host writer handoff with concurrent physical requests

Cross-check against W05/W10 confirmed that native EA overlaps next-generation
model calls with current-generation evaluation, while ToT uses concurrent
branches. Serializing whole callbacks would change that execution contract.
Receipts now perform preparation/reservation and completion on the Host writer,
with physical I/O on the calling worker between those boundaries. Model usage,
oracle job settlement and accepted generation/collection commits use the same
handoff. The current direct Campaign expansion runs through the live message
pump, so this is an exercised execution path rather than an unused native stub.

The Host drains a bounded queue while its stage runs in a background future.
Callbacks never mutate the ledger from worker threads. Stage cancellation
releases pending callers before executor shutdown; callbacks arriving after
closure are rejected. Queue backpressure releases the lifecycle condition while
waiting, avoiding a producer/Host shutdown lock inversion.

Complexity review: one task-local dispatcher and the existing receipt state
machine cover model and oracle operations. There is no alternate ledger, copied
runtime, per-worker budget cache or backend request retry. The synchronous
service receipt path uses the same state transitions without a dispatcher;
the Host's existing direct calls still execute on their owning thread.

Remote validation: the task suite has 143 passing tests. Concurrency tests cover
two requests contending for one remaining permit, overlapping physical calls,
model pre-reservation, immutable receipt/budget replay, unknown outcomes,
interruption without deadlock, late-callback rejection and 80 concurrent callers
against the bounded queue. The nine separate Assay tests also pass after the
receipt split. Native algorithm scheduling/recovery and the shared Harness
provider preauthorization protocol remain required next steps; this change does
not claim those adapters are complete.

## Change 11: pinned native algorithm execution and durable concurrent replay

Cross-check against W10/X06 confirmed that retaining the real source requires
both the nested execution order and native state transitions. The loader verifies
all five original algorithm files, applies only import/state instrumentation and
a ToT shared-best race fix, and saves the original/patched/runtime hashes. CoT
chains, ToT trees and EA populations execute through the official create_algo.
The original IC seed selection, survivor rules, thresholds, mutation/crossover
and EA generation overlap remain in that code.

The runtime records worker identities, leaf callback inputs/outputs, completion
delivery, lock order, UUIDs, clocks and actual chain/node/population snapshots
through the existing Campaign event writer. Completed callbacks replay without
dispatch; incomplete callbacks still require durable request receipts. Replay
validates the complete stored output set before execution, rejects changed
inputs and releases waiting workers on control signals. Budget and pause signals
escape upstream broad Exception handlers instead of becoming empty metrics.

Complexity review: no copied search loop, second ledger/checkpoint database,
source package installation or compatibility branch. State/clock/naming hooks
use one event primitive. Removed duplicate input/output validation during replay
after moving it to the earliest valid boundary. The four source callback seams
remain explicit. Official generator integration and the atomic search-batch
budget bridge are deliberately not advertised as finished or exposed through a
working-name fallback. The workflow still rejects these unfinished methods.

Remote validation: 164 task tests pass, with one Assay environment-only module
skipped as before. The 21 native tests include all three original-versus-patched
algorithms, exact completed replay, overlapping I/O, changed-input rejection,
source tampering, and budget signals in generation and evaluation. Nine tests
terminate real subprocesses after generation, evaluation or state commits and
resume with the shared receipt/ledger machinery; completed physical calls stay
at one. These crash points require no unknown in-flight receipt. Existing Host
tests separately verify that unknown requests pause without POST retransmission.
Scientific parity retains chain/generation/ranking order; only ToT's independent
parallel-history ordering is normalized for separate-run comparisons, while
same-run replay compares every field and list order exactly. W10 still needs the
actual generator, budget/validation/finalization bridge and source profiles.

## Change 12: official native generation, repair and auditable raw occurrences

Cross-check against W07/W10 found that a native search callback must retain the
official repair prompt, partial-success threshold and even the shuffle performed
when exactly N factors have been collected. The adapter verifies the complete
generator/prompt source hashes and executes the four required source functions.
It fixes empty-array division by zero and malformed single-object fields, with
the latter following the array normalizer's existing reason/type policy. It
does not replace the native loop with the direct Campaign's exact-K generator.

Model calls reuse Generator.request and its pre-reserved durable receipts;
dynamic checks reuse the same gateway and grammar. All raw occurrences, including
schema failures and the unvisited tail, are retained alongside normalized,
visited, checked and accepted facts. Original native quality/return values are
retained as JSON. Service failures, malformed receipts and configuration errors
escape the native catchers, so they cannot consume five more model repairs as
fake expression failures. Actual source algorithms now exercise this generator
in the fixture integration path.

UUIDs, set iteration, sample indices and RNG state use keyed events in the
existing Campaign journal. Completed generation callbacks skip these internal
events; an incomplete generation reconstructs them with its model/check receipts.
The global RNG resumes at its last committed state and is not rewound when an
old choice is reused. Accepted actions use the existing paired collection
publication. Review exposed a tuple/list mismatch in persisted native error
records; canonical JSON conversion now makes this partial-publication recovery
idempotent, including a generation with a rejected expression.

The frozen protocol now includes the requested temperature. The existing client
factory passes it and maps native JSON mode to Responses text.format. The
official DeepSeek Responses guide states that temperature is ignored in thinking
mode; generation records explicitly report the requested value and unavailable
effective value. User-selected deepseek-flash / Responses / max remains fixed.

Complexity review: kept only the required upstream definitions, omitting unused
helpers, heavy imports and demo entry points. No copied generator loop, SDK,
oracle implementation, second collection exporter or per-request client clone.
Removed an extra check wrapper and kept control conversion at the actual I/O
boundaries. The direct generator remains a separate required submission contract,
not a compatibility path. The main native workflow remains gated pending the
atomic batch budget, private validation/finalization bridge and source profiles.

Remote validation: 177 task tests pass, one backend-environment module is skipped,
and the nine tests in the pinned Assay environment pass separately. The 13 new
generator tests cover original-source parity, five repairs and both sides of the
partial-success threshold, raw/visited/checked divergence and tail quality audit,
Qlib/Assay prompt and dialect binding, all three actual native algorithms, RNG
continuation and paired collection after interruption, unknown/malformed check
responses, and actual HTTP Responses bodies with JSON/max thinking and one POST
on a 503 failure. These are implementation checks, not qualified market runs or
the user's final two-round real-endpoint acceptance.

## Change 13: atomic native batches and private evaluation settlement

Cross-check against W05/W10 and X05/X06 found that reserving E separately from
worker permits could spend an entire batch's search allowance before the first
request discovered a permit shortage. The shared ledger now commits several
cumulative usage keys in one transaction. The native adapter reserves all search
attempts and their job permits together, retaining the gateway's existing keys.
A failed group changes neither counters nor persisted usage. Single-request
confirmation and replay remain idempotent through the same implementation.

All three evaluation callback shapes use one native evaluator. Callback identity
and input position distinguish repeated expressions from replay. CoT/ToT seed
queries have a source-instrumented role and reuse the verified matched initial
observations; seed information, protocol, worker count and adapter implementation
are frozen. The source algorithms still select their own seeds and preserve
their chain, tree and population transitions. Empty ToT nodes now also emit the
existing state event before the source's early return.

Search receipts settle their actual jobs before private validation. Successful
validation completion precedes publication of each public observation; neither
validation scores nor validation failures become search scores. Observations
retain logical submission/input order independently of concurrent completion.
Failed search attempts retain their true status and cost. Malformed job records
pause through the native control signal instead of triggering upstream fallback
evaluation or appearing as invalid formulas.

Pre-dispatch authorization checks the shared stop flag inside the Host writer,
including each model repair and dynamic check. Completed receipts stay readable
after a stop. Stop assignment has its own short mutex; Host authorization does
not acquire a scheduler condition that a waiting worker may hold. After all
workers drain, settlement reconciles begun requests, finishes private validation
and lists reserved-but-unstarted candidates without fabricated observations.
An unknown in-flight result still pauses even when another worker first stopped
because E was exhausted. The durable stop record prevents new dispatch after
resuming a terminal matched budget stop.

Complexity review: the old single-key budget calculation was replaced by the
multi-key implementation, not retained as a second path. Request construction
and reservation identity each have one implementation shared by ordinary and
native evaluation. No copied native loop, alternative evaluator, separate
budget ledger, per-batch checkpoint database or expression-result cache was
added. The three callable shapes exist only to match the official four seams.
Formal workflow/finalization wiring, source profiles and native cold generation
remain gated; this component does not claim W10 or full T3 qualification.

Remote validation: 195 task tests pass with one backend-only module skipped;
491 shared tests pass with one existing skip; all nine pinned Assay tests pass
separately. Eighteen new task checks cover atomic E/permit rejection, competing
complete batches, private validation, seed reuse versus fresh re-proposals,
post-stop cancellation, unknown outcomes, provider repair authorization, failed
search billing, validation reconciliation, malformed job receipts and all three
actual algorithms with both metered generation and evaluation. Two checks kill
real subprocesses after ledger reservation or committed observation and verify
that resumed physical search/validation calls and charges occur only once.

## Change 14: native matched workflow, cold initialization and finalization

Cross-check against W01/W07-W10/W14, X05/X06/X09 and Y04: the official CoT,
ToT and EA now run through the registered task entry point on one shared search
runtime, without starting a search LDMEngine. All required native parameters and
fixed source hashes are checked before initialization incurs cost. The native
cold initializer executes the pinned pipeline function with the frozen seed
count; its extraction of partial factors is preserved even when the generator
reports unsuccessful completion. Verified shared initialization bundles can
cross method boundaries while preserving exactly the same initial information.

Both search paths use the existing private validation, frozen selection, full
test, combination, quality and collection finalizer. Reports distinguish the
actual native return and native pool from the matched comparison pool. A whole
batch budget stop uses only committed native state and does not invent a normal
algorithm return. Durable search completion lets finalization resume without
rerunning search or generation. Empty initialization follows the actual method
requirements: CoT/ToT pause, while EA can generate its first population.

Cross-check of EA's overlapping next-generation work found that terminal budget
exhaustion could otherwise hide a completed model response or an unknown check.
Settlement now inspects all begun model/check receipts after workers drain.
Unknown outcomes pause. Completed output from abandoned generation retains all
raw occurrences and reusable checks, has no fabricated native result, and is
reported separately as interrupted. Completed accepted actions use the same
publication function during execution and recovery. Canonical request JSON fixes
action identity changes caused by persisted message-key ordering.

The shared runner's named profiles now bind the complete protocol content digest
as well as locked CLI arguments and E. The exact runner contract is snapshotted
before paid work and its identity reaches the search runtime. Direct resume
without runner environment variables still verifies that frozen contract;
tampered snapshots and protocol contents are rejected. Three complete two-round
mock profiles are registered, with dependency checks for the actual fixed source.

Complexity review: replaced the engine-result wrapper in finalization with one
explicit execution record; no compatibility alias remains. Reused the existing
shared runtime, gateway, receipt store, collection exporter and finalizer.
Source hash verification and native raw-item extraction each have one shared
implementation. Removed duplicate accepted-action construction in recovery.
No second search engine, copied algorithm loop, standalone checkpoint store or
new dependency was introduced. Existing direct LLM/LDM behavior remains covered.

Remote validation: the task suite passed 214 tests with one backend-environment
module skipped. A subsequent focused run passed all 16 native workflow tests,
including the added direct-resume contract check and nonempty paired collection
assertions. Tests cover both oracle dialects, shared seed imports, empty seeds,
whole-batch exhaustion, unknown requests and finalization reconciliation. Task
registration and dependency validation passed. Actual shared-runner executions
completed under `tasks/alphabench/runs/mock_4` through `mock_7`: native CoT used
4 new search attempts, ToT and EA each used 8, and existing LDM used 2. All saved
runner/protocol identities matched; each run completed test and quality stages.

These runs use synthetic model/oracle responses. Native source-profile resolution,
market/data qualification, persistent Harness, compiled policy and the remaining
full-T3 gates are still pending. Neither W10 nor the requested two-round real
DeepSeek acceptance is marked complete by these results.

## Change 15: source entry resolution and repaired example execution

Cross-check against W01/W10 found material differences beyond the documented
constructor mismatch. The example evaluates all 42 Alpha158 factors before
selecting 13 kbar/price seeds for CoT/ToT or 29 rolling seeds for EA. Its four
workers limit concurrency; they do not select four top-IC seeds as the searcher
adapters do. Its convenience client also defaults to inclusive 2023-2024 full
backtests, while only baseline evaluation receives the requested market. The
entry itself supplies no private validation or held-out test stage. These facts
are now explicit; missing stages are not populated from another entry's defaults.

The source resolver verifies fixed inputs and records declared/effective values,
parameter origins and adapter deltas separately. Algorithm defaults, physical
client defaults and final-test parser defaults come from the pinned source AST.
The user-selected DeepSeek model and ignored temperature behavior remain explicit
departures from source model settings. Seven source entry/method combinations
have been resolved for both CSI300 and SP500, with complete server artifacts and
a committed evidence index. Resolutions remain non-executable until their full
oracle, filtering, final-pool and budget contracts are wired.

The example loader executes the original benchmark_main/run_batch and original
three algorithms. Repairs change obsolete callable names, remove unsupported
verbose arguments and inject the required single/list/name-indexed oracle shapes.
One explicit initialization seam accounts for the full baseline independently
of search. Original grouping, seed order, worker cap and native search decisions
remain in the original code. Control signals pass through the original worker
exception handlers; a budget stop cannot become an ordinary failed seed result.
Returned local summaries retain actual chains/trees/pools and worker errors.
The entry freezes its complete input configuration before baseline dispatch;
changed model parameters are rejected even during replay of completed callbacks.

Complexity review: reused the existing source verifier and recorded scheduler;
no replacement benchmark loop, provider, evaluator, persistence store or package
was introduced. Alpha158 module loading now has one implementation used by the
existing initializer and the example entry. Removed the unused process-executor
path from the extracted entry, whose actual caller always requests threads.
Source outputs stay separate from executable task protocols rather than adding
an implicit compatibility conversion or inventing validation/test defaults.

Remote validation: 229 task tests passed, with one backend-environment module
skipped. Fourteen new checks cover all seven parameter/seed resolutions, immutable
output and source tampering, the actual three example methods with all group
seeds, exact replay without new callbacks, and budget control through workers.
Initial replay assertions were corrected to exclude the shared runtime's explicit
campaign-resume event; the native journal itself remains byte-identical on replay.
All validation uses synthetic oracle responses. Source profiles, W10 as a whole
and full T3 are not claimed complete by this component.

## Change 16: source filtering and honest quality evidence

Cross-check against Spec 5.2 and W04/W06/W14 found two behavior mismatches:
Qlib code checks had lost the pinned FFO 1% NaN rejection, and Assay code checks
were evaluating market panels instead of calling the source's data-free lint.
The Qlib worker now invokes the original `_check_single_column`; Assay invokes
the original `lint` before accessing any asset. Source diagnostics remain in
the receipts. NaN and non-finite ratios are separate, so the source code rule
and the paper rule do not silently share different meanings of "NaN".

One protocol method validates the result boundary and applies the selected
filter. Both generators and final quality auditing use it. Lint has its own
counter and rejection status; it cannot produce measured missing-value ratios
or a paper dynamic success rate. Paper auditing also checks depth 5 when code
filtering admits depth 6. Incomplete audit coverage stays explicit. Invalid
check evidence pauses after physical-job accounting, and completed receipts
prevent duplicate dispatch on recovery. All four named mock protocol digests
were regenerated after adding the frozen lint budget.

Complexity review: removed the duplicated generator threshold implementations;
reused the existing gateway, receipts, Host ledger and upstream checkers.
No second checking service, compatibility path or runtime package was added.
The Qlib environment adds pytest only in its test dependency group. Numerical
tests build an explicit fixture with Qlib's storage API and compare the actual
worker against the original checker, including the exact 1% boundary, NaN vs
Inf and the 30-second paper limit. Missing ratios are never fabricated as zero.

Remote validation: 236 task tests passed; the two backend modules skipped in
the generic environment passed separately (10 Assay tests and one Qlib test
covering five numerical cases). The Qlib fixture initially needed Qlib's global
configuration initialized before using FileFeatureStorage; that fixture setup
was corrected. Registration validation and the procedure CLI passed. A fresh
shared-runner mock campaign completed at `tasks/alphabench/runs/mock_8`; replay
left all 187 files unchanged. Its refreshed evidence includes the current
protocol digest, quality contract, budget and both paired collection exports.
This campaign uses synthetic model/oracle responses, not the real-model gate.

The official download was rechecked against the fixed repository: four
precomputed IC/RankIC tables support T2/T4, while T3 requires separate market
data. DATA.md now distinguishes ZIP-directory inspection from a full download.
Local acquisition, hash-verified transfer and server-only numerical preparation
remain the data workflow. Existing real-data gaps, source interval/final-pool
integration, Harness/compiled methods and the complete T3 acceptance gates
remain open; this filtering change does not qualify those capabilities.

## Change 17: native interval boundaries and backend-specific label tails

Cross-check against W04/W06/W09/W10 and the pinned source found that both
workers always applied matched end exclusion and label purge. The source
searcher and example both use inclusive endpoints, but the backends differ:
FFO/Qlib Ref labels read later bars, whereas Assay service._build_engine reads
only its requested panel and forward_returns leaves unavailable tail labels.
Extending Assay's panel would change its source semantics and could change its
corporate-action adjustment basis. The adapter now preserves each behavior.

The existing profile determines endpoint inclusion. All seven pinned entry/
method resolutions are checked against the task's declared intervals. The
example keeps its actual 2023-2024 search dates; requesting its absent validation
or test interval raises explicitly. It cannot inherit matched dates. Qlib
computes the forward read end from its real calendar and queries only retained
label rows; matched evaluation no longer even requests labels beyond its split.
Source Qlib retains all inclusive signal dates. Assay source retains its
undefined tail IC rows and still backtests the complete inclusive interval.
Responses record requested/actual boundaries, label read end and tail treatment;
Qlib also records each horizon's valid cross-sectional sample count.

Calendar coverage failures use the existing EvaluationPaused control rather
than generating bad-factor feedback. A completed worker response preserves its
pause reason and physical-job receipt. The gateway charges that job exactly
once even on the search path before propagating the pause. Replaying the same
receipt does not dispatch or charge again. This applies to calendar coverage;
it does not claim that every other data-qualification gate is now resolved.

Complexity review: kept one worker path per backend and reused the frozen
profile, source numerical functions, existing receipt protocol and pause type.
Removed the obsolete end_exclusive-only output and the duplicated Qlib label
validation already enforced by T3Protocol. Label expressions are constructed
only for evaluation after the required calendar range is known. No compatibility
adapter, date-policy hierarchy, extra configuration surface or package was added.

Remote validation: 238 task tests passed; two environment-specific modules were
run separately, with 17 Assay and six Qlib tests passing. Qlib daily correlations
are checked against explicit multi-horizon arithmetic. Assay checks close/open
tail lengths, undefined IC with zero valid pairs and isolation from a later
split event. Full Qlib and Assay portfolios verify dates, positions, actions and
costs, including Qlib's CN 100-share lots. A bounded real Assay worker verifies
that missing-calendar output remains a metered pause.

Qlib emitted 17 upstream empty-mean warnings. A diagnostic run promoting them
to errors traced the first to the order-fulfillment indicator on an empty order
array. The fixture was also corrected to exclude its benchmark from the trading
universe, avoiding Qlib's adjusted-price mode; the final suite passed with the
ordinary upstream warnings visible. No warning suppression or source patch was
added. These are synthetic numerical/portfolio fixtures, not real-market or
model-endpoint acceptance. Source workflow initialization/final-pool/budget
wiring, remaining agent methods, data qualification and the full T3 gate remain
unfinished.

## Change 18: complete Qlib searcher workflow binding

Cross-check against Spec 3/6/7, W08-W10 and the pinned SearchPipeline found that
source execution was still gated after its lower-level intervals and filters
were implemented. The native entry now resolves the selected original YAML
before initialization costs, verifies its source hashes and binds algorithm
parameters without substituting matched defaults. CoT/ToT retain 30 cold seeds;
EA automatically loads all 125 distinct fixed seeds. A different seed file,
shared matched seed bundle, method, rounds, temperature, filter or final
portfolio contract is rejected. Source dry-run performs the same configuration
checks and displays the resolved entry. Data qualification is still required
for real execution.

SearchPipeline's ValEvalTracker also validates failed search evaluations; the
source adapter now does the same without exposing validation to the algorithm.
Finalization takes the algorithm's actual final_pool, preserving member order,
names and duplicate occurrences. Finite validation ties retain source order;
the tests execute the original rank_factors function to compare the selected
sequence. Source runs pause before test when validation is incomplete instead
of silently using search scores. This full-T3 requirement, provider override,
longer Host transport timeout and reuse of verified seed receipts are explicit
entry differences. Qlib source search/validation remain fast; test remains
50 factors / 50 stocks / drop 5 with full portfolio output.

The first full CoT test exposed an existing reporting assumption: a chain may
retain the same formula in multiple rounds, but signal diversity rejected
duplicate candidate IDs. Reporting now indexes pool occurrences directly and
records both candidate identities and pair positions. A three-member numeric
fixture proves duplicate weighting rather than collapsing a chain into a set.
No extra identity layer or duplicate-removal workaround was added.

Source E exhaustion pauses rather than publishing a truncated result under an
original-round profile. The strengthened test lets EA complete 20 evaluations
under a 25-evaluation cap before rejecting the next complete batch; replay
leaves both the budget and model receipts unchanged and publishes no result.
Matched E-stop finalization remains independently covered. All three source
workflows also replay completed runs without changing any file.

Complexity review: reused the existing resolver, native algorithms, evaluator,
initialization Campaign, shared ledger and finalization function. Removed the
obsolete matched-only entry/evaluator gates; no second search loop, budget
store, compatibility path, package or speculative profile abstraction was
introduced. Final-pool projection is a short branch over the same measured
receipt index. Source-only settings stay in the existing native_parameters
contract, leaving matched protocol identities unchanged.

Remote validation: 246 task tests passed, with two backend-specific modules
skipped in the generic environment; this change did not modify the workers.
After review, all 24 native campaign tests passed, and three additional dry-run
rejection checks passed. Shared regression: 491 passed, one skipped.
Registration, dependency check and shared-runner dry-run passed. A fresh
shared-runner mock completed at mock_9; replay preserved all 187 files and both
paired collection exports. Source workflows completed 40 CoT, 60 ToT and 200 EA
search evaluations, yielding 44/60/30 native final-pool members and 44/50/30 full
mock test evaluations. Compact evidence and archive hashes are recorded in
resources/evidence/native_searcher_workflow.json; the 14 source resolutions and
standard mock evidence were refreshed.

These runs use synthetic model and oracle responses. They do not establish
real-market numerical qualification or the requested final two-round model
acceptance. Assay's complete source contract, the example workflow's absent
validation/test extension, remaining agent methods, data gaps and the full T3
matrix remain open. The full goal and complete_t3=false are unchanged.

## Change 19: bind Assay source searcher defaults

Cross-check against frozen W06/W10 and the pinned AlphaBench bridge plus Assay
service confirmed that the source searcher still rejected the Assay backend.
Source resolution now verifies the Assay defaults and relevant AlphaBench
bridge, records `next_open`, split adjustment, single horizon, period-end
universe and the arguments the FFO route ignores, then validates the complete
native portfolio config before any initialization cost. The Python adapter
matches the source diagnostic failure behavior, including a `CONSTANT` warning
that the FFO wrapper treats as failure. Actual service fixtures compare daily
and aggregate IC for both CN and US, including explicit group input and a
non-trading period-end snapshot. The three native methods now replay through
both source backends with synthetic oracle responses. Source resolution reports
cover all 63 method/config/backend/market combinations; these are configuration
checks, not market evaluations.

The non-trading boundary exposed one portfolio mismatch: the source factor
universe is chosen at the requested end date, but the independent backtest
configuration used the last trading date. Source portfolio dates now retain the
requested inclusive interval and point-in-time cutoff; its daily NAV still
ends on the last actual session. Matched portfolio dates remain the retained
signal dates.

Complexity review: the source path uses the existing resolver, worker, native
algorithm, receipts and single `PortfolioBacktestConfig`. A small shared
validator moves preflight checks out of the evaluation function so startup and
execution apply the same rules. Matched panel selection remains unchanged. No
new service, fallback path or duplicate task schema was added.

Cross-check limitation: the plan requires a Host request to the Assay
`/v1/portfolio/backtest` route. The current worker invokes the pinned Python
portfolio entry directly to supply the frozen offline panel, so W06 remains
open until the REST route can use the qualified task data store and its complete
response is verified. Real data, operational recovery, remaining methods and
the final real-model run are also pending.

Remote verification on Python 3.10: Assay suite 21 passed; task suite 253 passed
with two backend-specific modules skipped in the generic environment; shared
suite 491 passed, one skipped. The shared runner completed as `mock_10`; replay
preserved all 187 artifacts. Task registration, dependency validation, and
runner dry-run passed. No real data or model endpoint was exercised.

## Change 20: reconcile source CN portfolio adjustment

Cross-check against pinned `AssayService` and `PortfolioBacktester` found that
source factor search evaluates CN and US on split-adjusted panels, while the
independent A-share portfolio selects total-return adjustment. The worker had
reused its split factor and price matrices for source CN portfolios despite a
native lineage field saying `total`. It now rebuilds the CN portfolio panel
and factor scores on the native total-return basis; source IC and score exports
stay split. The combination signal applies the same daily finite z-score rule
on the rebuilt panel. US and matched paths retain their established basis.

A split-plus-cash-dividend fixture verifies that CN panel values differ and
that the resulting NAV, actual-index benchmark series and trade log match the
pinned `PortfolioBacktester` using its `DataStore.get_panel` adjustment and
the same full trading inputs. All 21 Assay execution tests passed remotely.

Complexity review: a single optional portfolio basis in the existing panel
loader and one shared combination function removed duplicate z-score code.
The existing portfolio runner, config, worker and data assets remain the only
execution path; no adjustment-specific service or compatibility branch was
added. The test keeps a minimal native store fixture to compare observable
portfolio output, not the adapter's internal arithmetic.

The pinned REST route forwards only `expr/config/as_of`; its Python runner
accepts the custom benchmark and tradability mask that full T3 requires.
Calling that route unchanged would silently drop the real benchmark and CN
trading controls. W06 therefore remains open pending a qualified offline store
and an endpoint contract that carries those inputs. Market-data qualification,
remaining workflows and the final two-round real-model gate remain open.

## Change 21: execute the pinned example search entry

Cross-check against the pinned `benchmark_main`, example YAML and W10 confirms
that this is a separate source entry. It evaluates all 42 Alpha158 factors,
passes kbar+price (13) to each CoT/ToT worker and rolling (29) to EA, and uses
the source's full (`fast=False`) 2023-01-01 through 2024-01-01 search requests.
The YAML enables EA only; selecting CoT/ToT is an explicit method-enable delta.
The original convenience callbacks default to CSI300, so every task callback
is bound to the selected market. Original rounds, batch sizes, worker counts,
generation order and native state are retained. No second search loop was made.

The example source has no validation or held-out test. The task freezes
2024-01-15 through 2024-06-28 for private validation and 2024-07-15 through
2024-12-20 for held-out test, with calendar embargoes after search and
validation. The actual committed chain/tree/population is ranked on complete
finite private RankIC before the test set is frozen. Successful full oracle
requests now require a portfolio response. The Assay factor bridge cannot
produce one, so it uses the already explicit independent portfolio extension.
Incomplete baseline, batch budget or validation pauses without a shortened
source result.

Complexity review: both source profiles share the existing resolver, runtime,
callbacks, Host ledger, receipt store and finalizer. `prepare_searcher` became
`prepare_source` without an alias or compatibility path. The example-specific
entry uses the existing pinned `native_benchmark` adapter and source state;
the portfolio condition is one shared gateway check instead of a test-only
duplicate. No new task schema, search algorithm or fallback score was added.
The prior example refusal and stale documentation were removed.

Remote synthetic verification covers Qlib CoT/ToT/EA and Assay EA through
baseline, search, validation, test, combination and byte-identical replay:
four full example runs passed, with run paths and artifact hashes in
`native_example_workflow.json`. A separate SP500 EA run checked market binding
at every oracle phase. Targeted Host/source/campaign checks passed (58 tests),
including missing portfolio and budget-stop boundaries; the remaining task
suite passed 259 tests with two backend-specific modules skipped and four long
example cases selected separately. Qlib and Assay fixture suites passed six
and 21 tests respectively. The 63 source-resolution artifacts were regenerated
into a new immutable resolver-hash directory, then each hash and resolution
digest was independently checked. This does not
qualify real data or the Assay REST portfolio route, and it does not close
W12/W13, the full market-method matrix or the two real-model acceptance rounds.

## Change 22: introduce sending-time Harness provider authorization

Cross-check against W12 and the existing Python/Pi Harness confirmed that
turn-end `providerCalls` cannot enforce a finite request budget. The shared
wire protocol is now release 0.2.0: each proxied request identifies its
campaign, profile, turn, unique provider request ID and digest before any
upstream connection. The Python client validates the active turn and answers
through a task-neutral callback. The sidecar binds each answer to the pending
run-turn and exact ID/digest; malformed authorization results end the turn.
Denial returns 403 without forwarding. Request intent and authorized markers
survive turn recovery, so a rejected or interrupted request ID is not reused.
The authorized count is separate from rejected attempts in the turn summary.

Complexity review: this extends the one existing proxy, client and JSONL
protocol. It adds no second model gateway, budget ledger, retry path or
AlphaBench-specific code to the shared Harness. Other current tasks retain
their existing accounting and may use their currently built sidecar when they
do not supply the new callback. The next
W12 increment must bind AlphaBench's callback to its single Host writer and
durable `model_requests` budget, then reconcile committed and replayed turns.
Without that task wiring, this protocol change does not close W12.

Remote validation: Python shared suite 495 passed, one skipped; Pi build and
37 tests passed; AlphaBench Host/Campaign tests 21 passed. The tests cover the
control frame, turn identity, old-sidecar rejection, denial,
one remaining authorization across two concurrent proxy sessions, and ID
advance after reconnect. These are fake-sidecar and fake-upstream tests;
the actual guest and full AlphaBench session are still pending.

## Change 23: bind AlphaBench Harness accounting and submission semantics

Cross-check against W12 and the existing Campaign/Host contracts found that
the shared provider-authorization frame alone could not enforce an AlphaBench
budget. The task now reserves all session turns and their proposal attempts in
one Host ledger operation before dispatch. Each provider request is authorized
by that same Host before forwarding, with a durable request identity and digest.
An already authorized ID cannot buy a second send; a ledger write without its
receipt pauses for reconciliation. Committed sidecar usage is checked against
Host authorizations rather than charged again.

The internal expander enforces the two distinct method shapes: direct Harness
submits the effective tail batch in reservoir order; LDM Harness submits K per
fixed session, combines M occurrences into unique candidates, and computes
q0 as count/M. Submission validation reports indexed grammar, historical
duplicate, same-session duplicate and dynamic-check failures. All evaluated
keys, including failed observations, are excluded. Accepted batches have a
Host receipt before they can become proposals; replay verifies the receipt
without paying another check. A repaired candidate keeps its check receipt
even if its list index changes.

Complexity review: this uses the existing CampaignRuntime ledger,
HostDispatcher, OracleGateway, Receipts, FactorDomain and shared ReservoirBuilder.
Turn and attempt reservations share one existing proposal-attempt usage key;
cross-session aggregation makes one pass over occurrences. It adds no second
model transport, selection engine, generic workflow layer or random-fill path.
The task entry still rejects Harness methods until the full W12 tools, real Pi
client, guest isolation and accepted-action path are connected; this internal
module is not a claim of runnable or complete Harness.

Remote verification: 20 focused Host/Harness tests passed, including competing
requests under a one-request budget, atomic strict-barrier reservation, ID
reuse/ledger-only crash, exact direct tail size, cross-session q0, historical
failure rejection and replay without duplicate dynamic checks. The task's
locked Python environment passed 266 tests with four long example cases
deselected. A trial with the repository-root environment failed on its missing
locked `zss` dependency; rerunning with the task environment resolved those
failures. W12, real data qualification and the final real-model acceptance
rounds remain open.

## Change 24: run persistent T3 Harness tools and GP research in the guest

Cross-check against W12: direct Harness now runs one persistent Pi session and
submits the exact effective tail batch in order. LDM Harness runs the frozen
number of independent sessions, collects all K occurrences, computes q0 from
the fixed S x K denominator, and uses the existing FactorSelector for final B.
Both variants share the same Host provider authorization and check receipts.
The Host exposes operator contracts, static validation, paid checks, paged
public search history and a single-observation lookup through one socket-backed
MCP server. Explicit LDM query mode additionally exposes a frozen start-of-round
GP posterior with snapshot identity, fit status, mean, latent std and UCB.
The direct profile has no GP query capability. Submitted batches require a
Host acceptance receipt; accepted actions publish through the existing paired
IR/SFT journal after all sessions reach the strict barrier.

The actual Pi sidecar and KVM guest were built and exercised. The guest test
verified that Host private files and the MCP socket are absent, task resources
cannot be written, and an external request is rejected by network policy.
The actual MCP bridge reached the Host socket with bounded check and query
budgets. DeepSeek Responses with max reasoning completed two synthetic Oracle
rounds for each method: direct used 2 turns and 16 authorized requests; LDM
query mode used 4 turns, 31 authorized requests and 17 GP queries. All six
accepted actions were journaled. Result and manifest hashes are recorded in
`tasks/alphabench/resources/evidence/harness_w12.json`. These runs remain
`mock_verified` and `complete_t3=false`.

Complexity review: the new runtime only binds the existing HarnessClient, Pi
sidecar, Gondolin guest, CampaignRuntime ledger and FactorSelector. The same
Host check position now serves research checks and submission validation,
avoiding a second Oracle charge for one expression. Query budget and the
query-mode protocol field are confined to Harness methods; the initial global
field/budget change was removed after existing native runner identity tests
identified the regression. The public observation projection lives in one
place. No second GP, model transport, session transcript, compatibility path or
speculative policy adapter was added. Stale README text that called Harness
unimplemented was removed.

Remote validation after that correction: AlphaBench task suite 274 passed,
two backend imports skipped in the task environment, four previously verified
long native examples deselected; shared Harness suite 33 passed; Pi TypeScript
build and 37 tests passed under Node 24. The guest isolation and MCP tests ran
against the built sidecar. Real data/backend qualification, no-query LDM
full-flow evidence, compiled policy and the complete method-market matrix are
still open, so W12 and full T3 are not declared complete.

## Change 25: execute independent compiled-policy research and residual GP

Cross-check against W13: the compiled method now starts a second persistent Pi
session with its own profile, read-only skill, artifact root, MCP list and Host
provider authorization. The existing PolicyResearchController accepts the
task's authorization callback; AlphaBench's wrapper reserves a policy turn and
reconciles provider receipts for new, replayed and cached results. Initialization
observations use explicit round `-1`; later feedback aligns the frozen
pre-measurement predictions with measured candidate IDs and engine rounds.

The policy may submit replace, keep or disable under the shared contract. A
replacement is validated and executed by DockerPolicyExecutor with no network,
read-only inputs and resource limits; the Pi profile exposes static draft
validation but not the built-in sidecar draft execution tool. The Host retains
the exact RBF GP, residual fit, UCB and Gumbel selection. Mean inputs contain
public AST features and target scale; proposal counts, q0 and acquisition
summaries are confined to weight_context. The three capability contracts and
zero/nonzero prior behavior have focused tests.

The actual DeepSeek Responses/max two-round synthetic-Oracle run committed a
new policy epoch with `replace` in round 0 and accepted `keep` in round 1.
Both actions were non-degraded; the proposal session committed four factor
actions. Host preauthorized 73 model requests in total. Policy session/profile
hashes, source, actions, and result hashes are in
`tasks/alphabench/resources/evidence/policy_w13.json`. The second round read
one measured prediction-feedback record from the first round. This verifies
the method path, not real-market performance or data qualification.

Complexity review: the compiled selector extends FactorSelector's existing
sampling math and reuses RBFGPSurrogate's residual-prior parameters. There is
one task adapter and no second GP, budget ledger, provider client or policy
execution transport. The sidecar command is shared by proposal and policy
sessions. The unsafe sidecar draft-execution capability is absent from the
T3 policy tool inventory; the formal isolated executor remains the only
execution path. An obsolete test call to the former meter signature was
updated, and the README's stale compiled-policy status was corrected.

Remote verification after updating the obsolete meter test call: the task
suite passed 284 tests, with two optional backend skips and four previously
verified long examples deselected. The shared policy/MCP/executor suite passed
35 tests. Fault-matrix qualification remains separate from this two-round
smoke. `complete_t3=false`.

## Change 26: clarify the provenance of the T3 market-data mirror

Cross-check of the pinned AlphaBench and Qlib instructions confirms that the
AlphaBench download contains precomputed T2/T4 labels, while its T3 evaluator
requires a separately installed market-data backend. Qlib currently says its
official dataset is temporarily disabled and recommends the same community CN
archive already pinned in `tasks/alphabench/resources/data_sources.json`. The
task data guide now states that provenance explicitly. The existing local
acquire, hash-verified transfer and remote offline audit require no new path or
fallback. Complexity review found no code to add or delete for this finding;
the coverage and Assay gaps remain blocked rather than being inferred away.

## Change 27: collect accepted policy actions and audit Harness submissions

Cross-check against W14 and Spec sections 5.6 and 13: accepted compiled-policy
`replace`, `keep`, and `disable` actions now enter the same immutable journal and
paired ldm-2.0 IR/SFT export as factor proposals. The IR projects the frozen
public numerical input, ordered candidate expressions, prior policy source and
accepted action; runtime candidate IDs become positional indices in prediction
feedback. The policy artifact and every frozen input file are checked against
their committed SHA-256 before collection. The action is journaled before
selection/evaluation, and a replay uses the same action ID. Private holdout
files and later outcomes do not enter SFT instructions.

The collection publisher now verifies file names, hashes, action IDs and IR/SFT
row counts before writing `current.json`, including when a generation directory
already exists. Recovery tests cover interruption between IR/SFT writes and
between directory publication and pointer publication. Harness submission
validation records each raw candidate occurrence and format failure with a
stable turn/attempt/submission identity. A repeated submission reuses its
validation receipt; a changed artifact on a restarted turn is a new attempt.
Frozen quality audit includes these records, preserves schema-level static
rejections and reports zero occurrences as unavailable rather than complete.

Complexity review: the existing `Receipts`, `DataCollectionSink`, IR builder and
PolicyResearchController artifacts remain the only persistence/rendering and
policy-snapshot machinery. No second training schema, policy controller or
quality evaluator was introduced. Duplicate check-position computation was
removed, and generation verification uses one helper for staged and existing
directories. The project-wide W14 gaps remain open: complete scientific report,
per-method Search Cost, EA update rate, multi-run denominator, and qualified
market data. This change does not promote `complete_t3`.

Remote verification: the entire task suite passed 294 tests with two optional
backend skips before the final input-set tightening. After the quality changes,
the suite passed 292 tests with two skips and four previously verified full
benchmark examples deselected; after the final input-set check, all 25 focused
collection, Harness, policy and quality tests passed.

## Change 28: project T3 metrics with explicit populations and units

Cross-check against W14 and Spec section 13, including the pinned EA pool
updates: `U_t` counts new generation expressions entering the retained top pool
at each completed round; the best-factor update event is separate. The report
retains the actual round count, planned count and each entered expression.
Search threshold discovery uses only new measured search attempts; the
including-initialization view remains separate. CoE/ToT FracSuccess uses an
explicit registered roster, counts missing/model-failed runs in its denominator,
and excludes infrastructure-invalid runs only under the roster's fixed rule
with a hashed evidence file. Completed cohort reports must agree on all
protocol fields except random seed; draft or mock runs remain unqualified.

Harness Search Cost is recorded as provider calls per committed turn with its
submission-attempt ledger, while direct/native repair cost remains model
attempts per logical step. Mixed initialization and Harness costs retain both
components without an invalid combined count. Current-run model requests sum
the independent initialization and search ledgers; imported seed-creation cost
is labeled separately. Raw occurrence quality audits now retain direct/native
format failures, the original native quality, and the denominator of unique
admission. The result carries stage completeness, EA updates and run costs;
the Markdown report and search-attempt CSV expose the same units and missing
reasons. Report rendering uses a fixed stage order so offline rerender and
completed-run replay are byte-identical.

Complexity review: the new computation remains in the existing reporting and
finalization modules. The small aggregation CLI delegates only the actual
FracSuccess calculation and reads already committed run artifacts; it creates
no second runtime, persistence framework, endpoint, or compatibility branch.
Costs with different units are not forced through a shared total. Existing
oracle receipts, generation records and stage results remain the sources for
the projections. The review found and fixed three concrete defects before this
commit: omitted initialization model requests, nondeterministic Markdown
stage ordering, and a false `search=complete` when the shared engine stops
before its attempt target.

Remote verification: 298 task tests passed, two optional backend tests skipped,
and four long upstream example cases previously verified were deselected.
After the final aggregate identity and report-order changes, 13 focused
campaign/aggregation tests passed. The early-stop correction then passed 44
campaign/native tests with four long example cases deselected. W14 remains open for full offline rebuild
of `result.json` from original artifacts and the rest of the scientific/source
qualification matrix; real market data and Assay portfolio access remain
unqualified. This change does not promote `complete_t3`.

## Change 29: commit report publication before terminal status

Cross-check against W15 R12/R17 and the accepted-action publication boundary:
`result.json`, `report.md`, and `trajectory.csv` now receive a protocol-bound
SHA-256 manifest before the run can finish. A completed resume verifies all
three artifacts, the embedded protocol, and both initialization/search budget
ledgers before reporting replay success. A run that has published this manifest
but has not reached `completed` first replays the idempotent collection export,
then uses the existing `CampaignRuntime.finish` to commit the terminal status.
The ordinary path follows the same order, so collection failure cannot leave a
premature completed status. A completed run without the manifest is rejected;
there is no compatibility branch for earlier incomplete publication contracts.

Complexity review: no new runtime state machine or persistence layer was added.
The artifact manifest uses the existing atomic JSON writer; replay reuses the
existing runtime and collection APIs. Moving `finish` from `finalization.py` to
the workflow makes the actual publication order explicit. The fault tests use
one parameterized subprocess case for the two adjacent process-exit windows,
plus focused corruption and collection-failure cases; no duplicate synthetic
recovery harness was retained.

Remote verification: the task suite passed 303 tests with two optional backend
skips and four previously verified long example cases deselected. After the
budget consistency check, 21 focused campaign/collection/seed-bundle tests
passed. W15's remaining service, native, Harness, cancellation, concurrency and
isolation fault matrix is still open; the manifest is not an offline rebuild of
`result.json` from raw receipts. `complete_t3=false` remains correct.

## Change 30: rebuild final reports from frozen inputs

Cross-check against W14 and Spec section 13: the search execution summary is
accepted before finalization, so a later process can reconstruct the same
selection, test, quality audit, metrics and publication manifest from the
checkpoint, budget ledgers, generation records and completed Oracle receipts.
The offline command uses the original request construction and verifies receipt
identity, request and response digests. A missing required receipt fails closed;
only a missing quality check with exhausted frozen budget reproduces the
original incomplete-coverage result. Existing published files must match the
rebuild byte for byte, while a missing `result.json` can be regenerated from
the original inputs. The source run is never modified and the output must be a
fresh directory outside it.

Complexity review: this remains a task-local CLI around the existing finalizer,
not a second reporting implementation. The only read-only adapter supplies
frozen Oracle responses; no model client, Oracle endpoint, new persistence
layer or compatibility path was added. The review reduced copied inputs to the
JSON files actually read by the finalizer, excluding session traces and other
private files. Quality budget reasons now use a stable code rather than a
runtime exception message, which also makes the offline projection exact.

Remote verification: 307 task tests passed, two optional backend tests skipped,
and four long upstream examples deselected before the final input-copy change.
After that change, eight focused rebuild/quality tests and all six native
CoT/ToT/EA Qlib/Assay full-workflow cases passed. A persistent mock run rebuilt
all four published files byte for byte with the selective input set. This
closes the offline-report-rebuild item only; real market qualification, Assay
portfolio service and the W15/W16 matrices remain open. `complete_t3=false`
remains correct.

## Change 31: recover durable T3 receipts across actual process death

Cross-check against W15 R04, R06 and R09-R11: subprocess tests now exit after
a durable reservation, after search/validation/test Oracle receipts, and after
the quality occurrence set is frozen. Resumed runs match uninterrupted runs in
Oracle dispatch order, budget counters, observation identity, search metrics,
test metrics, quality coverage and stage completeness. Their reports also
rebuild byte for byte from frozen inputs. The report-publication process-exit
cases from Change 29 cover R12.

Two defects were fixed. A shared-engine restart after search completion set
`rounds_run` to the number of rounds in that invocation (zero), conflicting
with the execution summary frozen before test; T3 now projects the cumulative
`next_round` while leaving shared engine semantics intact. A `reserved` receipt
was authorized and reserved again on replay; it now advances to dispatch
intent using the existing reservation, while unknown states and a stored
request whose body fails its digest are refused before physical work.

Complexity review: these are local corrections at the task projection and
receipt state transition, with no new lifecycle layer or compatibility route.
One parameterized real-process test covers the four finalization boundaries;
one process test covers reservation and corrupt-record refusal. The Oracle
transport is deterministic mock data, but each interruption kills the OS
process after a durable write. The task suite
passed 312 tests, with two optional skips and four previously verified long
example cases deselected; after the last receipt integrity check, 137 focused
contract/Host/campaign/lifecycle tests passed. W15 remains open for real
service outage/cancel, Harness/policy process recovery and remaining isolation
evidence, and W16 remains blocked by market data and Assay service assets.
