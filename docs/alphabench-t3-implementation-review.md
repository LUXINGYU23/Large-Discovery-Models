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
