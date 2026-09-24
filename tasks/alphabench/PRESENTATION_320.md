# CSI300 320-evaluation comparison

This study compares `ldm_harness_compiled` (No GP Query), Pure Harness, and
the pinned AlphaBench CoE (`cot`), ToT, and EA implementations. All model calls
use `deepseek-flash` through Responses API with max reasoning. The four seeds
are 42, 43, 3407, and 10086. Each run has at most 20 native rounds/depths
and 320 new search evaluation attempts. The 42 Alpha158 initialization factors
are evaluated separately and shared by all five methods for the same seed.

LDM uses four persistent candidate sessions, 32 accepted occurrences per
session, a 64-candidate BO pool maintained by empirical-`q0` Gumbel sampling,
and 16 evaluations per round. The policy session sees that same BO pool; the
agent-facing surrogate query tool is disabled. Pure Harness uses one session
and a batch of 16. CoE uses 16 original search chains, ToT retains its four
workers, six candidates per node and `top_k=3`, and EA generates 16 offspring
per generation with an 8/8 mutation/crossover split. Native algorithms keep
their own update and branching cadence. Report actual evaluation attempts,
completed rounds/depth and unused allowance; never infer 320 measurements
from a configured upper bound.

The fixed input is the audited partial CSI300 Qlib archive documented in
[DATA.md](DATA.md). All runs use the same data digest, splits, factor grammar,
filter, search RankIC objective, private validation, held-out test and complete
portfolio finalization. This is a partial-data comparison, not a claim that
the full AlphaBench T3 market matrix is qualified. The official algorithms
are source-faithful with budget-adapted parameters, not published-score
replications.

On the configured server, from the repository root:

```bash
T3_PY=tasks/alphabench/.venv/bin/python
$T3_PY -m tasks.alphabench.presentation_320 prepare
$T3_PY -m tasks.alphabench.presentation_320 start-oracle
$T3_PY -m tasks.alphabench.presentation_320 initialize
$T3_PY -m tasks.alphabench.presentation_320 status
# After all four initialization statuses are completed:
$T3_PY -m tasks.alphabench.presentation_320 launch
$T3_PY -m tasks.alphabench.presentation_320 status
```

The frozen manifest, protocols, process registry, logs, Oracle receipts and
campaign directories are under
`/mnt/data1/Large-Discovery-Models/runs/alphabench-t3/presentation-320-20260924`.
The launcher refuses to replace a frozen protocol or start a registered
process twice. A paused campaign must be reconciled and resumed from its
existing run directory with the same protocol and service; it must not be
started again under a new run ID to conceal already dispatched work.
`start-oracle` can restart a stopped service against the same durable receipt
directory, recording the prior PID in the process registry.

For a paused run, inspect its log and private receipts first. Oracle requests
with completed server receipts reconcile during `resume`. The two dynamic-check
workers that exited on Qlib's scalar-expression `AttributeError` can be
settled as invalid checks from their recorded request, permit and worker
traceback, without dispatching another Qlib worker:

```bash
$T3_PY -m tasks.alphabench.presentation_320 resolve-worker --request-id c8fe9bb6f6b4c28a937b1c0cd1d974be5d4cb20309b162230598bd2262b3e71e
$T3_PY -m tasks.alphabench.presentation_320 resolve-worker --request-id 473c28f8d56e02a46737e87a2498f8a434dd4ace6bd4a8669ea9fb58aacb1f0f
```

After an Oracle POST times out, the client polls that request's durable
receipt every 20 seconds for up to twice the frozen worker timeout. A large
receipt has a 60-second read timeout. On resume, an existing
server receipt is reconciled without another POST. If the server confirms the
request ID is absent (404), the client posts the same ID and payload again;
the service serializes that ID and permits at most one physical worker.
An unresolved request still pauses for inspection.
Search HTTP responses omit the large per-instrument score array, which the
search controllers do not consume. New search receipts store that smaller
response and its link to the complete worker output in the same request's job
directory. Existing full receipts remain valid. Private validation and test
responses still return their scores for finalization.

The EA seed 42 model timeout has no retrievable provider response. After
confirming its process is dead and its single model receipt remains unknown,
authorize one new physical request. The original unknown request stays in the
audit trail and both requests count toward the model budget. The accepted
response is committed to the original logical generation only after the
retry receipt is durable:

```bash
$T3_PY -m tasks.alphabench.presentation_320 resolve-model --method alphabench_ea --seed 42
$T3_PY -m tasks.alphabench.presentation_320 resume --method alphabench_ea --seed 42
```

Use `resume --method METHOD --seed SEED` for any other paused campaign after
its outstanding receipts are reconcilable. The launcher checks that the old
process is dead and that the run protocol matches the frozen study.
A dead process whose status still says `running` can also resume from the same
directory; Oracle preflight timeouts now record a recoverable pause.

The primary figure uses actual newly measured search attempts on the x-axis
and best-so-far search RankIC relative to the shared seed pool on the y-axis.
Private validation and held-out test never feed the search controller. Report
held-out RankIC and portfolio metrics separately, alongside evaluation
coverage, model requests, Oracle jobs, and wall time for every run. Do not
extend a curve beyond its last measured attempt as if additional evaluations
had occurred.
