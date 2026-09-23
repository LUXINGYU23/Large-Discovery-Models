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

The primary figure uses actual newly measured search attempts on the x-axis
and best-so-far search RankIC relative to the shared seed pool on the y-axis.
Private validation and held-out test never feed the search controller. Report
held-out RankIC and portfolio metrics separately, alongside evaluation
coverage, model requests, Oracle jobs, and wall time for every run. Do not
extend a curve beyond its last measured attempt as if additional evaluations
had occurred.
