# AtomWorld and ReaSyn benchmark adapters

Both tasks use manifest discovery and the shared Campaign lifecycle. Scientific
candidate identities, proposal validation, legal history views, projection and
reporting remain inside their task packages.

| Task | Research and candidate generation | Scientific feedback and selection |
| --- | --- | --- |
| AtomWorld | Direct CIF/refinement, packaged geometry operations, persistent parallel Harness research, independent public-audit policy | Input CIF, public instruction and syntax feedback only. Target CIF and hidden judge metrics never enter research. The last scheduled submission determines accuracy. |
| ReaSyn reconstruction | Direct or persistent Harness projection queries, exact molecular GP with empirical proposal frequency and optional independent compiled policy | Query plus restart seed identifies a projection trial. Similarity is measured against the original reconstruction target. Failures remain in the requested target denominator. |
| ReaSyn TDC | Molecular research, frozen ReaSyn projection, canonical-product admission, empirical-q0/GP-UCB sampling | Only selected unique products consume oracle calls. Previously measured products trigger bounded replenishment. Full-budget AUC remains unavailable for partial campaigns. |

Read the [AtomWorld guide](../tasks/atomworld/README.md) and
[ReaSyn guide](../tasks/reasyn/README.md) for isolated installation, exact flags,
external assets, and real-run preflight. Geometry operations ship in this
repository; no developer sibling directory is required for task execution.
Official source archives, model weights and datasets remain explicit external
inputs with source/asset digest checks.

## Sampling and recovery

ReaSyn requests independent minibatches and retains legal occurrences before
canonical deduplication. Reconstruction probability identities preserve independent
restart seeds; TDC identities refer to canonical projected products. Its LDM rule
is `alpha * log(q0 + epsilon) + eta * robust_z(UCB)`, with a separately sized BO
pool and reproducible sampling without replacement. Direct and BO baselines are
separately named and do not claim these semantics.

Projection workers persist each completed target, including empty results. Serial
batch deadlines include the per-target allowance, and resume reuses verified
completed rows and deterministic seeds. Endpoint/projection recovery and proposal
replenishment have separate finite limits; neither increases scientific rounds
or the oracle budget. An incomplete campaign returns a nonzero status.

## Research Harness and pilots

Each task provides digest-bound roles, research Skills, structured history tools,
submission validators and a guest-image recipe. Shared Harness clients own native
session persistence, provider traces, MCP tools and validation/repair. Independent
policy sessions use the shared compiled-policy controller and isolated executor.
The AtomWorld policy sees public draft features only; it is a blind audit policy,
not a GP/LDM method using hidden correctness.

Pilot matrices are in `config/pilot_evaluation/atomworld.yaml` and
`config/pilot_evaluation/reasyn.yaml`. AtomWorld declares the task-neutral
`final_submission` protocol, which permits repeated answers and reports the last
scheduled score. ReaSyn uses matched initialization and best-so-far optimization
reporting. Small synthetic pilots test infrastructure, not scientific performance.

## Evidence status

Both experiment contracts remain **draft**. Historical local records are dated
snapshots, not evidence that the current code passed live qualification. New
regression tests cover sampling distributions, endpoint recovery, interrupted
projection, optional-dependency boundaries, packaged geometry, hidden-answer
isolation and persistent Harness protocol behavior. A real Harness acceptance run
requires a working model endpoint, Linux/KVM runtime, native/provider trace review,
and the task's official evaluator assets. Do not upgrade qualification or report
paper scores from mock/fake-sidecar tests.

The [2026-09-08 review verification record](validation/science_benchmarks_review_20260908.json)
maps all seven findings to their implementation and records the final test lanes:
60 AtomWorld tests, 71 ReaSyn tests with RDKit, and 508 shared/lightweight tests
with four optional skips. The lanes overlap. Live provider/KVM acceptance remains
pending, as stated in the record.

Developer-specific SSH troubleshooting and installation notes are kept outside
the repository. The frozen historical validation record retains only portable
evidence; it must not be read as a statement about the current workspace.
