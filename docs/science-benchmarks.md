# AtomWorld and ReaSyn benchmark adapters

Both tasks use manifest discovery and the shared Campaign lifecycle. Scientific
candidate identities, proposal validation, legal history views, projection and
reporting remain inside their task packages.

| Task | Research and candidate generation | Scientific feedback and selection |
| --- | --- | --- |
| AtomWorld blind baselines | Direct CIF/refinement, packaged geometry operations, persistent parallel Harness research, independent public-audit policy | Input CIF, public instruction and syntax feedback only; no judge feedback. The last scheduled submission determines accuracy. |
| AtomWorld LDM extension | Direct or persistent Harness proposal pools, fixed residual RBF GP/UCB, optional independent compiled policy | Same-question past scalar correctness is available for optimization. Private targets, judge diagnostics and cross-question history stay inaccessible. This is not the official no-feedback protocol; report the last scheduled submission, not the best observed answer. |
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
canonical deduplication. Reconstruction q0 groups canonical queries; independent
restart seeds distinguish scientific evaluation trials, not proposal groups.
TDC groups canonical projected products. The trial logit is
`alpha * log(q0_group + epsilon) - log(group_trial_count) + eta * robust_z(UCB)`.
Each surviving query's mass is divided across its retained trials after separate
BO-pool maintenance; TDC has one trial per product group. Sampling is reproducible
and without replacement. Compiled policies can change both the standardized
prior mean and alpha/eta in both ReaSyn benchmarks. Direct and BO baselines are
separately named and do not claim these LDM sampling semantics.

Projection workers persist each completed target, including empty results. Serial
batch deadlines include the per-target allowance, and resume reuses verified
completed rows and deterministic seeds. Endpoint/projection recovery and proposal
replenishment have separate finite limits; neither increases scientific rounds
or the oracle budget. An incomplete campaign returns a nonzero status.
Completed oracle/judge receipts replay without charging another evaluation;
fresh calls still obey both EngineConfig and durable ledger limits. Ambiguous
interrupted judge calls are surfaced, not silently repeated or scored incorrect.

## Research Harness and pilots

Each task provides digest-bound roles, research Skills, structured history tools,
submission validators and a guest-image recipe. Shared Harness clients own native
session persistence, provider traces, MCP tools and validation/repair. Independent
policy sessions use the shared compiled-policy controller and isolated executor.
AtomWorld implements `ldm`, `ldm_harness` and `ldm_harness_compiled` as an explicit
measured-feedback optimization extension: same-question past scalar correctness,
fixed residual RBF GP/UCB, empirical q0 and Gumbel sampling. Compiled sessions
control both prior_mean@1 and ldm_weights@1. Private targets, judge internals and
cross-question history stay inaccessible. Blind baselines remain available;
their protocol is distinct from feedback-enabled LDM. ReaSyn reconstruction
also enables both capabilities, grouping q0 by query while retaining independent
trial seeds and allocating each query's probability across its surviving trials.

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
maps that earlier review's findings to their implementation and records its test lanes:
60 AtomWorld tests, 71 ReaSyn tests with RDKit, and 508 shared/lightweight tests
with four optional skips. The lanes overlap. Live provider/KVM acceptance remains
pending, as stated in the record.

Developer-specific SSH troubleshooting and installation notes are kept outside
the repository. The frozen historical validation record retains only portable
evidence; it must not be read as a statement about the current workspace.
