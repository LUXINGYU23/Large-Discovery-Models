# NucleoBench LDM Task

This task adapts the shared LDM engine to the official
[NucleoBench](https://github.com/move37-labs/nucleobench) sequence-design
interface. It does not run the benchmark's reference optimizers. LDM proposes
start-relative nucleotide mutation patches, evaluates the rebuilt sequences
through the source-pinned official model wrapper, and returns sequences through
the unchanged official runner.

The upstream source is pinned to commit
`a6d8b040a4fa80b18266ee5416c904d01f842428`
(`nucleobench==2.0.10`).

## Task Boundary

```text
ldm_task/          stable shared-runner adapter
core/task_spec.py  declarative candidate, proposal, surrogate, and method contract
core/factory.py    task-local proposal and selector assembly
core/workflow.py   official runner, provider, Harness, budget, and resume orchestration
core/              mutation validation, prompts, Hamming GP, policy, oracle, reports
scripts/           source-pinned external-data preparation
resources/         case catalog, provenance, Harness roles, tools, guest, and policy skill
tests/             task-local scientific and integration tests
runs/              generated artifacts only; never committed
```

The task does not import another task. Shared orchestration remains in
`ldm_tts`; NucleoBench-specific identity, prompts, features, surrogate,
Harness validation, and official adapters remain under this directory.

## Cases and Qualification

| Family | Cases | Sequence length | Editable positions | Official time limit |
| --- | ---: | ---: | ---: | ---: |
| Malinois | 3 | 200 | 200 | 8 hours |
| BPNet-lite | 12 | 3,000 | 3,000 | 8 hours |
| RiNALMo MRL | 1 | 100 | 100 | 8 hours |
| Enformer | 1 | 196,608 | 256 | 12 hours |

The 17-case source contract is stored in
[`resources/cases/catalog.json`](resources/cases/catalog.json).
`malinois_k562` is qualified through a real start evaluation and tiny
campaign. Fifteen additional published cases have digest-pinned preparation
contracts. RiNALMo remains planned because the official 100-sequence paired
start set is not present in the published benchmark artifacts.

`prepared` means that source-pinned inputs and loaders are defined.
`qualified` additionally requires recorded real-oracle and tiny-campaign
evidence. A prepared case must pass the same gates before Pilot Evaluation or
official execution.

## Search Methods

| Method | Candidate generation | Selection before official evaluation |
| --- | --- | --- |
| `ldm` | Four independent model requests, each returning one `B`-candidate minibatch. | Empirical `q0`, task-local Hamming GP-UCB, and LDM acquisition tilt. |
| `ldm_harness` | Four persistent research Agents, each submitting `B` validated candidates. | The same `q0 + GP-UCB + LDM` path as direct LDM. |
| `ldm_harness_compiled` | The same four proposal Agents plus one independent policy Agent. | The policy may set the GP prior mean and round-specific `alpha`/`eta`; all other optimization components stay fixed. |
| `bo` | Task-local score-blind mutation search. | Hamming GP-UCB without a model proposal distribution. |
| `llm` | `B` independent one-candidate model requests. | No surrogate; evaluate the accepted candidates directly. |
| `harness` | One persistent research Agent submitting exactly `B` candidates. | No surrogate; evaluate the accepted minibatch in submission order. |

For the committed Pilot Evaluation, `B=16`. LDM proposal methods therefore
collect 64 valid occurrences per active round and maintain a BO pool of up to
48 unique candidates. Fixed LDM methods use `alpha=2` and `eta=0.25`;
Harness-Compiled LDM starts from those values and may adapt them from current
proposal, surrogate, and measurement evidence.

## LDM Semantics

Each active LDM round follows one task-local implementation of the shared
algorithm:

1. Build four independent proposal lineages. Direct LDM issues four concurrent
   model requests; Harness LDM advances four persistent role sessions.
2. Reject malformed patches, non-editable positions, unchanged bases, and
   candidates already present in authoritative measured history. Refill the
   affected lineage or Harness submission until all 64 required occurrences
   are valid.
3. Preserve deliberate equal candidates proposed within or across lineages and
   sessions as separate same-round occurrences. Their frequency defines
   `q0(x) = count(x) / valid_occurrences`.
4. Canonicalize the occurrence reservoir. If more than `3B` unique candidates
   remain, retain a `q0`-weighted Gumbel sample of size `3B`.
5. Encode each sequence at the official editable positions and fit the
   task-local exact normalized-Hamming GP to previous official measurements.
6. Compute GP-UCB, robustly standardize it, and sample without replacement from
   `pi(x) proportional to q0(x)^alpha * exp(eta * robust_z(UCB(x)))`.
7. Rebuild the selected full sequences and evaluate the complete minibatch in
   one official model call. Each candidate is still charged as one official
   oracle evaluation.

Only measured candidates are historical exclusions. A candidate proposed in an
earlier round but not selected for evaluation remains eligible. This preserves
the proposal distribution instead of introducing a private session-level
exclusion rule.

## Surrogate and Compiled Policy

The surrogate vector contains categorical DNA codes in the fixed official
editable-mask order. Its kernel is
`exp(-normalized_hamming_distance / length_scale)`; the length scale is chosen
from a fixed grid by marginal likelihood. The GP standardizes observed utility,
uses bounded best-plus-recent history, and applies GP-UCB.

`ldm_harness_compiled` keeps that kernel, variance model, UCB rule, pool,
candidate budget, and evaluator unchanged. A separate persistent
`policy_architect` session receives public mutation-summary features and
measured history, then submits a complete `optimization_policy.py`. The
validated policy can provide:

- a standardized prior mean, with the unchanged Hamming GP fitted to residuals;
- the current round's non-negative `alpha` and `eta`.

History utilities remain raw task values; the adapter supplies their current
location and scale, and the policy returns standardized conditional
expectations. Prior features are aggregate patch summaries: edit burden,
position and adjacency statistics, GC/transition and target-base composition,
and eight editable-region bins. They do not identify an exact sequence, motif,
or edited position. The task-local policy instructions describe the resulting
compositional collinearity and require regularized, coarse effects rather than
trying to reproduce the full Hamming representation. Draft evaluation compares
chronological measured-history holdouts with training-prefix GP hyperparameters
frozen. Read-only numeric inputs, exact sampling logits, and subsequent errors
of frozen predictions support policy research; historical holdouts are development
diagnostics rather than an untouched test set.
The task owns these calculations in `core/policy_diagnostics.py` and the
digest-pinned Pi hook `resources/harness/policy_diagnostics.py`.
Feedback preserves the measured point's original pool size, q0 ranks and
relative mass, acquisition ranks, first-draw probability, and actual alpha/eta.
Selected-point residuals do not validate whole-pool ranking or weight-policy
reward. Proposal multiplicity expresses preference, not evidence from being
selected or left unmeasured.

The policy role and its task-local Skill live under
`resources/harness/`. Accepted policy epochs and validation records are
stored under `<run_dir>/policy_harness/`.
The sidecar digest-verifies the selected profile and Skill, then exposes their
per-session snapshot read-only under `/workspace/.ldm-resources`; the task tree
remains the versioned source.

## Persistent Research Harness

The four proposal roles focus on target biology, regulatory grammar, sequence
landscape, and evidence criticism. Candidate-generation roles load committed
`AGENTS.md` files but no Skills. They can use the isolated task guest, web and
Context7 tools, configured MCP servers, and task-local structured tools for
paired-start context, sequence windows, mutable regions, and mutation
validation.

Every terminal submission has a strict dynamic JSON Schema. Task-local Python
then performs authoritative scientific validation. A rejected submission is
returned to the same session with JSON-Pointer paths, stable error codes,
specific reasons, and repair hints; the turn is committed only after its full
minibatch is valid. Network tools have configurable per-turn budgets, while
unlisted tools are unlimited.

Harness sessions receive released paired-start context and campaign
measurements, not model weights, hidden benchmark assets, or unqueried scores.
The task guest is content-addressed from
`resources/harness/image/guest-image.json` and must be built before a real
Harness campaign. It includes Biopython, pyfaidx, NumPy, SciPy, pandas,
scikit-learn, Matplotlib, Seaborn, Logomaker, and the SeqKit, BEDTools,
SAMtools, and MAFFT command-line tools, plus ViennaRNA Python bindings. The
official evaluator package, model weights, and benchmark data remain outside
the research guest.

See [the quick start](QUICKSTART.md) for commands and
[the shared Harness guide](../../docs/research-harness.md) for MCP, budgets,
traces, and sidecar contracts.

## Official Runner and Profiles

The source-pinned `docker_entrypoint.run_loop` remains the outer driver.
`NucleoBenchDesigner` exposes the shared `LDMEngine` through the official
`SequenceOptimizer` surface. The first paired start is evaluated once, active
rounds are checkpointed by the shared runtime, and the designer returns the
best unique sequences requested by the official runner.

The task defines two release profiles:

- `pilot_evaluation`: one shared initialization round plus eleven active
  optimization rounds, three seeds, and six methods.
- `official_benchmark`: the unchanged official wall-time limit, with no
  task-level round cap.

Pilot Evaluation is a development comparison, not an official benchmark
claim.

## Provider and External Data

Provider settings are user-defined:

```bash
export LLM_BASE_URL=https://your-provider.example/v1
export LLM_MODEL_NAME=your-model
export LLM_API_KEY=your-secret
```

`OPENAI_API_KEY` is accepted as a fallback, and `--api-key-file` may point
to an ignored protected file. Direct proposal methods support Chat Completions
and Responses through `--llm-wire-api`; the committed pilot uses Responses.
Harness methods use the Pi Responses sidecar. Endpoint fields and credentials
are never fixed in tracked configs.

Official source checkouts, model weights, paired starts, prepared manifests,
caches, traces, and runs must remain outside Git. On Linux, use an external work
root for all of them. Malinois uses the official CPU path; Enformer uses CUDA
when available.

## Outputs

Every campaign writes the shared `campaign.json`, `budget.json`,
`status.json`, `events.jsonl`, `checkpoint.json`, `summary.json`,
`result.json`, and `trajectory.csv` artifacts. Official runner output is
kept under `official/`; proposal Harness traces under `harness/`; compiled
policy traces and immutable policy epochs under `policy_harness/`.

Mock runs exercise integration without a model endpoint, official weights, or
benchmark claims.
