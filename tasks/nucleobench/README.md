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
`malinois_k562` and `enformer_muscle_not_liver` are qualified through real
start evaluations and tiny campaigns. Enformer qualification includes two
active Harness-Compiled rounds with physical inference minibatches; see
[`resources/enformer_qualification.json`](resources/enformer_qualification.json).
Fourteen additional published cases have digest-pinned preparation
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
| `ldm_harness` | Configured persistent research Agents; by default four, each submitting `B` validated occurrences. | The same `q0 + GP-UCB + LDM` path as direct LDM. |
| `ldm_harness_compiled` | The same proposal Agents plus one independent policy Agent. | The policy may set the GP prior mean and round-specific `alpha`/`eta`; all other optimization components stay fixed. |
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

1. Build independent proposal lineages. Direct LDM issues four concurrent
   model requests; Harness LDM advances the configured persistent role sessions.
2. Reject malformed patches, non-editable positions, unchanged bases, and
   candidates already present in authoritative measured history. Refill the
   affected lineage or Harness submission until all requested occurrences
   are valid.
3. Preserve accepted equal candidates across lineages and sessions as separate
   same-round occurrences. Within-session repeats are allowed by default;
   `--harness-unique-candidates` rejects them before commit. Accepted frequency defines
   `q0(x) = count(x) / valid_occurrences`.
4. Canonicalize the occurrence reservoir. If more than `bo-pool-size` unique
   candidates remain, retain a `q0`-weighted Gumbel sample up to that limit,
   which defaults to `3B`.
5. Encode each sequence at the official editable positions and fit the
   task-local exact normalized-Hamming GP to previous official measurements.
6. Compute GP-UCB, robustly standardize it, and sample without replacement from
   `pi(x) proportional to q0(x)^alpha * exp(eta * robust_z(UCB(x)))`.
7. Rebuild and evaluate the selected full sequences through the official model.
   `--oracle-batch-size` controls the inference minibatch independently of the
   selection batch. Enformer defaults to four sequences without gradients to
   bound memory use; other families default to the full evaluation batch.
   Every actual model call is recorded, and each candidate is charged as one
   official oracle evaluation. History advances only after the selected batch
   has completed.

Only measured candidates are historical exclusions. A candidate proposed in an
earlier round but not selected for evaluation remains eligible. This preserves
the proposal distribution instead of introducing a private session-level
exclusion rule.

Harness proposal counts and evaluation counts are independent. Repeat
`--harness-profile` once per session and set
`--harness-candidates-per-session` for their minibatch size. Repeating a role
creates separate persistent sessions sharing the same `AGENTS.md`, with distinct
session IDs, histories, and workspaces. `--proposal-samples`
must equal their product. LDM requires
`evaluations-per-round <= bo-pool-size < proposal-samples`.
If repeated occurrences leave fewer unique candidates than the evaluation
batch, evaluate those candidates without adding proposals or duplicate oracle
evaluations. Occurrence frequencies still define `q0`.

`--harness-unique-candidates` applies only within each parallel session's
submission, comparing canonical sequences rather than patch order. It does not
deduplicate across sessions or change the GP/selection path. Historical measured
candidates are still rejected, and rejected entries must be replaced before the
complete minibatch can commit.

## Surrogate and Compiled Policy

The surrogate vector contains categorical DNA codes in the fixed official
editable-mask order. Its kernel is
`exp(-normalized_hamming_distance / length_scale)`; the length scale is chosen
from a fixed grid by marginal likelihood. The GP standardizes observed utility,
uses bounded best-plus-recent history, and applies GP-UCB.

`ldm_harness_compiled` keeps that kernel, variance model, UCB rule, pool,
candidate budget, and evaluator unchanged. A separate persistent
`policy_architect` session receives public mutation-summary features and
measured history, then keeps an active policy, retains the task baseline, or
submits a complete `optimization_policy.py`. A validated file can provide:

- a standardized prior mean, with the unchanged Hamming GP fitted to residuals;
- the current round's non-negative `alpha` and `eta`.

The configured baseline (released defaults `alpha=2.0, eta=0.25`) is the starting
decision rule. The policy adapts from measured evidence, not fixed round ranges;
missing calibration alone is not a reason to remove both selection signals.
Zero prior mean retains the GP posterior and UCB exploration. The `disable`
action restores zero mean and configured weights, not uniform sampling. ESS
and logit contributions describe influence rather than an optimization target.

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

The default four proposal roles focus on target biology, regulatory grammar,
sequence landscape, and evidence criticism. Additional selectable roles cover
off-target selectivity, sequence composition, measured-module recombination,
and alternative regulatory programs. Candidate-generation roles load committed
`AGENTS.md` files and four task-local scientific Skills. They can use the isolated
task guest, web and
Context7 tools, configured MCP servers, and task-local structured tools for
paired-start context, sequence windows, mutable regions, mutation validation,
and measured research history.

The `comprehensive_research` role combines these perspectives in one lead
researcher. Repeat this role for independent complete-panel proposals instead
of dividing the research by specialty. Unlike the direct `harness` method,
these proposals still enter the shared LDM pool and only selected candidates
are measured.

Repeated `comprehensive_research` sessions load the same `AGENTS.md` and maintain
independent research notes. The instructions distinguish supported refinement,
competitive alternative programs, and diagnostic controls. Agents revisit
unresolved hypotheses after feedback and choose their balance from comparative
measurements and remaining time, without fixed role assignments or slot quotas.
Previously proposed but unmeasured candidates remain eligible unchanged.

All candidate roles, including the direct `harness` researcher, expose the same
on-demand Skills: `biopython` for sequence reconstruction and motif analysis,
`experimental-design` for matched controls and factorial contrasts,
`scientific-critical-thinking` for evidence and competing explanations, and
`statsmodels` for identifiable contrasts in measured history, not one-point
initialization or mandatory per-turn fitting. Pi lists their descriptions and
loads the full text when the Agent reads a Skill; the complete library is not
injected every turn. Sources, adaptation scope, and licenses are documented in
[the attribution](resources/harness/skills/ATTRIBUTION.md). The policy architect
continues to load only its task-local `compile-ldm-policy` Skill.

Candidate Agents write `candidates.json` in their workspace with code, then call
`submit_candidates({"artifact_path":"candidates.json"})`. The file contains only
a `candidates` array of the requested number of objects, each containing
`mutations`, `change_summary`, `rationale`, and optional `comparison_candidate_ids`. The two notes are concise English
sentences describing the actual change and its pre-evaluation hypothesis,
expected effect, or control purpose. Pi
snapshots the file; task-local Python validates that exact snapshot's digest,
count, patch legality, measured-history exclusion, and configured uniqueness.
The accepted snapshot also supplies the LDM occurrences, so later workspace
edits cannot change a committed batch.

Annotations are proposal metadata, not part of the mutation payload, canonical
identity, empirical `q0`, or numerical GP inputs. Every source annotation is
retained when independently proposed equal sequences become one reservoir item.
After evaluation, candidate and policy Agents receive a compact index of IDs,
utility, round and mutation count. `get_measured_history` filters by
`candidate_ids` or `round_index`, sorts by recency or utility, and pages with
`offset` and `limit` (default 16, maximum 128). The default `concise` response
returns the index; `response_format="detailed"` returns exact patches and original
annotations. Pages stop before exceeding 32 KB after the first complete record;
`next_offset` identifies the next page. Only measured candidates are shared; unmeasured
proposals remain in their private sessions and immutable submissions.

History queries return a read-only `guest_file` with a path and SHA-256. It
contains all matching detailed records, independently of response pagination.
An unfiltered query exports the complete evaluated set; filtered exports are
not complete exclusion sets. `get_task_context` exports the exact paired start
and editable mask, and `get_sequence_window` exports the exact requested window.
Research scripts load these files directly. Draft candidate files and private
proposal history are not measurement inputs.

Optional `comparison_candidate_ids` identify measured contrasts, not mandatory
parents. Unknown IDs return indexed `unknown_comparison_candidate` errors.
Accepted references stay in research metadata and measured history, not the
oracle payload or GP features. The policy's `weight_context.proposal_sampling`
records the actual session count, panel size and repeat rules.

`get_sequence_window` returns a zero-based half-open window of the original
start, or of a measured sequence when `candidate_id` is supplied. It returns
`bases`, `bases_sha256`, and editable positions. Agents can verify exact parent
bases in code before designing variants without transcribing mutation lists.
The checksum covers the returned window; long-sequence cases can use bounded
windows instead of loading a complete sequence into the conversation.

`harness/measured_history/observations.json` is a read-only tool view projected
from Campaign observations, not a second optimization history. It contains
candidate IDs, measurement rounds, patches, utility, and source-attributed
`research_annotations`. This view is refreshed before research and at campaign
completion. `validate_mutations` reports exact historical membership as
`already_evaluated`; final submission independently checks authoritative
observations. No full exclusion list is inserted into model context. Treat annotations as
hypotheses to test, not verified explanations or additional observations; keep
later interpretations in research notes without overwriting submitted reasons.

A rejected submission returns JSON-Pointer paths inside the file, error codes,
specific reasons, and repair hints to the same session. The Agent edits the
reported entries and resubmits the file path, without retransmitting the full
array through the model. `validate_mutations` remains available for inspecting
individual patches; complete-file validation is mandatory on submission.
Repairs must update the two notes to describe the final patch. Missing or blank
notes return an `invalid_annotation` error at the exact field path.
Network tools have configurable per-turn budgets; unlisted tools are unlimited.

Harness sessions receive released paired-start context and campaign
measurements, not model weights, hidden benchmark assets, or unqueried scores.
The task guest is content-addressed from
`resources/harness/image/guest-image.json` and must be built before a real
Harness campaign. It includes Biopython, pyfaidx, NumPy, SciPy, pandas,
scikit-learn, statsmodels/patsy, pyDOE3, Matplotlib, Seaborn, Logomaker, and the
SeqKit, BEDTools, SAMtools, and MAFFT command-line tools, plus ViennaRNA Python bindings. The
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

An official campaign starts from one published paired sequence. The 100 starts
are separate campaigns, not an initial population within one campaign.
`--start-index` selects the paired start; `--campaign-index` sets optimization
randomness. For Malinois, local start indices 0 through 99 correspond to rows
1300 through 1399 in the published start table.

The official runner checks wall time between optimization rounds. A final
round can finish after the limit. Wall-time campaigns record a task clock
immediately before entering that unchanged runner, excluding setup and including
all active proposal, policy, GP, and oracle work. `trajectory.csv` preserves
runtime `elapsed_seconds` and adds `benchmark_elapsed_seconds` from completed
oracle calls. Use `result.json.wall_time_result` for strictly in-budget results;
`best_found_utility` and official outputs retain the full result, including any
late measurements. Proposal and policy sessions receive remaining benchmark
time, and the policy also receives the effective evaluation batch size.
Use `args.resume-from=<campaign>` to resume a stopped wall-time run. The task
deducts elapsed runtime up to the previous failure or stop and preserves all
checkpointed measurements. Repair downtime is excluded. Results are labelled
`cumulative_runtime`, with `benchmark_comparable=false`, rather than presented
as uninterrupted official runs. A resume never grants another eight hours.
Runtime and resource identity changes require an explicitly recorded recovery
copy, not silently reusing pinned sessions. Record actual hardware rather than claiming identical paper hardware
from a configuration label alone.

Within a live wall-time campaign, a proposal or policy session timeout or transient provider
failure resumes the unfinished turn while benchmark time remains. Other profiles'
committed submissions are reused; the session, workspace, tool quotas, and turn
identity are preserved. No additional recovery starts once the official time
window expires, and recovery does not restart the benchmark clock. Selection
still waits for the complete configured occurrence count. Fatal provider or
protocol errors stop execution rather than silently dropping a profile.

Round-limited runs also continue transient proposal-session failures, with a
recovery-start window equal to `--harness-wall-time-seconds`. Each attempt keeps
the configured session deadline. Accepted submissions are reused, and recovery
does not request extra proposal occurrences or oracle evaluations.

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
