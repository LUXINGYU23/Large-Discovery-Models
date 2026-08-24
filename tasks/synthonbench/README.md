# SynthonBench LDM Task

This task applies the repository's shared `LDMEngine` to the official
[SynthonBench](https://github.com/mireklzicar/synthonbench) budgeted-search
interface. It does not run SynthonBench reference optimizers. Instead, it
adapts LDM to propose valid `(reaction_id, synthon_ids)` tuples, evaluate them
through the official fixed oracle, and score the resulting submission with the
official source-pinned scorer.

## Task Boundary

```text
ldm_task/  stable shared-runner adapter and dependency hook
core/      tuple validation, public slates, prompts, task-local Tanimoto GP, LDM policy, oracle adapter
scripts/   source-pinned official data preparation
resources/ immutable upstream and qualification contracts
tests/     task-local contract and algorithm tests
runs/      generated artifacts only; never committed
```

The adapter is self-contained under `tasks/synthonbench` and uses the shared
LDM interfaces without importing another task.

## Pinned Upstream Contract

| Artifact | Pinned revision |
| --- | --- |
| Official code | `mireklzicar/synthonbench` `4e89d72a19ebc5f9e59513bb57771ea8e8db4336` |
| Package API | `synthonbench==0.2.0` |
| Released dataset | `mireklzicar/synthonbench` `19f7bb669032f60b306139318d3f71a26f60134a` |
| Official audit | upstream `scripts/score_submission.py`, executed unchanged |

The code is MIT licensed and the dataset card declares CC-BY-4.0. See
[`resources/upstream_contract.json`](resources/upstream_contract.json) for the
task's machine-readable provenance.

## LDM Method

Each outer round uses the following task-local realization of the shared LDM
lifecycle:

1. Draw `M=64` independent reaction-conditioned public slates. Reactions are
   allocated in proportion to their official product-space sizes by default.
2. Issue one independent OpenAI-compatible request per slate. Each request may
   return exactly one tuple selected from its listed source-valid IDs.
3. Parse and validate every response against both its slate and the full
   official `SynthonSpace`. There are no replacement or refill requests.
4. Estimate the empirical proposal measure from valid unseen occurrences:
   `q0(x) = count(x) / valid_occurrences`.
5. If more than `K=32` unique candidates survive, retain a `q0`-weighted
   Gumbel sample of size `K`. Only this maintained BO pool is scored.
6. Build a task-local product proxy by summing raw-connector Count-Morgan
   fingerprints for the ordered synthons. A fixed, reaction-balanced set of
   public tuple landmarks defines a Nyström/FITC count-Tanimoto GP, which
   produces a GP-UCB value from prior oracle observations only.
7. Sample without replacement using the tilted policy
   `pi(x) proportional to q0(x)^alpha * exp(eta * robust_z(UCB(x)))` and
   Gumbel-top-k.
8. Send each selected unseen tuple to official `GlobalSynthonTask.score()`.
   One selected tuple is one charged official oracle call.

The public slate is necessary because full reaction slot catalogs contain
thousands of synthons and cannot be faithfully serialized into one LLM prompt.
It contains only released IDs and SMILES, never score-table values, top-k lists,
or unqueried oracle outcomes. Previous charged outcomes are the only feedback
shown in a prompt.

All reaction IDs, slot positions, and synthon IDs are canonically ordered before
any seeded sampling. A fixed campaign seed therefore defines the same shared
initial design and task-local GP basis across independently launched methods.
The shared design is evaluated in its generated order before any method-specific
acquisition step.

## Surrogate Representation

For a valid tuple \(x=(r,s_1,\ldots,s_k)\), the task computes
\(c(x)=\sum_i \operatorname{CountMorgan}(s_i)\) from released synthon SMILES.
The connector atoms in those SMILES are retained. The default kernel is the
count-Tanimoto similarity

\[
k_0(x,x') = T_{\min/\max}(c(x), c(x')).
\]

`--gp-reaction-weight` adds a reaction-template delta term,
\((k_0+\lambda_r\mathbf{1}[r=r'])/(1+\lambda_r)\); its committed value is
`1.0`. At startup, `--gp-landmarks` fixed public tuples are sampled with a
reaction-balanced seed. With `K_{ZZ}=LL^T`, each candidate is encoded as
`[L^-1 k(Z, x), 1 - k(x, Z) K_ZZ^-1 k(Z, x)]`. The final coordinate is the
FITC diagonal residual. The online posterior standardizes observed utilities,
treats the residual as candidate-specific noise, and stores only the resulting
fixed-size vector in the shared LDM checkpoint path.

The default representation uses 2,048 Count-Morgan bins, 256 landmarks, a
`1e-8` kernel jitter, and a unit signal, mean, and observation-noise scale.
These are declared in every committed profile and can be changed explicitly
for a separate experiment.

## Tracks

| Track | Oracle | Purpose |
| --- | --- | --- |
| Official Example | Official bundled example space + `PairwiseSynthonOracle` | Deterministic LDM lifecycle validation |
| Surrogate Oracle | Released surrogate tables at 1M, 10M, or 100M | Scalable benchmark track |
| Glide Ligand-Efficiency | Released 1M Glide ligand-efficiency table | Released real-oracle track |

The Surrogate Oracle and Glide Ligand-Efficiency tracks report exact official top-k recall and DCRF by running the pinned
upstream `score_submission.py` against the released table. The primary official
budget is 10,000 unique oracle calls. The committed full profiles use batches
of 16 selections per posterior update (`625 * 16 = 10,000`); this is an
explicit batch-BO policy. Set `--evaluations-per-round=1` for strictly
sequential feedback, with the corresponding increase in LLM requests.

## Installation and Official Example

Run commands from the repository root:

```bash
uv sync --locked --project tasks/synthonbench
uv run --locked --project tasks/synthonbench \
  python -m pytest tasks/synthonbench/tests
uv run --locked --project tasks/synthonbench \
  python scripts/run_ldm_tts.py config/synthonbench/mock.yaml
```

The Official Example track needs the pinned package and RDKit but no external model, downloaded score
table, GPU, or endpoint.

## Prepare Official Data

Keep data, official source, and runs outside the cloned repository:

```bash
export SYNTHONBENCH_WORK_ROOT=/path/to/synthonbench-workdir
export SYNTHONBENCH_DATA_ROOT="$SYNTHONBENCH_WORK_ROOT/data"
export SYNTHONBENCH_SOURCE_ROOT="$SYNTHONBENCH_WORK_ROOT/source"
export SYNTHONBENCH_RUNS_ROOT="$SYNTHONBENCH_WORK_ROOT/runs"

uv run --locked --project tasks/synthonbench \
  python -m tasks.synthonbench.scripts.prepare_official_data \
  --data-dir "$SYNTHONBENCH_DATA_ROOT" \
  --source-dir "$SYNTHONBENCH_SOURCE_ROOT" \
  --scale 1M
```

The preparation command checks out the official source commit, downloads the
fixed Hugging Face revision through the official endpoint, and writes a local
manifest with artifact digests. If a user's network requires a different
Hugging Face endpoint, pass `--hf-endpoint <URL>` explicitly.
Use `--scale all` to prepare the 1M/10M/100M surrogate tables. It never writes data
inside `tasks/synthonbench`.

## Model Endpoint

Real campaigns use the same generic OpenAI-compatible environment contract as
other LDM tasks:

```bash
export LLM_BASE_URL=https://your-provider.example/v1
export LLM_MODEL_NAME=your-model
export LLM_API_KEY=your-secret
```

`TTS_LLM_URL` / `TTS_LLM_MODEL` / `TTS_LLM_API_KEY`,
`LDM_LLM_URL` / `LDM_LLM_MODEL` / `LDM_LLM_API_KEY`, and `OPENAI_API_KEY` are
also accepted. Config files intentionally leave endpoint fields unset. Use
`--llm-json-mode` only when the selected provider supports the OpenAI JSON
response-format field. Provider-specific OpenAI-compatible request fields can
be passed as JSON through `--llm-extra-body-json`. The task default is
`{"thinking":{"type":"disabled"}}`, because each request must emit one short
JSON decision. Providers that do not recognize this extension can override it
with `--llm-extra-body-json '{}'` or their own request body.

## Real Runs

After preparation and endpoint configuration:

```bash
uv run --locked --project tasks/synthonbench \
  python scripts/check_task_dependencies.py \
  config/synthonbench/surrogate_1m_qualification.yaml --no-optional

uv run --locked --project tasks/synthonbench \
  python scripts/run_ldm_tts.py \
  config/synthonbench/surrogate_1m_qualification.yaml
```

Use `glide_1m_qualification.yaml` for one Glide ligand-efficiency query. The
full 10,000-call profiles are `surrogate_1m_budget10000_batch16.yaml` and
`glide_1m_budget10000_batch16.yaml`. Change target or surrogate scale through
config overrides; the task validates that the Glide oracle is only used at 1M.

Each completed run writes:

- `submission.csv`: exactly the unique official product IDs that were charged.
- `trajectory.csv`: official oracle trace in call order.
- `official_audit.json`: unmodified output from the pinned official scorer for
  real tracks.
- `result.json`: LDM summary, best utility, and official metrics.
- shared `status.json`, `budget.json`, events, and checkpoint artifacts.

## Fixed-Budget Quick Comparison

`config/quick_compare/synthonbench.yaml` runs a three-seed comparison on the
official 1M KIF11 surrogate track. One product-uniform shared initialization
batch and five optimization batches make six batches of 16 official calls
(96 calls per campaign).

`config/quick_compare/synthonbench_extended.yaml` preserves the same method,
seed, and candidate-budget settings but uses eleven optimization batches
(192 official calls per campaign). It is a separate confirmation profile for
posterior-convergence checks, not a replacement for the fixed six-batch screen.

LDM retains the current 64 independent public-slate requests, empirical `q0`,
32-candidate maintained pool, and task-local reaction-aware Nyström
count-Tanimoto GP over standardized utilities.
Pure BO uses the same GP-UCB but receives a fresh score-blind pool of 64 unseen
official tuples per batch and makes no model requests. Direct LLM sampling
issues 16 independent one-tuple requests per optimization batch and evaluates
the admitted tuples directly. Its `direct_v1` prompt is recorded under the
separate direct-LLM contract profile and does not invoke a GP selector. Each
direct request has one deterministic, source-valid anchor synthon so the 16
independent responses map to 16 distinct official product tuples.

```bash
uv run --locked --project tasks/synthonbench python \
  scripts/run_quick_compare.py config/quick_compare/synthonbench.yaml --dry-run

uv run --locked --project tasks/synthonbench python \
  scripts/run_quick_compare.py config/quick_compare/synthonbench.yaml

uv run --locked --project tasks/synthonbench python \
  scripts/run_quick_compare.py config/quick_compare/synthonbench_extended.yaml
```

The result directory contains standard child artifacts plus a portable matrix
manifest, round-level trajectories, CSV/JSON summaries, and a best-so-far
plot. Endpoint settings remain user-defined OpenAI-compatible environment
variables; the BO children do not require them.
For a source archive rather than a Git checkout, set
`LDM_QUICK_COMPARE_COMMIT` to the archive release commit so the manifest records
explicit provenance.

## Qualification Status

The current Nyström/FITC method is qualified through the source-pinned
Official Example campaign and one real LDM-selected query for each released 1M
oracle track. Full 10,000-call campaigns remain separate performance
experiments. The verification boundary is recorded in
[`resources/qualification_evidence.json`](resources/qualification_evidence.json)
and [`resources/verification_record.json`](resources/verification_record.json).
