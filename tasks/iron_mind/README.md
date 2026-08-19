# Iron Mind Reaction-Condition Optimization

This task evaluates the repository's LDM method on source-pinned Iron Mind
reaction-condition tables. It does not run the upstream Iron Mind optimizer.

Each external round follows the Iron Mind comparison protocol: exactly one
reaction condition is evaluated. The released LDM profile first asks the model
for a four-candidate internal reservoir, then uses a factor-aware categorical
GP-UCB selector to choose the one condition sent to the frozen oracle.

## Architecture

```text
shared runner
  -> tasks.iron_mind.ldm_task.procedure
  -> tasks.iron_mind.core.workflow
  -> ldm_tts.engine.LDMEngine
  -> Iron Mind domain, proposal, GP-selector, and oracle adapters
```

`ldm_task/` is only the stable shared-runner seam. The campaign lifecycle,
contracts, budgets, checkpoints, data collection, and acquisition interface
come from the outer `ldm_tts` core. The task-local `core/` package contains
only reaction-domain behavior: source-pinned schemas and tables, candidate
admission, proposal parsing, the factor-aware GP surrogate, and the frozen
oracle evaluator.

```text
ldm_task/   shared-runner adapter
core/       reaction-domain implementation
resources/  versioned fixtures and upstream provenance
scripts/    data preparation and result aggregation
tests/      task-local tests
```

## Campaign Protocol

The public Iron Mind paper reports a batch size of one and a budget of 20
experiments per campaign. This task preserves those externally evaluated
quantities:

- One frozen-oracle evaluation per round.
- Twenty rounds for every `ldm_20_<dataset>.yaml` campaign.
- Twenty independent campaigns per dataset in the suite configurations.

The four-item reservoir is an LDM implementation setting, not an Iron Mind
benchmark restriction. It is fixed in the released profiles so that published
LDM runs are directly reproducible. A future experiment may use a larger
internal reservoir while retaining one external evaluation per round, but it
must report that reservoir size as a separate method setting.

## Quick Start

Run these commands from the repository root:

```bash
uv sync --locked --project tasks/iron_mind
uv run --locked --project tasks/iron_mind \
  python -m pytest tasks/iron_mind/tests
uv run --locked --project tasks/iron_mind \
  python scripts/run_ldm_tts.py config/iron_mind/mock.yaml
```

The mock campaign is self-contained. It does not require an external model,
GPU, or the official data. See [QUICKSTART.md](QUICKSTART.md) for the full
first-run procedure.

## Model Endpoint Configuration

Real campaigns use an OpenAI-compatible Chat Completions endpoint. The
committed YAML files intentionally do not bind the task to a provider. Set the
endpoint, model, and credentials in your environment:

```bash
export LLM_BASE_URL=https://your-model-host.example/v1
export LLM_MODEL_NAME=your-served-model
export LLM_API_KEY=your-api-key
```

Command-line values (`--llm-url`, `--llm-model-name`, and `--api-key`)
override the environment. `LLM_API_KEY` may be omitted for a local endpoint
that does not require authentication. The provider settings are recorded
without persisting the API key.

## Prepare the Official Data

Keep upstream checkouts, generated data, and run outputs outside the source
repository:

```bash
export IRON_MIND_WORK_ROOT=/absolute/path/to/iron-mind-work
export IRON_MIND_DATA_ROOT="$IRON_MIND_WORK_ROOT/data/official-complete"
export IRON_MIND_RUNS_ROOT="$IRON_MIND_WORK_ROOT/runs"
mkdir -p "$IRON_MIND_WORK_ROOT/sources" "$IRON_MIND_RUNS_ROOT"

git clone https://github.com/gomesgroup/iron-mind-public \
  "$IRON_MIND_WORK_ROOT/sources/iron-mind-public"
git -C "$IRON_MIND_WORK_ROOT/sources/iron-mind-public" checkout \
  476c555e45e2556e2ee4b24c726e774c2bfb7762

git clone https://github.com/gomesgroup/olympus \
  "$IRON_MIND_WORK_ROOT/sources/olympus"
git -C "$IRON_MIND_WORK_ROOT/sources/olympus" checkout \
  7b4bb35c04eb31dc57a8e46cc79a9cab71dee06d

uv run --locked --project tasks/iron_mind python \
  tasks/iron_mind/scripts/prepare_official_data.py \
  --iron-mind-checkout "$IRON_MIND_WORK_ROOT/sources/iron-mind-public" \
  --olympus-checkout "$IRON_MIND_WORK_ROOT/sources/olympus" \
  --output "$IRON_MIND_DATA_ROOT"
```

## Run a Real Campaign

Validate the source-pinned data and endpoint configuration, then run a
one-round smoke test:

```bash
uv run --locked --project tasks/iron_mind python \
  scripts/check_task_dependencies.py config/iron_mind/real_smoke.yaml --no-optional

uv run --locked --project tasks/iron_mind python \
  scripts/run_ldm_tts.py config/iron_mind/real_smoke.yaml
```

Run a 20-evaluation campaign for one dataset:

```bash
uv run --locked --project tasks/iron_mind python \
  scripts/run_ldm_tts.py config/iron_mind/ldm_20_buchwald_hartwig.yaml
```

The paper suite expands to six datasets times 20 campaigns:

```bash
uv run --locked --project tasks/iron_mind python \
  scripts/run_ldm_tts.py config/iron_mind/paper_v2_ldm_20x20.yaml
```

The public-union suite adds the remaining public dataset:

```bash
uv run --locked --project tasks/iron_mind python \
  scripts/run_ldm_tts.py config/iron_mind/public_union_ldm_20x20.yaml
```

## Outputs

Each campaign creates `campaign.json`, `budget.json`, `checkpoint.json`,
`selection_record.json`, `result.json`, and `trajectory.csv`. The
selection record contains posterior predictions, GP kernel parameters,
effective UCB exploration weight, and the selected candidate.

Aggregate a completed paper suite with:

```bash
uv run --locked --project tasks/iron_mind python \
  tasks/iron_mind/scripts/aggregate_official_results.py \
  --runs-root "$IRON_MIND_RUNS_ROOT/official_complete" \
  --output-dir "$IRON_MIND_RUNS_ROOT/summary/paper_v2" \
  --suite paper_v2
```

The aggregate directory contains `summary.json`, `dataset_summary.csv`,
and `aggregate_trajectory.csv`.
