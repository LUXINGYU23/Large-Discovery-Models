# NucleoBench Task Quickstart

The current release qualifies `malinois_k562` and provides digest-pinned
preparation contracts for the other 15 published paired-start cases. RiNALMo
remains planned because its official 100-sequence paired-start set has not been
published.

Run these commands from the repository root.

## Install the Task Environment

```bash
uv sync --locked --project tasks/nucleobench
```

The mock and unit tests use the lightweight base environment. Before preparing
or running an official oracle on Linux, install the source-pinned runtime:

```bash
uv sync --locked --project tasks/nucleobench --extra official
```

## Validate Registration

```bash
uv run --locked --project tasks/nucleobench \
  python scripts/validate_tasks.py --task nucleobench \
  --require-qualified --require-stage tiny_campaign_verified

uv run --locked --project tasks/nucleobench \
  python -m pytest tasks/nucleobench/tests
```

## Inspect a Case

```bash
uv run --locked --project tasks/nucleobench \
  python -m tasks.nucleobench.ldm_task.procedure \
  --case-id malinois_k562 --dry-run
```

This prints the source-pinned case metadata and declarative LDM task contract.
Real commands additionally require the prepared external inputs and the
provider settings used by the selected method.

## Run the Mock Campaign

```bash
uv run --locked --project tasks/nucleobench \
  python scripts/run_ldm_tts.py config/nucleobench/mock.yaml
```

This deterministic two-round run exercises the shared engine and writes the
standard campaign, budget, status, event, checkpoint, summary, result,
trajectory, and data-collection artifacts under `tasks/nucleobench/runs/`.

## Prepare External Inputs

For Malinois and BPNet, supply the published Zenodo CSV and the corresponding
official model artifact. The preparation command selects the declared 100-row
block by source index and verifies both source and model bytes against the
tracked contract:

```bash
uv run --locked --project tasks/nucleobench \
  python -m tasks.nucleobench.scripts.prepare_official_data \
  --case malinois_k562 \
  --source-dir /external/nucleobench/source \
  --output-dir /external/nucleobench/data/malinois_k562 \
  --starts-file /external/nucleobench/inputs/start_sequences_df.csv \
  --model-artifact /external/nucleobench/models/malinois_artifacts.tar.gz \
  --bending-factor 1.0
```

Change `--case` and `--model-artifact` for either remaining Malinois target or
any BPNet target; no task-specific Python workflow is copied. For Enformer,
pass `start_sequences_enformer.parquet`. Preparation preserves its distinct
256-position mask for each paired start:

```bash
uv run --locked --project tasks/nucleobench --extra official \
  python -m tasks.nucleobench.scripts.prepare_official_data \
  --case enformer_muscle_not_liver \
  --source-dir /external/nucleobench/source \
  --output-dir /external/nucleobench/data/enformer_muscle_not_liver \
  --starts-file /external/nucleobench/inputs/start_sequences_enformer.parquet \
  --model-artifact /external/nucleobench/models/human.ckpt
```

The RiNALMo loader accepts an external 100-sequence JSON set only when its
SHA-256 is explicitly supplied with `--expected-start-set-sha256`. Such a run
is not an official paired-start benchmark until upstream defines that set.

Source checkouts, model weights, starts, prepared manifests, caches, and run
outputs must remain outside the Git repository.

## Run the Qualified Tiny Campaign

Configure external paths and any OpenAI-compatible provider:

```bash
export NUCLEOBENCH_WORK_ROOT=/absolute/path/to/nucleobench-work
export NUCLEOBENCH_SOURCE_ROOT="$NUCLEOBENCH_WORK_ROOT/source"
export NUCLEOBENCH_DATA_ROOT="$NUCLEOBENCH_WORK_ROOT/data"
export NUCLEOBENCH_RUNS_ROOT="$NUCLEOBENCH_WORK_ROOT/runs"
export NUCLEOBENCH_START_SET_SHA256=2b76fcdeaa0821b94cca642531145d6ee82aada4d378156d027ef92833ef33f5
export NUCLEOBENCH_API_KEY_FILE=/absolute/path/to/api-key
export LLM_BASE_URL=https://your-provider.example/v1
export LLM_MODEL_NAME=your-model

uv run --locked --project tasks/nucleobench --extra official \
  python scripts/run_ldm_tts.py \
  config/nucleobench/malinois_k562_tiny_campaign.yaml
```

The released K562 configs select the CPU reference path. The provider URL,
API key, and model remain user-defined.

Prepared but unqualified cases may run the `qualification` profile directly by
passing their case ID, prepared directory, source checkout, and prepared
start-set digest. Pilot and official profiles reject a case until its catalog
state is promoted to `qualified` with recorded evidence.
