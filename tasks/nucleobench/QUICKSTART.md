# NucleoBench Task Quickstart

The current release qualifies `malinois_k562` through a source-pinned seed
evaluation, three tiny real campaign paths, and a resume check. The remaining
16 cases are registered but not yet qualified.

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

Supply the official Zenodo start table (or the extracted 100-sequence JSON
array), a local official model artifact, and the pinned protocol values:

```bash
uv run --locked --project tasks/nucleobench \
  python -m tasks.nucleobench.scripts.prepare_official_data \
  --case malinois_k562 \
  --source-dir /external/nucleobench/source \
  --output-dir /external/nucleobench/data/malinois_k562 \
  --starts-file /external/nucleobench/inputs/start_sequences_df.csv \
  --model-artifact /external/nucleobench/models/malinois_artifacts.tar.gz \
  --model-sha256 06e926e42304b8207138f1fb871ec19e0654dcdb6b26a62ed23fe1e9ac8cc592 \
  --start-set-sha256 2b76fcdeaa0821b94cca642531145d6ee82aada4d378156d027ef92833ef33f5 \
  --bending-factor 1.0
```

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

The released Malinois configs select the CPU reference path. The provider URL,
API key, and model remain user-defined.
