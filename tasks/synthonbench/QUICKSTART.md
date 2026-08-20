# SynthonBench Quick Start

Run every command from the repository root.

## 1. Install and Verify the Official Example

```bash
uv sync --locked --project tasks/synthonbench
uv run --locked --project tasks/synthonbench \
  python -m pytest tasks/synthonbench/tests
uv run --locked --project tasks/synthonbench \
  python scripts/run_ldm_tts.py config/synthonbench/mock.yaml
```

This run uses the official bundled example space and deterministic proposal
client. It does not download a benchmark score table or call a model endpoint.
The committed configuration uses a 2,048-bit Count-Morgan proxy and 256 public
Nyström landmarks. For a separate ablation, override
`--gp-reaction-weight` or the other `--gp-*` options explicitly.

## 2. Prepare the Official 1M Track

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

The command pins both code and data, including the exact official scoring
script used after a real campaign. Pass `--hf-endpoint <URL>` only when a
user's network requires an alternate Hugging Face endpoint.

## 3. Configure an Endpoint

```bash
export LLM_BASE_URL=https://your-provider.example/v1
export LLM_MODEL_NAME=your-model
export LLM_API_KEY=your-secret
```

The task accepts any OpenAI-compatible Chat Completions provider. Credentials
remain outside tracked configuration and run metadata.

The default request body disables optional thinking for the one-tuple JSON
response: `{"thinking":{"type":"disabled"}}`. If the selected provider does
not support that extension, set `--llm-extra-body-json '{}'` or supply its own
compatible JSON body.

## 4. Check and Run the Surrogate Oracle Track

```bash
uv run --locked --project tasks/synthonbench \
  python scripts/check_task_dependencies.py \
  config/synthonbench/surrogate_1m_qualification.yaml --no-optional

uv run --locked --project tasks/synthonbench \
  python scripts/run_ldm_tts.py \
  config/synthonbench/surrogate_1m_qualification.yaml
```

Replace the config with `glide_1m_qualification.yaml` for the Glide
ligand-efficiency track. The full batch-16 10,000-call profiles are documented
in `README.md`.
