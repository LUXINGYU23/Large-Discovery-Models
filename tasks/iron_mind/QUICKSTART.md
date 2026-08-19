# Iron Mind: First Run Guide

This guide starts from a clean checkout and verifies the task in the same
order used for a reproducible release: local mock campaign, official data
preparation, endpoint configuration, and one real smoke run.

Run every command from the repository root.

## 1. Install and Verify the Mock Campaign

```bash
uv sync --locked --project tasks/iron_mind
uv run --locked --project tasks/iron_mind \
  python -m pytest tasks/iron_mind/tests
uv run --locked --project tasks/iron_mind \
  python scripts/run_ldm_tts.py config/iron_mind/mock.yaml
```

The mock path needs no external data, GPU, or model endpoint.

## 2. Configure a Model Endpoint

Real runs require an OpenAI-compatible Chat Completions endpoint:

```bash
export LLM_BASE_URL=https://your-model-host.example/v1
export LLM_MODEL_NAME=your-served-model
export LLM_API_KEY=your-api-key
```

The task accepts any compatible provider. The committed configuration files
leave these values unset, so users can select a provider through their own
environment. For a local endpoint without authentication, omit
`LLM_API_KEY`.

## 3. Prepare the Source-Pinned Data

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

## 4. Run One Real Round

```bash
uv run --locked --project tasks/iron_mind python \
  scripts/check_task_dependencies.py config/iron_mind/real_smoke.yaml --no-optional

uv run --locked --project tasks/iron_mind python \
  scripts/run_ldm_tts.py config/iron_mind/real_smoke.yaml
```

A successful run writes a timestamped directory below
`$IRON_MIND_RUNS_ROOT/smoke/`. The released profile produces 64 internal
proposals in one structured model response, ranks them with the task GP
selector, and evaluates one reaction condition. That one external evaluation
is the Iron Mind-compatible batch size.

After the smoke run, use `ldm_20_<dataset>.yaml` for a 20-evaluation
campaign or a suite configuration for the full benchmark. Set
`--set args.reservoir-size=<N>` when running the shared runner to change the
internal proposal count without changing the number of evaluated reactions.
