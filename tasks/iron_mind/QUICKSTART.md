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
`$IRON_MIND_RUNS_ROOT/smoke/`. The released profile sends 64 independent
one-candidate requests with up to 64 local workers, estimates empirical `q0`,
maintains a 32-candidate BO pool, and samples one reaction condition from the
GP-UCB-tilted LDM policy.
That one external evaluation is the Iron Mind-compatible batch size.
Malformed or duplicate responses are recorded and can reduce the admitted
reservoir.

The default `portfolio_v1` prompt assigns a distinct factor focus to every
request and records the policy, slot role, focus, and prompt digest in the run
events. To run the frozen earlier unallocated prompt as an ablation, use:

```bash
uv run --locked python scripts/run_ldm_tts.py config/iron_mind/prompt_baseline_smoke.yaml
```

For a full baseline campaign, override both the contract profile and policy on
an `ldm_20_<dataset>.yaml` configuration:

```bash
--set contract_profile=ldm_prompt_baseline_20 --set args.prompt-policy=baseline_v1
```

If a provider supports OpenAI-compatible JSON mode, add
`--set args.llm-json-mode=true`; this is an optional formatting aid, not a
provider requirement. Provider-specific request fields can be supplied without
editing a tracked config, for example:

```bash
--set 'args.llm-extra-body-json={"thinking":{"type":"disabled"}}'
```

Thinking is disabled by default for proposal-only generation. Use this override
only to replace the default request object for a different OpenAI-compatible
provider.

After the smoke run, use `ldm_20_<dataset>.yaml` for a 20-evaluation
campaign or a suite configuration for the full benchmark. Set both
`--set args.proposal-samples=<M>` and `--set args.bo-pool-size=<K>` to change
the internal search, with `M > K`, without changing the number of evaluated
reactions.

## 5. Run the Quick Three-Method Comparison

After the official data and endpoint are ready, run the fixed six-round matrix:

```bash
uv run --locked --project tasks/iron_mind python \
  scripts/run_quick_compare.py config/quick_compare/iron_mind.yaml --dry-run

uv run --locked --project tasks/iron_mind python \
  scripts/run_quick_compare.py config/quick_compare/iron_mind.yaml
```

The BO comparator is offline after data preparation. LDM and direct LLM use
the generic endpoint variables from step 2. The output root is
`$IRON_MIND_RUNS_ROOT/quick_compare/`; rerun an interrupted matrix with
`--resume` after confirming the repository and configurations are unchanged.
