---
name: collect-ldm-data
description: Collect, render, augment, audit, and validate LDM fine-tuning data through the repository's ldm-2.0 pipeline. Use when enabling data collection for an LDM task run, verifying or adding a DataCollectionSink task hook, converting accepted teacher actions into IR or LlamaFactory Alpaca rows, generating expert reasoning with an OpenAI-compatible endpoint, resuming augmentation, or checking a training corpus for action validity, leakage, provenance, and format errors.
---

# Collect LDM Data

Build auditable training data through the public `ldm_tts.data` interface. Keep
the collected IR immutable, keep hidden outcomes out of model-visible text, and
write every derived stage to a new path.

Read [references/pipeline-contract.md](references/pipeline-contract.md) before
running or changing collection. Also read `data/README.md`, `data/SCHEMA.md`, and
the selected task's README when present.

## Resolve The Request

1. Work from the repository root and inspect `git status --short`.
2. Identify the task, source run/config, collection mode, output campaign,
   reasoning policy, whether the source is an accepted direct action or a raw
   Harness session trace, and whether the user requested a mock, tiny real, or
   full run. Do not promote the requested execution level.
3. Choose a new ignored output directory under
   `data/generated/<campaign>/`, unless the user explicitly requests a safe
   resume. Never append a different run or schema to an existing campaign by
   accident.
4. Inspect the task for a live `DataCollectionSink` hook. Environment variables
   do nothing when the executed code path has no hook. Report that fact instead
   of claiming collection succeeded.

## Verify Or Add The Hook

Use only imports from `ldm_tts.data`. Construct `DataCollectionSink` once per
recorder or run, build IR through `make_complete_design_ir` or
`make_parameter_edit_ir`, and append one record per accepted teacher action.

Freeze the model-visible context and action after model output parsing and
validation, at the point where the action is known to be executable. The task
may persist that frozen row later so it can attach audit outcomes, but it must
not rebuild the training target from post-selection state. Do not collect
rejected attempts, the first unvalidated tool call, silently repaired outputs,
evaluator predictions emitted by the model, or post-BO candidates as if the
teacher proposed them.

Research-Harness session JSONL and redacted provider request/response files are
raw multi-turn traces, not accepted-action IR. Preserve them under the run's
Harness artifact root. Do not pass them to `DataCollectionSink`, the renderer,
or expert augmentation until a separate converter defines the model-visible
messages, terminal accepted action, rejected-attempt treatment, tool-result
policy, and leakage checks.

Keep run IDs, evaluator outcomes, selected candidates, acquisition values, and
drop counts under `collection.provenance` or `collection.outcome`. The renderer
must exclude them from `instruction`. Include a field in `task`, `search_state`,
`request`, or `raw_context` only when it was genuinely visible to the teacher.

When adding a hook, add a focused test that exercises the real task adapter and
asserts:

- the accepted action is preserved;
- `validate_ir_record` accepts the row;
- IR, SFT, and `dataset_info.json` are written;
- provenance/outcome markers do not appear in the rendered instruction;
- disabled collection performs no writes.

Run the shared tests before task execution:

```bash
python -m pytest -q tests/test_data_collection.py tests/test_data_augmentation.py
```

## Collect Progressively

Follow `$run-ldm-task` preflight and execution guardrails when that repository
skill is available. Start with the task's mock or contract path, then use a tiny
real run before a full real campaign.

Enable collection for the exact task process:

```bash
export LDM_DATA_COLLECTION_ENABLED=1
export LDM_DATA_COLLECTION_DIR="$PWD/data/generated/<campaign>"
export LDM_DATA_COLLECTION_RENDER=prose

python scripts/run_ldm_tts.py <config.yaml> [--set path=value ...]
```

Use the task's `uv run --locked --project tasks/<task_id>` environment when its
dependencies require it. Use the same effective overrides for dependency
checks, dry-run, and execution.

After the first action, require non-empty `ldm_ir.jsonl`, `ldm_sft.jsonl`, and a
matching `dataset_info.json`. Stop if no row was written; diagnose the executed
method, hook, collection environment, and accepted-attempt path before spending
more model or evaluator budget.

## Add Expert Reasoning

Augment the immutable `ldm_ir.jsonl`, not a copy edited in place. Load protected
credentials into environment variables without printing values or copying the
credential file into the repository.

First run one record with one worker:

```bash
python data/augment.py \
  --input data/generated/<campaign>/ldm_ir.jsonl \
  --output data/generated/<campaign>/ldm_ir_augmented.jsonl \
  --checkpoint data/generated/<campaign>/augmentation.checkpoint.jsonl \
  --sft-output data/generated/<campaign>/ldm_sft_augmented.jsonl \
  --limit 1 \
  --workers 1
```

Validate that smoke artifact before removing `--limit` and increasing workers.
Reuse the same checkpoint to resume unfinished rows. Changing model, endpoint,
temperature, or system prompt intentionally creates a different cache identity.

Do not use `--overwrite-reasoning` unless the user explicitly requests
replacement. Do not use `--include-reasoning-unavailable` for normal data
construction; it can fabricate unsupported protein or sequence rationales.
Reasoning must explain the fixed accepted action from visible evidence only and
must not use `collection.outcome` or other post-action results.

## Render And Validate

Choose the actual training artifact before registering it. For
reasoning-augmented training, regenerate the registration so it points to the
augmented SFT filename:

```bash
python data/build_ldm2.py render \
  --in-ir data/generated/<campaign>/ldm_ir_augmented.jsonl \
  --out data/generated/<campaign>/ldm_sft_augmented.jsonl \
  --render prose \
  --dataset-info data/generated/<campaign>/dataset_info.json
```

Run all applicable gates:

```bash
python data/build_ldm2.py audit \
  --in-ir data/generated/<campaign>/ldm_ir_augmented.jsonl

python data/verify.py validity \
  --in-ir data/generated/<campaign>/ldm_ir_augmented.jsonl

python data/verify.py alpaca \
  --sft data/generated/<campaign>/ldm_sft_augmented.jsonl \
  --dataset-info data/generated/<campaign>/dataset_info.json

python data/verify.py leakage \
  --sft data/generated/<campaign>/ldm_sft_augmented.jsonl
```

Compare source and output row counts and inspect task/action distributions.
Treat low design-space-expansion coverage, duplicated candidates, oversized
prompts, missing rationale evidence, and mixed task contracts as data-quality
findings rather than validator noise. Split train/evaluation data by run,
antigen, or seed when possible, not by individual row.

## Report The Result

Report the task and method, execution level, hook status, source runs, output
paths, IR/SFT/checkpoint row counts, generated/resumed/skipped/failed reasoning
counts, action distribution, validation results, and residual warnings. State
whether the corpus is action-level or underwent a separate acquisition-weighted
selection step. Do not describe current action-level collection as
acquisition-weighted candidate distillation unless such a step actually ran.
For a Harness source, explicitly report that only raw session/transport traces
were retained unless a named and validated conversion contract was executed.
