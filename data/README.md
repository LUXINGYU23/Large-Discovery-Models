# LDM Data Pipeline

This directory contains the complete offline pipeline for turning LDM-TTS
search trajectories into auditable training data: the schema, compact examples,
historical import, reasoning augmentation, post-processing, validation, and
fine-tuning configuration.

Task code uses the public `ldm_tts.data` module to emit accepted actions. The
files in this directory operate on those outputs; task implementations must not
import these command-line tools.

## Pipeline

```text
LDM-TTS task run
  -> collect accepted actions as ldm-2.0 IR
  -> normalize historical trajectories when needed
  -> add evidence-grounded expert reasoning
  -> render training rows
  -> validate provenance, action validity, leakage, and format
  -> train or analyze
```

## Files

| Path | Purpose |
| --- | --- |
| `SCHEMA.md` | Contract for the `ldm-2.0` intermediate representation. |
| `examples/` | Small committed IR and rendered fixtures. |
| `augment.py` | Add expert reasoning and render augmented training rows. |
| `build_ldm2.py` | Import historical traces, render IR, and audit records. |
| `fabricate.py` | Optional, provenance-marked rule-based post-processing. |
| `verify.py` | Independent IR, action, run, and Alpaca validation. |
| `ldm_lora_sft.yaml` | LlamaFactory LoRA fine-tuning baseline. |
| [`../finetune/`](../finetune/) | Full-parameter rationale-distillation recipe and grouped dataset preparation. |

This directory owns the canonical collection, augmentation, validation, and
generated-data contracts. The full-SFT recipe consumes augmented IR from this
pipeline; it does not define a parallel data format.

Generated data belongs in the ignored `data/generated/<campaign>/` directory.
Keep the stages visible in filenames instead of creating another directory for
each stage:

```text
data/generated/<campaign>/
|-- ldm_ir.jsonl                 # immutable task-emitted IR
|-- ldm_ir_augmented.jsonl       # reasoning-augmented IR
|-- augmentation.checkpoint.jsonl
|-- ldm_sft_augmented.jsonl      # rendered training rows
|-- dataset_info.json
`-- validation.txt
```

## Collect

Collection is opt-in. Point the task runner at one campaign directory:

```bash
export LDM_DATA_COLLECTION_ENABLED=1
export LDM_DATA_COLLECTION_DIR="$PWD/data/generated/my_campaign"
export LDM_DATA_COLLECTION_RENDER=prose

python scripts/run_ldm_tts.py <config.yaml>
```

`ldm_ir.jsonl` is the authoritative source. Preserve it unchanged; rendered
files written during collection are convenience artifacts that can be rebuilt.
The structured NanoGPT operation runtime, small-molecule direct proposer, and
antibody direct-sequence paths all support this sink contract. See the detailed
task mapping for boundaries that remain intentionally excluded, such as antibody
policy/DSL updates and random fallbacks.

Persistent research-Harness runs retain native session JSONL and redacted
provider transport under their run artifact directory. Those multi-turn traces
are not emitted through `DataCollectionSink` and are not valid `ldm-2.0` rows
without a separate converter and leakage contract.

## Add Expert Reasoning

Reasoning augmentation explains an already accepted action using only the
model-visible context. It must not change the action, use hidden outcomes, or
invent evidence. Keep credentials in environment variables:

```bash
export LLM_BASE_URL=https://your-model-host.example/v1
export LLM_MODEL_NAME=your-served-model
export LLM_API_KEY=your-secret

python data/augment.py \
  --input data/generated/my_campaign/ldm_ir.jsonl \
  --output data/generated/my_campaign/ldm_ir_augmented.jsonl \
  --checkpoint data/generated/my_campaign/augmentation.checkpoint.jsonl \
  --sft-output data/generated/my_campaign/ldm_sft_augmented.jsonl
```

The command is resumable. Re-running it reuses completed records and retries
unfinished ones.

## Validate

Run the public-module tests:

```bash
python -m pytest tests/test_data_collection.py tests/test_data_augmentation.py
```

Then validate the generated files:

```bash
python data/build_ldm2.py audit \
  --in-ir data/generated/my_campaign/ldm_ir_augmented.jsonl

python data/verify.py validity \
  --in-ir data/generated/my_campaign/ldm_ir_augmented.jsonl

python data/verify.py alpaca \
  --sft data/generated/my_campaign/ldm_sft_augmented.jsonl
```

See the [data-collection guide](../docs/data-collection.md) for task hooks and detailed
quality rules. See [`SCHEMA.md`](SCHEMA.md) for the IR fields.

## Invariants

1. Preserve collected IR; write transformations to new filenames.
2. Keep provenance and evaluator outcomes out of model-visible input.
3. Never train on rejected, invalid, or silently repaired actions.
4. Mark synthetic or rule-derived records and retain their derivation.
5. Do not infer unavailable reasoning or use post-action outcomes as rationale.
6. Never commit credentials, private endpoints, generated corpora, or checkpoints.
