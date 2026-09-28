# LDM-TTS Agent Execution Guide

This file is the operational contract for coding agents that need to inspect,
test, or run this repository. Humans should start with the [project README](../README.md).
Task-specific details live in each task's `README.md` and `QUICKSTART.md`.

## 1. Non-Negotiable Rules

1. Run commands from the repository root unless a task guide explicitly says otherwise.
2. Inspect `git status --short` before editing. Preserve unrelated user changes.
3. Never commit API keys, credential JSON, model tokens, or private endpoint URLs.
4. Keep provider credentials in process environment variables. Do not pass keys through YAML, `--set`, command arguments, logs, summaries, or plots.
5. Start with mock, then dry-run, then dependency preflight, then one real evaluation. Launch a full campaign only when the user has authorized the cost and scope.
6. Treat real evaluator outputs as predictions or computational scores, not wet-lab measurements.
7. Do not silently replace Vina, the activity model, Absolut, or nanoGPT training with mocks in a real run.
8. Do not retry an expensive remote submission when its status is unknown. Inspect persisted status and artifacts first.
9. Keep generated runs under task `runs/` directories or another ignored output path. Commit only curated examples under `assets/examples/`.
10. A running campaign is not complete. Validate terminal status, requested iteration count, acquisition metadata, finite objectives, and output counts before reporting success.

## 2. Repository Contract

The shared entry point is:

```bash
python scripts/run_ldm_tts.py <config.yaml> [--dry-run] [--set path=value ...]
```

Task registration is manifest-based. The shared runner calls:

```text
tasks.<task_id>.ldm_task.procedure:main
```

Task implementations belong in `tasks/<task>/core/`; shared proposal search,
acquisition, response parsing, task-space types, and trajectory helpers belong
in `ldm_tts/`. Do not reintroduce task-local copies of shared search methods.

## 3. First Inspection

```bash
git status --short
python scripts/validate_tasks.py
python scripts/run_ldm_tts.py --list
python scripts/run_ldm_tts.py config/suites/mock_all.yaml --dry-run
```

Read only the task documents relevant to the requested run:

- `tasks/antibody/QUICKSTART.md` and `tasks/antibody/README.md`
- `tasks/small_molecule/QUICKSTART.md` and `tasks/small_molecule/README.md`
- `tasks/nanogpt/QUICKSTART.md` and `tasks/nanogpt/README.md`

## 4. Environment And Secrets

All real tasks accept the common provider variables:

```bash
export LLM_BASE_URL=https://your-model-host.example/v1
export LLM_MODEL_NAME=your-served-model
export LLM_API_KEY=your-secret
```

The URL is the API root, normally ending in `/v1`, not
`/chat/completions`. nanoGPT also accepts `TTS_LLM_URL`, `TTS_LLM_MODEL`, and
`TTS_LLM_API_KEY` for compatibility.

When a protected JSON credential file contains `url`, `model`, and `key`, load
it into the environment without copying it into the repository:

```bash
export LDM_CREDENTIAL_FILE=/secure/path/api_credential.json
export LLM_BASE_URL="$(jq -r .url "$LDM_CREDENTIAL_FILE")"
export LLM_MODEL_NAME="$(jq -r .model "$LDM_CREDENTIAL_FILE")"
export LLM_API_KEY="$(jq -r .key "$LDM_CREDENTIAL_FILE")"
```

Never print the key. Verify that committed configs keep provider fields null.

## 5. Mock Gate

Build the isolated environments and run the deterministic mock paths before
introducing external evaluators:

```bash
uv sync --locked --project tasks/antibody
uv sync --locked --project tasks/small_molecule
uv sync --locked --project tasks/nanogpt

CUDA_VISIBLE_DEVICES='' uv run --locked --project tasks/antibody \
  python scripts/run_ldm_tts.py config/antibody/mock_ei.yaml

CUDA_VISIBLE_DEVICES='' uv run --locked --project tasks/small_molecule \
  python scripts/run_ldm_tts.py config/small_molecule/mock_m1_stratified_oversample.yaml

CUDA_VISIBLE_DEVICES='' uv run --locked --project tasks/nanogpt \
  python scripts/run_ldm_tts.py config/nanogpt/mock_best_of_n.yaml
```

Do not call a mock result a real scientific result.

## 6. Real-Run Gate

For the exact config and overrides that will be launched:

1. Run `scripts/check_task_dependencies.py`.
2. Run `scripts/run_ldm_tts.py ... --dry-run`.
3. Inspect the resolved task, mode, evaluator paths, acquisition, budget, output directory, and CUDA visibility.
4. Confirm no secret is present in the output.
5. Run one real evaluation using the task's `QUICKSTART.md`.
6. Inspect its persisted objective and status.
7. Only then launch the full budget.

## 7. Task Matrix

| Task | Objective | Real evaluator | Acquisition used in the example | Primary artifacts |
| --- | --- | --- | --- | --- |
| Antibody | Minimize CDRH3 binding energy | Absolut | UCB | `results.csv`, `llm_acq_decisions.jsonl`, `config.json` |
| Small molecule | Minimize Vina and maximize G12D activity | AutoDock Vina plus activity model | EHVI | `history.json`, `rounds.jsonl`, `summary.json`, `vina_cache/` |
| nanoGPT | Minimize validation bits per byte | Real GPU training command | LCB | `model_based_buffer.jsonl`, `model_based.log`, `summary.json`, state directories |

## 8. Example Full Campaign Commands

These commands express the validated campaign shapes. Replace deployment paths
and choose visible GPUs appropriate for the host. Do not reuse the output path
of an active campaign.

### Antibody: 100 Evaluations, UCB

```bash
export ABSOLUT_PATH=/path/to/Absolut
export CUDA_VISIBLE_DEVICES=0

uv run --locked --project tasks/antibody \
  python scripts/check_task_dependencies.py config/antibody/real_lcb.yaml \
  --set args.antigens-file=null \
  --set args.antigen=1ADQ_A \
  --set args.acq=ucb \
  --set args.budget=100 \
  --set args.parallel-budget=300

uv run --locked --project tasks/antibody \
  python scripts/run_ldm_tts.py config/antibody/real_lcb.yaml \
  --set args.antigens-file=null \
  --set args.antigen=1ADQ_A \
  --set args.method=policy_max \
  --set args.acq=ucb \
  --set args.acq-beta=2.0 \
  --set args.device=cpu \
  --set args.budget=100 \
  --set args.n-init=20 \
  --set args.parallel-budget=300 \
  --set args.fallback-random=true \
  --set args.out-dir=runs/antibody_ucb_100
```

`fallback-random=true` is required for robustness when the LLM repeats an
already evaluated sequence and deduplication empties the proposed batch.
The example keeps GP/acquisition fitting on CPU so it also runs with task
environments whose CUDA build does not support the deployment GPU.

### Small Molecule: 100 Evaluations, EHVI

```bash
export VINA_BIN=/path/to/vina
export G12D=/trusted/path/best_g12d_model.joblib
export CUDA_VISIBLE_DEVICES=0

uv run --locked --project tasks/small_molecule \
  python scripts/check_task_dependencies.py \
  config/small_molecule/real_m1_seed_analog.yaml \
  --set args.vina-bin="$VINA_BIN" \
  --set args.nn-model-path="$G12D" \
  --set args.gp-device=cuda \
  --no-optional

uv run --locked --project tasks/small_molecule \
  python scripts/run_ldm_tts.py \
  config/small_molecule/real_m1_seed_analog.yaml \
  --set contract_profile= \
  --set args.seed=42 \
  --set args.budget=100 \
  --set args.init-size=5 \
  --set args.batch-size=1 \
  --set args.m1-k-direct-llm=128 \
  --set args.max-candidates-per-round=128 \
  --set args.gp-device=cuda \
  --set args.acq=ehvi \
  --set args.ehvi-n-samples=128 \
  --set args.vina-bin="$VINA_BIN" \
  --set args.nn-model-path="$G12D" \
  --set args.trajectory-dir=runs/small_molecule_ehvi_100 \
  --set args.allow-early-stop=false
```

### nanoGPT: 100 Outer Iterations, LCB N4H4

Install the real training group and prepare the dataset first. Keep the Hugging
Face cache available locally; use offline lookup only after confirming the
required kernel and data artifacts are cached.

```bash
uv sync --locked --group train --project tasks/nanogpt
export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1

uv run --locked --group train --project tasks/nanogpt \
  python scripts/check_task_dependencies.py \
  config/nanogpt/real_operation_tool_best_of_n.yaml \
  --set args.iterations=100 \
  --set args.warmup=20 \
  --set args.breadth=4 \
  --set args.depth=4

uv run --locked --group train --project tasks/nanogpt \
  python scripts/run_ldm_tts.py \
  config/nanogpt/real_operation_tool_best_of_n.yaml \
  --set args.iterations=100 \
  --set args.warmup=20 \
  --set args.warmup-include-root=true \
  --set args.warmup-strategy=random_operation \
  --set args.warmup-seed=42 \
  --set args.breadth=4 \
  --set args.depth=4 \
  --set args.max-generated-per-iteration=64 \
  --set args.surrogate-mode=lcb \
  --set args.gp-beta=1.0 \
  --set args.acquisition-feedback=brief \
  --set args.buffer=runs/nanogpt_lcb_n4h4_i100_buffer.jsonl \
  --set args.out-dir=runs \
  --set args.run-name=nanogpt_lcb_n4h4_i100 \
  --set args.temperature=0.2 \
  --set args.disable-thinking=true \
  --set args.no-progress=true
```

One outer iteration evaluates one selected candidate for approximately five
minutes. A 20-attempt warm-up plus 100 outer iterations can take many hours.

## 9. Resume And Recovery

- Small molecule supports `args.resume=true` and `args.resume-from=<campaign-dir>`.
- nanoGPT supports `args.resume-from=<run-dir-or-summary>` and restores state counters, buffers, schema, and iteration numbering.
- Antibody writes per-evaluation artifacts but does not have the same general resume contract. Preserve the run directory and relaunch only with a documented recovery plan.
- Never delete or overwrite a partial run while diagnosing it.

Before resuming, validate the last complete record and use a new log file. Do
not count generated proposal states as completed real evaluations.

## 10. Completion Checks

### Antibody

- `results.csv` contains exactly the requested number of data rows.
- `llm_acq_decisions.jsonl` contains one decision per evaluation.
- The first `n_init` decisions have acquisition disabled or unused.
- Every later decision has the requested acquisition name and `used=true`.
- `BestValue` is finite and monotone non-increasing.

### Small molecule

- `summary.json.history_size` and `summary.json.round_count` equal the budget.
- `rounds.jsonl` indices are contiguous.
- `selection_mode` matches the requested acquisition, with only documented initialization fallbacks.
- `early_stop_reason` is null when early stopping was disabled.
- Vina and activity scores are finite; hypervolume is monotone non-decreasing.

### nanoGPT

- Terminal launcher status is `completed` with return code 0.
- The model-based summary contains exactly the requested outer iterations.
- Selected iteration numbers are contiguous and each has a persisted state.
- Every completed model-based record has `surrogate_mode=lcb` for an LCB run.
- Failed warm-up or selected evaluations are reported explicitly and excluded from GP fitting.
- The best state, score, active feature schema, and real evaluation count are present.

## 11. Data Collection And Reasoning Augmentation

Root `data/` is the offline training-data workspace, not a task runtime module.
Task code emits accepted actions through `ldm_tts.data`; never import
the command-line tools in `data/` into a task. Keep each campaign's generated
artifacts together under ignored `data/generated/<campaign>/`, as documented in
`data/README.md`.

Collect ldm-2.0 IR during an authorized task run:

```bash
export LDM_DATA_COLLECTION_ENABLED=1
export LDM_DATA_COLLECTION_DIR="$PWD/data/generated/my_campaign"
export LDM_DATA_COLLECTION_RENDER=prose

python scripts/run_ldm_tts.py <config.yaml>
```

Preserve collected IR as immutable source data. Add expert reasoning in a new
file, using only information that was visible when the accepted action was
proposed:

```bash
python data/augment.py \
  --input data/generated/my_campaign/ldm_ir.jsonl \
  --output data/generated/my_campaign/ldm_ir_augmented.jsonl \
  --checkpoint data/generated/my_campaign/augmentation.checkpoint.jsonl \
  --sft-output data/generated/my_campaign/ldm_sft_augmented.jsonl
```

Run unit tests plus the independent IR and rendered-data checks before using a
dataset for training:

```bash
python -m pytest tests/test_data_collection.py tests/test_data_augmentation.py
python data/build_ldm2.py audit \
  --in-ir data/generated/my_campaign/ldm_ir_augmented.jsonl
python data/verify.py validity \
  --in-ir data/generated/my_campaign/ldm_ir_augmented.jsonl
python data/verify.py alpaca \
  --sft data/generated/my_campaign/ldm_sft_augmented.jsonl
```

Do not expose post-action outcomes to reasoning generation, silently repair
invalid actions, augment records marked `reasoning_available=false`, or commit
generated corpora and checkpoints. See [Data Collection](data-collection.md) for the full
contract and `data/SCHEMA.md` for the IR schema.

## 12. Plotting

Run the maintained plotter in an environment containing Matplotlib and the
small-molecule plotting dependencies:

```bash
uv run --locked --project tasks/small_molecule \
  python scripts/plot_campaigns.py \
  --antibody-results /path/to/antibody/results.csv \
  --molecule-run /path/to/small_molecule/run_dir \
  --nanogpt-buffer /path/to/nanogpt/model_based_buffer.jsonl \
  --nanogpt-total-iterations 100 \
  --output-dir /path/to/plots
```

The script infers successful nanoGPT warm-up observations from buffer metadata
and labels an incomplete nanoGPT plot as interim. Verify PNG magic bytes, row
counts in the generated CSV files, and visual layout before publishing.

## 13. Known Improvement Priorities

1. Add a shared launcher/status format that reports live completed iterations, selected state, and progress for all tasks.
2. Add bounded LLM-call and trace-size controls to molecule reservoir refills; the example required 7,874 model calls and produced a 239 MB round trace.
3. Strengthen pre-evaluation nanoGPT operation constraints to reject invalid architectures before spawning training.
4. Add a robust antibody empty-batch policy as a tested invariant, not only a config choice.
5. Add multi-seed random, pure-LLM, BO-only, and acquisition-ablation suites before making causal claims about LDM.
6. Add automated end-to-end artifact validation and plotting to CI using compact mock fixtures.

## 14. Evidence And Claims

The examples under `assets/examples/real_100_20260809/` demonstrate that each
adapter can execute against its real evaluator and that the observed incumbent
improved. They do not isolate causality. Report the following distinction:

- Runnable: supported by successful end-to-end artifacts.
- Optimization progress: supported by improving trajectories.
- Causal LDM advantage: requires controlled baselines, ablations, and multiple seeds.
