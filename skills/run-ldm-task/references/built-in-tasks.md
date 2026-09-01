# Built-In Task Run Reference

## Contents

- [Common Rules](#common-rules)
- [nanoGPT](#nanogpt)
- [Small Molecule](#small-molecule)
- [Antibody](#antibody)
- [Iron Mind](#iron-mind)
- [SynthonBench](#synthonbench)

## Common Rules

Treat each task README, especially its **Minimal First Real Run** section, as
the source of truth for commands and artifacts. This reference identifies the
right configs and migration state; it intentionally does not duplicate volatile
real-run recipes.

Use identical `--set` overrides for dependency checks, runner dry-runs, and
execution. Runner `--dry-run` validates config resolution and any selected
experiment profile. A task-level `args.dry-run=true`, zero-iteration mode, or
similar option enters the task adapter and may write diagnostic artifacts.

All built-in tasks use the shared campaign engine. Conventional adapters call
`ldm_tts.campaign.run_campaign`; documented specialized lifecycles construct
`LDMEngine` with `CampaignRuntime` directly. Every executed campaign writes the shared
`campaign.json` / `events.jsonl` / `checkpoint.json` / `summary.json` /
`budget.json` / `status.json` artifact set. Tasks additionally re-export their
historical trajectory files from engine events:

- `nanogpt`: `model_based_summary.json`, `summary.json` (merged with
  `engine_summary`), `model_based_buffer.jsonl`, `states/`.
- `small_molecule`: `history.json`, `rounds.jsonl`, and legacy summary fields
  merged into `summary.json`.
- `antibody`: `results.csv`, `llm_acq_decisions.jsonl`, and legacy summary
  fields merged into `summary.json`.

Resume goes through the engine checkpoint (`checkpoint.json`); `small_molecule`
additionally accepts a legacy `history.json`/`rounds.jsonl` directory when no
campaign manifest exists.

When a config selects `contract_profile`, do not override locked budget or
method arguments. Use a checked-in smoke profile, or follow a task README that
explicitly clears `contract_profile` for a diagnostic run. Such a run is not a
qualified execution of the named profile.

All model-backed task modes accept OpenAI-compatible URL, model, and key
settings. Keep authenticated keys in environment variables or documented
ignored protected files, never tracked YAML or literal command arguments.

Direct proposal backends use Chat Completions. A task's research Harness may
use another compatible wire API; the Pi sidecar uses Responses and
must be checked through its documented capability smoke.

## nanoGPT

Files:

```text
tasks/nanogpt/README.md
config/nanogpt/mock_best_of_n.yaml
config/nanogpt/real_operation_tool_best_of_n.yaml
config/nanogpt/real_operation_tool_fixed_best_of_n.yaml
```

Model variables: `LLM_BASE_URL`, `LLM_MODEL_NAME`, and `LLM_API_KEY`. The
historical `TTS_LLM_URL`, `TTS_LLM_MODEL`, and `TTS_LLM_API_KEY` aliases remain
accepted.

The real config selects a profile that locks `method`, `iterations`, and
`warmup`. Follow `tasks/nanogpt/README.md` when making a zero-iteration or tiny
run; its diagnostic recipe explicitly clears `contract_profile` before changing
those values. Real evaluation requires prepared data/tokenizer artifacts and
the task's training dependency group.

The nanoGPT campaign runs through `run_campaign`: warm-up and each model-based
iteration are engine rounds. The task emits the surrogate-scored proposal pool,
the engine invokes selection, and all real evaluations, budgets, events, and
checkpoints belong to the shared campaign algorithm.
Inspect `events.jsonl` / `summary.json` for the shared contract and
`model_based_summary.json` for the task's iteration records.

## Small Molecule

Files:

```text
tasks/small_molecule/README.md
config/small_molecule/mock_m1_stratified_oversample.yaml
config/small_molecule/real_m1_seed_analog.yaml
```

Model variables: `LLM_BASE_URL`, `LLM_MODEL_NAME`, and `LLM_API_KEY`. Evaluator
variables include `VINA_BIN` and `G12D`; ReaSyn paths are needed only by methods
that generate analogues.

The real profile locks `budget`, `batch-size`, and `acq`. Follow
`tasks/small_molecule/README.md` for contract and tiny runs; it explicitly clears
`contract_profile` before reducing the budget. `--no-optional` may omit ReaSyn
checks only when the selected direct method cannot call ReaSyn.

The small-molecule campaign runs through `run_campaign`: the tilted EHVI/SIR
reservoir search lives in the task's expander and acquisition-selector adapters
(`tasks/small_molecule/core/engine_adapters.py`), and the engine owns budget,
events, checkpoints, and summaries. `history.json` / `rounds.jsonl` remain as
event re-exports for downstream tooling.

## Antibody

Files:

```text
tasks/antibody/README.md
tasks/antibody/resources/default_config.yaml
config/antibody/mock_ei.yaml
config/antibody/real_cpu_smoke.yaml
config/antibody/real_lcb.yaml
```

Model variables: `LLM_BASE_URL`, `LLM_MODEL_NAME`, and `LLM_API_KEY`. Real runs
also require an Absolut installation configured through `ABSOLUT_PATH` or the
selected task config.

Use `real_cpu_smoke.yaml` for the first real proposal and evaluation. It carries
the matching `real_cpu_smoke` contract profile and already fixes the one-run
budget, initialization count, parallel budget, and CPU device. Do not recreate
that smoke run by overriding `real_lcb.yaml`; reserve `real_lcb.yaml` for the
larger unprofiled run described by the task README.

The antibody campaign runs through `run_campaign`: one engine campaign per
(antigen, seed) pair, with the warmup/proposal/GP-acquisition components
adapted into the task's expander and selector
(`tasks/antibody/core/engine_adapters.py`). `results.csv` and
`llm_acq_decisions.jsonl` remain as event re-exports.

## Iron Mind

Files:

```text
tasks/iron_mind/README.md
tasks/iron_mind/QUICKSTART.md
config/iron_mind/mock.yaml
config/iron_mind/harness_smoke.yaml
config/pilot_evaluation/iron_mind.yaml
```

Both proposal backends use `LLM_BASE_URL`, `LLM_MODEL_NAME`, and `LLM_API_KEY`.
Direct sampling uses Chat Completions. Harness sampling uses the Pi Responses
sidecar and additionally requires Docker, Linux KVM, a built Harness image, and
writable external run/cache roots. A protected API-key file may be selected by
`--harness-api-key-file`.

Iron Mind constructs the shared `LDMEngine` directly around source-pinned
finite reaction tables. Four persistent Harness profiles each submit 16 exact
condition combinations. Task-local Python rejects invalid, historically
evaluated, and within-session duplicate candidates before commit; cross-session
agreement remains as proposal-frequency mass. Accepted occurrences then use
the same empirical `q0`, factor-aware categorical GP-UCB, acquisition tilt,
and frozen evaluator as direct LDM.

Follow `tasks/iron_mind/QUICKSTART.md`: validate the mock path, prepare the
official source-pinned data, build the sidecar, run `harness_smoke.yaml`, and
only then run the four-method pilot matrix. Inspect `<run_dir>/harness/` for
session/provider traces and the normal campaign artifacts for optimization
history and results.

## SynthonBench

Files:

```text
tasks/synthonbench/README.md
tasks/synthonbench/QUICKSTART.md
config/synthonbench/mock.yaml
config/synthonbench/ldm_harness_surrogate_smoke.yaml
```

The direct backend uses the common `LLM_BASE_URL`, `LLM_MODEL_NAME`, and
`LLM_API_KEY` settings. The Harness backend uses the same provider identity but
runs the Pi sidecar over the OpenAI Responses wire format. It additionally
requires Docker, Linux KVM, the configured Harness image, writable run and cache
directories, and read-only task profiles/tools. An ignored protected key file
may be selected by the task config instead of placing a key in process
arguments.

Follow `tasks/synthonbench/QUICKSTART.md` to prepare the official data and build
the image. Before a real Harness campaign, run the Pi unit tests and capability
smoke documented in `harnesses/pi/README.md`; a direct Chat Completions probe is
not sufficient.

SynthonBench constructs the shared `LDMEngine` directly for its specialized
official-oracle lifecycle. Four persistent task profiles submit provisional
minibatches through one `HarnessClient`. The task validates exact reaction and
ordered-synthon tuples before each turn commits, then routes accepted
occurrences through the same empirical `q0`, task-local GP-UCB, acquisition,
and official evaluator used by the ordinary LDM path. Inspect `<run_dir>/harness/`
alongside the shared campaign artifacts.
