# Built-In Task Run Reference

## Contents

- [Common Rules](#common-rules)
- [nanoGPT](#nanogpt)
- [Small Molecule](#small-molecule)
- [Antibody](#antibody)
- [Iron Mind](#iron-mind)
- [SynthonBench](#synthonbench)
- [NucleoBench](#nucleobench)
- [Other Registered Tasks](#other-registered-tasks)

## Common Rules

Treat each task README, especially its **Minimal First Real Run** section, as
the source of truth for commands and artifacts. This reference identifies the
right configs and runtime ownership; it intentionally does not duplicate volatile
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

For Iron Mind, SynthonBench, and NucleoBench pilot runs, select children from
`config/pilot_evaluation/<task>.yaml` using `scripts/run_pilot_evaluation.py`
with `--method`, `--case`, and `--seed`. Method arguments come from the matrix's
base config and `method_overrides`; do not create standalone method copies.
See `docs/pilot-evaluation.md` for six-round and twelve-round entry points.

Resolve the direct proposal wire API from the task config: Iron Mind and
SynthonBench use Chat Completions; NucleoBench also supports Responses.
The Pi sidecar uses Responses and must be checked through its documented
capability smoke.

## Other Registered Tasks

The following adapters use the same registration and shared-runner contracts,
but their task-local guides are the authoritative source for setup and runtime
requirements:

- [Adaptive KV-cache quantization](../../../tasks/llm_kv_adaptive_quantization/README.md)
- [AI4Bio mutation-effect prediction](../../../tasks/ai4bio_mutation_effect_prediction/README.md)
- [Discrete causal discovery](../../../tasks/causal_discovery_discrete/README.md)
- [AtomWorld](../../../tasks/atomworld/README.md)
- [ReaSyn](../../../tasks/reasyn/README.md)

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
config/iron_mind/ldm_harness_smoke.yaml
config/pilot_evaluation/iron_mind.yaml
```

All model-backed methods use `LLM_BASE_URL`, `LLM_MODEL_NAME`, and `LLM_API_KEY`.
Direct sampling uses Chat Completions. Harness methods use the Pi Responses
sidecar and additionally require Docker, Linux KVM, a built Harness image, and
writable external run/cache roots. A protected API-key file may be selected by
`--harness-api-key-file`.

Iron Mind constructs the shared `LDMEngine` directly around source-pinned
finite reaction tables. Four independent persistent sessions share one
comprehensive researcher template and each submit 16 distinct conditions in an
annotated `candidates.json`. Task-local Python rejects invalid and historically
evaluated candidates before commit; agreement across sessions retains its
proposal-frequency mass and all contributing research notes. Accepted occurrences then use
the same empirical `q0`, factor-aware categorical GP-UCB, acquisition tilt,
and frozen evaluator as direct LDM. The direct research Harness instead uses
one persistent session to choose the evaluated condition without `q0`, GP, or
acquisition.

Candidate sessions expose experimental-design, scientific-critical-thinking,
and statsmodels Skills on demand. Their task guest includes the corresponding
numerical and plotting packages. Compact measurement indexes link to
`get_measured_history` for exact conditions and original notes; task validation
checks the full evaluated set. Proposal and policy recovery reuse existing
sessions within a bounded window without resampling accepted peers.
The default single-session limit is 30 minutes; each proposal or policy call
has a recovery window of twice that limit, starting before its first attempt.

Harness-Compiled LDM keeps the four proposal sessions and adds one independent
`policy_architect` session. Its complete `optimization_policy.py` may supply a
standardized prior mean for the task-local residual GP and schedule LDM
`alpha`/`eta`; candidate generation, kernel, variance, UCB, pool, and evaluator
remain fixed. Inspect `<run_dir>/policy_harness/` for the separate manifest,
turns, validations, accepted epochs, and degraded/fallback state.

Follow `tasks/iron_mind/QUICKSTART.md`: validate the mock path, prepare the
official source-pinned data, build the sidecar, run `ldm_harness_smoke.yaml`, and
only then run the six-method Pilot Evaluation. Inspect `<run_dir>/harness/` for
proposal traces, `<run_dir>/policy_harness/` for compiled-policy traces, and the
normal campaign artifacts for optimization history and results. Optional MCP
servers and separate proposal/policy tool budgets use the shared Harness
configuration in `docs/research-harness.md`.

## SynthonBench

Files:

```text
tasks/synthonbench/README.md
tasks/synthonbench/QUICKSTART.md
config/synthonbench/mock.yaml
config/synthonbench/ldm_harness_surrogate_smoke.yaml
config/pilot_evaluation/synthonbench.yaml
```

Direct methods use the common `LLM_BASE_URL`, `LLM_MODEL_NAME`, and
`LLM_API_KEY` settings. Harness methods use the same provider identity but run
the Pi sidecar over the OpenAI Responses wire format. They additionally
require Docker, Linux KVM, the configured Harness image, writable run and cache
directories, and read-only task profiles/tools. An ignored protected key file
may be selected by the task config instead of placing a key in process
arguments.

Follow `tasks/synthonbench/QUICKSTART.md` to prepare the official data and build
the image. Before a real Harness campaign, run the Pi unit tests and capability
smoke documented in `harnesses/pi/README.md`; a direct Chat Completions probe is
not sufficient.

SynthonBench constructs the shared `LDMEngine` directly for its specialized
official-oracle lifecycle. Four independent persistent sessions share the
comprehensive researcher template and submit annotated `candidates.json`
files through one `HarnessClient`. Each minibatch contains 16 distinct tuples;
cross-session agreement retains empirical mass and all original notes.
The task validates exact reaction and ordered-synthon tuples before each turn commits, then routes accepted
occurrences through the same empirical `q0`, task-local GP-UCB, acquisition,
and official evaluator used by the ordinary LDM path. Direct research Harness
uses one persistent comprehensive session and evaluates its complete 16-tuple
minibatch without `q0`, GP, or acquisition. Inspect `<run_dir>/harness/`
alongside the shared campaign artifacts. Optional MCP servers and per-tool turn
budgets use `docs/research-harness.md`.

Candidate sessions load task-local RDKit, experimental-design,
scientific-critical-thinking, and statsmodels Skills on demand. The guest
preinstalls their chemistry and numerical tools. `get_measured_history` returns
paginated measured tuples, component SMILES, and original notes; turn messages
stay compact. Recoverable failed sessions resume within a bounded window,
while accepted peers are replayed. The default single-session limit is 30
minutes; each proposal or policy call uses a recovery window of twice that
limit, starting before its first attempt. Public synthons and measured outcomes are
the only scientific inputs; the history projection is not an oracle.

Harness-Compiled LDM retains the four proposal sessions and adds one independent
policy session. The task-local policy feature encoder uses released reaction,
slot, capacity, and synthon-descriptor data. Its complete Python artifact may
set the residual-GP prior mean and LDM `alpha`/`eta`, while the Nyström/FITC
kernel, uncertainty, acquisition, pool, and official evaluator remain fixed.
Inspect `<run_dir>/policy_harness/` and the matrix
`compiled_policy_rounds.csv`; policy tool budgets are separate from proposal
tool budgets.

## NucleoBench

Files:

```text
tasks/nucleobench/README.md
tasks/nucleobench/QUICKSTART.md
config/nucleobench/mock.yaml
config/nucleobench/malinois_k562_tiny_campaign.yaml
config/pilot_evaluation/nucleobench.yaml
```

Direct and Harness methods use `LLM_BASE_URL`, `LLM_MODEL_NAME`, and
`LLM_API_KEY`. Direct methods can select Chat Completions or Responses; the
committed real profiles use Responses. Harness methods require the Pi sidecar,
Linux KVM, the built task guest, and external cache and run roots. An ignored
protected key file may be selected with `--api-key-file`.

NucleoBench exposes the shared `LDMEngine` through the source-pinned official
`SequenceOptimizer` lifecycle. Candidates are mutation patches relative to
one paired official start. Direct LDM makes four independent minibatch requests;
Harness LDM advances the selected persistent sequence-research roles (four by
default). `--harness-profile`, `--harness-candidates-per-session`, and
`--bo-pool-size` configure its topology independently of the evaluation batch.
Repeated profile roles create independent sessions with a shared instruction
template. `--harness-unique-candidates` enforces canonical uniqueness within
each session; cross-session occurrences still contribute to empirical `q0`.
Candidate roles expose the task-local `biopython`, `experimental-design`,
`scientific-critical-thinking`, and `statsmodels` Skills on demand. The guest
preinstalls their sequence-analysis, DOE, and statistical-modeling dependencies;
skill sources and licenses are recorded under the task's Harness resources.
Invalid and
historically evaluated patches are repaired before commit, while cross-lineage
agreement remains empirical `q0` mass. The task maintains a bounded pool (`3B`
by default), fits
its exact normalized-Hamming GP-UCB, applies the LDM acquisition tilt, and
batch-evaluates selected sequences through the official model wrapper.

Direct research Harness uses one persistent session and evaluates its accepted
minibatch without `q0`, GP, or acquisition. Harness-Compiled LDM adds one
independent `policy_architect` session whose task-local Skill may set the
residual-GP prior mean and LDM `alpha`/`eta`; the kernel, variance, UCB,
pool, candidate budget, and evaluator remain fixed.

The official wall-time profile runs one published start per campaign;
`start-index` and optimization `campaign-index` are independent. Wall-time
resume deducts previously used runtime and excludes repair downtime; it must be
reported as cumulative runtime, not an uninterrupted official run. Never reset
the eight-hour budget. The runner checks time
between rounds: report `result.json.wall_time_result` and
`trajectory.csv.benchmark_elapsed_seconds` for the strict time window, separately
from late measurements in the complete official output. Confirm actual hardware
when comparing against published results.

NucleoBench sends compact measurement indexes. Use `get_measured_history`
filters and ranking before requesting detailed patches and research notes for
selected IDs. Exact historical exclusion is checked by task tools and submission,
not by relying on an exhaustive list in model context. Keep full records on disk.

Follow `tasks/nucleobench/QUICKSTART.md` to validate the mock path, prepare
digest-pinned external starts and model artifacts, build and smoke the task
guest, run the qualified Malinois K562 tiny campaign, and inspect the
six-method three-seed Pilot Evaluation. Proposal traces are under
`<run_dir>/harness/`, compiled-policy traces under
`<run_dir>/policy_harness/`, and optimization state in the shared campaign
artifacts.
