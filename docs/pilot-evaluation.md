# Fixed-Budget Pilot Evaluation

The Pilot Evaluation pipeline runs a small, reproducible comparison before a
full benchmark campaign. A registered task may expose six methods:

| Method | Candidate source | Pre-evaluation selection |
| --- | --- | --- |
| `ldm` | Direct model API | Task LDM surrogate and acquisition |
| `ldm_harness` | Multiple persistent research Agents | Task LDM surrogate and acquisition |
| `ldm_harness_compiled` | Multiple persistent proposal Agents plus one policy Agent | Task residual GP and compiled LDM weights |
| `bo` | Task-local score-blind search space | Task BO surrogate and acquisition |
| `llm` | Direct model API | None; evaluate the requested minibatch |
| `harness` | One persistent research Agent | None; evaluate the requested minibatch |

The supplied matrices use three seeds. Custom matrices may choose any non-empty
set of distinct seeds and supported methods. Methods share the same task case,
initialization, optimization rounds, oracle, and real-evaluation count. The
task continues to own candidate identity, prompts, proposal refill, surrogate,
acquisition, and scientific dependencies.

## Module Boundary

```text
config/pilot_evaluation/<task>.yaml
  -> ldm_tts.pilot_evaluation.config       validate the task-neutral matrix
  -> ldm_tts.pilot_evaluation.execution    invoke standard task runner configs
  -> tasks/<task>/ldm_task/procedure.py execute each task-owned campaign
  -> ldm_tts.pilot_evaluation.reporting    validate and aggregate artifacts
```

The shared pipeline contains no benchmark-specific imports or algorithms. A
child campaign always runs through the task's normal registered procedure and
the shared LDM engine.

## Running A Matrix

Inspect resolved child plans before endpoint calls:

```bash
uv run --locked python scripts/run_pilot_evaluation.py \
  config/pilot_evaluation/iron_mind.yaml --dry-run
```

Run or resume the exact same repository and configuration revision:

```bash
uv run --locked python scripts/run_pilot_evaluation.py \
  config/pilot_evaluation/iron_mind.yaml

uv run --locked python scripts/run_pilot_evaluation.py \
  config/pilot_evaluation/iron_mind.yaml --resume
```

Use `--case`, `--method`, and `--seed` to select children. Reports are written
only after every configured child is complete.

For example, run only Harness-Compiled LDM for one seed:

```bash
uv run --locked python scripts/run_pilot_evaluation.py \
  config/pilot_evaluation/iron_mind.yaml \
  --method ldm_harness_compiled --seed 0
```

## Configuration Layout

Each matrix references one task base configuration. Its `method_overrides`
selects the experiment profile and method-specific arguments; the runner sets
the method, proposal backend, case, seed, rounds, and output path. Use this
matrix for both single-method runs and comparisons, without separate copies
of each method's configuration. Scientific profiles remain in the task's
`experiment.json`.

| Matrix | Task base config | Campaign rounds |
| --- | --- | --- |
| `iron_mind.yaml` | `iron_mind/pilot_evaluation_base.yaml` | 6 |
| `iron_mind_extended.yaml` | `iron_mind/pilot_evaluation_extended.yaml` | 12 |
| `synthonbench.yaml` | `synthonbench/pilot_evaluation_base.yaml` | 6 |
| `synthonbench_extended.yaml` | `synthonbench/pilot_evaluation_extended.yaml` | 12 |
| `nucleobench.yaml` | `nucleobench/malinois_k562_pilot_base.yaml` | 12 |

Matrix paths are relative to `config/pilot_evaluation/`; task base paths are
relative to `config/`. Round one is shared initialization. Mock, qualification,
Harness smoke, and full official benchmark configs stay under `config/<task>/`
because they serve distinct validation or benchmark budgets.

## Outputs And Integrity

The output root contains:

- `evaluation_manifest.json`: matrix identity, repository provenance, child
  status, integrity results, and safe Harness manifest references;
- `campaigns/<case>/<method>/seed_<n>/`: standard task artifacts;
- `summary.csv`: per-run score, trajectory AUC, budgets, and task metrics;
- `trajectories.csv`: normalized round-level best-so-far values;
- `summary.json`: aggregate statistics and method verdicts;
- `best_so_far.png`: mean best-so-far trajectories with formal method labels.
- `compiled_policy_rounds.csv`: policy actions, epochs, weights, validation,
  usage, degraded/fallback state, and selection diagnostics when the compiled
  method is present.

Integrity checks require complete rounds, shared initialization, unique real
evaluations, zero model calls from BO, and method-specific proposal budgets.
`ldm_harness` must execute one turn per profile and optimization round. Direct
`harness` must execute one turn per optimization round and submit exactly the
real-evaluation minibatch. `ldm_harness_compiled` must also attempt one policy
turn per optimization round and retain separate proposal and policy manifests.
`policy_harness_turns` counts these attempts, including runtime failures that
produce a recorded degraded decision. Reports distinguish committed turns from
failed attempts; a missing policy result still fails integrity validation.
Known failed-call usage is included in budgets. Unknown usage remains empty in
round reports, with `usage_complete=false` and `policy_usage_incomplete_rounds`
in the summary; aggregate usage counters then represent measured lower bounds.
The top-level manifest records their digests and selected non-secret
provenance.
Policy reports read `candidates_selected` events from the shared engine's
`events.jsonl`, not task-specific exports. Initialization candidates come from
checkpoint round zero. The expected evaluation count is that initial count
plus the optimization-round count times the configured evaluation batch size;
initialization need not have the same batch size as subsequent rounds.
Proposal and policy pools may use different providers, models, and reasoning
settings. Provenance checks require matching campaign, task, and seed identity,
not matching model configuration or a fixed case-name suffix.

Scientific policy diagnostics are configured by the task matrix:

```yaml
policy_fields:
  objective_diagnostics: compiled_policy.objective_diagnostics
  residual_means: base_selection.residual_target_mean
  rank_agreement: compiled_policy.rank_agreement
policy_mean_fields: [rank_agreement]
```

Paths are relative to selection metadata. The pipeline preserves mapped values;
arrays and objects are encoded as JSON in CSV cells. Absent diagnostics produce
empty cells. Only explicitly named scalar fields receive a
`policy_mean_<field>` summary. Mappings cannot overwrite lifecycle columns.
Objective direction, scaling, scalarization, and scientific diagnostics remain
task-owned. A multiobjective task selects an explicit progress metric (such as
hypervolume) for the existing scalar trajectory plot.

Direct model methods may yield fewer valid evaluations when their fixed request
budget produces malformed or repeated candidates. Harness methods instead use
in-session rejection and refill to deliver their complete accepted minibatch.

Automatic method verdicts compare against both `bo` and `llm`, and require a
strict majority of paired seeds to win. Matrices without both baselines still
produce scores and trajectories, but omit those verdicts.

## Registering Another Task

1. Implement the applicable method names in the normal task procedure. All
   methods must emit `trajectory.csv`, `result.json`, and standard runtime
   artifacts.
2. Add a real task base config under `config/<task>/`. Keep provider settings
   user-defined and enforce scientific profiles through `experiment.json`.
   A compiled method also needs a task-local feature and policy adapter,
   residual-GP path, independent policy profile, isolated artifact runner, and
   explicit policy budgets.
3. Add `config/pilot_evaluation/<task>.yaml` with cases, distinct seeds,
   method profiles, trajectory columns, and optional result fields. Do not add
   task-specific branches to `ldm_tts.pilot_evaluation`.
4. Add config and execution coverage, run `--dry-run`, and verify a task-local
   mock campaign for every registered method before real endpoint evaluation.

Iron Mind, SynthonBench, and NucleoBench are the reference matrices:

- `config/pilot_evaluation/iron_mind.yaml`
- `config/pilot_evaluation/synthonbench.yaml`
- `config/pilot_evaluation/nucleobench.yaml`
