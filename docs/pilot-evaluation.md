# Fixed-Budget Pilot Evaluation

The Pilot Evaluation pipeline runs a small, reproducible comparison before a
full benchmark campaign. A registered task may expose five methods:

| Method | Candidate source | Pre-evaluation selection |
| --- | --- | --- |
| `ldm` | Direct model API | Task LDM surrogate and acquisition |
| `ldm_harness` | Multiple persistent research Agents | Task LDM surrogate and acquisition |
| `bo` | Task-local score-blind search space | Task BO surrogate and acquisition |
| `llm` | Direct model API | None; evaluate the requested minibatch |
| `harness` | One persistent research Agent | None; evaluate the requested minibatch |

Every committed matrix uses three seeds. Methods share the same task case,
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

## Outputs And Integrity

The output root contains:

- `evaluation_manifest.json`: matrix identity, repository provenance, child
  status, integrity results, and safe Harness manifest references;
- `campaigns/<case>/<method>/seed_<n>/`: standard task artifacts;
- `summary.csv`: per-run score, trajectory AUC, budgets, and task metrics;
- `trajectories.csv`: normalized round-level best-so-far values;
- `summary.json`: aggregate statistics and method verdicts;
- `best_so_far.png`: mean best-so-far trajectories with formal method labels.

Integrity checks require complete rounds, shared initialization, unique real
evaluations, zero model calls from BO, and method-specific proposal budgets.
`ldm_harness` must execute one turn per profile and optimization round. Direct
`harness` must execute one turn per optimization round and submit exactly the
real-evaluation minibatch. Harness children must retain a sidecar manifest; the
top-level manifest records its digest and selected non-secret provenance.

Direct model methods may yield fewer valid evaluations when their fixed request
budget produces malformed or repeated candidates. Harness methods instead use
in-session rejection and refill to deliver their complete accepted minibatch.

## Registering Another Task

1. Implement the applicable method names in the normal task procedure. All
   methods must emit `trajectory.csv`, `result.json`, and standard runtime
   artifacts.
2. Add real task profiles under `config/<task>/`. Keep provider settings
   user-defined and enforce scientific settings through `experiment.json`.
3. Add `config/pilot_evaluation/<task>.yaml` with cases, exactly three seeds,
   method profiles, trajectory columns, and optional result fields. Do not add
   task-specific branches to `ldm_tts.pilot_evaluation`.
4. Add config and execution coverage, run `--dry-run`, and verify a task-local
   mock campaign for every registered method before real endpoint evaluation.

Iron Mind and SynthonBench are the reference matrices:

- `config/pilot_evaluation/iron_mind.yaml`
- `config/pilot_evaluation/synthonbench.yaml`
