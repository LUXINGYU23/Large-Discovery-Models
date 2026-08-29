# Fixed-Round Quick Comparison

The quick-comparison pipeline runs the same benchmark case and seed schedule
with three methods: LDM, task-local BO, and direct LLM sampling. It is intended
for a small, reproducible effectiveness check before a full benchmark run.

Every matrix uses exactly three seeds. The configured initialization is shared
across methods, while each task retains ownership of its candidate domain,
surrogate, acquisition rule, evaluator, prompts, and scientific dependencies.

## Module Boundary

```text
config/quick_compare/<task>.yaml
  -> ldm_tts.quick_compare.config       validate the task-neutral matrix
  -> ldm_tts.quick_compare.execution    invoke standard task runner configs
  -> tasks/<task>/ldm_task/procedure.py execute each task-owned campaign
  -> ldm_tts.quick_compare.reporting    verify and aggregate standard artifacts
```

The shared `ldm_tts.quick_compare` package contains no benchmark-specific
imports or algorithms. Task-specific method settings belong in the referenced
base config and in `method_overrides`. A child campaign still runs through the
normal task procedure and shared LDM engine; the comparison layer does not
implement a second optimization path.

## Running A Matrix

Inspect the fully resolved child plans before making endpoint calls:

```bash
uv run --locked python scripts/run_quick_compare.py \
  config/quick_compare/iron_mind.yaml --dry-run
```

Run a complete matrix or resume the exact same repository/config revision:

```bash
uv run --locked python scripts/run_quick_compare.py \
  config/quick_compare/iron_mind.yaml

uv run --locked python scripts/run_quick_compare.py \
  config/quick_compare/iron_mind.yaml --resume
```

Use `--case`, `--method`, or `--seed` to execute selected children. Reports are
written only after every configured case/method/seed child is complete.

## Outputs

The configured output root contains:

- `comparison_manifest.json`: matrix identity, repository provenance, child
  status, and integrity results.
- `campaigns/<case>/<method>/seed_<n>/`: unmodified standard task artifacts.
- `summary.csv`: per-run final score, trajectory AUC, budgets, and task metrics.
- `trajectories.csv`: normalized round-level best-so-far values.
- `summary.json`: aggregate statistics and the simple comparison verdict.
- `best_so_far.png`: mean best-so-far trajectories by case and method.

Integrity validation requires every child to complete the configured rounds,
shared initial candidates, unique evaluated candidates, no model calls from BO,
and the configured proposal counts for LDM and direct LLM. A round may contain
fewer successful evaluations when generated candidates are invalid, previously
evaluated, or duplicated. Reports retain the actual per-round and cumulative
evaluation counts, while round-AUC weights completed rounds equally.

## Registering Another Task

1. Implement `ldm`, `bo`, and `llm` as task-local `--search-method` values in
   the normal registered procedure. All methods must emit the standard runtime
   artifacts, including `trajectory.csv` and `result.json`.
2. Add a real base runner config under `config/<task>/`. Keep provider settings
   user-defined and lock scientific method arguments in `experiment.json`.
3. Add `config/quick_compare/<task>.yaml` with benchmark cases, exactly three
   seeds, method overrides, trajectory columns, and optional result-field
   mappings. Do not add task-specific branches to `ldm_tts.quick_compare`.
4. Add config validation coverage, run `--dry-run`, and verify one task-local
   mock campaign for every method before starting real endpoint evaluations.

Iron Mind and SynthonBench are the reference integrations:

- `config/quick_compare/iron_mind.yaml`
- `config/quick_compare/synthonbench.yaml`
