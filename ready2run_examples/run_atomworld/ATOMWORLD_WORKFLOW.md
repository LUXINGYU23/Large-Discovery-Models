# Minimal AtomWorld Campaign Runbook

## Purpose

Run the existing AtomWorld task on a small released-data subset with **one
question from each of the ten supported actions and one direct CIF answer per
question**. A model reads the public instruction and source structure; the
pinned official AtomWorld evaluator scores its final structure against private
target data.

- [Existing one-shot configuration](../../config/atomworld/one_shot.yaml)
- [Full task setup and scientific contract](../../tasks/atomworld/README.md)

## Evidence Boundary

This is a real, seed-0 **integration example**, not a reproduction of the
paper's test split or a paper-scale benchmark. The prepared ten-question subset
contains one released row per action. It must be reported as
`local_released_subset`, not as a verified paper subset.

The workflow uses an existing model endpoint and a host-local checkout of the
official AtomWorldBench release. It does not require Delta-Infra, a Harness
sidecar, Docker/KVM, bounded geometry tools, or a compiled policy. The model
sees only public questions and source CIFs; private targets remain in the task
runner and are used only by the evaluator.

Runtime results are deliberately not checked in with this runbook.

## Configuration

| Setting | Value |
| --- | --- |
| Dataset | Released AtomWorldBench data, one row per supported action |
| Questions / seed | 10 / 0 |
| Proposal model | `DeepSeek-V4-Flash` in the reference run |
| Method | `direct_reasoning` / one-shot CIF response |
| Scheduled answers | 1 per question, 10 total |
| Maximum output | 8,192 tokens |
| Temperature | 0 |
| Evaluator | Source-pinned official AtomWorld evaluator |

The ten actions are change, remove, add, move, move-towards, insert-between,
swap, delete-below, rotate-around, and super-cell. Every question remains in
the denominator even when the response is malformed or evaluation fails.

## Prepare the Released Subset

The host must have the official AtomWorldBench release identified by
`tasks/atomworld/resources/source_manifest.json`. Choose a new directory for
the prepared data; preparation refuses to overwrite an existing manifest.

From the repository root:

```bash
export ATOMWORLD_UPSTREAM_ROOT=/absolute/path/to/AtomWorldBench
export ATOMWORLD_DATA_ROOT=/absolute/path/to/prepared-atomworld-example

uv sync --locked --project tasks/atomworld --group dev
uv run --locked --project tasks/atomworld \
  python -m tasks.atomworld.scripts.prepare_official_data \
  --source-dir "$ATOMWORLD_UPSTREAM_ROOT/src/data" \
  --out-dir "$ATOMWORLD_DATA_ROOT" \
  --per-action 1 \
  --seed 0
```

The command should report `sample_count: 10`. The prepared directory contains
public and private files; keep `private.jsonl` and all evaluator output away
from model or Harness processes and do not commit them.

## Run

Configure an OpenAI-compatible Chat Completions endpoint. The values below
match the reference setup; adapt the URL and credentials for another host.

```bash
export LLM_BASE_URL=http://127.0.0.1:52306/v1
export LLM_MODEL_NAME=DeepSeek-V4-Flash
export LLM_API_KEY=EMPTY

example_args=(config/atomworld/one_shot.yaml
  --set name=atomworld_ready2run_tiny
  --set args.out-dir=runs/atomworld_ready2run_tiny)

uv run --locked --project tasks/atomworld \
  python scripts/validate_tasks.py --task atomworld
uv run --locked --project tasks/atomworld \
  python scripts/check_task_dependencies.py "${example_args[@]}"
uv run --locked --project tasks/atomworld \
  python scripts/run_ldm_tts.py "${example_args[@]}" --dry-run
uv run --locked --project tasks/atomworld \
  python scripts/run_ldm_tts.py "${example_args[@]}"
```

`EMPTY` is appropriate only for a local service that does not authenticate.
Keep real credentials in environment variables. Use a fresh output directory
for every experiment instead of resuming with changed data, model, or settings.

## Verify Outputs

The printed run directory contains `status.json`, `result.json`, `summary.json`,
`budget.json`, `trajectory.csv`, `attempts/`, and private `evaluations/`.
Confirm that:

- `status.json` reports `status=completed`;
- the budget records ten scheduled and ten successful evaluations;
- `result.json` contains ten samples and one submitted attempt per sample; and
- action-level and overall accuracy use the final scheduled answer without
  best-of-attempt selection.

The official distance fields are normalized, dimensionless values and must not
be labeled as ångströms without a separate justified conversion. A ten-question
result is an integration check, not an estimate of full-dataset accuracy.
