# synthonbench

Closed-loop optimization over a source-pinned synthon space and score tables.

## Mock Run

From the repository root:

```bash
uv sync --project tasks/synthonbench --group dev
uv run --project tasks/synthonbench \
  python scripts/run_ldm_tts.py config/synthonbench/mock.yaml
```

The mock path already exercises the shared `LDMEngine`. Replace the generated
candidate-domain admission adapter, reservoir expander, evaluator, surrogate
representation, objective, and response contract before adding a real-run
config. Complete `experiment.json`
with source-pinned benchmark provenance, metric roles, evaluator limits, and a
named campaign profile. Keep `qualification` set to `draft` until a real seed
and tiny LDM-selected evaluation pass.
