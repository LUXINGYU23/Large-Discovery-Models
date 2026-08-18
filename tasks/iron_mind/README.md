# iron_mind

Closed-loop optimization over source-pinned reaction-condition tables.

## Mock Run

From the repository root:

```bash
uv sync --project tasks/iron_mind --group dev
uv run --project tasks/iron_mind \
  python scripts/run_ldm_tts.py config/iron_mind/mock.yaml
```

The mock path already exercises the shared `LDMEngine`. Replace the generated
candidate-domain admission adapter, reservoir expander, evaluator, surrogate
representation, objective, and response contract before adding a real-run
config. Complete `experiment.json`
with source-pinned benchmark provenance, metric roles, evaluator limits, and a
named campaign profile. Keep `qualification` set to `draft` until a real seed
and tiny LDM-selected evaluation pass.
