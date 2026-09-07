# Quickstart

See [README.md](README.md) for scientific protocol, data provenance, environment setup and real launch commands.

From the repository root:

```bash
uv sync --locked --project tasks/atomworld --group dev
uv run --locked --project tasks/atomworld python -m pytest -q tasks/atomworld/tests
uv run --locked --project tasks/atomworld python scripts/check_task_dependencies.py config/atomworld/mock.yaml --no-optional
uv run --locked --project tasks/atomworld python scripts/run_ldm_tts.py config/atomworld/mock.yaml --dry-run
uv run --locked --project tasks/atomworld python scripts/run_ldm_tts.py config/atomworld/mock.yaml
```

The mock intentionally changes from an incorrect first answer to a correct revision; it proves shared lifecycle, metric separation and collection, and is not a model score.
