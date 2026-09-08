# Quickstart

See [README.md](README.md) for the scientific protocol, official assets, Linux Harness setup and resume contract.

```bash
uv sync --locked --project tasks/atomworld --group dev
uv run --locked --project tasks/atomworld python -m pytest -q tasks/atomworld/tests
uv run --locked --project tasks/atomworld python scripts/run_ldm_tts.py config/atomworld/mock.yaml
```

The mock checks lifecycle and fixed final-answer reporting. It is not a model benchmark score.

After preparing official data and the provider environment, use `config/atomworld/one_shot.yaml`, `extended_reasoning.yaml`, or `extended_operations.yaml`. Bounded geometry code ships with the task. For persistent agent research, build the shared sidecar and task guest, set `ATOMWORLD_HARNESS_CACHE`, then use `harness.yaml` or `blind_harness_compiled.yaml`.
