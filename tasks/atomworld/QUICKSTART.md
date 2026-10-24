# Quickstart

LDM Harness defaults to **MainHistoryForks 1+3**. Use `--harness-research-strategy-config` with `config/research_strategies/` to select another strategy; see the [shared tree guide](../../docs/research-search-architecture.md) for configuration, budgets and recovery.

See [README.md](README.md) for the scientific protocol, official assets, Linux Harness setup and resume contract.

```bash
uv sync --locked --project tasks/atomworld --group dev
uv run --locked --project tasks/atomworld python -m pytest -q tasks/atomworld/tests
uv run --locked --project tasks/atomworld python scripts/run_ldm_tts.py config/atomworld/mock.yaml
```

The mock checks lifecycle and fixed final-answer reporting. It is not a model benchmark score.

After preparing official data and the provider environment, use `config/atomworld/one_shot.yaml`, `extended_reasoning.yaml`, or `extended_operations.yaml` for blind baselines. For measured-feedback LDM, use `ldm.yaml`, `ldm_harness.yaml` or `ldm_harness_compiled.yaml`. Harness methods require the shared sidecar and task guest on Linux/KVM and `ATOMWORLD_HARNESS_CACHE`. Compiled LDM uses measured scalar correctness, a fixed GP and both prior/weight capabilities; it never exposes private target structures.
