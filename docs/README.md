# Documentation

Start with the [project README](../README.md) for installation, supported tasks,
and runnable examples. The documents here cover the repository's deeper
technical and operational contracts.

## Core Guides

- [LDM discovery concepts](concepts.md): canonical task-neutral terminology
  for candidates, reservoirs, evaluation, acquisition, and campaigns.
- [Testing and coverage](testing.md): isolated environments, test lanes, and
  coverage thresholds.
- [Data collection](data-collection.md): the shared `ldm-2.0` collection,
  augmentation, rendering, and validation workflow.
- [Agent execution](agent-execution.md): machine-oriented rules for safely
  inspecting, testing, and running the repository.

## Task Guides

- [nanoGPT](../tasks/nanogpt/README.md)
- [Small molecule](../tasks/small_molecule/README.md)
- [Antibody](../tasks/antibody/README.md)
- [Adaptive KV-cache quantization](../tasks/llm_kv_adaptive_quantization/README.md)
- [AI4Bio mutation-effect prediction](../tasks/ai4bio_mutation_effect_prediction/README.md)
- [Discrete causal discovery](../tasks/causal_discovery_discrete/README.md)
- [Task registration](../tasks/README.md)

## Runbooks And Agent Skills

- [Ready-to-run Delta-Infra examples](../ready2run_examples/README.md)
- [Agent skills](../skills/README.md): `collect-ldm-data`, `register-ldm-task`,
  and `run-ldm-task`.

## Project Policies

- [Contributing](../CONTRIBUTING.md)
- [Security](../SECURITY.md)
- [Code of Conduct](../CODE_OF_CONDUCT.md)
- [Changelog](../CHANGELOG.md)
