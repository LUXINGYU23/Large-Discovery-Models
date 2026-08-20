# Ready-to-Run LDM Examples with Delta-Infra

This directory contains reproducible runbooks and artifacts from running Large
Discovery Model (LDM) campaigns on **Delta-Infra**. Start here when you want to
run an existing small-molecule, antibody, AI4Bio, or discrete causal-discovery
campaign, or use an existing run as a reference for registering a new LDM task.

## What is Delta-Infra?

[Delta-Infra](https://delta-infra-dashboard.yangtzeailab.com/) is the
infrastructure layer used by these examples. It gives local AI agents access
to isolated cloud CPU/GPU sandboxes, shared data, model endpoints, and
scientific tools without requiring those resources to run on the local
machine. Its main client is `delta-cli`, which manages authentication,
sandbox lifecycles, file transfer, remote commands, and scientific service
calls.

The examples use the infrastructure in two complementary ways:

- `delta-cli sandbox` allocates isolated compute, transfers code, and runs
  model inference or other long-running jobs.
- `delta-cli science` invokes managed scientific evaluators when a workflow
  requires one.

## Install Delta-Infra

The [Delta-Infra quickstart](https://delta-infra-dashboard.yangtzeailab.com/docs/quickstart/account)
recommends the interactive installer below. It installs the CLI globally,
deploys the `delta-*` AI-agent skills, initializes configuration, and guides
you through authentication:

```bash
npx @delta-infra/cli@latest install
delta-cli --version
```

This method requires Node.js and `npx`. The quickstart also documents these
alternative installation methods:

```bash
# Install from npm.
npm install -g @delta-infra/cli

# Or use the installation script.
curl -L https://raw.githubusercontent.com/yzailab/delta-infra-cli/main/install.sh | bash
```

Prebuilt binaries are available from the
[delta-infra-cli releases page](https://github.com/yzailab/delta-infra-cli/releases).

### Authenticate

Obtain a Delta-Infra Bearer token from the platform console or your
Delta-Infra administrator, then log in and verify the configuration:

```bash
delta-cli auth login --token <your-token>
delta-cli auth status
delta-cli config show
```

Do not commit, print, or share the token. `config show` displays a redacted
configuration and is the preferred diagnostic command.

To check for or install later CLI and skill updates:

```bash
delta-cli upgrade --check
delta-cli upgrade
```

### Verify sandbox access

The following quickstart command creates a GPU sandbox with a two-hour maximum
lifetime. Resource allocation may consume account credits, so skip this smoke
test if you only need to read the runbooks.

```bash
delta-cli sandbox create \
  --image image.yangtzeailab.com/opensandbox/pytorch-cuda13:latest \
  --cpu 4 \
  --memory 16Gi \
  --gpu 1 \
  --max-life 120
```

Record the `sandbox_id` from the JSON response. You can then run a command and
release the sandbox when finished:

```bash
delta-cli sandbox run <sandbox_id> \
  --command "python -c 'import torch; print(torch.cuda.is_available())'" \
  --timeout 300

delta-cli sandbox kill <sandbox_id>
```

## Choose an Example

| Goal | Start here | What the recorded run demonstrates |
| --- | --- | --- |
| Run a small-molecule campaign | [Small-molecule workflow](./run_small_molecule_w_delta_infra/DELTA_CLI_WORKFLOW.md) | Real Qwen inference, AutoDock Vina scoring against KRAS G12D, activity prediction, and EHVI search. The included result is a deliberately stopped partial 30/100 campaign. |
| Run an antibody campaign | [Antibody workflow](./run_antibody_w_delta_infra/DELTA_CLI_WORKFLOW.md) | Real Qwen CDRH3 proposals and 20 managed AntBO/Absolut evaluations for antigen `1ADQ_A`. |
| Register and run a custom task | [Task-registration workflow](run_customized_llm_kv_adaptive_quantization/TASK_REGISTRATION_WORKFLOW.md) | Manifest-based registration and a 20-iteration diagnostic campaign for adaptive LLM KV-cache quantization. The recorded campaign is non-official and the task remains `draft`. |
| Register, qualify, and run the AI4Bio mutation-effect task | [AI4Bio registration and Delta workflow](run_customized_ai4bio_mutation_effect_prediction/REGISTER_AND_DELTA_WORKFLOW.md) | Source-pinned MLS-Bench registration, staged qualification, official three-assay evaluation, and a separately labeled 20-iteration extended-budget campaign with CSV/PDF/PNG progress artifacts. |
| Register, qualify, and run discrete causal discovery | [Causal-discovery quickstart](../tasks/causal_discovery_discrete/QUICKSTART.md) and [recorded artifacts](./run_customized_causal_discovery_discrete/) | Source-pinned MLS-Bench registration, staged qualification, and a 20-iteration extended-budget GP-UCB campaign. All 20 candidates completed the five-network evaluator for 100 benchmark jobs; the best official score was `0.02766568667561009`, first reached at iteration 6. |

Each linked workflow or task guide documents its exact configuration, commands,
validation gates, failure recovery, evidence boundary, and generated artifacts.
Read the chosen guide before allocating resources: its image, CPU/GPU requirements,
timeouts, model paths, and evaluator dependencies are more specific than the
generic smoke test above.

Run documented repository commands from the repository root unless a workflow
explicitly says otherwise:

```bash
cd /path/to/large-discovery-models
delta-cli config show
delta-cli auth status
```

The checked-in JSON, CSV, and PNG files are evidence from the recorded runs;
they are not newly generated by installing Delta-Infra. New campaigns should
write to a separate output directory so the reference artifacts remain
unchanged.

## Further Reading

- [Delta-Infra documentation](https://delta-infra-dashboard.yangtzeailab.com/docs/quickstart/account)
- [Delta-Infra CLI reference](https://delta-infra-dashboard.yangtzeailab.com/docs/cli)
- [Delta-Infra agent integration](https://delta-infra-dashboard.yangtzeailab.com/docs/agents)
- [Main LDM README](../README.md)
