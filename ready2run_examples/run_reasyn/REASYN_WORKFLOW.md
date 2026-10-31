# Minimal ReaSyn Campaign Runbook

## Purpose

Run the existing ReaSyn task with **one Enamine target and two real projection
trials**. DeepSeek-V4-Flash proposes molecular SMILES; LDM/GP-UCB selects queries;
the frozen ReaSyn AR/EB models generate synthesis pathways for evaluation.

- [Existing tiny configuration](../../config/reasyn/reconstruction_tiny.yaml)
- [Full task setup and scientific contract](../../tasks/reasyn/README.md)

## Evidence Boundary

This is a real, seed-0 **tiny integration example**, not a reproduction of the
paper's 1,000-target, multi-seed reconstruction table. It uses the existing GPU
host over SSH, not Delta-Infra. No Docker/KVM, Harness, compiled policy, new model
deployment, or model training is needed. LDM here is the repository's search
workflow using DeepSeek, not a separately fine-tuned LDM checkpoint.

## Configuration

| Setting | Value |
| --- | --- |
| Benchmark / dataset | Reconstruction / Enamine |
| Proposal model | `DeepSeek-V4-Flash` |
| Search method | `ldm` (Tanimoto GP/UCB tilted sampling) |
| Targets / seed | 1 / 0 |
| Rounds / selected evaluations per round | 2 / 1 |
| Valid proposal occurrences per round | 4 |
| Frozen projector | Released ReaSyn AR-166M + EB-174M |
| Width / exhaustiveness | 8 / 4 |
| Cycles per trial / EB samples per cycle | 1 / 4 |

## Run

The host must already have the source-pinned upstream ReaSyn checkout, both
checkpoints, the converted `comp_2048` fingerprint/reaction indices, the Enamine
target file, and a working GPU projector environment. These are external assets,
not bundled with this small example; see the full task guide above for setup.

Set `LDM_SSH_PORT` and `LDM_SSH_TARGET` on your local machine and connect:

```bash
ssh -p "${LDM_SSH_PORT:?}" "${LDM_SSH_TARGET:?}"
```

On the server, set `LDM_PROJECT_ROOT`, `REASYN_ROOT`, `LLM_BASE_URL`,
`LLM_API_KEY`, and `CUDA_VISIBLE_DEVICES` for that environment, then run:

```bash
cd "${LDM_PROJECT_ROOT:?}"
: "${REASYN_ROOT:?}"
: "${LLM_BASE_URL:?}"
: "${LLM_API_KEY:?}"
: "${CUDA_VISIBLE_DEVICES:?}"

export REASYN_PYTHON="$REASYN_ROOT/.venv/bin/python"
export LLM_MODEL_NAME=DeepSeek-V4-Flash

uv sync --locked --project tasks/reasyn --group dev --extra chemistry

example_args=(config/reasyn/reconstruction_tiny.yaml
  --set name=reasyn_ready2run_tiny
  --set args.dataset=enamine
  --set args.targets-file=synformer/data/enamine_smiles_1k.txt
  --set args.out-dir=runs/reasyn_ready2run_tiny)

tasks/reasyn/.venv/bin/python scripts/validate_tasks.py --task reasyn
tasks/reasyn/.venv/bin/python scripts/check_task_dependencies.py "${example_args[@]}"
tasks/reasyn/.venv/bin/python scripts/run_ldm_tts.py "${example_args[@]}" --dry-run
tasks/reasyn/.venv/bin/python scripts/run_ldm_tts.py "${example_args[@]}"
```

`LLM_API_KEY=EMPTY` is only appropriate for a local service without
authentication. Keep real credentials in environment variables. Choose a fresh
output directory for each new experiment. Runtime result files are deliberately
not checked in with this runbook.

The dependency report can warn that Torch is absent from the lightweight runner:
the real projector uses the separate `REASYN_PYTHON` environment. A `draft`
qualification warning is expected for this tiny run, not a paper certification.

## Outputs

The printed run directory contains `status.json`, `result.json`,
`benchmark_result.json`, `budget.json`, `trajectory.csv`, proposal traces and
`projections/` with requests, GPU worker logs, and replay-verified pathways.
Check `status=completed` and two successful evaluations before reading metrics.

Reconstruction rate, best target similarity, analogous-product diversity and
building-block diversity are separate metrics. A single-target success must not
be presented as an estimate of full-dataset performance. Model requests and
projector evaluations are real and consume server resources.
