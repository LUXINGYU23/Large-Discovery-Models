# NucleoBench LDM Task

This task adapts the repository's shared LDM workflow to the official
[NucleoBench](https://github.com/move37-labs/nucleobench) sequence-design
interface. The upstream source is pinned to commit
`a6d8b040a4fa80b18266ee5416c904d01f842428` (`nucleobench==2.0.10`).

The adapter does not modify official model wrappers, energy functions,
editable-position masks, start sequences, runner termination, or result
serialization. LDM is exposed as a NucleoBench `SequenceOptimizer`, while
the source-pinned `docker_entrypoint.run_loop` remains the outer benchmark
driver.

## Current Status

The experiment contract and `malinois_k562` case are qualified through the
`tiny_campaign_verified` gate. Qualification used the source-pinned official
start set and Malinois model for a seed evaluation, a direct LDM tiny campaign,
a four-session LDM Harness tiny campaign, a one-session direct Harness tiny
campaign, and a no-duplicate-evaluation resume check. The other 16 public cases
remain `planned`.

The same task package will be extended to the remaining cases without
duplicating the workflow or optimization methods. No 12-round Pilot Evaluation
or official wall-time result is included in this qualification claim.

## Case Families

| Family | Cases | Sequence length | Editable positions | Official time limit |
| --- | ---: | ---: | ---: | ---: |
| Malinois | 3 | 200 | 200 | 8 hours |
| BPNet-lite | 12 | 3,000 | 3,000 | 8 hours |
| RiNALMo MRL | 1 | 100 | 100 | 8 hours |
| Enformer | 1 | 196,608 | 256 | 12 hours |

The catalog is stored in
[`resources/cases/catalog.json`](resources/cases/catalog.json). Its state is a
release claim: a case may only move from `planned` after the corresponding
official artifacts and validation evidence exist.

## Inspect the Registered Contract

From the repository root:

```bash
uv sync --locked --project tasks/nucleobench
uv run --locked --project tasks/nucleobench \
  python scripts/validate_tasks.py --task nucleobench \
  --require-qualified --require-stage tiny_campaign_verified
uv run --locked --project tasks/nucleobench \
  python -m tasks.nucleobench.ldm_task.procedure \
  --case-id malinois_k562 --dry-run
```

Real official-oracle runs require Linux and the pinned upstream runtime:

```bash
uv sync --locked --project tasks/nucleobench --extra official
```

The released Malinois configs hide CUDA and use the official CPU reference
path, including on accelerator-equipped hosts.

For a local Docker daemon on POSIX, Harness containers run as the invoking
UID:GID and receive the host KVM group. This keeps session traces readable by
the user while retaining KVM access. Use `--harness-container-user` when a
remote daemon or a custom identity requires an explicit value.

The dry run is inspection-only. Executable mock and official workflows are not
equivalent: the mock verifies repository integration but is not a benchmark
result.

## Run the Deterministic Mock

```bash
uv run --locked --project tasks/nucleobench \
  python scripts/run_ldm_tts.py config/nucleobench/mock.yaml
```

The run uses the shared `LDMEngine`, candidate admission, batch evaluator,
budget ledger, checkpoint, event log, result export, trajectory export, and
data-collection sink without network, model weights, or GPU access.

## Prepare Malinois K562 Inputs

Preparation validates a clean checkout at the pinned revision, the official
Zenodo start table or an equivalent JSON array, the editable positions, and a
local copy of the official model artifact. It writes only small prepared files
and a digest-bound manifest to an external directory.

```bash
uv run --locked --project tasks/nucleobench \
  python -m tasks.nucleobench.scripts.prepare_official_data \
  --case malinois_k562 \
  --source-dir /external/nucleobench/source \
  --output-dir /external/nucleobench/data/malinois_k562 \
  --starts-file /external/nucleobench/inputs/start_sequences_df.csv \
  --model-artifact /external/nucleobench/models/malinois_artifacts.tar.gz \
  --model-sha256 06e926e42304b8207138f1fb871ec19e0654dcdb6b26a62ed23fe1e9ac8cc592 \
  --start-set-sha256 2b76fcdeaa0821b94cca642531145d6ee82aada4d378156d027ef92833ef33f5 \
  --bending-factor 1.0
```

The official start table is published as `start_sequences_df.csv` in Zenodo
record `17079936`. The model may be obtained from the official URI recorded in
[`resources/upstream_contract.json`](resources/upstream_contract.json) or from
a user-selected mirror. The script never stores weights or starts in Git. A
prepared manifest is an input-integrity record and is required before any real
campaign.

## Execution Profiles

The contract separates two drivers that share the same candidate, surrogate,
proposal, and selection implementation:

- `pilot_evaluation`: 12 total rounds for fast three-seed method comparison.
- `official_benchmark`: the unchanged official runner terminates by wall time,
  with no method-level round cap.

Pilot results are development diagnostics and are not official benchmark
results. See [`QUICKSTART.md`](QUICKSTART.md) for the current validation path
and [`resources/upstream_contract.json`](resources/upstream_contract.json) for
the pinned protocol boundary.
