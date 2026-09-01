# NucleoBench LDM Task

This task adapts the repository's shared LDM workflow to the official
[NucleoBench](https://github.com/move37-labs/nucleobench) sequence-design
interface. The upstream source is pinned to commit
`a6d8b040a4fa80b18266ee5416c904d01f842428` (`nucleobench==2.0.10`).

The adapter does not modify official model wrappers, energy functions,
editable-position masks, start sequences, runner termination, or result
serialization. LDM will be exposed as a NucleoBench `SequenceOptimizer`, while
the source-pinned `docker_entrypoint.run_loop` remains the outer benchmark
driver.

## Current Status

The task is registered with a draft experiment contract. Its versioned case
catalog covers the 17 public NucleoBench tasks, all currently marked
`planned`. The mutation-patch candidate contract and external data preparation
command are implemented; executable campaigns are not yet qualified.

The first implementation target is `malinois_k562`. After that case passes
source preparation, seed evaluation, a tiny campaign, and qualification, the
same task package will be extended to the remaining cases without duplicating
the workflow or optimization methods.

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
  python scripts/validate_tasks.py --task nucleobench
uv run --locked --project tasks/nucleobench \
  python -m tasks.nucleobench.ldm_task.procedure \
  --case-id malinois_k562 --dry-run
```

The dry run is inspection-only. Executable mock and official workflows are not
available until their corresponding qualification stages are implemented.

## Prepare Malinois K562 Inputs

Preparation validates a clean checkout at the pinned revision, a JSON array of
100 unique 200-base starts, the editable positions, and a local copy of the
official model artifact. It writes only small prepared files and a digest-bound
manifest to an external directory.

```bash
uv run --locked --project tasks/nucleobench \
  python -m tasks.nucleobench.scripts.prepare_official_data \
  --case malinois_k562 \
  --source-dir /external/nucleobench/source \
  --output-dir /external/nucleobench/data/malinois_k562 \
  --starts-file /external/nucleobench/inputs/malinois_k562_starts.json \
  --model-artifact /external/nucleobench/models/malinois_artifacts.tar.gz \
  --model-sha256 <verified-sha256> \
  --start-set-sha256 <verified-canonical-start-set-sha256> \
  --bending-factor <verified-official-value>
```

The model may be obtained from the official URI recorded in
[`resources/upstream_contract.json`](resources/upstream_contract.json) or from
a user-selected mirror. The script never stores weights in Git. A prepared
manifest is an input-integrity record; the case remains `planned` until its
authoritative starts and complete model arguments pass qualification.

## Execution Profiles

The draft contract separates two drivers that will share the same candidate,
surrogate, proposal, and selection implementation:

- `pilot_evaluation`: 12 total rounds for fast three-seed method comparison.
- `official_benchmark`: the unchanged official runner terminates by wall time,
  with no method-level round cap.

Pilot results are development diagnostics and are not official benchmark
results. See [`QUICKSTART.md`](QUICKSTART.md) for the current validation path
and [`resources/upstream_contract.json`](resources/upstream_contract.json) for
the pinned protocol boundary.
