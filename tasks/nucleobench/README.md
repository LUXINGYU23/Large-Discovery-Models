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

## Usage and Isolation

[`QUICKSTART.md`](QUICKSTART.md) is the single operational guide for installing
the task environment, validating registration, preparing external inputs, and
running mock or real campaigns.

The released Malinois configs use the official CPU reference path. Harness
sessions receive the paired-start design context rather than official case or
model identity, and a task-local network policy blocks the registered upstream
source and model-artifact hosts while leaving public-literature research
available. For a local Docker daemon on POSIX, the Harness uses the invoking
UID:GID and host KVM group; `--harness-container-user` overrides that identity.

Mock runs exercise repository integration without network access, model
weights, or official benchmark claims. Official source checkouts, model
artifacts, prepared inputs, caches, traces, and run outputs remain outside Git.

## Execution Profiles

The contract separates two drivers that share the same candidate, surrogate,
proposal, and selection implementation:

- `pilot_evaluation`: 12 total rounds for fast three-seed method comparison.
- `official_benchmark`: the unchanged official runner terminates by wall time,
  with no method-level round cap.

Pilot results are development diagnostics and are not official benchmark
results. The pinned protocol boundary is recorded in
[`resources/upstream_contract.json`](resources/upstream_contract.json).
