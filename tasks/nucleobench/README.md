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

One workflow now supports all four official model families. The two remaining
Malinois cases, all 12 BPNet cases, and Enformer have digest-pinned public start
and model artifacts and are marked `prepared`. `malinois_k562` remains the only
case qualified through a real seed evaluation and tiny campaigns. Prepared
cases must pass those same gates before Pilot Evaluation or official execution.

The source-pinned RiNALMo oracle loader is implemented, but the published
NucleoBench artifacts do not define its 100 paired starts. That case therefore
remains `planned`. No 12-round Pilot Evaluation or official wall-time result is
included in these status claims.

## Case Families

| Family | Cases | Sequence length | Editable positions | Official time limit |
| --- | ---: | ---: | ---: | ---: |
| Malinois | 3 | 200 | 200 | 8 hours |
| BPNet-lite | 12 | 3,000 | 3,000 | 8 hours |
| RiNALMo MRL | 1 | 100 | 100 | 8 hours |
| Enformer | 1 | 196,608 | 256 | 12 hours |

The catalog is stored in
[`resources/cases/catalog.json`](resources/cases/catalog.json). Its state is a
release claim: `prepared` means that the source-pinned inputs and loader are
defined, while `qualified` additionally requires real oracle and tiny-campaign
evidence.

## Usage and Isolation

[`QUICKSTART.md`](QUICKSTART.md) is the single operational guide for installing
the task environment, validating registration, preparing external inputs, and
running mock or real campaigns.

The released K562 configs use the official CPU reference path. Other cases may
use the device behavior of their unchanged official model wrapper. Harness
sessions receive the paired-start design context rather than official case or
model identity, and a task-local network policy blocks the registered upstream
source and model-artifact hosts while leaving public-literature research
available. For a local Docker daemon on POSIX, the Harness uses the invoking
UID:GID and host KVM group; `--harness-container-user` overrides that identity.

Mock runs exercise repository integration without network access, model
weights, or official benchmark claims. Official source checkouts, model
artifacts, prepared inputs, caches, traces, and run outputs remain outside Git.

## Execution Profiles

The contract separates two case-independent profiles that share the same
candidate, surrogate, proposal, and selection implementation:

- `pilot_evaluation`: 12 total rounds for fast three-seed method comparison.
- `official_benchmark`: the unchanged official runner terminates by wall time,
  with no method-level round cap.

Pilot results are development diagnostics and are not official benchmark
results. The pinned protocol boundary is recorded in
[`resources/upstream_contract.json`](resources/upstream_contract.json).
